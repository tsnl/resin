//! Compile Resin programs and retain their analysis for editor queries.
//!
//! `Session` owns source updates and caches. `Compilation` is an immutable result
//! for one entry and its imports; it remains valid after subsequent session edits.
//! Language crates own their trees and transformations. This crate sequences them.
//! Loading, cache state, and retained implementation data remain private.
//!
//! ```compile_fail,E0603
//! use resin_compiler::{loading, session, compilation};
//! ```

use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

mod build;
mod compilation;
mod error;
mod loading;
mod passes;
mod queries;
mod request;
mod session;
mod shaders;
mod source;
mod syntax;

pub use resin_common::source::{SourceError, SourceLocation, SourceNote};
pub use resin_hir::{Completion, DefinitionKind, Hover};

/// A source file and the exported entry to execute, usually `main`.
#[derive(Clone, Debug)]
pub struct Input {
    pub path: PathBuf,
    pub entry: String,
}

/// Explicit build settings; tool resolution happens before compilation.
pub struct Options {
    pub profile: resin_platform_toolchain::CProfile,
    pub tools: resin_platform_toolchain::Toolchain,
}

/// A validated request, including optional executable publication.
pub struct Request {
    input: Input,
    destination: Option<PathBuf>,
    options: Options,
}
impl Request {
    /// Resolve directory destinations and reject outputs that overwrite the source.
    pub fn new(
        input: Input,
        destination: Option<PathBuf>,
        options: Options,
    ) -> Result<Self, Error> {
        Self::validate(input, destination, options)
    }
    pub fn destination(&self) -> Option<&Path> {
        self.destination.as_deref()
    }
}

/// A host-controlled source database with parsed-tree and dependency caches.
pub struct Session {
    state: session::State,
}
impl Default for Session {
    fn default() -> Self {
        Self::new(stdlib_path())
    }
}
impl Session {
    pub fn new(stdlib: PathBuf) -> Self {
        Self {
            state: session::State::new(stdlib),
        }
    }
    /// Analyze and build; the returned executable retains its cache lock through use.
    pub fn compile(
        &mut self,
        request: &Request,
    ) -> Result<resin_platform_toolchain::Executable, Error> {
        let compilation = self.analyze(&request.input.path)?;
        build::generate(request, compilation.data.verified()?)
    }
    pub fn analyze(&mut self, entry: &Path) -> io::Result<Arc<Compilation>> {
        self.state.analyze(entry)
    }
    pub fn revision(&self) -> u64 {
        self.state.revision()
    }
    pub fn set_overlay(&mut self, path: &Path, text: String) -> io::Result<()> {
        self.state.set_overlay(path, text)
    }
    pub fn remove_overlay(&mut self, path: &Path) -> io::Result<()> {
        self.state.remove_overlay(path)
    }
    /// Notify a disk change; an open overlay continues to supply the source until closed.
    pub fn file_changed(&mut self, path: &Path) -> io::Result<()> {
        self.state.file_changed(path)
    }
    pub fn set_stdlib(&mut self, stdlib: PathBuf) {
        self.state.set_stdlib(stdlib);
    }
    /// Release roots the host no longer needs while retaining shared dependencies.
    pub fn retain_entries(&mut self, entries: &BTreeSet<PathBuf>) {
        self.state.retain_entries(entries);
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: SourceLocation,
    pub message: String,
    pub related: Vec<SourceNote>,
}

/// Completed phase products and editor facts for one source entry and its imports.
/// Failed later passes preserve the successfully completed earlier products.
/// Borrowed phase products cannot be edited through a retained compilation.
///
/// ```compile_fail,E0596
/// fn edit(compilation: &mut resin_compiler::Compilation) {
///     compilation.module().unwrap().functions.clear();
/// }
/// ```
///
/// Editor queries do not expose the internal document-provider implementation.
///
/// ```compile_fail,E0277
/// fn requires_documents<T: resin_hir::Documents>() {}
/// requires_documents::<resin_compiler::Compilation>();
/// ```
pub struct Compilation {
    data: compilation::Data,
}
impl Compilation {
    /// Analyze once with custom source access; use Session to reuse work across edits.
    pub fn new(entry: &Path, sources: &impl SourceProvider, stdlib: &Path) -> Self {
        Self {
            data: compilation::Data::new(entry, sources, stdlib),
        }
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.data.diagnostics
    }
    pub fn dependencies(&self) -> &BTreeSet<PathBuf> {
        &self.data.dependencies
    }
    pub fn sources(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.data.sources()
    }
    pub fn program(&self) -> Result<&resin_ast::Program, SourceError> {
        self.data.program()
    }
    pub fn hir(&self) -> Result<&resin_hir::Module, SourceError> {
        self.data.hir()
    }
    pub fn module(&self) -> Result<&resin_lir::Module, SourceError> {
        self.data.module()
    }
    pub fn recovered_file(&self, path: &Path) -> Option<&resin_ast::SourceFile> {
        self.data.recovered_file(path)
    }
    pub fn definition(&self, path: &Path, offset: usize) -> Option<SourceLocation> {
        self.data.semantics.definition(&self.data, path, offset)
    }
    pub fn hover(&self, path: &Path, offset: usize) -> Option<Hover> {
        self.data.semantics.hover(&self.data, path, offset)
    }
    pub fn completions(&self, path: &Path, offset: usize) -> Vec<Completion> {
        self.data.semantics.completions(&self.data, path, offset)
    }
}

/// Custom source identity and content access for a one-shot Compilation.
pub trait SourceProvider {
    fn resolve(&self, path: &Path) -> io::Result<PathBuf>;
    fn read(&self, path: &Path) -> io::Result<String>;
}
/// Open buffers override disk, including files that have never been saved.
#[derive(Clone, Default)]
pub struct Sources {
    pub overlays: BTreeMap<PathBuf, String>,
}

/// Canonicalize existing ancestors, preserving identities for unsaved files under symlinks.
pub fn normalize_path(path: &Path) -> io::Result<PathBuf> {
    source::normalize_path(path)
}
/// Bundled library path. Frontends may resolve their own configuration overrides.
pub fn stdlib_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib"))
}

#[derive(Debug)]
pub struct Error(String);
