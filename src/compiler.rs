//! Compilation requests and long-lived sessions shared by command-line tools and editor adapters.
//!
//! Hosts supply buffer/disk changes; the session owns parsed trees, dependencies,
//! and checked snapshots. Queries without intervening changes reuse the same
//! snapshot. Existing snapshots remain valid when the session advances.

use crate::{ast, backend, toolchain};
mod snapshot;
mod source;
pub(crate) mod syntax;
pub use snapshot::{Diagnostic, Snapshot};
pub use source::{Sources, normalize_path};
use syntax::Document;

use std::{
    collections::{BTreeMap, BTreeSet},
    io,
    path::{Path, PathBuf},
    sync::Arc,
};

/// A source file and its selected exported entry, independent of CLI selector syntax.
#[derive(Clone, Debug)]
pub struct Input {
    pub path: PathBuf,
    pub entry: String,
}

/// Explicit code-generation settings, resolved by the caller before compilation.
pub struct Options {
    pub profile: toolchain::CProfile,
    pub tools: toolchain::Settings,
}

/// A validated compilation request. Constructing one resolves executable-directory
/// destinations and rejects destinations that would overwrite the source file.
pub struct Request {
    input: Input,
    destination: Option<PathBuf>,
    options: Options,
}

impl Request {
    pub fn new(
        mut input: Input,
        destination: Option<PathBuf>,
        options: Options,
    ) -> Result<Self, backend::Error> {
        let source = normalize_path(&input.path)?;
        let destination = destination
            .map(|path| {
                let path = executable_destination(&input, path)?;
                validate_destination_ancestors(&path)?;
                if source == normalize_path(&path)? {
                    return Err(backend::Error(
                        "output would overwrite the source file".into(),
                    ));
                }
                Ok::<_, backend::Error>(std::path::absolute(path)?)
            })
            .transpose()?;
        input.path = source;
        Ok(Self {
            input,
            destination,
            options,
        })
    }

    pub fn input(&self) -> &Input {
        &self.input
    }
    pub fn destination(&self) -> Option<&Path> {
        self.destination.as_deref()
    }
    pub fn options(&self) -> &Options {
        &self.options
    }
}

// Check each prefix before normalization: Windows may report NotFound for a
// child of a regular file. Checking only the final parent misses that case.
fn validate_destination_ancestors(path: &Path) -> io::Result<()> {
    // Preserve raw components until they have been checked. In particular,
    // Windows absolute-path resolution can collapse `file/..` prematurely.
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut prefix = PathBuf::new();
    let components: Vec<_> = absolute.components().collect();
    for part in &components[..components.len().saturating_sub(1)] {
        prefix.push(part.as_os_str());
        match std::fs::metadata(&prefix) {
            Ok(metadata) if !metadata.is_dir() => {
                return Err(io::Error::new(
                    io::ErrorKind::NotADirectory,
                    format!("output ancestor is not a directory: {}", prefix.display()),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                if std::fs::symlink_metadata(&prefix).is_ok_and(|metadata| metadata.is_symlink()) {
                    return Err(io::Error::new(
                        io::ErrorKind::NotFound,
                        format!(
                            "output ancestor is a dangling symlink: {}",
                            prefix.display()
                        ),
                    ));
                }
            }
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn executable_destination(input: &Input, path: PathBuf) -> Result<PathBuf, backend::Error> {
    let trailing_separator = path
        .as_os_str()
        .as_encoded_bytes()
        .last()
        .is_some_and(|&b| b == b'/' || b == std::path::MAIN_SEPARATOR as u8);
    if !path.is_dir() && !trailing_separator {
        return Ok(path);
    }
    let mut name = input
        .path
        .file_stem()
        .ok_or_else(|| backend::Error("source file needs a name".into()))?
        .to_os_string();
    if input.entry != "main" {
        name.push(format!("-{}", input.entry));
    }
    name.push(std::env::consts::EXE_SUFFIX);
    Ok(path.join(name))
}

pub struct Session {
    sources: Sources,
    stdlib: PathBuf,
    parsed: BTreeMap<PathBuf, Arc<Document>>,
    checked: BTreeMap<PathBuf, Arc<Snapshot>>,
    dependents: BTreeMap<PathBuf, BTreeSet<PathBuf>>,
    revision: u64,
}

impl Default for Session {
    fn default() -> Self {
        Self::new(ast::stdlib_path())
    }
}

impl Session {
    pub fn new(stdlib: PathBuf) -> Self {
        Self {
            sources: Sources::default(),
            stdlib,
            parsed: BTreeMap::new(),
            checked: BTreeMap::new(),
            dependents: BTreeMap::new(),
            revision: 0,
        }
    }

    /// Analyze using this session's sources and cached snapshots, then generate
    /// the executable. Execution remains a separate operation.
    pub fn compile(&mut self, request: &Request) -> Result<backend::Executable, backend::Error> {
        let snapshot = self.analyze(&request.input.path)?;
        backend::generate(request, snapshot.verified()?)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn set_overlay(&mut self, path: &Path, text: String) -> io::Result<()> {
        let path = normalize_path(path)?;
        if self.sources.overlays.get(&path) == Some(&text) {
            return Ok(());
        }
        self.sources.overlays.insert(path.clone(), text);
        self.invalidate(&path);
        Ok(())
    }

    pub fn remove_overlay(&mut self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        if self.sources.overlays.remove(&path).is_some() {
            self.invalidate(&path);
        }
        Ok(())
    }

    /// Notify the compiler after an external modification, creation, or deletion.
    /// Open buffers continue to own their contents until their overlay is removed.
    pub fn file_changed(&mut self, path: &Path) -> io::Result<()> {
        let path = normalize_path(path)?;
        if !self.sources.overlays.contains_key(&path) {
            self.invalidate(&path);
        }
        Ok(())
    }

    pub fn analyze(&mut self, entry: &Path) -> io::Result<Arc<Snapshot>> {
        let entry = normalize_path(entry)?;
        if let Some(snapshot) = self.checked.get(&entry) {
            return Ok(snapshot.clone());
        }
        let snapshot = Arc::new(Snapshot::build(
            &entry,
            &self.sources,
            &self.stdlib,
            &mut self.parsed,
        ));
        for dependency in &snapshot.dependencies {
            self.dependents
                .entry(dependency.clone())
                .or_default()
                .insert(entry.clone());
        }
        self.checked.insert(entry, snapshot.clone());
        Ok(snapshot)
    }

    pub fn set_stdlib(&mut self, stdlib: PathBuf) {
        if self.stdlib == stdlib {
            return;
        }
        self.stdlib = stdlib;
        self.checked.clear();
        self.dependents.clear();
        self.revision += 1;
    }

    /// Release entries a host no longer needs while retaining shared dependencies.
    pub fn retain_entries(&mut self, entries: &BTreeSet<PathBuf>) {
        let removed = self
            .checked
            .keys()
            .filter(|p| !entries.contains(*p))
            .cloned()
            .collect::<Vec<_>>();
        for entry in removed {
            self.forget(&entry);
        }
        self.parsed.retain(|path, _| {
            self.dependents.contains_key(path) || self.sources.overlays.contains_key(path)
        });
    }

    fn invalidate(&mut self, path: &Path) {
        self.revision += 1;
        let mut entries = self.dependents.get(path).cloned().unwrap_or_default();
        entries.insert(path.to_path_buf());
        for entry in entries {
            self.forget(&entry);
        }
    }

    fn forget(&mut self, entry: &Path) {
        self.checked.remove(entry);
        self.dependents.retain(|_, entries| {
            entries.remove(entry);
            !entries.is_empty()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_snapshots_are_shared_and_only_dependents_are_invalidated() {
        let root = std::env::temp_dir().join("resin-compiler-session-test");
        let mut compiler = Session::new(root.join("std"));
        let entry = root.join("main.resin");
        let library = root.join("lib.resin");
        let independent = root.join("other.resin");
        compiler
            .set_overlay(
                &entry,
                "import { \"lib.resin\" }; def main () -> int = { value() };".into(),
            )
            .unwrap();
        compiler
            .set_overlay(
                &library,
                "export { value }; def value () -> int = { 1 };".into(),
            )
            .unwrap();
        compiler
            .set_overlay(&independent, "def other () -> int = { 2 };".into())
            .unwrap();
        let old = compiler.analyze(&entry).unwrap();
        let other = compiler.analyze(&independent).unwrap();
        assert!(old.module().is_ok(), "{:?}", old.diagnostics);
        assert!(Arc::ptr_eq(&old, &compiler.analyze(&entry).unwrap()));
        compiler
            .set_overlay(
                &library,
                "export { renamed }; def renamed () -> int = { 1 };".into(),
            )
            .unwrap();
        let new = compiler.analyze(&entry).unwrap();
        assert!(!Arc::ptr_eq(&old, &new));
        assert!(new.module().is_err());
        assert!(old.module().is_ok(), "previous snapshots must remain valid");
        assert!(Arc::ptr_eq(
            &other,
            &compiler.analyze(&independent).unwrap()
        ));
    }

    #[test]
    fn creating_a_missing_dependency_recovers_the_cached_error() {
        let root = std::env::temp_dir().join("resin-compiler-missing-test");
        let mut compiler = Session::default();
        compiler
            .set_overlay(&root.join("main.resin"), "import { \"new.resin\" };".into())
            .unwrap();
        assert!(
            compiler
                .analyze(&root.join("main.resin"))
                .unwrap()
                .module()
                .is_err()
        );
        compiler
            .set_overlay(&root.join("new.resin"), "export {};".into())
            .unwrap();
        let snapshot = compiler.analyze(&root.join("main.resin")).unwrap();
        assert!(snapshot.module().is_ok(), "{:?}", snapshot.diagnostics);
    }
}

#[cfg(test)]
mod verification_tests {
    use super::*;
    use crate::{
        backend::{c, glsl},
        ir::verify::ANALYSES,
    };

    #[test]
    fn snapshots_reuse_verification_for_multiple_backends_and_invalidate_on_edit() {
        let path = std::env::temp_dir().join("resin-verification-cache.resin");
        let source = "export { main, a, b }; def main() -> int = { 0 }; @compute_shader def a(invocation: ulong, p: Ptr<uint>) = { var i = uint(invocation); p.* := i; }; @compute_shader def b(invocation: ulong, p: Ptr<uint>) = { var i = uint(invocation); p.* := i + 1_ui; };";
        let mut session = Session::default();
        session.set_overlay(&path, source.into()).unwrap();
        ANALYSES.set(0);
        let snapshot = session.analyze(&path).unwrap();
        let checked = snapshot.verified().unwrap();
        assert_eq!(ANALYSES.get(), 1);
        let mut shaders = Vec::new();
        for name in ["a", "b"] {
            let id = checked.module().entries[name];
            shaders.push(glsl::emit_verified(checked, id, glsl::Stage::Compute).unwrap());
        }
        let host = c::emit_verified(checked, "main", &[]).unwrap();
        assert_eq!(
            ANALYSES.get(),
            1,
            "all internal emissions share one analysis"
        );
        assert!(std::ptr::eq(
            checked.analysis(),
            snapshot.verified().unwrap().analysis()
        ));
        assert_eq!(host, c::emit(checked.module(), "main").unwrap());
        for (name, expected) in ["a", "b"].into_iter().zip(shaders) {
            assert_eq!(
                expected,
                glsl::emit(checked.module(), name, glsl::Stage::Compute).unwrap()
            );
        }
        session
            .set_overlay(&path, source.replace("{ 0 }", "{ 1 }"))
            .unwrap();
        ANALYSES.set(0);
        let changed = session.analyze(&path).unwrap();
        let new = changed.verified().unwrap();
        assert_eq!(ANALYSES.get(), 1);
        assert!(!std::ptr::eq(checked.module(), new.module()));
        assert_ne!(host, c::emit_verified(new, "main", &[]).unwrap());
        assert_eq!(
            host,
            c::emit_verified(snapshot.verified().unwrap(), "main", &[]).unwrap()
        );
        assert_eq!(ANALYSES.get(), 1);
    }
}
