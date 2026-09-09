//! Immutable named sources, source locations, and import loading.
//! A source owns one version of its text; loaders preserve unchanged versions.

use std::{
    collections::BTreeMap,
    fmt, io,
    path::{Component, Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

mod paths;

//
// Immutable sources and locations
//

/// Stable logical module identity, independent of its name and text version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(usize);

/// Immutable named text. Clones share a version; names need not be paths or unique.
/// Equality and ordering identify versions, rather than comparing their contents.
#[derive(Debug, Clone)]
pub struct Source(Arc<Text>);

impl Source {
    pub fn new(name: impl Into<std::sync::Arc<str>>, text: impl Into<std::sync::Arc<str>>) -> Self {
        create(name.into(), text.into())
    }
    pub fn id(&self) -> SourceId {
        self.0.id
    }
    pub fn name(&self) -> &str {
        &self.0.name
    }
    pub fn text(&self) -> &str {
        &self.0.text
    }
    /// Create a new immutable version of this module. The original remains valid.
    pub fn with_text(&self, text: impl Into<std::sync::Arc<str>>) -> Self {
        replace(self, text.into())
    }
}

pub type Ident = Spanned<Arc<str>>;

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Span,
}
impl<T> Spanned<T> {
    pub fn new(val: T, span: Span) -> Self {
        Self { val, span }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.val
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceLocation {
    pub source: Source,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct SourceNote {
    pub location: SourceLocation,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct SourceError {
    pub source: Source,
    pub span: Option<Span>,
    /// Human-readable CLI rendering, including the import chain.
    pub message: Box<str>,
    /// The diagnostic itself, without path prefixes or an import chain.
    pub diagnostic: String,
    pub related: Vec<SourceNote>,
}

impl SourceError {
    pub fn new(source: Source, span: Option<Span>, diagnostic: String) -> Self {
        Self {
            message: format!("{}: {diagnostic}", source.name()).into(),
            source,
            span,
            diagnostic,
            related: Vec::new(),
        }
    }
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for SourceError {}

#[derive(Debug)]
struct Text {
    pub version: usize,
    pub id: SourceId,
    pub name: Arc<str>,
    pub text: Arc<str>,
}

fn next_version() -> usize {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
        .expect("source identities exhausted")
}

fn create(name: Arc<str>, text: Arc<str>) -> Source {
    let version = next_version();
    Source(Arc::new(Text {
        version,
        id: SourceId(version),
        name,
        text,
    }))
}

fn replace(source: &Source, text: Arc<str>) -> Source {
    Source(Arc::new(Text {
        version: next_version(),
        id: source.id(),
        name: source.0.name.clone(),
        text,
    }))
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for Source {}
impl PartialOrd for Source {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Source {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.version.cmp(&other.0.version)
    }
}
impl std::hash::Hash for Source {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.version.hash(state);
    }
}

/// Source vocabulary shared by compiler phases. Import privately.
pub mod prelude {
    pub use crate::{
        Ident, Source, SourceError, SourceId, SourceLocation, SourceNote, Span, Spanned,
    };
}

//
// Import resolution and filesystem loading
//

/// Filesystem sources and named-source import bindings for one compilation environment.
pub struct Loader {
    stdlib: PathBuf,
    files: BTreeMap<PathBuf, File>,
    origins: BTreeMap<SourceId, PathBuf>,
    imports: BTreeMap<(SourceId, String), Source>,
}

impl Loader {
    pub fn new(stdlib: PathBuf) -> Self {
        Self {
            stdlib,
            files: BTreeMap::new(),
            origins: BTreeMap::new(),
            imports: BTreeMap::new(),
        }
    }

    /// Read current disk contents without replacing authoritative supplied text.
    pub fn load_file(&mut self, path: &Path) -> io::Result<Source> {
        let path = normalize_path(path)?;
        let text: Arc<str> = read_file(&path)?.into();
        let file = self.file(path, text.clone());
        let source = file.version(text);
        file.cached = source.clone();
        Ok(source)
    }

    /// Supply text for a file's imports until `remove_source` restores disk loading.
    pub fn source_from_text(
        &mut self,
        path: &Path,
        text: impl Into<Arc<str>>,
    ) -> io::Result<Source> {
        // An open document retains its identity when the filesystem changes underneath it.
        let path = if self
            .files
            .get(path)
            .is_some_and(|file| file.supplied.is_some())
        {
            path.to_path_buf()
        } else {
            normalize_path(path)?
        };
        let text = text.into();
        let file = self.file(path, text.clone());
        let source = file.version(text);
        file.supplied = Some(source.clone());
        Ok(source)
    }

    /// Stop supplying a file's text while retaining its identity and cached versions.
    pub fn remove_source(&mut self, path: &Path) -> io::Result<()> {
        // Closing a registered file must preserve the identity chosen when it opened.
        let path = if self.files.contains_key(path) {
            path.to_path_buf()
        } else {
            normalize_path(path)?
        };
        if let Some(file) = self.files.get_mut(&path) {
            file.supplied = None;
        }
        Ok(())
    }

    /// Bind one logical source's import reference to an immutable target source.
    pub fn set_import(
        &mut self,
        source: &Source,
        reference: &str,
        target: Source,
    ) -> io::Result<()> {
        validate_reference(reference)?;
        self.imports.insert((source.id(), reference.into()), target);
        Ok(())
    }

    /// Load an explicit binding, supplied file text, or the current file contents.
    pub fn load_import(&mut self, source: &Source, reference: &str) -> io::Result<Source> {
        validate_reference(reference)?;
        if let Some(target) = self.imports.get(&(source.id(), reference.into())) {
            return Ok(target.clone());
        }
        let path = self.resolve_import(source, reference)?;
        if let Some(supplied) = self.files.get(&path).and_then(|file| file.supplied.clone()) {
            return Ok(supplied);
        }
        self.load_file(&path)
    }

    /// Recover an exact OS path from a source produced by this loader.
    pub fn path(&self, source: &Source) -> Option<&Path> {
        self.origins.get(&source.id()).map(PathBuf::as_path)
    }

    /// Resolve a filesystem reference; `$/std/` also works for named in-memory sources.
    pub fn resolve_import(&self, source: &Source, reference: &str) -> io::Result<PathBuf> {
        validate_reference(reference)?;
        if let Some(relative) = reference.strip_prefix("$/std/") {
            return normalize_path(&self.stdlib.join(relative));
        }
        let origin = self.path(source).ok_or_else(unknown_source)?;
        normalize_path(&origin.parent().unwrap_or(Path::new(".")).join(reference))
    }

    fn file(&mut self, path: PathBuf, text: Arc<str>) -> &mut File {
        let file = self.files.entry(path.clone()).or_insert_with(|| File {
            cached: Source::new(path.to_string_lossy().into_owned(), text),
            supplied: None,
        });
        self.origins.insert(file.cached.id(), path);
        file
    }
}

/// Canonicalize existing ancestors, retaining identities for unsaved files under symlinks.
pub fn normalize_path(path: &Path) -> io::Result<PathBuf> {
    paths::normalize(path)
}

/// Bundled library path; applications may supply their own configured directory.
pub fn stdlib_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resin/std"))
}

struct File {
    cached: Source,
    supplied: Option<Source>,
}

impl File {
    fn version(&self, text: Arc<str>) -> Source {
        if let Some(source) = &self.supplied
            && source.text() == text.as_ref()
        {
            return source.clone();
        }
        if self.cached.text() == text.as_ref() {
            return self.cached.clone();
        }
        self.cached.with_text(text)
    }
}

fn read_file(path: &Path) -> io::Result<String> {
    std::fs::read_to_string(path)
        .map_err(|error| io::Error::new(error.kind(), format!("{}: {error}", path.display())))
}

fn validate_reference(reference: &str) -> io::Result<()> {
    if let Some(relative) = reference.strip_prefix("$/std/") {
        if relative.is_empty()
            || Path::new(relative)
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "standard-library imports must stay under $/std/",
            ));
        }
    } else if reference.starts_with('$') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("unknown import namespace: {reference}"),
        ));
    }
    Ok(())
}

fn unknown_source() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "relative imports require a registered file source or an explicit import binding",
    )
}
