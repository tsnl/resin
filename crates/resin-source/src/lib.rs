//! Immutable named sources, source locations, and import loading.
//! A source owns one version of its text; loaders preserve unchanged versions.

use resin_executor::{Cancellation, Execution};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt, io,
    path::{Component, Path, PathBuf},
    sync::Arc,
};

mod logical;
mod paths;

//
// Immutable sources and locations
//

/// Reconstructible logical module identity, independent of display name and text.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceId(Arc<Identity>);

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Identity {
    Logical { name: Arc<str> },
    File { path: PathBuf },
}

impl SourceId {
    /// Identify the same logical module across independent acquisitions.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self(Arc::new(Identity::Logical { name: name.into() }))
    }

    fn file(path: PathBuf) -> Self {
        Self(Arc::new(Identity::File { path }))
    }
}

/// Immutable named text. Equality uses logical identity, retained name, and exact text.
/// Independently reconstructed equal sources can address the same retained compiler facts.
#[derive(Debug, Clone)]
pub struct Source(Arc<Text>);

impl Source {
    /// Use the name as this source's logical module identity.
    pub fn new(name: impl Into<Arc<str>>, text: impl Into<Arc<str>>) -> Self {
        let name = name.into();
        Self::with_identity(SourceId::new(name.clone()), name, text)
    }

    /// Give a module an identity independent of its diagnostic display name.
    pub fn with_identity(
        id: SourceId,
        name: impl Into<Arc<str>>,
        text: impl Into<Arc<str>>,
    ) -> Self {
        let text = text.into();
        Self(Arc::new(Text {
            id,
            name: name.into(),
            digest: *blake3::hash(text.as_bytes()).as_bytes(),
            text,
        }))
    }
    pub fn id(&self) -> SourceId {
        self.0.id.clone()
    }
    pub fn name(&self) -> &str {
        &self.0.name
    }
    pub fn text(&self) -> &str {
        &self.0.text
    }
    /// BLAKE3 over the exact UTF-8 text bytes; equality additionally checks the text.
    pub fn content_hash(&self) -> &[u8; 32] {
        &self.0.digest
    }
    /// Create a new immutable version of this module. The original remains valid.
    pub fn with_text(&self, text: impl Into<std::sync::Arc<str>>) -> Self {
        let text = text.into();
        if self.text() == text.as_ref() {
            return self.clone();
        }
        Self::with_identity(self.id(), self.0.name.clone(), text)
    }

    /// Apply UTF-8 byte edits in order, each relative to the text after earlier edits.
    /// `None` starts with empty text, so a full upload is one insertion at `0..0`.
    pub fn from_edits(
        id: SourceId,
        name: impl Into<Arc<str>>,
        previous: Option<&Source>,
        edits: &[SourceEdit],
    ) -> Result<Self, EditError> {
        if let Some(previous) = previous
            && previous.id() != id
        {
            return Err(EditError::WrongPredecessor {
                expected: id,
                actual: previous.id(),
            });
        }
        let mut text = previous.map_or("", Source::text).to_owned();
        for (index, edit) in edits.iter().enumerate() {
            validate_edit(&text, edit.range, index)?;
            text.replace_range(edit.range.start..edit.range.end, &edit.text);
        }
        Ok(Self::with_identity(id, name, text))
    }
}

/// One replacement in the current text, using UTF-8 byte offsets.
#[derive(Debug, Clone)]
pub struct SourceEdit {
    pub range: Span,
    pub text: Arc<str>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditError {
    WrongPredecessor {
        expected: SourceId,
        actual: SourceId,
    },
    InvalidRange {
        edit: usize,
        range: Span,
        length: usize,
    },
    InvalidUtf8Boundary {
        edit: usize,
        offset: usize,
    },
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WrongPredecessor { expected, actual } => {
                write!(
                    f,
                    "source predecessor has identity {actual:?}, expected {expected:?}"
                )
            }
            Self::InvalidRange {
                edit,
                range,
                length,
            } => {
                write!(
                    f,
                    "edit {edit} range {}..{} is invalid for {length} bytes",
                    range.start, range.end
                )
            }
            Self::InvalidUtf8Boundary { edit, offset } => {
                write!(
                    f,
                    "edit {edit} offset {offset} is not a UTF-8 character boundary"
                )
            }
        }
    }
}

impl std::error::Error for EditError {}

fn validate_edit(text: &str, range: Span, index: usize) -> Result<(), EditError> {
    if range.start > range.end || range.end > text.len() {
        return Err(EditError::InvalidRange {
            edit: index,
            range,
            length: text.len(),
        });
    }
    for offset in [range.start, range.end] {
        if !text.is_char_boundary(offset) {
            return Err(EditError::InvalidUtf8Boundary {
                edit: index,
                offset,
            });
        }
    }
    Ok(())
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
    pub id: SourceId,
    pub name: Arc<str>,
    pub digest: [u8; 32],
    pub text: Arc<str>,
}

impl PartialEq for Source {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.id == other.0.id
                && self.0.name == other.0.name
                && self.0.digest == other.0.digest
                && self.0.text == other.0.text)
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
        if Arc::ptr_eq(&self.0, &other.0) {
            return std::cmp::Ordering::Equal;
        }
        self.0
            .id
            .cmp(&other.0.id)
            .then_with(|| self.0.name.cmp(&other.0.name))
            .then_with(|| self.0.digest.cmp(&other.0.digest))
            .then_with(|| self.0.text.cmp(&other.0.text))
    }
}
impl std::hash::Hash for Source {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.id.hash(state);
        self.0.name.hash(state);
        self.0.digest.hash(state);
    }
}

//
// Immutable source graphs
//

/// An explicit import edge; resolving it never consults a filesystem or another graph.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImportBinding {
    pub source: SourceId,
    pub reference: Arc<str>,
    pub target: SourceId,
}

/// One entry and its immutable, explicitly bound source closure.
/// Construction canonicalizes order and drops sources/edges unreachable from the entry.
/// Cycles remain representable so syntax-aware callers can diagnose them with source spans.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceGraph {
    entry: SourceId,
    sources: BTreeMap<SourceId, Source>,
    bindings: BTreeMap<(SourceId, Arc<str>), ImportBinding>,
}

impl SourceGraph {
    /// Validate all supplied endpoints, then retain only the entry's reachable closure.
    /// Callers must also check parsed import declarations against these explicit bindings.
    pub fn new(
        entry: Source,
        sources: impl IntoIterator<Item = Source>,
        bindings: impl IntoIterator<Item = ImportBinding>,
    ) -> Result<Self, GraphError> {
        let mut graph = Self {
            entry: entry.id(),
            sources: BTreeMap::new(),
            bindings: BTreeMap::new(),
        };
        for source in std::iter::once(entry).chain(sources) {
            graph.insert_source(source)?;
        }
        for binding in bindings {
            graph.insert_binding(binding)?;
        }
        graph.retain_reachable();
        Ok(graph)
    }

    pub fn entry(&self) -> &Source {
        &self.sources[&self.entry]
    }

    pub fn source(&self, id: &SourceId) -> Option<&Source> {
        self.sources.get(id)
    }

    /// Iterate in canonical logical-identity order, independently of discovery order.
    pub fn sources(&self) -> impl Iterator<Item = &Source> {
        self.sources.values()
    }

    pub fn bindings(&self) -> impl Iterator<Item = &ImportBinding> {
        self.bindings.values()
    }

    /// Resolve only this graph's captured binding. An absent binding stays absent.
    pub fn resolve(&self, source: &SourceId, reference: &str) -> Option<&Source> {
        let binding = self.bindings.get(&(source.clone(), reference.into()))?;
        self.sources.get(&binding.target)
    }

    fn insert_source(&mut self, source: Source) -> Result<(), GraphError> {
        if let Some(previous) = self.sources.get(&source.id())
            && previous != &source
        {
            return Err(GraphError::ConflictingSource {
                source: source.id(),
            });
        }
        self.sources.insert(source.id(), source);
        Ok(())
    }

    fn insert_binding(&mut self, binding: ImportBinding) -> Result<(), GraphError> {
        for id in [&binding.source, &binding.target] {
            if !self.sources.contains_key(id) {
                return Err(GraphError::MissingSource { source: id.clone() });
            }
        }
        validate_reference(&binding.reference).map_err(|error| GraphError::InvalidReference {
            source: binding.source.clone(),
            reference: binding.reference.clone(),
            message: error.to_string(),
        })?;
        let key = (binding.source.clone(), binding.reference.clone());
        if let Some(previous) = self.bindings.get(&key)
            && previous != &binding
        {
            return Err(GraphError::ConflictingImport {
                source: binding.source,
                reference: binding.reference,
            });
        }
        self.bindings.insert(key, binding);
        Ok(())
    }

    fn retain_reachable(&mut self) {
        let mut children: BTreeMap<SourceId, Vec<SourceId>> = BTreeMap::new();
        for binding in self.bindings.values() {
            children
                .entry(binding.source.clone())
                .or_default()
                .push(binding.target.clone());
        }
        let mut reachable = BTreeSet::new();
        let mut pending = vec![self.entry.clone()];
        while let Some(source) = pending.pop() {
            if reachable.insert(source.clone())
                && let Some(targets) = children.get(&source)
            {
                pending.extend(targets.iter().cloned());
            }
        }
        self.sources.retain(|source, _| reachable.contains(source));
        self.bindings
            .retain(|(source, _), _| reachable.contains(source));
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    ConflictingSource {
        source: SourceId,
    },
    MissingSource {
        source: SourceId,
    },
    ConflictingImport {
        source: SourceId,
        reference: Arc<str>,
    },
    InvalidReference {
        source: SourceId,
        reference: Arc<str>,
        message: String,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingSource { source } => {
                write!(f, "conflicting versions of source {source:?}")
            }
            Self::MissingSource { source } => {
                write!(f, "import refers to uncaptured source {source:?}")
            }
            Self::ConflictingImport { source, reference } => {
                write!(
                    f,
                    "conflicting targets for import {reference:?} in source {source:?}"
                )
            }
            Self::InvalidReference {
                source,
                reference,
                message,
            } => {
                write!(
                    f,
                    "invalid import {reference:?} in source {source:?}: {message}"
                )
            }
        }
    }
}

impl std::error::Error for GraphError {}

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
    library_root: PathBuf,
    files: BTreeMap<PathBuf, File>,
    origins: BTreeMap<SourceId, PathBuf>,
    imports: BTreeMap<(SourceId, String), Source>,
}

/// File acquisition failures distinguish source I/O from cancellation or worker failure.
#[derive(Debug)]
pub enum LoadError {
    Io { error: io::Error },
    Execution { error: resin_executor::Error },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { error } => error.fmt(f),
            Self::Execution { error } => error.fmt(f),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { error } => Some(error),
            Self::Execution { error } => Some(error),
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(error: io::Error) -> Self {
        Self::Io { error }
    }
}

impl From<resin_executor::Error> for LoadError {
    fn from(error: resin_executor::Error) -> Self {
        Self::Execution { error }
    }
}

impl Loader {
    pub fn new(library_root: PathBuf) -> Self {
        Self {
            library_root,
            files: BTreeMap::new(),
            origins: BTreeMap::new(),
            imports: BTreeMap::new(),
        }
    }

    /// Copy current supplied registrations and explicit import bindings for a new request.
    ///
    /// Disk-only and closed file records are discarded. Supplied files keep their
    /// registered physical identities without normalizing paths again, even when
    /// the filesystem has changed since they opened. Each retained file caches only
    /// its current supplied source, not an older disk or editor version.
    ///
    /// Explicit bindings are caller-owned configuration and remain available with
    /// their registered origins. This performs no I/O and changes neither loader;
    /// applications can replace a registration loader with its filtered snapshot
    /// after closing buffers, then give independent snapshots to concurrent requests.
    pub fn supplied_snapshot(&self) -> Self {
        let files: BTreeMap<_, _> = self
            .files
            .iter()
            .filter_map(|(path, file)| {
                let source = file.supplied.as_ref()?;
                Some((
                    path.clone(),
                    File {
                        cached: source.clone(),
                        supplied: Some(source.clone()),
                    },
                ))
            })
            .collect();
        let mut retained: BTreeSet<_> = files.values().map(|file| file.cached.id()).collect();
        for ((source, _), target) in &self.imports {
            retained.insert(source.clone());
            retained.insert(target.id());
        }
        Self {
            library_root: self.library_root.clone(),
            files,
            origins: self
                .origins
                .iter()
                .filter(|(source, _)| retained.contains(*source))
                .map(|(source, path)| (source.clone(), path.clone()))
                .collect(),
            imports: self.imports.clone(),
        }
    }

    /// Assign checkout-independent names before selecting compiler cache entries.
    /// The returned map associates each original identity with its logical source;
    /// callers retain the originals separately for local paths and diagnostics.
    ///
    /// Files under the configured library root use `$/relative/path`; other files
    /// use paths relative to `root`, including `../` for sources outside it. Names
    /// use `/` separators and reversible percent escapes for reserved characters
    /// and invalid UTF-8 bytes. Sources without a loader origin remain unchanged.
    /// User files and `root` must have a common filesystem prefix (drive on Windows).
    /// This operation changes neither the loader nor any supplied source.
    pub async fn logical_sources(
        &self,
        sources: impl IntoIterator<Item = Source>,
        root: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<BTreeMap<SourceId, Source>, LoadError> {
        cancellation.check()?;
        let sources = sources
            .into_iter()
            .map(|source| {
                let path = self.path(&source).map(Path::to_path_buf);
                (source, path)
            })
            .collect();
        let root = root.to_path_buf();
        let library = self.library_root.clone();
        execution
            .run(cancellation, move |cancellation| {
                logical::sources(sources, &root, &library, cancellation)
            })
            .await?
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

    /// Read current disk text asynchronously without replacing authoritative editor text.
    /// Path normalization runs on a worker; cancelled reads publish no new cached source.
    pub async fn load_file_async(
        &mut self,
        path: &Path,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Source, LoadError> {
        let path = normalize_path_async(path, execution, cancellation).await?;
        self.read_path_async(path, execution, cancellation).await
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

    /// Resolve an explicit binding or supplied text before asynchronously reading disk.
    pub async fn load_import_async(
        &mut self,
        source: &Source,
        reference: &str,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Source, LoadError> {
        cancellation.check()?;
        validate_reference(reference)?;
        if let Some(target) = self.imports.get(&(source.id(), reference.into())) {
            return Ok(target.clone());
        }
        let unresolved = self.import_path(source, reference)?;
        let path = normalize_path_async(&unresolved, execution, cancellation).await?;
        if let Some(source) = self.files.get(&path).and_then(|file| file.supplied.clone()) {
            return Ok(source);
        }
        self.read_path_async(path, execution, cancellation).await
    }

    /// Recover an exact OS path from a source produced by this loader.
    pub fn path(&self, source: &Source) -> Option<&Path> {
        self.origins.get(&source.id()).map(PathBuf::as_path)
    }

    /// Resolve a filesystem reference; `$/` also works for named in-memory sources.
    pub fn resolve_import(&self, source: &Source, reference: &str) -> io::Result<PathBuf> {
        normalize_path(&self.import_path(source, reference)?)
    }

    fn import_path(&self, source: &Source, reference: &str) -> io::Result<PathBuf> {
        validate_reference(reference)?;
        if let Some(relative) = reference.strip_prefix("$/") {
            return Ok(self.library_root.join(relative));
        }
        let origin = self.path(source).ok_or_else(unknown_source)?;
        Ok(origin.parent().unwrap_or(Path::new(".")).join(reference))
    }

    async fn read_path_async(
        &mut self,
        path: PathBuf,
        execution: &Execution,
        cancellation: &Cancellation,
    ) -> Result<Source, LoadError> {
        let text = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(resin_executor::Error::Cancelled.into()),
            text = tokio::fs::read_to_string(&path) => text.map_err(|error| file_error(&path, error))?,
        };
        let previous = self.files.get(&path).cloned();
        let source_path = path.clone();
        let source = execution
            .run(cancellation, move |_| {
                let text: Arc<str> = text.into();
                match previous {
                    Some(file) => file.version(text),
                    None => Source::with_identity(
                        SourceId::file(source_path.clone()),
                        source_path.to_string_lossy().into_owned(),
                        text,
                    ),
                }
            })
            .await?;
        let file = self.files.entry(path.clone()).or_insert_with(|| File {
            cached: source.clone(),
            supplied: None,
        });
        file.cached = source.clone();
        self.origins.insert(source.id(), path);
        Ok(source)
    }

    fn file(&mut self, path: PathBuf, text: Arc<str>) -> &mut File {
        let file = self.files.entry(path.clone()).or_insert_with(|| File {
            cached: Source::with_identity(
                SourceId::file(path.clone()),
                path.to_string_lossy().into_owned(),
                text,
            ),
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
pub fn library_root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../resin"))
}

#[derive(Clone)]
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
    std::fs::read_to_string(path).map_err(|error| file_error(path, error))
}

fn file_error(path: &Path, error: io::Error) -> io::Error {
    io::Error::new(error.kind(), format!("{}: {error}", path.display()))
}

async fn normalize_path_async(
    path: &Path,
    execution: &Execution,
    cancellation: &Cancellation,
) -> Result<PathBuf, LoadError> {
    let path = path.to_path_buf();
    Ok(execution
        .run(cancellation, move |_| normalize_path(&path))
        .await??)
}

fn validate_reference(reference: &str) -> io::Result<()> {
    if let Some(relative) = reference.strip_prefix("$/") {
        if relative.is_empty()
            || Path::new(relative)
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "library imports must stay under $/",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_digests_still_compare_the_exact_content() {
        let first = Source::new("same", "first");
        // Force a collision privately; callers can construct only computed digests.
        let second = Source(Arc::new(Text {
            id: first.id(),
            name: first.0.name.clone(),
            digest: *first.content_hash(),
            text: "second".into(),
        }));
        assert_ne!(first, second);
        assert_ne!(first.cmp(&second), std::cmp::Ordering::Equal);
        let hashed: std::collections::HashSet<_> = [first.clone(), second.clone()].into();
        let ordered: BTreeSet<_> = [first, second].into();
        assert_eq!(hashed.len(), 2);
        assert_eq!(ordered.len(), 2);
    }
}
