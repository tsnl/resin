use std::{
    collections::HashMap,
    fmt, fs, io,
    path::{Component, Path, PathBuf},
};

use super::{AstError, AstGen, SourceFile, Span};

#[derive(Debug, Clone)]
pub struct Program {
    /// Dependencies precede their consumers; the entry file is last.
    pub modules: Vec<SourceModule>,
}

#[derive(Debug, Clone)]
pub struct SourceModule {
    pub path: PathBuf,
    pub source: String,
    pub file: SourceFile,
    pub imports: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SourceLocation {
    pub path: PathBuf,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct SourceNote {
    pub location: SourceLocation,
    pub message: String,
}

impl SourceModule {
    pub fn location(&self, span: Span) -> String {
        let mut start = span.start.min(self.source.len());
        while !self.source.is_char_boundary(start) {
            start -= 1;
        }
        let prefix = &self.source[..start];
        let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let column = prefix.rsplit('\n').next().unwrap().chars().count() + 1;
        format!("{}:{line}:{column}", self.path.display())
    }

    pub fn error(&self, span: Span, message: impl fmt::Display) -> SourceError {
        let mut error = SourceError::new(self.path.clone(), Some(span), message.to_string());
        error.message = format!("{}: {}", self.location(span), error.diagnostic);
        error
    }
}

#[derive(Debug, Clone)]
pub struct SourceError {
    pub path: PathBuf,
    pub span: Option<Span>,
    /// Human-readable CLI rendering, including the import chain.
    pub message: String,
    /// The diagnostic itself, without path prefixes or an import chain.
    pub diagnostic: String,
    pub related: Vec<SourceNote>,
}

impl SourceError {
    pub fn new(path: PathBuf, span: Option<Span>, diagnostic: String) -> Self {
        Self {
            message: format!("{}: {diagnostic}", path.display()),
            path,
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

/// Source access shared by the CLI and editor. An editor may resolve/read a file
/// from memory before it exists on disk.
pub trait SourceProvider {
    fn resolve(&self, path: &Path) -> io::Result<PathBuf>;
    fn read(&self, path: &Path) -> io::Result<String>;
    /// A cached parse of the exact text returned by `read`, when available.
    fn parsed(&self, _path: &Path) -> Option<Result<SourceFile, AstError>> {
        None
    }
}

pub struct FileSystem;
impl SourceProvider for FileSystem {
    fn resolve(&self, path: &Path) -> io::Result<PathBuf> {
        fs::canonicalize(path)
    }
    fn read(&self, path: &Path) -> io::Result<String> {
        fs::read_to_string(path)
    }
}

pub fn stdlib_path() -> PathBuf {
    std::env::var_os("RESIN_STDLIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/stdlib")))
}

pub fn resolve_import(source: &Path, import: &str, stdlib: &Path) -> Result<PathBuf, &'static str> {
    if let Some(relative) = import.strip_prefix("std/") {
        let path = Path::new(relative);
        if relative.is_empty()
            || path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
        {
            return Err("standard-library imports must stay under std/");
        }
        Ok(stdlib.join(path))
    } else {
        Ok(source.parent().unwrap_or(Path::new(".")).join(import))
    }
}

pub fn load(path: &Path) -> Result<Program, SourceError> {
    load_with(path, &stdlib_path(), &FileSystem)
}

pub fn load_with(
    path: &Path,
    stdlib: &Path,
    sources: &impl SourceProvider,
) -> Result<Program, SourceError> {
    let mut loader = Loader {
        stdlib,
        sources,
        modules: Vec::new(),
        loaded: HashMap::new(),
        active: Vec::new(),
    };
    loader.visit(path)?;
    Ok(Program {
        modules: loader.modules,
    })
}

struct Loader<'a, S> {
    stdlib: &'a Path,
    sources: &'a S,
    modules: Vec<SourceModule>,
    loaded: HashMap<PathBuf, usize>,
    active: Vec<PathBuf>,
}

impl<S: SourceProvider> Loader<'_, S> {
    fn visit(&mut self, path: &Path) -> Result<usize, SourceError> {
        let error = |message: String| SourceError::new(path.to_path_buf(), None, message);
        let canonical = self
            .sources
            .resolve(path)
            .map_err(|e| error(e.to_string()))?;
        if self.active.contains(&canonical) {
            let chain = self
                .active
                .iter()
                .chain(std::iter::once(&canonical))
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(" -> ");
            return Err(error(format!("cyclic source import: {chain}")));
        }
        if let Some(&id) = self.loaded.get(&canonical) {
            return Ok(id);
        }
        let source = self
            .sources
            .read(&canonical)
            .map_err(|e| error(e.to_string()))?;
        let file = self
            .sources
            .parsed(&canonical)
            .unwrap_or_else(|| {
                let mut parser = tree_sitter::Parser::new();
                parser
                    .set_language(&tree_sitter_resin::LANGUAGE.into())
                    .expect("Resin parser");
                let tree = parser.parse(&source, None).expect("parser language is set");
                AstGen::new(&source).gen_source_file(tree.root_node())
            })
            .map_err(|e| SourceError::new(canonical.clone(), Some(e.span), e.to_string()))?;
        let mut module = SourceModule {
            path: canonical.clone(),
            source,
            file,
            imports: Vec::new(),
        };
        self.active.push(canonical.clone());
        for import in &module.file.imports {
            let path = resolve_import(&canonical, &import.val, self.stdlib)
                .map_err(|e| module.error(import.span, e))?;
            let id = self.visit(&path).map_err(|mut e| {
                if e.span.is_none() {
                    return module.error(import.span, e);
                }
                e.message = format!("{}: {}", module.location(import.span), e.message);
                e.related.push(SourceNote {
                    location: SourceLocation {
                        path: canonical.clone(),
                        span: import.span,
                    },
                    message: "imported here".into(),
                });
                e
            })?;
            module.imports.push(id);
        }
        self.active.pop();
        let id = self.modules.len();
        self.modules.push(module);
        self.loaded.insert(canonical, id);
        Ok(id)
    }
}
