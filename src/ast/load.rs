use std::{
    collections::{BTreeSet, HashMap},
    fmt, fs, io,
    path::{Component, Path, PathBuf},
};

use super::{AstGen, SourceFile, Span};

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
    pub imports: Vec<(Span, usize)>,
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
        error.message = format!("{}: {}", self.location(span), error.diagnostic).into();
        error
    }
}

#[derive(Debug, Clone)]
pub struct SourceError {
    pub path: PathBuf,
    pub span: Option<Span>,
    /// Human-readable CLI rendering, including the import chain.
    pub message: Box<str>,
    /// The diagnostic itself, without path prefixes or an import chain.
    pub diagnostic: String,
    pub related: Vec<SourceNote>,
}

impl SourceError {
    pub fn new(path: PathBuf, span: Option<Span>, diagnostic: String) -> Self {
        Self {
            message: format!("{}: {diagnostic}", path.display()).into(),
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

/// The bundled standard library. Executable frontends resolve environment overrides.
pub fn stdlib_path() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/stdlib"))
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
    let loaded = load_parsed(path, stdlib, sources, &mut |_, source| {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .expect("Resin parser");
        let tree = parser.parse(source, None).expect("parser language is set");
        let generator = AstGen::new(source);
        (
            generator.source_file(tree.root_node()),
            generator
                .errors(tree.root_node())
                .into_iter()
                .map(|error| (error.span, error.to_string()))
                .collect(),
        )
    });
    match loaded.errors.into_iter().next() {
        Some(error) => Err(error),
        None => Ok(loaded.program),
    }
}

pub(crate) struct Loaded {
    pub program: Program,
    pub errors: Vec<SourceError>,
    pub dependencies: BTreeSet<PathBuf>,
}

/// Loading retains accessible modules and failed dependencies for the next edit.
/// The compiler supplies its parse cache; strict callers use the same traversal.
pub(crate) fn load_parsed(
    path: &Path,
    stdlib: &Path,
    sources: &impl SourceProvider,
    parse: &mut impl FnMut(&Path, &str) -> (SourceFile, Vec<(Span, String)>),
) -> Loaded {
    let mut loader = Loader {
        stdlib,
        sources,
        parse,
        result: Loaded {
            program: Program {
                modules: Vec::new(),
            },
            errors: Vec::new(),
            dependencies: BTreeSet::new(),
        },
        loaded: HashMap::new(),
        active: Vec::new(),
    };
    if let Err(error) = loader.visit(path) {
        loader.result.errors.push(error);
    }
    loader.result
}

struct Loader<'a, S, P> {
    stdlib: &'a Path,
    sources: &'a S,
    parse: &'a mut P,
    result: Loaded,
    loaded: HashMap<PathBuf, usize>,
    active: Vec<PathBuf>,
}

impl<S: SourceProvider, P: FnMut(&Path, &str) -> (SourceFile, Vec<(Span, String)>)>
    Loader<'_, S, P>
{
    fn visit(&mut self, path: &Path) -> Result<usize, SourceError> {
        let error = |message: String| SourceError::new(path.to_path_buf(), None, message);
        self.result.dependencies.insert(path.to_path_buf());
        let canonical = self
            .sources
            .resolve(path)
            .map_err(|e| error(e.to_string()))?;
        self.result.dependencies.insert(canonical.clone());
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
        let (file, errors) = (self.parse)(&canonical, &source);
        let mut module = SourceModule {
            path: canonical.clone(),
            source,
            file,
            imports: Vec::new(),
        };
        self.result.errors.extend(
            errors
                .into_iter()
                .map(|(span, error)| module.error(span, error)),
        );
        self.active.push(canonical.clone());
        for import in &module.file.imports {
            let start = self.result.errors.len();
            let imported = resolve_import(&canonical, &import.val, self.stdlib)
                .map_err(|e| module.error(import.span, e))
                .and_then(|path| self.visit(&path));
            match imported {
                Ok(id) => module.imports.push((import.span, id)),
                Err(error) => self.result.errors.push(error),
            }
            for error in &mut self.result.errors[start..] {
                if error.span.is_none() {
                    *error = module.error(import.span, &*error);
                } else if error.path != canonical {
                    error.message =
                        format!("{}: {}", module.location(import.span), error.message).into();
                    error.related.push(SourceNote {
                        location: SourceLocation {
                            path: canonical.clone(),
                            span: import.span,
                        },
                        message: "imported here".into(),
                    });
                }
            }
        }
        self.active.pop();
        let id = self.result.program.modules.len();
        self.result.program.modules.push(module);
        self.loaded.insert(canonical, id);
        Ok(id)
    }
}
