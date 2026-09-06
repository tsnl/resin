use std::{
    collections::HashMap,
    fmt, fs,
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
    pub imports: Vec<usize>,
}

impl SourceModule {
    pub fn location(&self, span: Span) -> String {
        let prefix = &self.source[..span.start.min(self.source.len())];
        let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let column = prefix.rsplit('\n').next().unwrap().chars().count() + 1;
        format!("{}:{line}:{column}", self.path.display())
    }

    pub fn error(&self, span: Span, message: impl fmt::Display) -> SourceError {
        SourceError {
            path: self.path.clone(),
            message: format!("{}: {message}", self.location(span)),
        }
    }
}

#[derive(Debug)]
pub struct SourceError {
    pub path: PathBuf,
    pub message: String,
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SourceError {}

pub fn load(path: &Path) -> Result<Program, SourceError> {
    let stdlib = std::env::var_os("RESIN_STDLIB")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/stdlib")));
    let mut loader = Loader {
        stdlib,
        modules: Vec::new(),
        loaded: HashMap::new(),
        active: Vec::new(),
    };
    loader.visit(path)?;
    Ok(Program {
        modules: loader.modules,
    })
}

struct Loader {
    stdlib: PathBuf,
    modules: Vec<SourceModule>,
    loaded: HashMap<PathBuf, usize>,
    active: Vec<PathBuf>,
}

impl Loader {
    fn visit(&mut self, path: &Path) -> Result<usize, SourceError> {
        let error = |message: String| SourceError {
            path: path.to_path_buf(),
            message: format!("{}: {message}", path.display()),
        };
        let canonical = fs::canonicalize(path).map_err(|e| error(e.to_string()))?;
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
        let source = fs::read_to_string(&canonical).map_err(|e| error(e.to_string()))?;
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .expect("Resin parser");
        let tree = parser.parse(&source, None).expect("parser language is set");
        let file = AstGen::new(&source)
            .gen_source_file(tree.root_node())
            .map_err(|e| error(e.to_string()))?;
        let mut module = SourceModule {
            path: canonical.clone(),
            source,
            file,
            imports: Vec::new(),
        };
        self.active.push(canonical.clone());
        for import in &module.file.imports {
            let path = if let Some(relative) = import.val.strip_prefix("std/") {
                let path = Path::new(relative);
                if relative.is_empty()
                    || path
                        .components()
                        .any(|c| !matches!(c, Component::Normal(_)))
                {
                    return Err(
                        module.error(import.span, "standard-library imports must stay under std/")
                    );
                }
                self.stdlib.join(path)
            } else {
                canonical.parent().unwrap().join(import.val.as_ref())
            };
            let id = self
                .visit(&path)
                .map_err(|e| module.error(import.span, e))?;
            module.imports.push(id);
        }
        self.active.pop();
        let id = self.modules.len();
        self.modules.push(module);
        self.loaded.insert(canonical, id);
        Ok(id)
    }
}
