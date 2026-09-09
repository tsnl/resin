use crate::SourceProvider;
use crate::ast::{Program, SourceFile, SourceModule, Span};
use crate::common_source::{SourceError, SourceLocation, SourceNote};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::{Component, Path, PathBuf},
};

pub(crate) fn resolve_import(
    source: &Path,
    import: &str,
    stdlib: &Path,
) -> Result<PathBuf, &'static str> {
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

pub(crate) struct Loaded {
    pub program: Program,
    pub errors: Vec<SourceError>,
    pub dependencies: BTreeSet<PathBuf>,
    pub resolutions: BTreeMap<PathBuf, PathBuf>,
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
            resolutions: BTreeMap::new(),
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
        let spelling = std::path::absolute(path).map_err(|e| error(e.to_string()))?;
        self.result.dependencies.insert(spelling.clone());
        let canonical = self
            .sources
            .resolve(path)
            .map_err(|e| error(e.to_string()))?;
        self.result.dependencies.insert(canonical.clone());
        self.result.resolutions.insert(spelling, canonical.clone());
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
