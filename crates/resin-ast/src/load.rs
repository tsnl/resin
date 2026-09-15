//! Sources and imports → a program of recovered AST modules.
use super::{Parsed, Program, SourceModule};
use resin_source::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

/// One recovered file together with the CST it was built from.
pub struct ModuleDocument {
    pub source: Source,
    pub syntax: Arc<resin_cst::Document>,
    pub file: crate::SourceFile,
    pub errors: Vec<(Span, String)>,
}

impl ModuleDocument {
    fn build(source: Source, previous: Option<&Self>) -> Self {
        let syntax = resin_cst::build_cst(
            source.text(),
            previous.map(|document| document.syntax.as_ref()),
        );
        let Parsed { file, errors } = crate::build_ast(&syntax);
        Self {
            source,
            syntax: Arc::new(syntax),
            file,
            errors,
        }
    }
}

/// AST for an entry and its imports, with CST documents and parse diagnostics.
pub struct BuiltProgram {
    pub source: Source,
    pub program: Program,
    pub documents: BTreeMap<Source, Arc<ModuleDocument>>,
    pub diagnostics: Vec<SourceError>,
}

impl BuiltProgram {
    pub fn graph(&self) -> Vec<(Source, Vec<(Span, usize)>)> {
        self.program
            .modules
            .iter()
            .map(|module| (module.source.clone(), module.imports.clone()))
            .collect()
    }

    pub fn syntax(&self) -> BTreeMap<Source, Arc<resin_cst::Document>> {
        self.documents
            .iter()
            .map(|(source, document)| (source.clone(), document.syntax.clone()))
            .collect()
    }
}

/// Build a program by resolving imports, reusing previous CST documents when the
/// source version is unchanged.
pub fn build_program(
    source: Source,
    loader: &mut resin_source::Loader,
    previous: Option<&BTreeMap<Source, Arc<ModuleDocument>>>,
) -> BuiltProgram {
    let mut traversal = Traversal {
        loader,
        previous,
        result: Loaded {
            program: Program {
                modules: Vec::new(),
            },
            errors: Vec::new(),
            documents: BTreeMap::new(),
        },
        loaded: BTreeMap::new(),
        active: Vec::new(),
        versions: BTreeMap::new(),
        resolutions: BTreeMap::new(),
    };
    if let Err(error) = traversal.visit(source.clone()) {
        traversal.result.errors.push(error);
    }
    BuiltProgram {
        source,
        program: traversal.result.program,
        documents: traversal.result.documents,
        diagnostics: traversal.result.errors,
    }
}

struct Loaded {
    program: Program,
    errors: Vec<SourceError>,
    documents: BTreeMap<Source, Arc<ModuleDocument>>,
}

struct Traversal<'a> {
    loader: &'a mut resin_source::Loader,
    previous: Option<&'a BTreeMap<Source, Arc<ModuleDocument>>>,
    result: Loaded,
    loaded: BTreeMap<SourceId, usize>,
    active: Vec<Source>,
    versions: BTreeMap<SourceId, Source>,
    resolutions: BTreeMap<(SourceId, String), Result<Source, String>>,
}

impl Traversal<'_> {
    fn check_version(&mut self, source: &Source) -> Result<(), SourceError> {
        if let Some(old) = self.versions.get(&source.id())
            && old != source
        {
            return Err(SourceError::new(
                source.clone(),
                None,
                "loader returned different versions of the same module during one compilation"
                    .into(),
            ));
        }
        self.versions.insert(source.id(), source.clone());
        Ok(())
    }

    fn parse(&self, source: &Source) -> Arc<ModuleDocument> {
        if let Some(old) = self
            .previous
            .and_then(|documents| documents.get(source))
            .filter(|old| old.source == *source)
        {
            return old.clone();
        }
        Arc::new(ModuleDocument::build(
            source.clone(),
            self.previous.and_then(|documents| {
                documents
                    .iter()
                    .find(|(candidate, _)| candidate.id() == source.id())
                    .map(|(_, document)| document.as_ref())
            }),
        ))
    }

    fn visit(&mut self, source: Source) -> Result<usize, SourceError> {
        self.check_version(&source)?;
        if self.active.iter().any(|active| active.id() == source.id()) {
            let chain = self
                .active
                .iter()
                .chain(std::iter::once(&source))
                .map(Source::name)
                .collect::<Vec<_>>()
                .join(" -> ");
            return Err(SourceError::new(
                source,
                None,
                format!("cyclic source import: {chain}"),
            ));
        }
        if let Some(&id) = self.loaded.get(&source.id()) {
            return Ok(id);
        }
        let document = self.parse(&source);
        let mut module = SourceModule {
            source: source.clone(),
            file: document.file.clone(),
            imports: Vec::new(),
        };
        self.result.errors.extend(
            document
                .errors
                .iter()
                .map(|(span, error)| module.error(*span, error)),
        );
        self.result.documents.insert(source.clone(), document);
        self.active.push(source.clone());
        self.visit_imports(&mut module);
        self.active.pop();
        let id = self.result.program.modules.len();
        self.result.program.modules.push(module);
        self.loaded.insert(source.id(), id);
        Ok(id)
    }

    fn resolve(&mut self, source: &Source, reference: &str) -> Result<Source, String> {
        self.resolutions
            .entry((source.id(), reference.into()))
            .or_insert_with(|| {
                self.loader
                    .load_import(source, reference)
                    .map_err(|error| error.to_string())
            })
            .clone()
    }

    fn visit_imports(&mut self, module: &mut SourceModule) {
        for import in &module.file.imports {
            let start = self.result.errors.len();
            let imported = self
                .resolve(&module.source, &import.val)
                .map_err(|error| module.error(import.span, error))
                .and_then(|source| self.visit(source));
            match imported {
                Ok(id) => module.imports.push((import.span, id)),
                Err(error) => self.result.errors.push(error),
            }
            for error in &mut self.result.errors[start..] {
                imported_at(error, module, import.span);
            }
        }
    }
}

fn imported_at(error: &mut SourceError, importer: &SourceModule, span: Span) {
    if error.span.is_none() {
        *error = importer.error(span, &*error);
    } else if error.source != importer.source {
        error.message = format!("{}: {}", importer.location(span), error.message).into();
        error.related.push(SourceNote {
            location: SourceLocation {
                source: importer.source.clone(),
                span,
            },
            message: "imported here".into(),
        });
    }
}
