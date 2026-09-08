//! A compiler result retained for executable generation and source queries.
use super::syntax::Document;
use crate::{
    ast::{self, SourceLocation, SourceNote, SourceProvider, Span},
    ir,
    ir::generate::semantic::SemanticData,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: SourceLocation,
    pub message: String,
    pub related: Vec<SourceNote>,
}

/// An immutable analysis of one entry and its transitive dependencies.
pub struct Snapshot {
    pub(crate) documents: BTreeMap<PathBuf, Arc<Document>>,
    pub diagnostics: Vec<Diagnostic>,
    pub dependencies: BTreeSet<PathBuf>,
    pub(crate) semantics: SemanticData,
    program: Result<ast::Program, ast::SourceError>,
    module: Result<ir::Module, ast::SourceError>,
    verification: ir::verify::ModuleTypes,
}

impl Snapshot {
    pub fn new(entry: &Path, sources: &impl SourceProvider, stdlib: &Path) -> Self {
        Self::build(entry, sources, stdlib, &mut BTreeMap::new())
    }

    pub(crate) fn build(
        entry: &Path,
        sources: &impl SourceProvider,
        stdlib: &Path,
        parsed: &mut BTreeMap<PathBuf, Arc<Document>>,
    ) -> Self {
        let mut documents = BTreeMap::new();
        let loaded = ast::load_parsed(entry, stdlib, sources, &mut |path, text| {
            let document = if let Some(old) = parsed.get(path).filter(|d| d.text == text) {
                old.clone()
            } else {
                Arc::new(Document::reparse(
                    text.to_owned(),
                    parsed.get(path).map(Arc::as_ref),
                ))
            };
            parsed.insert(path.to_path_buf(), document.clone());
            let file = document.file.clone();
            let errors = document.errors.clone();
            documents.insert(path.to_path_buf(), document);
            (file, errors)
        });
        let load_error = loaded.errors.first().cloned();
        let checked = ir::analyze_program(&loaded.program);
        let errors = loaded
            .errors
            .into_iter()
            .chain(checked.diagnostics)
            .collect::<Vec<_>>();
        let compile_error = errors.first().cloned();
        Self {
            documents,
            diagnostics: errors
                .into_iter()
                .map(|error| Diagnostic {
                    location: SourceLocation {
                        path: error.path,
                        span: error.span.unwrap_or(Span { start: 0, end: 0 }),
                    },
                    message: error.diagnostic,
                    related: error.related,
                })
                .collect(),
            dependencies: loaded.dependencies,
            semantics: checked.semantics,
            program: load_error.map_or(Ok(loaded.program), Err),
            module: compile_error.map_or_else(
                || {
                    Ok(checked
                        .module
                        .expect("successful compilation has verified IR"))
                },
                Err,
            ),
            verification: checked.verification,
        }
    }

    /// The compiler AST, possibly containing holes. `program` and `module` stay strict.
    pub fn recovered_file(&self, path: &Path) -> Option<&ast::SourceFile> {
        self.documents.get(path).map(|d| &d.file)
    }

    pub fn program(&self) -> Result<&ast::Program, ast::SourceError> {
        self.program.as_ref().map_err(Clone::clone)
    }

    pub fn module(&self) -> Result<&ir::Module, ast::SourceError> {
        self.module.as_ref().map_err(Clone::clone)
    }

    pub(crate) fn verified(&self) -> Result<ir::verify::Verified<'_>, crate::backend::Error> {
        let module = self.module()?;
        Ok(ir::verify::Verified {
            module,
            analysis: &self.verification,
        })
    }

    pub fn syntax_tree(&self, path: &Path) -> Option<&tree_sitter::Tree> {
        self.documents.get(path).map(|d| &d.tree)
    }

    pub fn sources(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.documents
            .iter()
            .map(|(path, doc)| (path.as_path(), doc.text.as_str()))
    }
}
