//! A compiler result retained for executable generation and source queries.
use super::syntax::Document;
use crate::lir_verifier;
use crate::{
    ast::{self, Span},
    hir::Analysis as HirAnalysis,
    lir, {Diagnostic, SourceLocation, SourceProvider},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::Arc,
};

/// An immutable analysis of one entry and its transitive dependencies.
pub(super) struct Data {
    pub(super) documents: BTreeMap<PathBuf, Arc<Document>>,
    pub diagnostics: Vec<Diagnostic>,
    pub dependencies: BTreeSet<PathBuf>,
    pub(super) resolutions: BTreeMap<PathBuf, PathBuf>,
    pub(super) semantics: HirAnalysis,
    program: Result<ast::Program, ast::SourceError>,
    hir: Result<crate::hir::Module, ast::SourceError>,
    module: Result<lir_verifier::VerifiedModule, ast::SourceError>,
}

impl Data {
    pub(super) fn new(entry: &Path, sources: &impl SourceProvider, stdlib: &Path) -> Self {
        Self::build(entry, sources, stdlib, &mut BTreeMap::new())
    }

    pub(super) fn build(
        entry: &Path,
        sources: &impl SourceProvider,
        stdlib: &Path,
        parsed: &mut BTreeMap<PathBuf, Arc<Document>>,
    ) -> Self {
        let mut documents = BTreeMap::new();
        let loaded = crate::loading::load_parsed(entry, stdlib, sources, &mut |path, text| {
            let document = if let Some(old) = parsed.get(path).filter(|d| d.source() == text) {
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
        let mut checked = crate::hir::analyze_program(&loaded.program);
        let hir_error = load_error
            .clone()
            .or_else(|| checked.diagnostics.first().cloned());
        let mut lowered = None;
        if let Some(hir) = &checked.module {
            match super::passes::lower(&loaded.program, hir) {
                Ok(module) => lowered = Some(module),
                Err(errors) => checked.diagnostics.extend(errors),
            }
        }
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
            resolutions: loaded.resolutions,
            semantics: checked.semantics,
            program: load_error.map_or(Ok(loaded.program), Err),
            hir: hir_error.map_or_else(|| Ok(checked.module.expect("successful HIR")), Err),
            module: compile_error.map_or_else(
                || Ok(lowered.expect("successful compilation has verified LIR")),
                Err,
            ),
        }
    }

    /// The compiler AST, possibly containing holes. `program` and `module` stay strict.
    pub(super) fn recovered_file(&self, path: &Path) -> Option<&ast::SourceFile> {
        self.documents.get(path).map(|d| &d.file)
    }

    pub(super) fn program(&self) -> Result<&ast::Program, ast::SourceError> {
        self.program.as_ref().map_err(Clone::clone)
    }

    /// Resolved tree before storage and control-flow lowering.
    pub(super) fn hir(&self) -> Result<&crate::hir::Module, ast::SourceError> {
        self.hir.as_ref().map_err(Clone::clone)
    }

    pub(super) fn module(&self) -> Result<&lir::Module, ast::SourceError> {
        self.module
            .as_ref()
            .map(|checked| checked.view().module())
            .map_err(Clone::clone)
    }

    pub(super) fn verified(&self) -> Result<lir_verifier::Verified<'_>, crate::Error> {
        Ok(self.module.as_ref().map_err(Clone::clone)?.view())
    }

    pub(super) fn sources(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.documents
            .iter()
            .map(|(path, doc)| (path.as_path(), doc.source()))
    }
}
