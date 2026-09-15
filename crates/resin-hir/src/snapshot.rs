//! HIR for an entry source and its imports, including editor facts.
use crate::{Analysis, Completion, Hover, Module};
use resin_source::prelude::*;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub location: SourceLocation,
    pub message: String,
    pub related: Vec<SourceNote>,
}

/// Immutable HIR and editor facts for a source and its imports.
///
/// ```compile_fail,E0596
/// fn edit(output: &mut resin_hir::Hir) {
///     output.hir().unwrap().functions.clear();
/// }
/// ```
pub struct Hir {
    inner: Arc<Inner>,
}

struct Inner {
    source: Source,
    documents: BTreeMap<Source, Arc<resin_ast::ModuleDocument>>,
    syntax: BTreeMap<Source, Arc<resin_cst::Document>>,
    diagnostics: Vec<Diagnostic>,
    semantics: Analysis,
    graph: Vec<(Source, Vec<(Span, usize)>)>,
    program: Result<resin_ast::Program, SourceError>,
    module: Result<Module, SourceError>,
}

impl Clone for Hir {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Hir {
    /// Build HIR from a source and its imports. Pass a previous result to reuse
    /// unchanged CST documents and, when the import graph matches, the HIR itself.
    pub fn build(
        source: Source,
        loader: &mut resin_source::Loader,
        previous: Option<&Self>,
    ) -> Self {
        let built = resin_ast::build_program(
            source.clone(),
            loader,
            previous.map(|hir| &hir.inner.documents),
        );
        if let Some(previous) = previous
            && previous.inner.program.is_ok()
            && previous.inner.graph == built.graph()
            && built.diagnostics.is_empty()
        {
            return previous.clone();
        }
        Self::from_built(built)
    }

    fn from_built(built: resin_ast::BuiltProgram) -> Self {
        let graph = built.graph();
        let syntax = built.syntax();
        let load_error = built.diagnostics.first().cloned();
        let checked = crate::build_hir(&built.program);
        let hir_error = load_error.clone().or(checked.diagnostics.first().cloned());
        let errors = built
            .diagnostics
            .into_iter()
            .chain(checked.diagnostics)
            .collect::<Vec<_>>();
        Self {
            inner: Arc::new(Inner {
                source: built.source,
                documents: built.documents,
                syntax,
                diagnostics: errors.into_iter().map(diagnostic).collect(),
                semantics: checked.semantics,
                graph,
                program: load_error.map_or(Ok(built.program), Err),
                module: hir_error.map_or_else(|| Ok(checked.module.expect("successful HIR")), Err),
            }),
        }
    }

    pub fn source(&self) -> &Source {
        &self.inner.source
    }
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.inner.diagnostics
    }
    pub fn sources(&self) -> impl Iterator<Item = &Source> {
        self.inner.documents.keys()
    }
    pub fn program(&self) -> Result<&resin_ast::Program, SourceError> {
        self.inner.program.as_ref().map_err(Clone::clone)
    }
    pub fn hir(&self) -> Result<&Module, SourceError> {
        self.inner.module.as_ref().map_err(Clone::clone)
    }
    pub fn recovered_file(&self, source: &Source) -> Option<&resin_ast::SourceFile> {
        self.inner
            .documents
            .get(source)
            .map(|document| &document.file)
    }
    pub fn definition(&self, source: &Source, offset: usize) -> Option<SourceLocation> {
        self.inner
            .semantics
            .definition(&self.inner.syntax, source, offset)
    }
    pub fn hover(&self, source: &Source, offset: usize) -> Option<Hover> {
        self.inner
            .semantics
            .hover(&self.inner.syntax, source, offset)
    }
    pub fn completions(&self, source: &Source, offset: usize) -> Vec<Completion> {
        self.inner
            .semantics
            .completions(&self.inner.syntax, source, offset)
    }
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }
}

fn diagnostic(error: SourceError) -> Diagnostic {
    Diagnostic {
        location: SourceLocation {
            source: error.source,
            span: error.span.unwrap_or(Span { start: 0, end: 0 }),
        },
        message: error.diagnostic,
        related: error.related,
    }
}
