//! Join completed semantic facts with their immutable syntax inputs.
use crate::{CheckedProgram, Diagnostic, Hir};
use resin_source::prelude::*;
use std::sync::Arc;

pub(super) fn complete(inputs: Arc<resin_ast::BuiltProgram>, checked: CheckedProgram) -> Hir {
    let error = inputs
        .diagnostics
        .first()
        .cloned()
        .or_else(|| checked.diagnostics.first().cloned());
    let diagnostics = inputs
        .diagnostics
        .iter()
        .cloned()
        .chain(checked.diagnostics)
        .map(diagnostic)
        .collect::<Vec<_>>()
        .into();
    let module = error.map_or_else(
        || Ok(Arc::new(checked.module.expect("successful HIR"))),
        Err,
    );
    Hir {
        syntax: Arc::new(inputs.syntax()),
        inputs,
        diagnostics,
        semantics: Arc::new(checked.semantics),
        module,
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
