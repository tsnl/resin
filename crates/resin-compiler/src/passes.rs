//! The compiler pipeline, read from source to machine-independent instructions.
use crate::diagnostic::{GenerateError, GenerateErrorKind};
use crate::{ast, hir, lir, lir_verifier};

pub(super) fn lower(
    program: &ast::Program,
    hir: &hir::Module,
) -> Result<lir_verifier::VerifiedModule, Vec<ast::SourceError>> {
    let lir = lir::analyze(hir).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lowering_error(program, hir, error))
            .collect::<Vec<_>>()
    })?;
    lir_verifier::VerifiedModule::new(lir).map_err(|error| {
        vec![ast::SourceError::new(
            Default::default(),
            None,
            invalid_lir(error).to_string(),
        )]
    })
}

fn lowering_error(
    program: &ast::Program,
    hir: &hir::Module,
    error: lir::Error,
) -> ast::SourceError {
    let origin = &hir.origins.functions[&error.function];
    if let Some(source) = program
        .modules
        .iter()
        .find(|source| source.path == origin.path)
    {
        return source.error(error.error.span, error.error);
    }
    ast::SourceError::new(
        origin.path.clone(),
        Some(error.error.span),
        error.to_string(),
    )
}

fn invalid_lir(error: lir_verifier::VerifyError) -> GenerateError {
    GenerateError {
        span: ast::Span { start: 0, end: 0 },
        kind: GenerateErrorKind::InvalidIr(error.to_string().into()),
    }
}
