//! The compiler pipeline, read from source to machine-independent instructions.
use resin_common::diagnostic::{GenerateError, GenerateErrorKind};

pub(super) fn lower(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
) -> Result<resin_lir_verifier::VerifiedModule, Vec<resin_ast::SourceError>> {
    let lir = resin_lir::analyze(hir).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lowering_error(program, hir, error))
            .collect::<Vec<_>>()
    })?;
    resin_lir_verifier::VerifiedModule::new(lir).map_err(|error| {
        vec![resin_ast::SourceError::new(
            Default::default(),
            None,
            invalid_lir(error).to_string(),
        )]
    })
}

fn lowering_error(
    program: &resin_ast::Program,
    hir: &resin_hir::Module,
    error: resin_lir::Error,
) -> resin_ast::SourceError {
    let origin = &hir.origins.functions[&error.function];
    if let Some(source) = program
        .modules
        .iter()
        .find(|source| source.path == origin.path)
    {
        return source.error(error.error.span, error.error);
    }
    resin_ast::SourceError::new(
        origin.path.clone(),
        Some(error.error.span),
        error.to_string(),
    )
}

fn invalid_lir(error: resin_lir_verifier::VerifyError) -> GenerateError {
    GenerateError {
        span: resin_ast::Span { start: 0, end: 0 },
        kind: GenerateErrorKind::InvalidIr(error.to_string().into()),
    }
}
