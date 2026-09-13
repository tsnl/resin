#![allow(dead_code)]
use resin_compiler::Compiler;
use resin_hir::{GenerateError, GenerateErrorKind};
use resin_source::library_root;
use resin_source::prelude::*;
use std::path::Path;

pub fn generate(file: &resin_ast::SourceFile) -> Result<resin_lir::Module, GenerateError> {
    let tree = resin_hir::generate(file)?;
    let module = resin_lir::generate(&tree).map_err(lowering_error)?;
    Ok(resin_lir::VerifiedModule::new(module)
        .expect("lowering produces valid LIR")
        .into_module())
}

// Source-level tests share diagnostic assertions across the two frontend passes.
fn lowering_error(error: resin_lir::Error) -> GenerateError {
    let kind = match error.kind {
        resin_lir::ErrorKind::Type { kind } => GenerateErrorKind::Type { kind },
        resin_lir::ErrorKind::UnboundValue { name } => GenerateErrorKind::UnboundValue { name },
        kind @ (resin_lir::ErrorKind::MonomorphLimit { .. }
        | resin_lir::ErrorKind::TypeExpansionLimit { .. }
        | resin_lir::ErrorKind::TypeSizeLimit { .. }) => {
            panic!("unexpected resource limit in source test: {kind:?}")
        }
        resin_lir::ErrorKind::NotAPlace => GenerateErrorKind::NotAPlace,
        resin_lir::ErrorKind::InvalidHir { message } => {
            panic!("HIR generation produced an invalid tree: {message}")
        }
    };
    GenerateError {
        span: error.span,
        kind,
    }
}
pub fn generate_program(program: &resin_ast::Program) -> Result<resin_lir::Module, SourceError> {
    let tree = resin_hir::generate_program(program)?;
    let module = resin_lir::generate(&tree).map_err(|error| {
        let Some(source) = &error.source else {
            return SourceError::new(
                program.modules.last().expect("entry module").source.clone(),
                None,
                error.to_string(),
            );
        };
        if let Some(module) = program
            .modules
            .iter()
            .find(|module| module.source == *source)
        {
            return module.error(error.span, &error);
        }
        SourceError::new(source.clone(), Some(error.span), error.to_string())
    })?;
    resin_lir::VerifiedModule::new(module)
        .map(|v| v.into_module())
        .map_err(|error| {
            SourceError::new(
                program.modules.last().unwrap().source.clone(),
                None,
                error.to_string(),
            )
        })
}
pub fn load(path: &Path) -> Result<resin_ast::Program, SourceError> {
    let mut loader = resin_source::Loader::new(library_root());
    let source = loader.load_file(path).map_err(|error| {
        SourceError::new(
            Source::new(path.display().to_string(), ""),
            None,
            error.to_string(),
        )
    })?;
    Compiler::new()
        .compile(source, &mut loader)
        .program()
        .cloned()
}
