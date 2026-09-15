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
        resin_lir::ErrorKind::UnsupportedProfile { profile, message } => {
            panic!("unexpected target failure in source test ({profile:?}): {message}")
        }
        resin_lir::ErrorKind::InvalidInstance { message } => {
            panic!("unexpected concrete instance failure in source test: {message}")
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

/// Lower an AST constructed or edited by a test. Unchanged files use `file_module`.
pub fn generate_program(program: &resin_ast::Program) -> Result<resin_lir::Module, SourceError> {
    let tree = resin_hir::generate_program(program)?;
    lower_program(&tree, &program.modules.last().expect("entry module").source)
}

/// Resolve only the imports explicitly declared by this test's source.
pub fn source_module(text: &str) -> Result<resin_lir::Module, SourceError> {
    let source = Source::new("test.resin", text);
    let mut loader = resin_source::Loader::new(library_root());
    let compilation = Compiler::new().analyze(source.clone(), &mut loader);
    lower_program(compilation.hir()?, &source)
}

/// Load a file and its explicit imports, then lower the HIR already built by analysis.
pub fn file_module(path: &Path) -> Result<resin_lir::Module, SourceError> {
    let compilation = analyze_file(path)?;
    lower_program(compilation.hir()?, compilation.source())
}

fn lower_program(
    tree: &resin_hir::Module,
    entry: &Source,
) -> Result<resin_lir::Module, SourceError> {
    let module = resin_lir::generate(tree).map_err(|error| {
        SourceError::new(
            error.source.clone().unwrap_or_else(|| entry.clone()),
            Some(error.span),
            error.to_string(),
        )
    })?;
    resin_lir::VerifiedModule::new(module)
        .map(|verified| verified.into_module())
        .map_err(|error| SourceError::new(entry.clone(), None, error.to_string()))
}

/// Load an AST for inspection or mutation, preserving syntax even if HIR is invalid.
pub fn load(path: &Path) -> Result<resin_ast::Program, SourceError> {
    analyze_file(path)?.program().cloned()
}

fn analyze_file(path: &Path) -> Result<std::sync::Arc<resin_compiler::Compilation>, SourceError> {
    let mut loader = resin_source::Loader::new(library_root());
    let source = loader.load_file(path).map_err(|error| {
        SourceError::new(
            Source::new(path.display().to_string(), ""),
            None,
            error.to_string(),
        )
    })?;
    Ok(Compiler::new().analyze(source, &mut loader))
}

pub fn shader_error(source: &str) -> String {
    let source = Source::new("shader-test.resin", source);
    let mut loader = resin_source::Loader::new(library_root());
    let compilation = Compiler::new().compile(
        source,
        &mut loader,
        &[resin_compiler::Target::Shader {
            entry: "kernel".into(),
        }],
    );
    match compilation.module() {
        Err(error) => error.to_string(),
        Ok(_) => {
            let directory = tempfile::TempDir::new().unwrap();
            resin_codegen::generate(compilation.verified().unwrap(), None, directory.path())
                .unwrap_err()
                .to_string()
        }
    }
}

/// Find a source nominal identity without depending on catalog insertion order.
pub fn nominal(module: &resin_lir::Module, name: &str) -> resin_types::TypeId {
    let index = module
        .types
        .iter()
        .position(|ty| ty.name().map(|name| name.as_ref()) == Some(name))
        .unwrap_or_else(|| panic!("missing nominal type {name}"));
    resin_types::TypeId::from_index(index)
}
