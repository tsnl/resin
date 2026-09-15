#![allow(dead_code)]
use resin_hir::{GenerateError, GenerateErrorKind, Hir};
use resin_source::library_root;
use resin_source::prelude::*;
use std::path::Path;

pub fn generate(file: &resin_ast::SourceFile) -> Result<resin_lir::Module, GenerateError> {
    let tree = resin_hir::generate(file)?;
    let module = resin_lir::build_lir(&tree, &[], &resin_lir::LoweringOptions::default())
        .map_err(|mut errors| lowering_error(errors.remove(0)))?;
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
    let tree = resin_hir::build_hir(program).into_module()?;
    lower_program(&tree, &program.modules.last().expect("entry module").source)
}

/// Resolve only the imports explicitly declared by this test's source.
pub fn source_module(text: &str) -> Result<resin_lir::Module, SourceError> {
    let source = Source::new("test.resin", text);
    let mut loader = resin_source::Loader::new(library_root());
    let compilation = Hir::build(source.clone(), &mut loader, None);
    lower_program(compilation.hir()?, &source)
}

/// Load a file and its explicit imports, then lower the HIR already built by `build_hir`.
pub fn file_module(path: &Path) -> Result<resin_lir::Module, SourceError> {
    let compilation = build_hir_file(path)?;
    lower_program(compilation.hir()?, compilation.source())
}

fn lower_program(
    tree: &resin_hir::Module,
    entry: &Source,
) -> Result<resin_lir::Module, SourceError> {
    let module = resin_lir::build_lir(tree, &[], &resin_lir::LoweringOptions::default()).map_err(
        |mut errors| {
            let error = errors.remove(0);
            SourceError::new(
                error.source.clone().unwrap_or_else(|| entry.clone()),
                Some(error.span),
                error.to_string(),
            )
        },
    )?;
    resin_lir::VerifiedModule::new(module)
        .map(|verified| verified.into_module())
        .map_err(|error| SourceError::new(entry.clone(), None, error.to_string()))
}

/// Load an AST for inspection or mutation, preserving syntax even if HIR is invalid.
pub fn load(path: &Path) -> Result<resin_ast::Program, SourceError> {
    build_hir_file(path)?.program().cloned()
}

fn build_hir_file(path: &Path) -> Result<Hir, SourceError> {
    let mut loader = resin_source::Loader::new(library_root());
    let source = loader.load_file(path).map_err(|error| {
        SourceError::new(
            Source::new(path.display().to_string(), ""),
            None,
            error.to_string(),
        )
    })?;
    Ok(Hir::build(source, &mut loader, None))
}

pub fn shader_error(source: &str) -> String {
    let source = Source::new("shader-test.resin", source);
    let mut loader = resin_source::Loader::new(library_root());
    let output = Hir::build(source, &mut loader, None);
    match verified_lir(&output, "kernel", resin_lir::Profile::Shader) {
        Err(errors) => errors
            .into_iter()
            .map(|error| error.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
        Ok(lir) => {
            let directory = tempfile::TempDir::new().unwrap();
            resin_codegen::generate(lir.view(), None, directory.path())
                .unwrap_err()
                .to_string()
        }
    }
}

pub fn verified_lir(
    output: &Hir,
    entry: &str,
    profile: resin_lir::Profile,
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    verified_lir_with_options(
        output,
        entry,
        profile,
        &resin_lir::LoweringOptions::default(),
    )
}

pub fn verified_lir_with_options(
    output: &Hir,
    entry: &str,
    profile: resin_lir::Profile,
    options: &resin_lir::LoweringOptions,
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    let hir = output.hir().map_err(|error| vec![error])?;
    let request = resin_lir::Entry::exported(hir, entry, profile)
        .map_err(|error| vec![lir_source_error(output, error)])?;
    verified_entries(output, hir, &[request], options)
}

pub fn verified_entries(
    output: &Hir,
    hir: &resin_hir::Module,
    entries: &[resin_lir::Entry],
    options: &resin_lir::LoweringOptions,
) -> Result<resin_lir::VerifiedModule, Vec<SourceError>> {
    let lir = resin_lir::build_lir(hir, entries, options).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| lir_source_error(output, error))
            .collect::<Vec<_>>()
    })?;
    resin_lir::VerifiedModule::new(lir).map_err(|error| {
        vec![SourceError::new(
            output.source().clone(),
            None,
            format!("invalid LIR: {error}"),
        )]
    })
}

fn lir_source_error(output: &Hir, error: resin_lir::Error) -> SourceError {
    let mut diagnostic = SourceError::new(
        error
            .source
            .clone()
            .unwrap_or_else(|| output.source().clone()),
        Some(error.span),
        error.to_string(),
    );
    diagnostic
        .related
        .extend(error.applications.into_iter().filter_map(|application| {
            application.location.map(|location| SourceNote {
                location,
                message: format!(
                    "while instantiating {} with {:?} for {:?}",
                    application.function, application.arguments, application.profile
                ),
            })
        }));
    diagnostic
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
