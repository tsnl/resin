#![allow(dead_code)]
use resin_common::prelude::*;
use resin_compiler::{Compilation, SourceProvider, Sources, stdlib_path};
use std::path::Path;

pub fn generate(file: &resin_ast::SourceFile) -> Result<resin_lir::Module, GenerateError> {
    let tree = resin_hir::generate(file)?;
    let module = resin_lir::generate(&tree).map_err(|error| error.error)?;
    resin_lir_verifier::VerifiedModule::new(module)
        .map(|v| v.into_module())
        .map_err(|error| GenerateError {
            span: Span { start: 0, end: 0 },
            kind: GenerateErrorKind::InvalidIr(error.to_string().into()),
        })
}
pub fn generate_program(program: &resin_ast::Program) -> Result<resin_lir::Module, SourceError> {
    let tree = resin_hir::generate_program(program)?;
    let module = resin_lir::generate(&tree).map_err(|error| {
        let origin = &tree.origins.functions[&error.function];
        program
            .modules
            .iter()
            .find(|source| source.path == origin.path)
            .unwrap()
            .error(error.error.span, error.error)
    })?;
    resin_lir_verifier::VerifiedModule::new(module)
        .map(|v| v.into_module())
        .map_err(|error| SourceError::new(Default::default(), None, error.to_string()))
}
pub fn load(path: &Path) -> Result<resin_ast::Program, SourceError> {
    load_with(path, &stdlib_path(), &Sources::default())
}
pub fn load_with(
    path: &Path,
    stdlib: &Path,
    sources: &impl SourceProvider,
) -> Result<resin_ast::Program, SourceError> {
    Compilation::new(path, sources, stdlib).program().cloned()
}

pub fn build_shaders(
    module: &resin_lir::Module,
    tools: &resin_toolchain::Toolchain,
) -> Result<Vec<resin_codegen::Shader>, Box<dyn std::error::Error>> {
    module
        .shaders
        .iter()
        .filter(|(_, shader)| shader.embedded)
        .map(|(&function, shader)| {
            let stage = shader.stage.parse()?;
            let source = resin_codegen::emit_glsl_function(module, function, stage)?;
            let bytes = tools.build_glsl(&source, stage)?;
            Ok(resin_codegen::Shader {
                function,
                stage,
                words: bytes
                    .chunks_exact(4)
                    .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                    .collect(),
            })
        })
        .collect()
}
