//! Lower declaration-selected shader artifacts for embedding in host code.
use crate::{c::Shader, codegen::Error};
use crate::{
    lir::Module,
    toolchain::{self, Settings},
};

pub fn build_shaders(module: &Module, settings: &Settings) -> Result<Vec<Shader>, Error> {
    crate::lir_verifier::with_verified(module, |checked| build_verified(checked, settings))
}

pub(crate) fn build_verified(
    checked: crate::lir_verifier::Verified<'_>,
    settings: &Settings,
) -> Result<Vec<Shader>, Error> {
    let module = checked.module();
    let sources = module
        .shaders
        .iter()
        .filter(|(_, entry)| entry.embedded)
        .map(|(&function, entry)| {
            let stage = entry.stage.parse()?;
            let source = emit_glsl(checked, function, stage)?;
            Ok((function, stage, source))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    sources
        .into_iter()
        .map(|(function, stage, source)| {
            let bytes = toolchain::build_glsl(&source, stage, settings)?;
            Ok(Shader {
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

pub(super) fn emit_glsl(
    checked: crate::lir_verifier::Verified<'_>,
    entry: crate::types::FunctionId,
    stage: crate::glsl::Stage,
) -> Result<String, crate::codegen::Error> {
    crate::glsl::lower::generate(checked, entry, stage)
        .map(|module| crate::glsl::print::module(&module))
}
