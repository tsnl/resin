//! Lower declaration-selected shader artifacts for embedding in host code.
use crate::Error;
use crate::codegen::Shader;
use crate::toolchain::Toolchain;

pub(crate) fn build_verified(
    checked: crate::lir_verifier::Verified<'_>,
    settings: &Toolchain,
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
            let bytes = settings.build_glsl(&source, stage)?;
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
    stage: crate::codegen::Stage,
) -> Result<String, crate::Error> {
    crate::codegen::generate_glsl(checked, entry, stage)
        .map(|module| crate::codegen::print_glsl(&module))
        .map_err(Into::into)
}
