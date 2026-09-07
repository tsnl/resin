//! Lower declaration-selected shader artifacts for embedding in host code.
use super::{Error, c::Shader, glsl};
use crate::{
    ir::Module,
    toolchain::{self, Settings},
};

pub fn build_shaders(module: &Module, settings: &Settings) -> Result<Vec<Shader>, Error> {
    let sources = module
        .shaders
        .iter()
        .filter(|(_, entry)| entry.embedded)
        .map(|(&function, entry)| {
            let stage = entry.stage.parse()?;
            let source = glsl::emit_function(module, function, stage)?;
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
