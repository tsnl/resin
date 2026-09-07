//! Lower declaration-selected shader artifacts for embedding in host code.
use super::{Error, c::Shader, glsl};
use crate::ir::Module;
use crate::toolchain::Settings;

pub fn build_shaders(module: &Module, settings: &Settings) -> Result<Vec<Shader>, Error> {
    let mut shaders: Vec<Shader> = Vec::new();
    for (&function, entry) in &module.shaders {
        if !entry.embedded {
            continue;
        }
        let stage = entry.stage.parse()?;
        let source = glsl::emit_function(module, function, stage)?;
        let bytes = crate::toolchain::build_glsl(&source, stage, settings)?;
        shaders.push(Shader {
            function,
            stage,
            words: bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect(),
        });
    }
    Ok(shaders)
}
