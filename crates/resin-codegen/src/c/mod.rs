//! C11 target language. Lower verified LIR, then print the resulting source tree.
mod language;
pub mod lower;
pub mod print;
pub use language::*;
pub use lower::Shader;

use crate::{Error, lir, lir_verifier};

pub fn emit(module: &lir::Module, entry: &str) -> Result<String, Error> {
    emit_with_shaders(module, entry, &[])
}

pub fn emit_with_shaders(
    module: &lir::Module,
    entry: &str,
    shaders: &[Shader],
) -> Result<String, Error> {
    lir_verifier::with_verified(module, |checked| {
        lower::generate(checked, entry, shaders).map(|module| print::module(&module))
    })
}
