//! GLSL target language. Lower verified LIR, then print a shader source tree.
mod language;
pub mod lower;
pub mod print;
pub use crate::types::shader::Stage;
pub use language::*;

use crate::{Error, lir, lir_verifier};

pub fn emit(module: &lir::Module, entry: &str, stage: Stage) -> Result<String, Error> {
    let Some(function) = module.entries.get(entry) else {
        return Err(Error(format!(
            "expected an exported shader function named {entry:?}"
        )));
    };
    emit_function(module, *function, stage)
}

pub fn emit_function(
    module: &lir::Module,
    entry: lir::FunctionId,
    stage: Stage,
) -> Result<String, Error> {
    lir_verifier::with_verified(module, |checked| {
        lower::generate(checked, entry, stage).map(|module| print::module(&module))
    })
}
