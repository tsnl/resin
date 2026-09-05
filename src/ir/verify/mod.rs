//! Stack and control-flow verification for typed IR.

use crate::ir::{FunctionId, GlobalId, Module};

mod error;
mod flow;
mod instructions;
mod types;

pub use error::{VerifyError, VerifyErrorKind, VerifyLocation};

use error::Location;
use flow::check_function;
use types::{check_definitions, check_type};

/// Check block and edge types; incoming edges must agree on the entry stack.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    check_definitions(&module.types)?;

    for (index, global) in module.globals.iter().enumerate() {
        check_type(
            &module.types,
            &global.ty,
            Location::global(GlobalId::from_index(index)),
        )?;
    }

    for (index, function) in module.functions.iter().enumerate() {
        check_function(module, FunctionId::from_index(index), function)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests;
