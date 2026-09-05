//! Stack and control-flow verification for typed IR.

use crate::ir::{FunctionId, GlobalId, Module, Ty};

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
    analyze(module).map(|_| ())
}

pub(crate) struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

pub(crate) fn analyze(module: &Module) -> Result<Vec<FunctionTypes>, VerifyError> {
    check_definitions(&module.types)?;

    for (index, global) in module.globals.iter().enumerate() {
        check_type(
            &module.types,
            &global.ty,
            Location::global(GlobalId::from_index(index)),
        )?;
    }

    module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| check_function(module, FunctionId::from_index(index), function))
        .collect()
}

#[cfg(test)]
mod tests;
