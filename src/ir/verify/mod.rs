//! Stack and control-flow verification for typed IR.

use crate::ir::{Entry, FunctionId, GlobalId, Module, Ty};

mod error;
mod flow;
mod instructions;
mod types;

pub use error::{VerifyError, VerifyErrorKind, VerifyLocation};

use error::Location;
use flow::check_function;
use types::{check_definitions, check_value};

/// Check block and edge types; incoming edges must agree on the entry stack.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    analyze(module).map(|_| ())
}

pub(crate) struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

pub(crate) fn analyze(module: &Module) -> Result<Vec<FunctionTypes>, VerifyError> {
    for entry in module.entries.values() {
        match *entry {
            Entry::Function(function) if function.index() >= module.functions.len() => {
                return Err(
                    Location::function(function).error(VerifyErrorKind::InvalidFunction {
                        function: function.index(),
                    }),
                );
            }
            Entry::Global(global) if global.index() >= module.globals.len() => {
                return Err(
                    Location::global(global).error(VerifyErrorKind::InvalidGlobal {
                        global: global.index(),
                    }),
                );
            }
            _ => {}
        }
    }
    check_definitions(&module.types)?;
    for (index, definition) in module.types.iter().enumerate() {
        check_value(
            &module.types,
            definition.body().unwrap(),
            Location::type_definition(crate::ir::TypeId::from_index(index)),
        )?;
    }

    for (index, global) in module.globals.iter().enumerate() {
        check_value(
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
