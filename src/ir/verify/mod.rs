//! Stack and control-flow verification for typed IR.

use crate::ir::{FunctionId, Module, Ty};

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

#[derive(Clone, Copy)]
pub(crate) struct Verified<'a> {
    pub module: &'a Module,
    pub analysis: &'a [FunctionTypes],
}

pub(crate) struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

pub(crate) fn analyze(module: &Module) -> Result<Vec<FunctionTypes>, VerifyError> {
    #[cfg(test)]
    ANALYSES.set(ANALYSES.get() + 1);
    for &function in module.entries.values() {
        if function.index() >= module.functions.len() {
            return Err(
                Location::function(function).error(VerifyErrorKind::InvalidFunction {
                    function: function.index(),
                }),
            );
        }
    }
    check_definitions(&module.types)?;
    let typer = crate::ir::TyperContext::from_definitions(module.types.clone());
    for (&id, entry) in &module.shaders {
        let function = module
            .functions
            .get(id.index())
            .ok_or_else(|| Location::function(id).error(VerifyErrorKind::InvalidShader))?;
        crate::ir::shader::validate(&typer, function, &entry.stage)
            .map_err(|_| Location::function(id).error(VerifyErrorKind::InvalidShader))?;
    }

    for (index, definition) in module.types.iter().enumerate() {
        check_value(
            &module.types,
            definition.body().unwrap(),
            Location::type_definition(crate::ir::TypeId::from_index(index)),
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

#[cfg(test)]
thread_local! { pub(crate) static ANALYSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
