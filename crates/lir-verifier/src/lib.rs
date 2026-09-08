//! Stack and control-flow verification for typed IR.
use resin_common::types;
use resin_lir as lir;

use crate::lir::{FunctionId, Module, Ty};

mod error;
mod flow;
mod instructions;
mod module;
mod rules;
mod table;

pub use error::{VerifyError, VerifyErrorKind, VerifyLocation};

use flow::check_function;

/// Check block and edge types; incoming edges must agree on the entry stack.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    analyze(module).map(|_| ())
}

/// A borrow of LIR and the analysis certifying that exact immutable module.
#[derive(Clone, Copy)]
pub struct Verified<'a> {
    module: &'a Module,
    analysis: &'a ModuleTypes,
}

impl<'a> Verified<'a> {
    pub fn module(self) -> &'a Module {
        self.module
    }

    pub fn analysis(self) -> &'a ModuleTypes {
        self.analysis
    }
}

/// The checked module and the analysis that certifies this exact immutable IR.
pub struct VerifiedModule {
    module: Module,
    analysis: ModuleTypes,
}

impl VerifiedModule {
    pub fn new(mut module: Module) -> Result<Self, VerifyError> {
        let analysis = analyze(&module)?;
        module.types = analysis.types.clone();
        Ok(Self { module, analysis })
    }

    pub fn view(&self) -> Verified<'_> {
        Verified {
            module: &self.module,
            analysis: &self.analysis,
        }
    }

    /// Discard the certificate and recover editable LIR.
    pub fn into_module(self) -> Module {
        self.module
    }
}

/// Direct IR clients borrow their module only for the duration of validation/use.
pub fn with_verified<T, E: From<VerifyError>>(
    module: &Module,
    use_module: impl FnOnce(Verified<'_>) -> Result<T, E>,
) -> Result<T, E> {
    let analysis = analyze(module)?;
    use_module(Verified {
        module,
        analysis: &analysis,
    })
}

pub struct ModuleTypes {
    pub functions: Vec<FunctionTypes>,
    pub types: crate::lir::TypeTable,
}

pub struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

fn analyze(module: &Module) -> Result<ModuleTypes, VerifyError> {
    module::check(module)?;
    let functions = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| check_function(module, FunctionId::from_index(index), function))
        .collect::<Result<Vec<_>, _>>()?;
    let types = table::collect(module, &functions);
    Ok(ModuleTypes { functions, types })
}

#[cfg(test)]
mod tests;
