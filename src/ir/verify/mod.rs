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
pub(crate) struct VerifiedModule {
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

    pub fn into_module(self) -> Module {
        self.module
    }
}

/// Direct IR clients borrow their module only for the duration of validation/use.
pub(crate) fn with_verified<T, E: From<VerifyError>>(
    module: &Module,
    use_module: impl FnOnce(Verified<'_>) -> Result<T, E>,
) -> Result<T, E> {
    let analysis = analyze(module)?;
    use_module(Verified {
        module,
        analysis: &analysis,
    })
}

pub(crate) struct ModuleTypes {
    pub functions: Vec<FunctionTypes>,
    pub types: crate::ir::TypeTable,
}

pub(crate) struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

fn analyze(module: &Module) -> Result<ModuleTypes, VerifyError> {
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
        if definition.name().is_some() {
            check_value(
                &module.types,
                definition.body().unwrap(),
                Location::type_definition(crate::ir::TypeId::from_index(index)),
            )?;
        }
    }

    for (index, definition) in module.types.iter().enumerate() {
        if let Some(hook) = definition.drop_hook() {
            let ty = crate::ir::TypeId::from_index(index);
            let location = Location::type_definition(ty);
            let Some(function) = module.functions.get(hook.index()) else {
                return Err(location.error(VerifyErrorKind::InvalidDropHook));
            };
            if function.locals.first().map(|local| &local.ty)
                != Some(&Ty::Pointer {
                    pointee: Box::new(Ty::Defined { definition: ty }),
                })
                || function.result != Ty::Unit
            {
                return Err(location.error(VerifyErrorKind::InvalidDropHook));
            }
        }
    }

    let functions = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| check_function(module, FunctionId::from_index(index), function))
        .collect::<Result<Vec<_>, _>>()?;
    let types = crate::ir::TypeTable::collect(module, &functions);
    Ok(ModuleTypes { functions, types })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
thread_local! { pub(crate) static ANALYSES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
