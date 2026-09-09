//! Validate module-wide identities before walking instruction graphs.
use crate::{
    VerifyError, VerifyErrorKind,
    error::Location,
    rules::{check_definitions, check_value},
};
use resin_common::types::{TyperContext, shader};
use resin_lir::{FunctionId, Module, Ty, TypeId};

pub(super) fn check(module: &Module) -> Result<(), VerifyError> {
    check_entries(module)?;
    check_definitions(&module.types)?;
    check_shaders(module)?;
    check_bodies(module)?;
    check_drop_hooks(module)
}

fn check_entries(module: &Module) -> Result<(), VerifyError> {
    for &function in module.entries.values() {
        if function.index() >= module.functions.len() {
            return Err(
                Location::function(function).error(VerifyErrorKind::InvalidFunction {
                    function: function.index(),
                }),
            );
        }
    }
    Ok(())
}

fn check_shaders(module: &Module) -> Result<(), VerifyError> {
    let typer = TyperContext::from_definitions(module.types.clone());
    for (&id, entry) in &module.shaders {
        check_shader(module, &typer, id, &entry.stage)?;
    }
    Ok(())
}

fn check_shader(
    module: &Module,
    typer: &TyperContext,
    id: FunctionId,
    stage: &str,
) -> Result<(), VerifyError> {
    let location = Location::function(id);
    let function = module
        .functions
        .get(id.index())
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidShader))?;
    let parameter = function
        .locals
        .first()
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidLocal { local: 0 }))?;
    shader::validate(
        typer,
        &parameter.ty,
        &function.result,
        function.foreign.is_some(),
        stage,
    )
    .map(|_| ())
    .map_err(|_| location.error(VerifyErrorKind::InvalidShader))
}

fn check_bodies(module: &Module) -> Result<(), VerifyError> {
    for (index, definition) in module.types.iter().enumerate() {
        if definition.name().is_some() {
            let location = Location::type_definition(TypeId::from_index(index));
            check_value(&module.types, definition.body().unwrap(), location)?;
        }
    }
    Ok(())
}

fn check_drop_hooks(module: &Module) -> Result<(), VerifyError> {
    for (index, definition) in module.types.iter().enumerate() {
        if let Some(hook) = definition.drop_hook() {
            check_drop_hook(module, TypeId::from_index(index), hook)?;
        }
    }
    Ok(())
}

fn check_drop_hook(module: &Module, ty: TypeId, hook: FunctionId) -> Result<(), VerifyError> {
    let error = || Location::type_definition(ty).error(VerifyErrorKind::InvalidDropHook);
    let function = module.functions.get(hook.index()).ok_or_else(error)?;
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Defined { definition: ty }),
    };
    if function.locals.first().map(|local| &local.ty) != Some(&pointer)
        || function.result != Ty::Unit
    {
        return Err(error());
    }
    Ok(())
}
