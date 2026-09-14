//! Validate module-wide identities before walking instruction graphs.
use super::{
    error::Location,
    rules::{check_definitions, check_value},
};
use crate::Module;
use crate::{VerifyError, VerifyErrorKind};
use resin_types::prelude::*;

pub(super) fn check(module: &Module) -> Result<(), VerifyError> {
    check_entries(module)?;
    check_definitions(&module.types)?;
    check_shaders(module)?;
    check_bodies(module)?;
    check_profiles(module)?;
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
    let parameters = function
        .locals
        .get(..function.parameter_count)
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidLocal { local: 0 }))?;
    resin_types::shader::validate(
        typer,
        &parameters
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
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
    if function.profile != crate::Profile::Host
        || function.parameter_count != 1
        || function.locals.first().map(|local| &local.ty) != Some(&pointer)
        || function.result != Ty::Unit
    {
        return Err(error());
    }
    Ok(())
}

// Function values retain the caller's semantic profile. Shader artifacts have a
// separate instruction and cannot be called through a host function value.
fn check_profiles(module: &Module) -> Result<(), VerifyError> {
    for &entry in module.entries.values() {
        check_profile(
            module,
            entry,
            crate::Profile::Host,
            Location::function(entry),
        )?;
    }
    for &entry in module.shaders.keys() {
        check_profile(
            module,
            entry,
            crate::Profile::Shader,
            Location::function(entry),
        )?;
    }
    for (index, function) in module.functions.iter().enumerate() {
        for (block, body) in function.blocks.iter().enumerate() {
            for (instruction, op) in body.instrs.iter().enumerate() {
                let location = Location::instruction(
                    FunctionId::from_index(index),
                    crate::BlockId::from_index(block),
                    instruction,
                );
                check_instruction_profiles(module, function.profile, op, location)?;
            }
        }
    }
    Ok(())
}

fn check_profile(
    module: &Module,
    id: FunctionId,
    expected: crate::Profile,
    location: Location,
) -> Result<(), VerifyError> {
    let function = module.functions.get(id.index()).ok_or_else(|| {
        location.error(VerifyErrorKind::InvalidFunction {
            function: id.index(),
        })
    })?;
    if function.profile != expected {
        return Err(location.error(VerifyErrorKind::InvalidProfile {
            expected,
            found: function.profile,
        }));
    }
    Ok(())
}

fn check_instruction_profiles(
    module: &Module,
    profile: crate::Profile,
    op: &crate::Instr,
    location: Location,
) -> Result<(), VerifyError> {
    let host = |id| check_profile(module, id, crate::Profile::Host, location);
    match op {
        crate::Instr::Function { function } => check_profile(module, *function, profile, location),
        crate::Instr::GpuNew { allocator, .. } | crate::Instr::GpuAllocate { allocator, .. } => {
            host(*allocator)
        }
        crate::Instr::GpuComputePipeline { factory, .. }
        | crate::Instr::GpuGraphicsPipeline { factory, .. } => host(*factory),
        crate::Instr::GpuDispatch {
            context,
            allocator,
            record,
        } => {
            host(*context)?;
            host(*allocator)?;
            host(*record)
        }
        crate::Instr::GpuDraw {
            context,
            allocator,
            record,
        } => {
            host(*context)?;
            if let Some(allocator) = allocator {
                host(*allocator)?;
            }
            host(*record)
        }
        _ => Ok(()),
    }
}
