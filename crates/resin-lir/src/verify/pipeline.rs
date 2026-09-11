//! Pipeline creation establishes a shader contract; recording checks that contract
//! together with the host argument shape and the source ownership bridges.
use super::{error::Location, gpu::allocation_error, instructions::pop, rules::expect_type};
use crate::{Function, Instr, Module, VerifyError, VerifyErrorKind};
use resin_types::prelude::*;

pub(super) fn check(
    module: &Module,
    instr: &Instr,
    stack: &mut Vec<Ty>,
    location: Location,
) -> Result<(), VerifyError> {
    let args = pop(stack, super::stack_effect(instr).pops, location)?;
    let result = match instr {
        Instr::GpuComputePipeline { factory, shader } => {
            create(module, *factory, &[*shader], &args[0], location)?
        }
        Instr::GpuGraphicsPipeline {
            factory,
            vertex,
            fragment,
        } => create(module, *factory, &[*vertex, *fragment], &args[0], location)?,
        Instr::GpuDispatch {
            context,
            allocator,
            record,
        } => record_call(
            module,
            *context,
            Some(*allocator),
            *record,
            &args,
            false,
            location,
        )?,
        Instr::GpuDraw {
            context,
            allocator,
            record,
        } => record_call(module, *context, *allocator, *record, &args, true, location)?,
        _ => unreachable!("pipeline instruction dispatch"),
    };
    stack.push(result);
    Ok(())
}

fn function(module: &Module, id: FunctionId, location: Location) -> Result<&Function, VerifyError> {
    module
        .functions
        .get(id.index())
        .filter(|f| !f.locals.is_empty())
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidGpuOperation))
}

fn create(
    module: &Module,
    factory: FunctionId,
    shaders: &[FunctionId],
    gpu: &Ty,
    location: Location,
) -> Result<Ty, VerifyError> {
    let invalid = || location.error(VerifyErrorKind::InvalidGpuOperation);
    let mut stages = Vec::new();
    for &shader in shaders {
        let entry = module
            .shaders
            .get(&shader)
            .filter(|e| e.embedded)
            .ok_or_else(invalid)?;
        let shader = function(module, shader, location)?;
        stages.push((&shader.locals[0].ty, &shader.result, entry.stage.as_ref()));
    }
    let typer = TyperContext::from_definitions(module.types.clone());
    let root = resin_types::shader::pipeline_root(&typer, &stages).map_err(|_| invalid())?;
    let factory = function(module, factory, location)?;
    let mut params = vec![gpu.clone()];
    params.extend(shaders.iter().map(|_| Ty::Span {
        element: Box::new(Ty::UInt8),
    }));
    expect_type(
        Ty::parameter(&params),
        factory.locals[0].ty.clone(),
        location,
    )?;
    let Ty::Result {
        value: owner,
        error,
    } = &factory.result
    else {
        return Err(invalid());
    };
    if !matches!(&**owner, Ty::Arc { .. }) {
        return Err(invalid());
    }
    let value = if shaders.len() == 1 {
        Ty::GpuComputePipeline {
            root: Box::new(root),
            owner: owner.clone(),
        }
    } else {
        Ty::GpuGraphicsPipeline {
            root: Box::new(root),
            owner: owner.clone(),
        }
    };
    if value.gpu_pipeline_argument(&module.types).is_none() {
        return Err(invalid());
    }
    Ok(Ty::Result {
        value: Box::new(value),
        error: error.clone(),
    })
}

fn record_call(
    module: &Module,
    context: FunctionId,
    allocator: Option<FunctionId>,
    record: FunctionId,
    args: &[Ty],
    draw: bool,
    location: Location,
) -> Result<Ty, VerifyError> {
    let invalid = || location.error(VerifyErrorKind::InvalidGpuOperation);
    let pipeline = &args[1];
    if !matches!(
        (draw, pipeline),
        (false, Ty::GpuComputePipeline { .. }) | (true, Ty::GpuGraphicsPipeline { .. })
    ) {
        return Err(invalid());
    }
    let (root, owner) = pipeline.gpu_pipeline().ok_or_else(invalid)?;
    expect_type(
        pipeline
            .gpu_pipeline_argument(&module.types)
            .ok_or_else(invalid)?,
        args[2].clone(),
        location,
    )?;
    let context = function(module, context, location)?;
    expect_type(owner.clone(), context.locals[0].ty.clone(), location)?;
    let record = function(module, record, location)?;
    let root_arg = if draw {
        Ty::union_of([Ty::GpuArguments, Ty::None])
    } else {
        Ty::GpuArguments
    };
    let mut params = vec![args[0].clone(), owner.clone(), root_arg];
    for ty in &args[3..] {
        expect_type(Ty::UInt32, ty.clone(), location)?;
        params.push(ty.clone());
    }
    expect_type(
        Ty::parameter(&params),
        record.locals[0].ty.clone(),
        location,
    )?;
    let Ty::Result { value, error } = &record.result else {
        return Err(invalid());
    };
    expect_type(Ty::Unit, *value.clone(), location)?;
    match (root == &Ty::None, allocator) {
        (true, None) => {}
        (false, Some(allocator)) => expect_type(
            *error.clone(),
            allocation_error(module, allocator, &context.result, location)?,
            location,
        )?,
        _ => return Err(invalid()),
    }
    Ok(record.result.clone())
}
