//! Pipeline creation establishes a shader contract; recording checks that contract
//! together with the host argument shape and the source ownership bridges.
use super::{
    error::Location,
    gpu::allocation_error,
    instructions::pop,
    rules::{expect_type, expect_types},
};
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
        Instr::GpuComputePipeline {
            pipeline,
            factory,
            shader,
        } => create(module, pipeline, *factory, &[*shader], &args[0], location)?,
        Instr::GpuGraphicsPipeline {
            pipeline,
            factory,
            vertex,
            fragment,
        } => create(
            module,
            pipeline,
            *factory,
            &[*vertex, *fragment],
            &args[0],
            location,
        )?,
        Instr::GpuDispatch {
            projection,
            context,
            allocator,
            record,
        } => record_call(
            module,
            (*context, Some(*allocator), *record),
            &args,
            Some(projection),
            false,
            location,
        )?,
        Instr::GpuDraw {
            projection,
            context,
            allocator,
            record,
        } => record_call(
            module,
            (*context, *allocator, *record),
            &args,
            projection.as_ref(),
            true,
            location,
        )?,
        _ => unreachable!("pipeline instruction dispatch"),
    };
    stack.push(result);
    Ok(())
}

fn function(module: &Module, id: FunctionId, location: Location) -> Result<&Function, VerifyError> {
    module
        .functions
        .get(id.index())
        .filter(|f| f.parameter_count <= f.locals.len())
        .ok_or_else(|| location.error(VerifyErrorKind::InvalidGpuOperation))
}

fn create(
    module: &Module,
    pipeline: &Ty,
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
        stages.push((
            shader.locals[..shader.parameter_count]
                .iter()
                .map(|local| local.ty.clone())
                .collect::<Vec<_>>(),
            &shader.result,
            entry.stage.as_ref(),
        ));
    }
    let typer = TyperContext::from_definitions(module.types.clone());
    let root = resin_types::shader::pipeline_root(
        &typer,
        &stages
            .iter()
            .map(|(params, result, stage)| (params.as_slice(), *result, *stage))
            .collect::<Vec<_>>(),
    )
    .map_err(|_| invalid())?;
    let factory = function(module, factory, location)?;
    let mut params = vec![gpu.clone()];
    params.extend(shaders.iter().map(|_| Ty::byte_span()));
    expect_types(
        &params,
        &factory.locals[..factory.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
        location,
    )?;
    let Ty::Result {
        value: owner,
        error,
    } = &factory.result
    else {
        return Err(invalid());
    };
    let metadata =
        resin_types::gpu_pipeline_contract(&module.types, pipeline).map_err(|_| invalid())?;
    let kind = if shaders.len() == 1 {
        resin_types::GpuPipelineKind::Compute
    } else {
        resin_types::GpuPipelineKind::Graphics
    };
    if metadata.root != root || metadata.owner != **owner || metadata.kind != kind {
        return Err(invalid());
    }
    Ok(Ty::Result {
        value: Box::new(pipeline.clone()),
        error: error.clone(),
    })
}

fn record_call(
    module: &Module,
    bridges: (FunctionId, Option<FunctionId>, FunctionId),
    args: &[Ty],
    projection: Option<&resin_types::GpuProjectionPlan>,
    draw: bool,
    location: Location,
) -> Result<Ty, VerifyError> {
    let invalid = || location.error(VerifyErrorKind::InvalidGpuOperation);
    let (context, allocator, record) = bridges;
    let pipeline = &args[1];
    let metadata =
        resin_types::gpu_pipeline_contract(&module.types, pipeline).map_err(|_| invalid())?;
    let kind = if draw {
        resin_types::GpuPipelineKind::Graphics
    } else {
        resin_types::GpuPipelineKind::Compute
    };
    if metadata.kind != kind {
        return Err(invalid());
    }
    let (root, owner) = (&metadata.root, &metadata.owner);
    if *root == Ty::None {
        if projection.is_some() || args[2] != Ty::None {
            return Err(invalid());
        }
    } else {
        let plan = resin_types::gpu_projection_plan(&module.types, &args[2], root)
            .map_err(|_| invalid())?;
        if projection != Some(&plan) {
            return Err(invalid());
        }
    }
    let context = function(module, context, location)?;
    expect_types(
        std::slice::from_ref(owner),
        &context.locals[..context.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
        location,
    )?;
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
    expect_types(
        &params,
        &record.locals[..record.parameter_count]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
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
