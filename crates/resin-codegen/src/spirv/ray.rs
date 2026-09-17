//! The ray-stage ABI is separate from ordinary device-address parameters.
use super::{Context, build_error, symbols::Slot};
use crate::Error;
use resin_types::prelude::*;
use rspirv::{dr::Operand, spirv::*};

#[derive(Clone)]
pub(super) struct Interface {
    push: Word,
    payload: Option<(Ty, Word)>,
    launch: Option<Word>,
    size: Option<Word>,
    hit: Vec<Word>,
    variables: Vec<Word>,
}

fn variable(context: &mut Context<'_>, ty: Word, storage: StorageClass) -> Word {
    let pointer = context.builder.type_pointer(None, storage, ty);
    context.builder.variable(pointer, None, storage, None)
}

fn builtin(context: &mut Context<'_>, ty: Word, name: BuiltIn) -> Word {
    let id = variable(context, ty, StorageClass::Input);
    context
        .builder
        .decorate(id, Decoration::BuiltIn, [Operand::BuiltIn(name)]);
    id
}

pub(super) fn declare(
    context: &mut Context<'_>,
    entry: FunctionId,
    stage: Stage,
    reachable: &[FunctionId],
) -> Result<(), Error> {
    if !matches!(
        stage,
        Stage::RayGeneration | Stage::Miss | Stage::ClosestHit
    ) {
        return Ok(());
    }
    context.builder.extension("SPV_KHR_ray_tracing");
    context.builder.capability(Capability::RayTracingKHR);
    let word = context.ty(&Ty::UInt64)?;
    let structure = context.builder.type_struct([word, word]);
    context.builder.decorate(structure, Decoration::Block, []);
    for i in 0..2 {
        context.builder.member_decorate(
            structure,
            i,
            Decoration::Offset,
            [Operand::LiteralBit32(i * 8)],
        );
    }
    let push = variable(context, structure, StorageClass::PushConstant);
    let mut interface = Interface {
        push,
        payload: None,
        launch: None,
        size: None,
        hit: vec![],
        variables: vec![push, context.failed],
    };
    let payload = if stage == Stage::RayGeneration {
        let mut payload = None;
        for id in reachable {
            for block in &context.module.functions[id.index()].blocks {
                for instruction in &block.instrs {
                    if let resin_lir::Instr::TraceRay { payload: ty } = instruction {
                        if payload.as_ref().is_some_and(|previous| previous != ty) {
                            return Err(Error::unsupported(
                                "a ray pipeline has one payload type".into(),
                            ));
                        }
                        payload = Some(ty.clone());
                    }
                }
            }
        }
        payload
    } else {
        Some(context.module.functions[entry.index()].locals[0].ty.clone())
    };
    if let Some(payload) = payload {
        let ty = context.ty(&payload)?;
        let storage = if stage == Stage::RayGeneration {
            StorageClass::RayPayloadKHR
        } else {
            StorageClass::IncomingRayPayloadKHR
        };
        let id = variable(context, ty, storage);
        context
            .builder
            .decorate(id, Decoration::Location, [Operand::LiteralBit32(0)]);
        interface.variables.push(id);
        interface.payload = Some((payload, id));
    }
    if stage == Stage::RayGeneration {
        let uint = context.ty(&Ty::UInt32)?;
        let vector = context.builder.type_vector(uint, 3);
        let launch = builtin(context, vector, BuiltIn::LaunchIdKHR);
        let size = builtin(context, vector, BuiltIn::LaunchSizeKHR);
        interface.launch = Some(launch);
        interface.size = Some(size);
        interface.variables.extend([launch, size]);
    }
    if stage == Stage::ClosestHit {
        let float = context.ty(&Ty::Float32)?;
        let uint = context.ty(&Ty::UInt32)?;
        interface
            .hit
            .push(builtin(context, float, BuiltIn::RayTmaxKHR));
        interface
            .hit
            .push(builtin(context, uint, BuiltIn::PrimitiveId));
        interface
            .hit
            .push(builtin(context, uint, BuiltIn::InstanceCustomIndexKHR));
        let vector = context.builder.type_vector(float, 2);
        interface
            .hit
            .push(variable(context, vector, StorageClass::HitAttributeKHR));
        interface.variables.extend_from_slice(&interface.hit);
    }
    context.ray = Some(interface);
    Ok(())
}

fn push_value(context: &mut Context<'_>, index: u32) -> Result<Word, Error> {
    let push = context.ray.as_ref().unwrap().push;
    let pointer = context.pointer_type(StorageClass::PushConstant, &Ty::UInt64)?;
    let index = context.constant_u32(index);
    let address = context
        .builder
        .access_chain(pointer, None, push, [index])
        .map_err(build_error)?;
    let word = context.ty(&Ty::UInt64)?;
    context
        .builder
        .load(word, None, address, None, [])
        .map_err(build_error)
}

pub(super) fn trace(context: &mut Context<'_>, args: &[Slot], payload: &Ty) -> Result<Word, Error> {
    let (_, target) = context
        .ray
        .as_ref()
        .and_then(|ray| ray.payload.as_ref())
        .ok_or_else(|| Error::unsupported("missing ray payload".into()))?
        .clone();
    context
        .builder
        .store(target, args[8].id, None, [])
        .map_err(build_error)?;
    let address = push_value(context, 1)?;
    let acceleration_type = context.builder.type_acceleration_structure_khr();
    let acceleration = context
        .builder
        .convert_u_to_acceleration_structure_khr(acceleration_type, None, address)
        .map_err(build_error)?;
    let float = context.ty(&Ty::Float32)?;
    let vector = context.builder.type_vector(float, 3);
    let origin = context
        .builder
        .composite_construct(vector, None, args[..3].iter().map(|arg| arg.id))
        .map_err(build_error)?;
    let direction = context
        .builder
        .composite_construct(vector, None, args[3..6].iter().map(|arg| arg.id))
        .map_err(build_error)?;
    let zero = context.constant_u32(0);
    let opaque = context.constant_u32(1);
    let mask = context.constant_u32(255);
    context
        .builder
        .trace_ray_khr(
            acceleration,
            opaque,
            mask,
            zero,
            zero,
            zero,
            origin,
            args[6].id,
            direction,
            args[7].id,
            target,
        )
        .map_err(build_error)?;
    let ty = context.ty(payload)?;
    context
        .builder
        .load(ty, None, target, None, [])
        .map_err(build_error)
}

pub(super) fn hit_info(context: &mut Context<'_>, result: &Ty) -> Result<Word, Error> {
    let hit = context.ray.as_ref().unwrap().hit.clone();
    let float = context.ty(&Ty::Float32)?;
    let uint = context.ty(&Ty::UInt32)?;
    let mut values = Vec::new();
    for (id, ty) in hit[..3].iter().zip([float, uint, uint]) {
        values.push(
            context
                .builder
                .load(ty, None, *id, None, [])
                .map_err(build_error)?,
        );
    }
    let vector = context.builder.type_vector(float, 2);
    let barycentric = context
        .builder
        .load(vector, None, hit[3], None, [])
        .map_err(build_error)?;
    for i in 0..2 {
        values.push(
            context
                .builder
                .composite_extract(float, None, barycentric, [i])
                .map_err(build_error)?,
        );
    }
    let ty = context.ty(result)?;
    context
        .builder
        .composite_construct(ty, None, values)
        .map_err(build_error)
}

pub(super) fn entry(
    context: &mut Context<'_>,
    entry: FunctionId,
    stage: Stage,
) -> Result<(), Error> {
    let interface = context.ray.as_ref().unwrap().clone();
    let void = context.builder.type_void();
    let signature = context.builder.type_function(void, []);
    let wrapper = context
        .builder
        .begin_function(void, None, FunctionControl::NONE, signature)
        .map_err(build_error)?;
    context.builder.begin_block(None).map_err(build_error)?;
    let input = if let Some(launch) = interface.launch {
        let uint = context.ty(&Ty::UInt32)?;
        let word = context.ty(&Ty::UInt64)?;
        let vector = context.builder.type_vector(uint, 3);
        let launch = context
            .builder
            .load(vector, None, launch, None, [])
            .map_err(build_error)?;
        let size = context
            .builder
            .load(vector, None, interface.size.unwrap(), None, [])
            .map_err(build_error)?;
        let mut coordinates = Vec::new();
        for value in [launch, size] {
            for i in 0..3 {
                let component = context
                    .builder
                    .composite_extract(uint, None, value, [i])
                    .map_err(build_error)?;
                coordinates.push(
                    context
                        .builder
                        .u_convert(word, None, component)
                        .map_err(build_error)?,
                );
            }
        }
        let zy = context
            .builder
            .i_mul(word, None, coordinates[2], coordinates[4])
            .map_err(build_error)?;
        let row = context
            .builder
            .i_add(word, None, zy, coordinates[1])
            .map_err(build_error)?;
        let offset = context
            .builder
            .i_mul(word, None, row, coordinates[3])
            .map_err(build_error)?;
        context
            .builder
            .i_add(word, None, offset, coordinates[0])
            .map_err(build_error)?
    } else {
        let (ty, payload) = interface.payload.as_ref().unwrap();
        let ty = context.ty(ty)?;
        context
            .builder
            .load(ty, None, *payload, None, [])
            .map_err(build_error)?
    };
    let root = push_value(context, 0)?;
    let result = context.module.functions[entry.index()].result.clone();
    let result = context.ty(&result)?;
    let value = context
        .builder
        .function_call(
            result,
            None,
            context.functions[entry.index()],
            [input, root],
        )
        .map_err(build_error)?;
    if stage != Stage::RayGeneration {
        context
            .builder
            .store(interface.payload.unwrap().1, value, None, [])
            .map_err(build_error)?;
    }
    context.builder.ret().map_err(build_error)?;
    context.builder.end_function().map_err(build_error)?;
    let model = match stage {
        Stage::RayGeneration => ExecutionModel::RayGenerationKHR,
        Stage::Miss => ExecutionModel::MissKHR,
        Stage::ClosestHit => ExecutionModel::ClosestHitKHR,
        _ => unreachable!(),
    };
    context
        .builder
        .entry_point(model, wrapper, "main", interface.variables);
    Ok(())
}
