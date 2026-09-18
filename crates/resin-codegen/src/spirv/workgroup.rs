//! Explicit shared-state operations; source control flow stays per invocation.
use super::{Context, build_error, symbols::Slot};
use crate::Error;
use resin_lir::Instr;
use resin_types::prelude::*;
use rspirv::spirv::*;
use std::collections::HashSet;

pub(super) struct Builtins {
    pub lane: Word,
    pub width: Word,
}

/// These helpers have no physical-address implementation. Emit them at calls
/// with a known Workgroup reference, using the existing root/path specialization.
pub(super) fn required_functions(
    module: &resin_lir::Module,
    order: &[FunctionId],
    ids: &[Word],
) -> HashSet<Word> {
    let mut required = HashSet::new();
    for &function in order {
        let needs_group = module.functions[function.index()]
            .blocks
            .iter()
            .flat_map(|block| &block.instrs)
            .any(|instruction| match instruction {
                Instr::Workgroup { .. } => true,
                Instr::Function { function } => required.contains(&ids[function.index()]),
                _ => false,
            });
        if needs_group {
            required.insert(ids[function.index()]);
        }
    }
    required
}

pub(super) fn operation(
    context: &mut Context<'_>,
    operation: WorkgroupOperation,
    state: &Slot,
) -> Result<Word, Error> {
    if state
        .local
        .as_ref()
        .is_none_or(|local| local.storage != StorageClass::Workgroup)
    {
        return Err(Error::unsupported("workgroup operations require the compute entry's shared state, not private locals or device pointers".into()));
    }
    let Some(builtins) = &context.workgroup else {
        return Err(Error::unsupported(
            "workgroup operations are only available in compute shaders".into(),
        ));
    };
    let (lane, width) = (builtins.lane, builtins.width);
    match operation {
        WorkgroupOperation::Sync => {
            barrier(context)?;
            Ok(context.constant_u32(0))
        }
        WorkgroupOperation::LaneIndex => {
            let value = lane_index(context, lane)?;
            let word = context.ty(&Ty::UInt64)?;
            context
                .builder
                .u_convert(word, None, value)
                .map_err(build_error)
        }
        WorkgroupOperation::LaneCount => {
            let word = context.ty(&Ty::UInt64)?;
            context
                .builder
                .u_convert(word, None, width)
                .map_err(build_error)
        }
    }
}

fn lane_index(context: &mut Context<'_>, input: Word) -> Result<Word, Error> {
    let uint = context.ty(&Ty::UInt32)?;
    let vector = context.builder.type_vector(uint, 3);
    let value = context
        .builder
        .load(vector, None, input, None, [])
        .map_err(build_error)?;
    context
        .builder
        .composite_extract(uint, None, value, [0])
        .map_err(build_error)
}

pub(super) fn initialize(context: &mut Context<'_>, state: Word, ty: &Ty) -> Result<(), Error> {
    let lane = lane_index(context, context.workgroup.as_ref().unwrap().lane)?;
    let boolean = context.ty(&Ty::Bool)?;
    let zero = context.constant_u32(0);
    let first = context
        .builder
        .i_equal(boolean, None, lane, zero)
        .map_err(build_error)?;
    let active = context.builder.id();
    let merge = context.builder.id();
    context
        .builder
        .selection_merge(merge, SelectionControl::NONE)
        .map_err(build_error)?;
    context
        .builder
        .branch_conditional(first, active, merge, [])
        .map_err(build_error)?;
    context
        .builder
        .begin_block(Some(active))
        .map_err(build_error)?;
    let initial = context.zero(ty)?;
    context
        .builder
        .store(state, initial, None, [])
        .map_err(build_error)?;
    context.builder.branch(merge).map_err(build_error)?;
    context
        .builder
        .begin_block(Some(merge))
        .map_err(build_error)?;
    barrier(context)
}

fn barrier(context: &mut Context<'_>) -> Result<(), Error> {
    let scope = context.constant_u32(Scope::Workgroup as u32);
    let semantics = context.constant_u32(
        (MemorySemantics::ACQUIRE_RELEASE
            | MemorySemantics::WORKGROUP_MEMORY
            | MemorySemantics::UNIFORM_MEMORY)
            .bits(),
    );
    context
        .builder
        .control_barrier(scope, scope, semantics)
        .map_err(build_error)
}
