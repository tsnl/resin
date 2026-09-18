//! Borrowed binding access. The initial wire format retains source field offsets
//! but replaces each opaque owner slot with a device address and zero padding.
use super::{Context, build_error, symbols::Slot};
use crate::Error;
use resin_lir::{Instr, Module};
use resin_types::prelude::*;
use rspirv::spirv::*;

pub(super) fn address_layout(
    module: &Module,
    ty: &Ty,
) -> Result<resin_types::layout::Layout, Error> {
    if ty.needs_drop(&module.types)
        && resin_types::gpu_projection_plan(&module.types, ty, ty).is_ok()
    {
        resin_types::layout::value(&module.types, ty).map_err(|e| Error::invalid(e.to_string()))
    } else {
        crate::layout::layout(module, ty)
    }
}

pub(super) fn access(
    context: &mut Context<'_>,
    instr: &Instr,
    args: &[Slot],
) -> Result<Word, Error> {
    let Ty::Reference { referent, .. } = &args[0].ty else {
        unreachable!("verified binding");
    };
    let (element, _) = resin_types::gpu_buffer_binding(&context.module.types, referent)
        .map_err(Error::unsupported)?;
    let element = element.clone();
    let layout = address_layout(context.module, referent)?;
    let uint = context.ty(&Ty::UInt64)?;
    let offset = context.constant_u64(layout.offsets[1] as u64);
    let length_address = context
        .builder
        .i_add(uint, None, args[0].id, offset)
        .map_err(build_error)?;
    let length = context.physical_load(&Ty::UInt64, length_address)?;
    let boolean = context.ty(&Ty::Bool)?;
    let valid = context
        .builder
        .u_less_than(boolean, None, args[1].id, length)
        .map_err(build_error)?;
    let hit = context.builder.id();
    let miss = context.builder.id();
    let merge = context.builder.id();
    context
        .builder
        .selection_merge(merge, SelectionControl::NONE)
        .map_err(build_error)?;
    context
        .builder
        .branch_conditional(valid, hit, miss, [])
        .map_err(build_error)?;
    context
        .builder
        .begin_block(Some(hit))
        .map_err(build_error)?;
    let base = context.physical_load(&Ty::UInt64, args[0].id)?;
    let stride = context.constant_u64(crate::layout::layout(context.module, &element)?.size as u64);
    let offset = context
        .builder
        .i_mul(uint, None, args[1].id, stride)
        .map_err(build_error)?;
    let address = context
        .builder
        .i_add(uint, None, base, offset)
        .map_err(build_error)?;
    let loaded = if matches!(instr, Instr::GpuBufferStore) {
        context.physical_store(&element, address, args[2].id)?;
        None
    } else {
        Some(context.physical_load(&element, address)?)
    };
    context.builder.branch(merge).map_err(build_error)?;
    context
        .builder
        .begin_block(Some(miss))
        .map_err(build_error)?;
    context.builder.branch(merge).map_err(build_error)?;
    context
        .builder
        .begin_block(Some(merge))
        .map_err(build_error)?;
    if let Some(loaded) = loaded {
        let ty = context.ty(&element)?;
        let zero = context.zero(&element)?;
        context
            .builder
            .phi(ty, None, [(loaded, hit), (zero, miss)])
            .map_err(build_error)
    } else {
        Ok(context.constant_u32(0))
    }
}
