//! A resource reference is a static path into the entry's binding schema, never
//! a device address. Binding zero carries constants and view offset/length data;
//! subsequent storage-buffer bindings follow source field order.
use super::{Context, build_error, ops, symbols::Slot};
use crate::Error;
use resin_lir::{Instr, Module};
use resin_types::prelude::*;
use rspirv::{dr::Operand, spirv::*};
use std::collections::BTreeMap;

pub(super) struct Resources {
    pub push: Word,
    pub variables: BTreeMap<(u32, Ty), Word>,
    bindings: BTreeMap<usize, (u32, bool)>,
}

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

pub(super) fn declare(context: &mut Context<'_>, root: &Ty) -> Result<(), Error> {
    let plan = resin_types::gpu_projection_plan(&context.module.types, root, root)
        .map_err(Error::unsupported)?;
    let mut bindings = BTreeMap::new();
    binding_fields(context.module, &plan, 0, &mut bindings)?;
    let word = context.ty(&Ty::UInt64)?;
    let block = context.builder.id();
    context.builder.type_struct_id(Some(block), [word]);
    context.builder.decorate(block, Decoration::Block, []);
    context
        .builder
        .member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let pointer = context
        .builder
        .type_pointer(None, StorageClass::PushConstant, block);
    let push = context
        .builder
        .variable(pointer, None, StorageClass::PushConstant, None);
    context.builder.name(push, "parameter_byte_offset");
    context.resources = Some(Resources {
        push,
        variables: BTreeMap::new(),
        bindings,
    });
    Ok(())
}

fn binding_fields(
    module: &Module,
    plan: &resin_types::GpuProjectionPlan,
    offset: usize,
    bindings: &mut BTreeMap<usize, (u32, bool)>,
) -> Result<(), Error> {
    use resin_types::GpuProjectionOperation;
    match &plan.operation {
        GpuProjectionOperation::Buffer { writable, .. } => {
            let binding = u32::try_from(bindings.len() + 1)
                .map_err(|_| Error::unsupported("too many resource bindings".into()))?;
            bindings.insert(offset, (binding, *writable));
        }
        GpuProjectionOperation::Record { fields } => {
            let layout = address_layout(module, &plan.target)?;
            for (field, field_offset) in fields.iter().zip(layout.offsets) {
                binding_fields(module, field, offset + field_offset, bindings)?;
            }
        }
        GpuProjectionOperation::Array { element, length } => {
            let stride = address_layout(module, &element.target)?.size;
            for index in 0..*length {
                binding_fields(module, element, offset + index * stride, bindings)?;
            }
        }
        GpuProjectionOperation::Copy => (),
        _ => {
            return Err(Error::unsupported(
                "resource bundles cannot contain physical device pointers".into(),
            ));
        }
    }
    Ok(())
}

// Scalar aliases of one SSBO binding preserve the packed shared layout, including
// byte fields, without reinterpreting logical pointers or expanding word stores
// into read/modify/write operations that race with neighboring bytes. Vulkan allows
// multiple storage variables at one binding; Aliased preserves their dependencies.
// https://docs.vulkan.org/spec/latest/chapters/interfaces.html#interfaces-resources
fn variable(
    context: &mut Context<'_>,
    binding: u32,
    scalar: &Ty,
    writable: bool,
) -> Result<Word, Error> {
    let key = (binding, scalar.clone());
    if let Some(&id) = context.resources.as_ref().unwrap().variables.get(&key) {
        return Ok(id);
    }
    let ty = context.ty(scalar)?;
    let array = context.builder.id();
    context.builder.type_runtime_array_id(Some(array), ty);
    let stride = crate::layout::layout(context.module, scalar)?.size as u32;
    context.builder.decorate(
        array,
        Decoration::ArrayStride,
        [Operand::LiteralBit32(stride)],
    );
    let block = context.builder.id();
    context.builder.type_struct_id(Some(block), [array]);
    context.builder.decorate(block, Decoration::Block, []);
    context
        .builder
        .member_decorate(block, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
    let pointer = context
        .builder
        .type_pointer(None, StorageClass::StorageBuffer, block);
    let function = context.builder.selected_function();
    let selected_block = context.builder.selected_block();
    context.builder.select_function(None).map_err(build_error)?;
    let id = context
        .builder
        .variable(pointer, None, StorageClass::StorageBuffer, None);
    context
        .builder
        .select_function(function)
        .map_err(build_error)?;
    context
        .builder
        .select_block(selected_block)
        .map_err(build_error)?;
    context
        .builder
        .decorate(id, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
    context
        .builder
        .decorate(id, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    context.builder.decorate(id, Decoration::Aliased, []);
    if !writable {
        context.builder.decorate(id, Decoration::NonWritable, []);
    }
    context
        .builder
        .name(id, format!("binding_{binding}_{scalar:?}"));
    context
        .resources
        .as_mut()
        .unwrap()
        .variables
        .insert(key, id);
    Ok(id)
}

fn byte_offset(context: &mut Context<'_>, base: Word, offset: usize) -> Result<Word, Error> {
    let offset = context.constant_u64(offset as u64);
    ops::emit(context, Op::IAdd, &Ty::UInt64, &[base, offset])
}

fn scalar_pointer(
    context: &mut Context<'_>,
    binding: u32,
    ty: &Ty,
    offset: Word,
    writable: bool,
) -> Result<Word, Error> {
    let variable = variable(context, binding, ty, writable)?;
    let stride = context.constant_u64(crate::layout::layout(context.module, ty)?.size as u64);
    let index = ops::emit(context, Op::UDiv, &Ty::UInt64, &[offset, stride])?;
    let pointer = context.pointer_type(StorageClass::StorageBuffer, ty)?;
    let zero = context.constant_u32(0);
    context
        .builder
        .access_chain(pointer, None, variable, [zero, index])
        .map_err(build_error)
}

fn load(
    context: &mut Context<'_>,
    binding: u32,
    ty: &Ty,
    offset: Word,
    writable: bool,
) -> Result<Word, Error> {
    let fields = match context.shape(ty) {
        Ty::Record { fields } => Some(fields.iter().map(|f| f.ty.clone()).collect::<Vec<_>>()),
        Ty::Array { element, length } => Some(vec![*element.clone(); *length]),
        _ => None,
    };
    if let Some(fields) = fields {
        let layout = crate::layout::layout(context.module, ty)?;
        let offsets = if let Ty::Array { element, .. } = context.shape(ty) {
            let stride = crate::layout::layout(context.module, element)?.size;
            (0..fields.len()).map(|i| i * stride).collect()
        } else {
            layout.offsets
        };
        let mut values = Vec::new();
        for (field, relative) in fields.iter().zip(offsets) {
            let address = byte_offset(context, offset, relative)?;
            values.push(load(context, binding, field, address, writable)?);
        }
        if values.is_empty() {
            values.push(context.constant_u32(0));
        }
        let ty = context.ty(ty)?;
        return context
            .builder
            .composite_construct(ty, None, values)
            .map_err(build_error);
    }
    let pointer = scalar_pointer(context, binding, ty, offset, writable)?;
    let ty = context.ty(ty)?;
    context
        .builder
        .load(ty, None, pointer, None, [])
        .map_err(build_error)
}

fn store(
    context: &mut Context<'_>,
    binding: u32,
    ty: &Ty,
    offset: Word,
    value: Word,
) -> Result<(), Error> {
    let fields = match context.shape(ty) {
        Ty::Record { fields } => Some(fields.iter().map(|f| f.ty.clone()).collect::<Vec<_>>()),
        Ty::Array { element, length } => Some(vec![*element.clone(); *length]),
        _ => None,
    };
    if let Some(fields) = fields {
        let offsets = if let Ty::Array { element, .. } = context.shape(ty) {
            let stride = crate::layout::layout(context.module, element)?.size;
            (0..fields.len()).map(|i| i * stride).collect()
        } else {
            crate::layout::layout(context.module, ty)?.offsets
        };
        for (index, (field, relative)) in fields.iter().zip(offsets).enumerate() {
            let field_type = context.ty(field)?;
            let field_value = context
                .builder
                .composite_extract(field_type, None, value, [index as u32])
                .map_err(build_error)?;
            let address = byte_offset(context, offset, relative)?;
            store(context, binding, field, address, field_value)?;
        }
        return Ok(());
    }
    let pointer = scalar_pointer(context, binding, ty, offset, true)?;
    context
        .builder
        .store(pointer, value, None, [])
        .map_err(build_error)
}

pub(super) fn constant(context: &mut Context<'_>, ty: &Ty, offset: usize) -> Result<Word, Error> {
    let push = context.resources.as_ref().unwrap().push;
    let pointer = context.pointer_type(StorageClass::PushConstant, &Ty::UInt64)?;
    let zero = context.constant_u32(0);
    let pointer = context
        .builder
        .access_chain(pointer, None, push, [zero])
        .map_err(build_error)?;
    let word = context.ty(&Ty::UInt64)?;
    let base = context
        .builder
        .load(word, None, pointer, None, [])
        .map_err(build_error)?;
    let address = byte_offset(context, base, offset)?;
    load(context, 0, ty, address, false)
}

pub(super) fn access(
    context: &mut Context<'_>,
    instr: &Instr,
    args: &[Slot],
) -> Result<Word, Error> {
    let offset = args[0].resource.ok_or_else(|| {
        Error::unsupported("buffer access requires a statically selected resource binding".into())
    })?;
    let referent = args[0].ty.deref_target().unwrap();
    let (element, _) = resin_types::gpu_buffer_binding(&context.module.types, referent)
        .map_err(Error::unsupported)?;
    let element = element.clone();
    let (binding, writable) = context.resources.as_ref().unwrap().bindings[&offset];
    let length_offset = address_layout(context.module, referent)?.offsets[1];
    let length = constant(context, &Ty::UInt64, offset + length_offset)?;
    let valid = ops::emit(context, Op::ULessThan, &Ty::Bool, &[args[1].id, length])?;
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
    let base = constant(context, &Ty::UInt64, offset)?;
    let stride = context.constant_u64(crate::layout::layout(context.module, &element)?.size as u64);
    let index = ops::emit(context, Op::IMul, &Ty::UInt64, &[args[1].id, stride])?;
    let address = ops::emit(context, Op::IAdd, &Ty::UInt64, &[base, index])?;
    let loaded = if matches!(instr, Instr::GpuBufferStore) {
        store(context, binding, &element, address, args[2].id)?;
        None
    } else {
        Some(load(context, binding, &element, address, writable)?)
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
