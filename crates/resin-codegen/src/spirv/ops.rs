//! Stack values become SPIR-V IDs; only addressable storage becomes variables.
use super::{
    Context,
    symbols::{LocalAddress, LocalIndex, Slot},
};
use crate::Error;
use resin_lir::Instr;
use resin_types::prelude::*;
use rspirv::{
    dr::{InsertPoint, Instruction, Operand},
    spirv::{GlslStd450Op, Op, StorageClass, Word},
};
use std::collections::HashMap;

pub(super) fn instruction(
    context: &mut Context<'_>,
    instr: &Instr,
    args: &[Slot],
    result: Option<&Ty>,
    locals: &[Word],
    function: &resin_lir::Function,
    arrays: &HashMap<Ty, Word>,
) -> Result<Option<Slot>, Error> {
    let id = match instr {
        Instr::GpuBufferLoad { .. } | Instr::GpuBufferStore => {
            super::buffers::access(context, instr, args)?
        }
        Instr::TraceRay { payload } => super::ray::trace(context, args, payload)?,
        Instr::RayHitInfo => super::ray::hit_info(context, result.unwrap())?,
        Instr::ForgetLocal { .. } | Instr::Discard => return Ok(None),
        Instr::TakeLocal { local } => load(context, result.unwrap(), locals[local.index()])?,
        Instr::TakeField { local, path } | Instr::SetField { local, path } => {
            let pointee = if matches!(instr, Instr::TakeField { .. }) {
                result.unwrap()
            } else {
                &args[0].ty
            };
            let address = Slot {
                ty: Ty::Reference {
                    mutable: true,
                    referent: Box::new(pointee.clone()),
                },
                id: locals[local.index()],
                local: Some(LocalAddress {
                    root: locals[local.index()],
                    root_type: function.locals[local.index()].ty.clone(),
                    indices: path
                        .iter()
                        .map(|index| LocalIndex::Static {
                            index: *index as u32,
                        })
                        .collect(),
                }),
            };
            if matches!(instr, Instr::SetField { .. }) {
                store(context, &address, args[0].id)?;
                return Ok(None);
            }
            dereference(context, &address)?
        }
        Instr::SetLocal { local } => {
            context
                .builder
                .store(locals[local.index()], args[0].id, None, [])
                .unwrap();
            return Ok(None);
        }
        Instr::LocalRef { local } => {
            return Ok(Some(Slot {
                ty: result.unwrap().clone(),
                id: locals[local.index()],
                local: Some(LocalAddress {
                    root: locals[local.index()],
                    root_type: function.locals[local.index()].ty.clone(),
                    indices: vec![],
                }),
            }));
        }
        Instr::Borrow | Instr::ReadOnly => {
            return Ok(Some(Slot {
                ty: result.unwrap().clone(),
                ..args[0].clone()
            }));
        }
        Instr::Function { function } => context.functions[function.index()],
        Instr::Call { .. } => {
            let ty = context.ty(result.unwrap())?;
            context
                .builder
                .function_call(ty, None, args[0].id, args[1..].iter().map(|arg| arg.id))
                .unwrap()
        }
        Instr::Push { value } => literal(context, result.unwrap(), value)?,
        Instr::TransferLoad | Instr::Load => dereference(context, &args[0])?,
        Instr::Store | Instr::Replace => {
            let previous = if matches!(instr, Instr::Replace) {
                dereference(context, &args[0])?
            } else {
                args[1].id
            };
            store(context, &args[0], args[1].id)?;
            if matches!(instr, Instr::Store) {
                literal(context, &Ty::Unit, &Value::Unit)?
            } else {
                previous
            }
        }
        Instr::MakeVariant { ty, tag } => variant(context, ty, tag, args[0].id)?,
        Instr::IsVariant { tag } => {
            if let Ty::Pointer { pointee }
            | Ty::Reference {
                referent: pointee, ..
            } = &args[0].ty
            {
                let value = dereference(context, &args[0])?;
                is_variant(context, pointee, tag, value)?
            } else {
                is_variant(context, &args[0].ty, tag, args[0].id)?
            }
        }
        Instr::VariantPayload { tag } => payload(context, &args[0].ty, tag, args[0].id)?,
        Instr::ExcludeNone | Instr::Widen { .. } => {
            widen(context, &args[0].ty, result.unwrap(), args[0].id)?
        }
        Instr::NumericCast { ty } => numeric_cast(context, &args[0].ty, ty, args[0].id)?,
        Instr::Ascribe { ty } => ascribe(context, &args[0].ty, ty, args[0].id)?,
        Instr::MakeArray { .. } | Instr::MakeRecord { .. } => {
            let fields: Vec<_> = args.iter().map(|arg| arg.id).collect();
            construct(context, result.unwrap(), fields)?
        }
        Instr::AccessStatic { index } => {
            return project(context, &args[0], *index, result.unwrap()).map(Some);
        }
        Instr::PointerIndex => {
            let Ty::Pointer { pointee } = &args[0].ty else {
                unreachable!("verified pointer indexing")
            };
            let size = crate::layout::layout(context.module, pointee)?.size;
            let stride = context.constant_u64(size as u64);
            let offset = emit(context, Op::IMul, &Ty::UInt64, &[args[2].id, stride])?;
            let id = emit(context, Op::IAdd, &Ty::UInt64, &[args[0].id, offset])?;
            return Ok(Some(Slot::value(result.unwrap().clone(), id)));
        }
        Instr::AccessDynamic => {
            return index(context, &args[0], &args[1], result.unwrap(), arrays).map(Some);
        }
        Instr::CallBuiltin { name, result, .. } => builtin(context, name, args, result)?,
        _ => {
            return Err(Error::unsupported(format!(
                "shader profile does not support {instr:?}"
            )));
        }
    };
    Ok(result.map(|ty| Slot::value(ty.clone(), id)))
}

pub(super) fn emit(
    context: &mut Context<'_>,
    op: Op,
    ty: &Ty,
    args: &[Word],
) -> Result<Word, Error> {
    let ty = context.ty(ty)?;
    let id = context.builder.id();
    context
        .builder
        .insert_into_block(
            InsertPoint::End,
            Instruction::new(
                op,
                Some(ty),
                Some(id),
                args.iter().copied().map(Operand::IdRef).collect(),
            ),
        )
        .unwrap();
    Ok(id)
}

pub(super) fn load(context: &mut Context<'_>, ty: &Ty, address: Word) -> Result<Word, Error> {
    let ty = context.ty(ty)?;
    Ok(context.builder.load(ty, None, address, None, []).unwrap())
}

fn local_pointer(
    context: &mut Context<'_>,
    pointee: &Ty,
    local: &LocalAddress,
) -> Result<Word, Error> {
    if local.indices.is_empty() {
        return Ok(local.root);
    }
    let ty = context.pointer_type(StorageClass::Function, pointee)?;
    let indices: Vec<_> = local
        .indices
        .iter()
        .map(|index| match index {
            LocalIndex::Static { index } => context.constant_u32(*index),
            LocalIndex::Dynamic { id, .. } => *id,
        })
        .collect();
    Ok(context
        .builder
        .access_chain(ty, None, local.root, indices)
        .unwrap())
}

fn dereference(context: &mut Context<'_>, slot: &Slot) -> Result<Word, Error> {
    let (Ty::Pointer { pointee }
    | Ty::Reference {
        referent: pointee, ..
    }) = &slot.ty
    else {
        unreachable!()
    };
    if let Some(local) = &slot.local {
        let pointer = local_pointer(context, pointee, local)?;
        load(context, pointee, pointer)
    } else {
        context.physical_load(pointee, slot.id)
    }
}

fn store(context: &mut Context<'_>, slot: &Slot, value: Word) -> Result<(), Error> {
    let (Ty::Pointer { pointee }
    | Ty::Reference {
        referent: pointee, ..
    }) = &slot.ty
    else {
        unreachable!()
    };
    if let Some(local) = &slot.local {
        let pointer = local_pointer(context, pointee, local)?;
        context.builder.store(pointer, value, None, []).unwrap();
        Ok(())
    } else {
        context.physical_store(pointee, slot.id, value)
    }
}

fn extract(context: &mut Context<'_>, ty: &Ty, value: Word, index: usize) -> Result<Word, Error> {
    let ty = context.ty(ty)?;
    Ok(context
        .builder
        .composite_extract(ty, None, value, [index as u32])
        .unwrap())
}

fn construct(context: &mut Context<'_>, ty: &Ty, mut fields: Vec<Word>) -> Result<Word, Error> {
    if fields.is_empty() {
        fields.push(context.constant_u32(0));
    }
    let ty = context.ty(ty)?;
    Ok(context
        .builder
        .composite_construct(ty, None, fields)
        .unwrap())
}

fn project(
    context: &mut Context<'_>,
    base: &Slot,
    index: usize,
    result: &Ty,
) -> Result<Slot, Error> {
    if let Some(mut local) = base.local.clone() {
        local.indices.push(LocalIndex::Static {
            index: index as u32,
        });
        return Ok(Slot {
            ty: result.clone(),
            id: local.root,
            local: Some(local),
        });
    }
    let id = if let Ty::Pointer { pointee }
    | Ty::Reference {
        referent: pointee, ..
    } = &base.ty
    {
        let offset = super::buffers::address_layout(context.module, pointee)?.offsets[index];
        let offset = context.constant_u64(offset as u64);
        emit(context, Op::IAdd, &Ty::UInt64, &[base.id, offset])?
    } else {
        extract(context, result, base.id, index)?
    };
    Ok(Slot::value(result.clone(), id))
}

fn index(
    context: &mut Context<'_>,
    base: &Slot,
    index: &Slot,
    result: &Ty,
    arrays: &HashMap<Ty, Word>,
) -> Result<Slot, Error> {
    if let Some(mut local) = base.local.clone() {
        local.indices.push(LocalIndex::Dynamic {
            id: index.id,
            ty: index.ty.clone(),
        });
        return Ok(Slot {
            ty: result.clone(),
            id: local.root,
            local: Some(local),
        });
    }
    let (address, element) = match context.shape(&base.ty).clone() {
        Ty::Pointer { pointee }
        | Ty::Reference {
            referent: pointee, ..
        } => {
            let Ty::Array { element, .. } = context.shape(&pointee).clone() else {
                return Err(Error::unsupported("array pointer required".into()));
            };
            (base.id, element)
        }
        Ty::Array { .. } => {
            let scratch = arrays[&base.ty];
            context.builder.store(scratch, base.id, None, []).unwrap();
            let local = LocalAddress {
                root: scratch,
                root_type: base.ty.clone(),
                indices: vec![LocalIndex::Dynamic {
                    id: index.id,
                    ty: index.ty.clone(),
                }],
            };
            let pointer = local_pointer(context, result, &local)?;
            return Ok(Slot::value(result.clone(), load(context, result, pointer)?));
        }
        _ => return Err(Error::unsupported("array required".into())),
    };
    let size = super::buffers::address_layout(context.module, &element)?.size;
    let size = context.constant_u64(size as u64);
    let index = numeric_cast(context, &index.ty, &Ty::UInt64, index.id)?;
    let offset = emit(context, Op::IMul, &Ty::UInt64, &[index, size])?;
    let id = emit(context, Op::IAdd, &Ty::UInt64, &[address, offset])?;
    Ok(Slot::value(result.clone(), id))
}

pub(super) fn is_variant(
    context: &mut Context<'_>,
    ty: &Ty,
    case: &Case,
    value: Word,
) -> Result<Word, Error> {
    if matches!(case, Case::Type(member) if member == ty) {
        return Ok(context.constant_bool(true));
    }
    let tag = extract(context, &Ty::UInt32, value, 0)?;
    let expected = context.constant_u32(context.tag(case));
    emit(context, Op::IEqual, &Ty::Bool, &[tag, expected])
}

pub(super) fn payload(
    context: &mut Context<'_>,
    ty: &Ty,
    case: &Case,
    value: Word,
) -> Result<Word, Error> {
    if matches!(case, Case::Type(member) if member == ty) {
        return Ok(value);
    }
    let payloads = ty.payloads().unwrap();
    let index = payloads
        .iter()
        .position(|(candidate, _)| candidate == case)
        .unwrap();
    extract(context, &payloads[index].1, value, index + 1)
}

fn variant(context: &mut Context<'_>, ty: &Ty, case: &Case, value: Word) -> Result<Word, Error> {
    if matches!(case, Case::Type(member) if member == ty) {
        return Ok(value);
    }
    let mut fields = vec![context.constant_u32(context.tag(case))];
    for (candidate, ty) in ty.payloads().unwrap() {
        fields.push(if &candidate == case {
            value
        } else {
            context.zero(&ty)?
        });
    }
    construct(context, ty, fields)
}

fn widen(context: &mut Context<'_>, from: &Ty, to: &Ty, value: Word) -> Result<Word, Error> {
    if from == to {
        return Ok(value);
    }
    if matches!(to, Ty::Union { variants } if variants.contains(from)) {
        return variant(context, to, &Case::Type(from.clone()), value);
    }
    if let (Ty::Error { payload: source }, Ty::Error { payload: target }) = (from, to) {
        return widen(context, source, target, value);
    }
    if !matches!(from, Ty::Union { .. })
        && let Ty::Union { variants } = to
        && let Some(target) = variants
            .iter()
            .find(|ty| *ty == from)
            .or_else(|| variants.iter().find(|ty| from.widens_to(ty)))
    {
        let value = widen(context, from, target, value)?;
        return variant(context, to, &Case::Type(target.clone()), value);
    }
    let mut widened = context.zero(to)?;
    for (case, source) in from.payloads().unwrap_or_default() {
        let target = to.payload(&case).map(|ty| (case.clone(), ty)).or_else(|| {
            to.payloads()?
                .into_iter()
                .find(|(_, target)| source.widens_to(target))
        });
        let Some((target_case, target)) = target else {
            continue;
        };
        let data = payload(context, from, &case, value)?;
        let data = widen(context, &source, &target, data)?;
        let constructed = variant(context, to, &target_case, data)?;
        let test = is_variant(context, from, &case, value)?;
        widened = select(context, to, test, constructed, widened)?;
    }
    Ok(widened)
}

fn select(
    context: &mut Context<'_>,
    ty: &Ty,
    condition: Word,
    yes: Word,
    no: Word,
) -> Result<Word, Error> {
    let fields: Option<Vec<Ty>> = match context.shape(ty) {
        Ty::Record { fields } => Some(fields.iter().map(|field| field.ty.clone()).collect()),
        Ty::Array { element, length } => Some(vec![element.as_ref().clone(); *length]),
        Ty::Union { .. } => Some(
            std::iter::once(Ty::UInt32)
                .chain(ty.payloads().unwrap().into_iter().map(|(_, ty)| ty))
                .collect(),
        ),
        _ => None,
    };
    let Some(fields) = fields else {
        return emit(context, Op::Select, ty, &[condition, yes, no]);
    };
    let mut values = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        let a = extract(context, field, yes, index)?;
        let b = extract(context, field, no, index)?;
        values.push(select(context, field, condition, a, b)?);
    }
    construct(context, ty, values)
}

fn ascribe(context: &mut Context<'_>, from: &Ty, to: &Ty, value: Word) -> Result<Word, Error> {
    if context.ty(from)? == context.ty(to)? {
        return Ok(value);
    }
    let fields: Vec<_> = match context.shape(to) {
        Ty::Record { fields } => fields.iter().map(|field| field.ty.clone()).collect(),
        _ => return Err(Error::unsupported("unsupported shader ascription".into())),
    };
    let mut values = Vec::new();
    for (index, ty) in fields.iter().enumerate() {
        values.push(extract(context, ty, value, index)?);
    }
    construct(context, to, values)
}

fn builtin(
    context: &mut Context<'_>,
    name: &str,
    args: &[Slot],
    result: &Ty,
) -> Result<Word, Error> {
    if name == "assert" {
        return context.zero(&Ty::Unit);
    }
    let unsupported = || Error::unsupported(format!("unsupported shader builtin {name:?}"));
    let Some(first) = args.first() else {
        return Err(unsupported());
    };
    if args.iter().any(|arg| arg.ty != first.ty) {
        return Err(unsupported());
    }
    let ty = context.shape(&first.ty);
    let bool_result = matches!(
        name,
        "==" | "!=" | "<" | "<=" | ">" | ">=" | "!" | "&&" | "||"
    );
    if result != if bool_result { &Ty::Bool } else { &first.ty } {
        return Err(unsupported());
    }
    let float = *ty == Ty::Float32;
    let signed = matches!(ty, Ty::Int32 | Ty::Int64);
    let boolean = *ty == Ty::Bool;
    let args: Vec<_> = args.iter().map(|arg| arg.id).collect();
    let op = match (name, args.len()) {
        ("sqrt" | "sin" | "cos", 1) if float => {
            let op = match name {
                "sqrt" => GlslStd450Op::Sqrt,
                "sin" => GlslStd450Op::Sin,
                _ => GlslStd450Op::Cos,
            };
            let ty = context.ty(result)?;
            return Ok(context
                .builder
                .ext_inst(ty, None, context.glsl, op as u32, [Operand::IdRef(args[0])])
                .unwrap());
        }
        ("+", 1) if ty.is_numeric() => return Ok(args[0]),
        ("-", 1) if float => Op::FNegate,
        ("-", 1) if ty.is_integer() => {
            let zero = context.zero(result)?;
            return emit(context, Op::ISub, result, &[zero, args[0]]);
        }
        ("~", 1) if ty.is_integer() => Op::Not,
        ("!", 1) if boolean => Op::LogicalNot,
        ("+", 2) if ty.is_numeric() => {
            if float {
                Op::FAdd
            } else {
                Op::IAdd
            }
        }
        ("-", 2) if ty.is_numeric() => {
            if float {
                Op::FSub
            } else {
                Op::ISub
            }
        }
        ("*", 2) if ty.is_numeric() => {
            if float {
                Op::FMul
            } else {
                Op::IMul
            }
        }
        ("/", 2) if float => Op::FDiv,
        ("/" | "%", 2) if ty.is_integer() => {
            return integer_division(context, name, result, &args);
        }
        ("<<", 2) if ty.is_integer() => Op::ShiftLeftLogical,
        (">>", 2) if signed => Op::ShiftRightArithmetic,
        (">>", 2) if ty.is_integer() => Op::ShiftRightLogical,
        ("&", 2) if ty.is_integer() => Op::BitwiseAnd,
        ("|", 2) if ty.is_integer() => Op::BitwiseOr,
        ("^", 2) if ty.is_integer() => Op::BitwiseXor,
        ("==", 2)
            if ty.is_numeric()
                || boolean
                || matches!(ty, Ty::Pointer { .. } | Ty::Reference { .. }) =>
        {
            if float {
                Op::FOrdEqual
            } else if boolean {
                Op::LogicalEqual
            } else {
                Op::IEqual
            }
        }
        ("!=", 2)
            if ty.is_numeric()
                || boolean
                || matches!(ty, Ty::Pointer { .. } | Ty::Reference { .. }) =>
        {
            if float {
                Op::FUnordNotEqual
            } else if boolean {
                Op::LogicalNotEqual
            } else {
                Op::INotEqual
            }
        }
        ("<", 2) if ty.is_numeric() => {
            if float {
                Op::FOrdLessThan
            } else if signed {
                Op::SLessThan
            } else {
                Op::ULessThan
            }
        }
        ("<=", 2) if ty.is_numeric() => {
            if float {
                Op::FOrdLessThanEqual
            } else if signed {
                Op::SLessThanEqual
            } else {
                Op::ULessThanEqual
            }
        }
        (">", 2) if ty.is_numeric() => {
            if float {
                Op::FOrdGreaterThan
            } else if signed {
                Op::SGreaterThan
            } else {
                Op::UGreaterThan
            }
        }
        (">=", 2) if ty.is_numeric() => {
            if float {
                Op::FOrdGreaterThanEqual
            } else if signed {
                Op::SGreaterThanEqual
            } else {
                Op::UGreaterThanEqual
            }
        }
        ("&&", 2) if boolean => Op::LogicalAnd,
        ("||", 2) if boolean => Op::LogicalOr,
        _ => return Err(unsupported()),
    };
    emit(context, op, result, &args)
}

fn literal(context: &mut Context<'_>, ty: &Ty, value: &Value) -> Result<Word, Error> {
    let type_id = context.ty(ty)?;
    Ok(match value {
        Value::None | Value::Unit => context.constant_u32(0),
        Value::Bool { value } => context.constant_bool(*value),
        Value::Int32 { value } => context.builder.constant_bit32(type_id, *value as u32),
        Value::UInt8 { value } => context.builder.constant_bit32(type_id, *value as u32),
        Value::UInt32 { value } => context.constant_u32(*value),
        Value::UInt64 { value } => context.constant_u64(*value),
        Value::Int64 { value } => context.builder.constant_bit64(type_id, *value as u64),
        Value::Float32 { value } if value.is_finite() => {
            context.builder.constant_bit32(type_id, value.to_bits())
        }
        Value::Record { value } => {
            let Ty::Record { fields } = context.shape(ty).clone() else {
                unreachable!()
            };
            let mut values = Vec::new();
            for (field, value) in fields.iter().zip(&value.fields) {
                values.push(literal(context, &field.ty, &value.value)?);
            }
            construct(context, ty, values)?
        }
        Value::Str { .. } => return Err(super::str_storage_error()),
        _ => return Err(Error::unsupported("unsupported shader literal".into())),
    })
}

fn integer(ty: &Ty) -> Option<(u32, bool)> {
    Some(match ty {
        Ty::UInt8 => (8, false),
        Ty::UInt32 => (32, false),
        Ty::Int32 => (32, true),
        Ty::UInt64 => (64, false),
        Ty::Int64 => (64, true),
        _ => return None,
    })
}

fn numeric_cast(context: &mut Context<'_>, from: &Ty, to: &Ty, value: Word) -> Result<Word, Error> {
    if from == to {
        return Ok(value);
    }
    let op = match (integer(from), integer(to)) {
        (Some((source, signed)), Some((target, target_signed))) => {
            if source == target {
                Op::Bitcast
            } else {
                // Width conversion preserves signedness; a separate bitcast
                // changes the integer interpretation after the checked resize.
                let intermediate = context.builder.type_int(target, u32::from(signed));
                let resized = if signed {
                    context
                        .builder
                        .s_convert(intermediate, None, value)
                        .unwrap()
                } else {
                    context
                        .builder
                        .u_convert(intermediate, None, value)
                        .unwrap()
                };
                return if signed == target_signed {
                    Ok(resized)
                } else {
                    emit(context, Op::Bitcast, to, &[resized])
                };
            }
        }
        (Some((_, true)), None) => Op::ConvertSToF,
        (Some((_, false)), None) => Op::ConvertUToF,
        (None, Some((_, true))) => Op::ConvertFToS,
        (None, Some((_, false))) => Op::ConvertFToU,
        _ => {
            return Err(Error::unsupported(
                "unsupported shader numeric conversion".into(),
            ));
        }
    };
    emit(context, op, to, &[value])
}

fn integer_division(
    context: &mut Context<'_>,
    name: &str,
    ty: &Ty,
    args: &[Word],
) -> Result<Word, Error> {
    let (bits, signed) = integer(context.shape(ty)).expect("checked integer operation");
    if !signed {
        let op = if name == "/" { Op::UDiv } else { Op::UMod };
        return emit(context, op, ty, args);
    }
    let divisor = signed_divisor(context, ty, bits, args)?;
    // The runtime requires maintenance8, which defines SRem for negative operands.
    let op = if name == "/" { Op::SDiv } else { Op::SRem };
    emit(context, op, ty, &[args[0], divisor])
}

// SPIR-V signed division is undefined for MIN / -1. Replacing that divisor
// with 1 produces Resin's wrapping quotient and zero remainder.
fn signed_divisor(
    context: &mut Context<'_>,
    ty: &Ty,
    bits: u32,
    args: &[Word],
) -> Result<Word, Error> {
    let minimum = integer_constant(context, ty, -(1_i128 << (bits - 1)))?;
    let minus_one = integer_constant(context, ty, -1)?;
    let first = emit(context, Op::IEqual, &Ty::Bool, &[args[0], minimum])?;
    let second = emit(context, Op::IEqual, &Ty::Bool, &[args[1], minus_one])?;
    let overflow = emit(context, Op::LogicalAnd, &Ty::Bool, &[first, second])?;
    let one = integer_constant(context, ty, 1)?;
    select(context, ty, overflow, one, args[1])
}

fn invalid_integer_operation(
    context: &mut Context<'_>,
    name: &str,
    args: &[Slot],
) -> Result<Option<Word>, Error> {
    let Some((bits, _)) = integer(context.shape(&args[0].ty)) else {
        return Ok(None);
    };
    let (op, bound) = match name {
        "/" | "%" => (Op::IEqual, 0),
        // Unsigned comparison also rejects negative signed shift counts.
        "<<" | ">>" => (Op::UGreaterThanEqual, i128::from(bits)),
        _ => return Ok(None),
    };
    let bound = integer_constant(context, &args[1].ty, bound)?;
    emit(context, op, &Ty::Bool, &[args[1].id, bound]).map(Some)
}

/// Return a predicate that must abort this invocation before the instruction.
pub(super) fn invalid(
    context: &mut Context<'_>,
    instr: &Instr,
    args: &[Slot],
) -> Result<Option<Word>, Error> {
    match instr {
        Instr::CallBuiltin { name, .. } if name.as_ref() == "assert" => {
            emit(context, Op::LogicalNot, &Ty::Bool, &[args[0].id]).map(Some)
        }
        Instr::CallBuiltin { name, .. } => invalid_integer_operation(context, name, args),
        Instr::ExcludeNone => {
            is_variant(context, &args[0].ty, &Case::Type(Ty::None), args[0].id).map(Some)
        }
        Instr::NumericCast { ty } => invalid_cast(context, &args[0].ty, ty, args[0].id),
        _ => Ok(None),
    }
}

fn invalid_cast(
    context: &mut Context<'_>,
    from: &Ty,
    to: &Ty,
    value: Word,
) -> Result<Option<Word>, Error> {
    let Some((bits, signed)) = integer(to) else {
        return Ok(None);
    };
    let (low, high) = range(bits, signed);
    if let Some((source_bits, source_signed)) = integer(from) {
        let (source_low, source_high) = range(source_bits, source_signed);
        let mut comparisons = Vec::new();
        if low > source_low {
            let constant = integer_constant(context, from, low)?;
            let op = if source_signed {
                Op::SLessThan
            } else {
                Op::ULessThan
            };
            comparisons.push(emit(context, op, &Ty::Bool, &[value, constant])?);
        }
        if high < source_high {
            let constant = integer_constant(context, from, high)?;
            let op = if source_signed {
                Op::SGreaterThan
            } else {
                Op::UGreaterThan
            };
            comparisons.push(emit(context, op, &Ty::Bool, &[value, constant])?);
        }
        return match comparisons.as_slice() {
            [] => Ok(None),
            [one] => Ok(Some(*one)),
            [a, b] => emit(context, Op::LogicalOr, &Ty::Bool, &[*a, *b]).map(Some),
            _ => unreachable!(),
        };
    }
    let ty = context.ty(from)?;
    let truncated = context
        .builder
        .ext_inst(
            ty,
            None,
            context.glsl,
            GlslStd450Op::Trunc as u32,
            [Operand::IdRef(value)],
        )
        .unwrap();
    let low = context.builder.constant_bit32(ty, (low as f32).to_bits());
    let high = context
        .builder
        .constant_bit32(ty, ((high + 1) as f32).to_bits());
    let above = emit(
        context,
        Op::FOrdGreaterThanEqual,
        &Ty::Bool,
        &[truncated, low],
    )?;
    let below = emit(context, Op::FOrdLessThan, &Ty::Bool, &[truncated, high])?;
    let valid = emit(context, Op::LogicalAnd, &Ty::Bool, &[above, below])?;
    emit(context, Op::LogicalNot, &Ty::Bool, &[valid]).map(Some)
}

fn range(bits: u32, signed: bool) -> (i128, i128) {
    if signed {
        (-(1i128 << (bits - 1)), (1i128 << (bits - 1)) - 1)
    } else {
        (0, (1i128 << bits) - 1)
    }
}

fn integer_constant(context: &mut Context<'_>, ty: &Ty, value: i128) -> Result<Word, Error> {
    let id = context.ty(ty)?;
    Ok(if matches!(ty, Ty::UInt64 | Ty::Int64) {
        context.builder.constant_bit64(id, value as u64)
    } else {
        context.builder.constant_bit32(id, value as u32)
    })
}
