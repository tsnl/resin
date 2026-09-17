use crate::{Function, Instr, Module};
use resin_types::prelude::*;

use super::error::Location;
use super::rules::{
    ascribe, check_type, expect_type, expect_types, function_type, is_integer, shape,
};
use crate::{VerifyError, VerifyErrorKind};

pub(super) fn check_instr(
    module: &Module,
    function: &Function,
    instr: &Instr,
    stack: &mut Vec<Ty>,
    location: Location,
) -> Result<(), VerifyError> {
    match instr {
        Instr::Workgroup { operation } => {
            let argument = pop_one(stack, location)?;
            if !matches!(argument, Ty::Reference { mutable: true, .. }) {
                return Err(location.error(VerifyErrorKind::InvalidWorkgroupOperation));
            }
            stack.push(operation.result());
        }
        Instr::TraceRay { payload } => {
            if function.profile != crate::Profile::Shader {
                return Err(location.error(VerifyErrorKind::InvalidGpuOperation));
            }
            resin_types::shader::ray_payload(&module.types, payload)
                .map_err(|_| location.error(VerifyErrorKind::InvalidGpuOperation))?;
            let args = pop(stack, 9, location)?;
            let mut params = vec![Ty::Float32; 8];
            params.push(payload.clone());
            expect_types(&params, &args, location)?;
            stack.push(payload.clone());
        }
        Instr::RayHitInfo => {
            if function.profile != crate::Profile::Shader {
                return Err(location.error(VerifyErrorKind::InvalidGpuOperation));
            }
            stack.push(Ty::Record {
                fields: [
                    Ty::Float32,
                    Ty::UInt32,
                    Ty::UInt32,
                    Ty::Float32,
                    Ty::Float32,
                ]
                .into_iter()
                .enumerate()
                .map(|(i, ty)| RecordField {
                    name: format!("_{i}").into(),
                    ty,
                })
                .collect(),
            });
        }
        Instr::GpuRayTracingPipeline { .. }
        | Instr::GpuComputePipeline { .. }
        | Instr::GpuGraphicsPipeline { .. }
        | Instr::GpuDispatch { .. }
        | Instr::GpuDraw { .. } => {
            super::pipeline::check(module, instr, stack, location)?;
        }

        Instr::GpuViewAllocate
        | Instr::GpuViewRange { .. }
        | Instr::GpuViewOffset
        | Instr::GpuViewRestrict
        | Instr::GpuViewLoad { .. }
        | Instr::GpuViewStore
        | Instr::GpuViewReplace
        | Instr::GpuViewCopyTo
        | Instr::GpuViewCopyFrom
        | Instr::GpuViewCopyImage
        | Instr::GpuArgumentsTraceRays
        | Instr::GpuArgumentsDispatch
        | Instr::GpuArgumentsDraw => super::gpu::check(module, instr, stack, location)?,
        Instr::ForgetLocal { local } | Instr::DropLocal { local } | Instr::TakeLocal { local } => {
            let target = function.locals.get(local.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidLocal {
                    local: local.index(),
                })
            })?;
            if matches!(instr, Instr::TakeLocal { .. }) {
                stack.push(target.ty.clone());
            }
        }
        Instr::TakeField { local, path } | Instr::SetField { local, path } => {
            let mut ty = function
                .locals
                .get(local.index())
                .ok_or_else(|| {
                    location.error(VerifyErrorKind::InvalidLocal {
                        local: local.index(),
                    })
                })?
                .ty
                .clone();
            let invalid = || {
                location.error(VerifyErrorKind::InvalidOwnedField {
                    local: local.index(),
                    path: path.clone(),
                })
            };
            if path.is_empty() {
                return Err(invalid());
            }
            for index in path {
                if matches!(instr, Instr::TakeField { .. })
                    && let Ty::Defined { definition } = ty
                    && module
                        .types
                        .get(definition.index())
                        .is_some_and(|definition| definition.drop_hook().is_some())
                {
                    return Err(invalid());
                }
                let Ty::Record { fields } = shape(&module.types, ty, location)? else {
                    return Err(invalid());
                };
                ty = fields.get(*index).ok_or_else(invalid)?.ty.clone();
            }
            if matches!(instr, Instr::TakeField { .. }) {
                stack.push(ty);
            } else {
                expect_type(ty, pop_one(stack, location)?, location)?;
            }
        }
        Instr::WeakEmpty => stack.push(Ty::WeakOwner),
        Instr::OwnerCreate { element } => {
            check_type(&module.types, element, location)?;
            super::rules::check_value(&module.types, element, location)?;
            expect_type(element.clone(), pop_one(stack, location)?, location)?;
            stack.push(Ty::union_of([Ty::StrongOwner, Ty::None]));
        }
        Instr::OwnerAllocate { element } => {
            check_type(&module.types, element, location)?;
            super::rules::check_value(&module.types, element, location)?;
            if !element.copies_implicitly(&module.types) {
                return Err(location.error(VerifyErrorKind::InvalidCopy {
                    ty: element.clone(),
                }));
            }
            let values = pop(stack, 2, location)?;
            expect_types(&[Ty::UInt64, element.clone()], &values, location)?;
            stack.push(Ty::union_of([Ty::StrongOwner, Ty::None]));
        }
        Instr::OwnerData { pointee } => {
            check_type(&module.types, pointee, location)?;
            expect_type(
                Ty::Reference {
                    mutable: false,
                    referent: Box::new(Ty::StrongOwner),
                },
                pop_one(stack, location)?,
                location,
            )?;
            stack.push(Ty::Pointer {
                pointee: Box::new(pointee.clone()),
            });
        }
        Instr::OwnerLength | Instr::OwnerDowngrade | Instr::OwnerUpgrade => {
            let owner = if matches!(instr, Instr::OwnerUpgrade) {
                Ty::WeakOwner
            } else {
                Ty::StrongOwner
            };
            expect_type(
                Ty::Reference {
                    mutable: false,
                    referent: Box::new(owner),
                },
                pop_one(stack, location)?,
                location,
            )?;
            stack.push(match instr {
                Instr::OwnerLength => Ty::UInt64,
                Instr::OwnerDowngrade => Ty::WeakOwner,
                Instr::OwnerUpgrade => Ty::union_of([Ty::StrongOwner, Ty::None]),
                _ => unreachable!(),
            });
        }
        Instr::SetLocal { local } => {
            let target = function.locals.get(local.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidLocal {
                    local: local.index(),
                })
            })?;
            expect_type(target.ty.clone(), pop_one(stack, location)?, location)?;
        }
        Instr::Widen { ty } => {
            check_type(&module.types, ty, location)?;
            let from = pop_one(stack, location)?;
            if !from.widens_to(ty) {
                return Err(location.error(VerifyErrorKind::TypeMismatch {
                    expected: ty.clone(),
                    found: from,
                }));
            }
            stack.push(ty.clone());
        }
        Instr::MakeVariant { ty, tag } => {
            check_type(&module.types, ty, location)?;
            let payload = ty
                .payload(tag)
                .ok_or_else(|| location.error(VerifyErrorKind::InvalidVariant))?;
            expect_type(payload, pop_one(stack, location)?, location)?;
            stack.push(ty.clone());
        }
        Instr::ExcludeNone => {
            let from = pop_one(stack, location)?;
            let remaining = from
                .without_none()
                .ok_or_else(|| location.error(VerifyErrorKind::InvalidVariant))?;
            stack.push(remaining);
        }
        Instr::IsVariant { tag } => {
            let from = pop_one(stack, location)?;
            let from = if let Ty::Pointer { pointee }
            | Ty::Reference {
                referent: pointee, ..
            } = from
            {
                *pointee
            } else {
                from
            };
            if from.payload(tag).is_none() {
                return Err(location.error(VerifyErrorKind::InvalidVariant));
            }
            stack.push(Ty::Bool);
        }
        Instr::VariantPayload { tag } => {
            let from = pop_one(stack, location)?;
            let payload = from
                .payload(tag)
                .ok_or_else(|| location.error(VerifyErrorKind::InvalidVariant))?;
            stack.push(payload);
        }
        Instr::Eliminate { result } => {
            expect_type(Ty::union([]), pop_one(stack, location)?, location)?;
            check_type(&module.types, result, location)?;
            // Check the dead continuation without constructing a runtime value.
            stack.push(result.clone());
        }
        Instr::NumericCast { ty } => {
            let from = pop_one(stack, location)?;
            if !from.is_numeric() || !ty.is_numeric() {
                return Err(location.error(VerifyErrorKind::TypeMismatch {
                    expected: ty.clone(),
                    found: from,
                }));
            }
            stack.push(ty.clone());
        }
        Instr::PointerCast { ty } => {
            check_type(&module.types, ty, location)?;
            let from = pop_one(stack, location)?;
            if !from.pointer_cast(ty) {
                return Err(location.error(VerifyErrorKind::InvalidPointerCast {
                    from,
                    to: ty.clone(),
                }));
            }
            stack.push(ty.clone());
        }
        Instr::Push { value } => stack.push(immediate_ty(&module.types, value, location)?),
        Instr::LocalRef { local } => {
            let local = function.locals.get(local.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidLocal {
                    local: local.index(),
                })
            })?;
            stack.push(Ty::Reference {
                mutable: true,
                referent: Box::new(local.ty.clone()),
            });
        }
        Instr::Borrow => {
            let source = pop_one(stack, location)?;
            let Ty::Pointer { pointee } = source else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: source }));
            };
            stack.push(Ty::Reference {
                mutable: true,
                referent: pointee,
            });
        }
        Instr::ReadOnly => {
            let source = pop_one(stack, location)?;
            let Ty::Reference {
                mutable: true,
                referent,
            } = source
            else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: source }));
            };
            stack.push(Ty::Reference {
                mutable: false,
                referent,
            });
        }
        Instr::AccessStatic { index } => {
            let source = pop_one(stack, location)?;
            let field = project_static(&module.types, source, *index, location)?;
            if !field.copies_implicitly(&module.types) {
                return Err(location.error(VerifyErrorKind::InvalidCopy { ty: field }));
            }
            stack.push(field);
        }
        Instr::PointerBytes => {
            let args = pop(stack, 2, location)?;
            if !matches!(&args[0], Ty::Pointer { pointee } if pointee.is_numeric()) {
                return Err(location.error(VerifyErrorKind::TypeMismatch {
                    expected: Ty::Pointer {
                        pointee: Box::new(Ty::UInt8),
                    },
                    found: args[0].clone(),
                }));
            }
            expect_types(&[Ty::UInt64], &args[1..], location)?;
            stack.push(Ty::byte_span());
        }
        Instr::PointerIndex | Instr::PointerRange => {
            let count = if matches!(instr, Instr::PointerRange) {
                4
            } else {
                3
            };
            let args = pop(stack, count, location)?;
            let Ty::Pointer { pointee } = &args[0] else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer {
                    found: args[0].clone(),
                }));
            };
            // Stepping a typed pointer requires a concrete element representation;
            // opaque native handles may be passed around only behind pointers.
            super::rules::check_value(&module.types, pointee, location)?;
            expect_types(&vec![Ty::UInt64; count - 1], &args[1..], location)?;
            stack.push(args[0].clone());
        }
        Instr::AccessDynamic => {
            let index = pop_one(stack, location)?;
            if !is_integer(&module.types, &index, location)? {
                return Err(location.error(VerifyErrorKind::ExpectedInteger { found: index }));
            }
            let source = pop_one(stack, location)?;
            let element = project_dynamic(&module.types, source, location)?;
            if !element.copies_implicitly(&module.types) {
                return Err(location.error(VerifyErrorKind::InvalidCopy { ty: element }));
            }
            stack.push(element);
        }
        Instr::Load | Instr::TransferLoad => {
            let address = pop_one(stack, location)?;
            let shape = shape(&module.types, address.clone(), location)?;
            let (Ty::Pointer { pointee }
            | Ty::Reference {
                referent: pointee, ..
            }) = shape
            else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            if matches!(instr, Instr::Load) && !pointee.copies_implicitly(&module.types) {
                return Err(location.error(VerifyErrorKind::InvalidCopy { ty: *pointee }));
            }
            stack.push(*pointee);
        }
        Instr::Store | Instr::Replace => {
            let value = pop_one(stack, location)?;
            let address = pop_one(stack, location)?;
            let shape = shape(&module.types, address.clone(), location)?;
            if matches!(shape, Ty::Reference { mutable: false, .. })
                || (matches!(instr, Instr::Replace) && !matches!(shape, Ty::Pointer { .. }))
            {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            }
            let (Ty::Pointer { pointee }
            | Ty::Reference {
                referent: pointee, ..
            }) = shape
            else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            expect_type(*pointee, value.clone(), location)?;
            stack.push(if matches!(instr, Instr::Store) {
                Ty::Unit
            } else {
                value
            });
        }
        Instr::Discard => {
            pop_one(stack, location)?;
        }
        Instr::Ascribe { ty } => {
            check_type(&module.types, ty, location)?;
            let found = pop_one(stack, location)?;
            ascribe(&module.types, ty, found, location)?;
            stack.push(ty.clone());
        }
        Instr::MakeRecord { fields } => {
            let values = pop(stack, fields.len(), location)?;
            stack.push(Ty::Record {
                fields: fields
                    .iter()
                    .cloned()
                    .zip(values)
                    .map(|(name, ty)| RecordField { name, ty })
                    .collect(),
            });
        }
        Instr::MakeArray { elements, element } => {
            check_type(&module.types, element, location)?;
            let values = pop(stack, *elements, location)?;
            for value in values {
                expect_type(element.clone(), value, location)?;
            }
            stack.push(Ty::Array {
                element: Box::new(element.clone()),
                length: *elements,
            });
        }
        Instr::Function { function } => {
            let target = module.functions.get(function.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidFunction {
                    function: function.index(),
                })
            })?;
            stack.push(function_type(target, location)?);
        }
        Instr::Call { arguments } => {
            let args = pop(stack, *arguments, location)?;
            let callee = pop_one(stack, location)?;
            let shape = shape(&module.types, callee.clone(), location)?;
            let Ty::Function { params, result } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedFunction { found: callee }));
            };
            expect_types(&params, &args, location)?;
            stack.push(*result);
        }
        Instr::CallBuiltin {
            name,
            params,
            result,
        } => {
            for param in params {
                check_type(&module.types, param, location)?;
            }
            check_type(&module.types, result, location)?;
            let signature = TyperContext::from_definitions(module.types.clone())
                .type_builtin_call(name, params)
                .map_err(|error| {
                    location.error(if error.kind == TypeErrorKind::PointerArithmetic {
                        VerifyErrorKind::PointerArithmetic
                    } else {
                        VerifyErrorKind::InvalidBuiltin(error)
                    })
                })?;
            expect_type(signature.result, result.clone(), location)?;
            let values = pop(stack, params.len(), location)?;
            expect_types(params, &values, location)?;
            stack.push(result.clone());
        }
    }
    Ok(())
}

fn immediate_ty(table: &[TypeDef], value: &Value, location: Location) -> Result<Ty, VerifyError> {
    let ty = match value {
        Value::Type { ty } => {
            check_type(table, ty, location)?;
            Ty::Type
        }
        Value::Str { .. } => Ty::Str,
        Value::Unit => Ty::Unit,
        Value::None => Ty::None,
        Value::Bool { .. } => Ty::Bool,
        Value::Int8 { .. } => Ty::Int8,
        Value::Int16 { .. } => Ty::Int16,
        Value::Int32 { .. } => Ty::Int32,
        Value::Int64 { .. } => Ty::Int64,
        Value::UInt8 { .. } => Ty::UInt8,
        Value::UInt16 { .. } => Ty::UInt16,
        Value::UInt32 { .. } => Ty::UInt32,
        Value::UInt64 { .. } => Ty::UInt64,
        Value::Float32 { .. } => Ty::Float32,
        Value::Float64 { .. } => Ty::Float64,
        Value::Array { value } => {
            check_type(table, &value.element_ty, location)?;
            for element in &value.elements {
                let found = immediate_ty(table, element, location)?;
                expect_type(value.element_ty.clone(), found, location)?;
            }
            Ty::Array {
                element: Box::new(value.element_ty.clone()),
                length: value.elements.len(),
            }
        }
        Value::Record { value } => Ty::Record {
            fields: value
                .fields
                .iter()
                .map(|field| {
                    immediate_ty(table, &field.value, location).map(|ty| RecordField {
                        name: field.name.clone(),
                        ty,
                    })
                })
                .collect::<Result<_, _>>()?,
        },
        Value::StaticAddress { .. } | Value::DynamicAddress { .. } => {
            return Err(location.error(VerifyErrorKind::InvalidImmediate));
        }
    };
    Ok(ty)
}

fn project_static(
    table: &[TypeDef],
    source: Ty,
    index: usize,
    location: Location,
) -> Result<Ty, VerifyError> {
    match source {
        Ty::Reference { referent, mutable } => Ok(Ty::Reference {
            mutable,
            referent: Box::new(project_static(table, *referent, index, location)?),
        }),
        Ty::Pointer { pointee } => Ok(Ty::Pointer {
            pointee: Box::new(project_static(table, *pointee, index, location)?),
        }),
        Ty::Defined { .. } => {
            project_static(table, shape(table, source, location)?, index, location)
        }
        view @ Ty::Str => project_static(table, view.view_record().unwrap(), index, location),
        Ty::Record { fields } => fields
            .get(index)
            .map(|field| field.ty.clone())
            .ok_or_else(|| {
                location.error(VerifyErrorKind::StaticIndexOutOfBounds {
                    index,
                    length: fields.len(),
                })
            }),
        Ty::Array { element, length } => {
            if index >= length {
                Err(location.error(VerifyErrorKind::StaticIndexOutOfBounds { index, length }))
            } else {
                Ok(*element)
            }
        }
        found => Err(location.error(VerifyErrorKind::ExpectedAggregate { found })),
    }
}

fn project_dynamic(table: &[TypeDef], source: Ty, location: Location) -> Result<Ty, VerifyError> {
    match source {
        Ty::Reference { referent, mutable } => match shape(table, *referent, location)? {
            Ty::Array { element, .. } => Ok(Ty::Reference {
                mutable,
                referent: element,
            }),
            found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
        },
        Ty::Pointer { pointee } => match shape(table, *pointee, location)? {
            Ty::Array { element, .. } => Ok(Ty::Pointer { pointee: element }),
            found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
        },
        Ty::Defined { .. } => project_dynamic(table, shape(table, source, location)?, location),
        Ty::Str => Ok(Ty::Pointer {
            pointee: Box::new(Ty::UInt8),
        }),
        Ty::Array { element, .. } => Ok(*element),
        found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
    }
}

pub(super) fn pop_one(stack: &mut Vec<Ty>, location: Location) -> Result<Ty, VerifyError> {
    stack.pop().ok_or_else(|| {
        location.error(VerifyErrorKind::StackUnderflow {
            needed: 1,
            available: 0,
        })
    })
}

pub(super) fn pop(
    stack: &mut Vec<Ty>,
    count: usize,
    location: Location,
) -> Result<Vec<Ty>, VerifyError> {
    if stack.len() < count {
        return Err(location.error(VerifyErrorKind::StackUnderflow {
            needed: count,
            available: stack.len(),
        }));
    }
    Ok(stack.split_off(stack.len() - count))
}
