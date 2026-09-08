use crate::ir::{Function, Instr, Module, RecordField, Ty, TypeDef, Value};

use super::error::Location;
use super::types::{
    ascribe, check_type, expect_type, expect_types, function_type, is_integer, shape,
};
use super::{VerifyError, VerifyErrorKind};

pub(super) fn check_instr(
    module: &Module,
    function: &Function,
    instr: &Instr,
    stack: &mut Vec<Ty>,
    location: Location,
) -> Result<(), VerifyError> {
    match instr {
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
        Instr::WeakEmpty { pointee } => {
            check_type(&module.types, pointee, location)?;
            stack.push(Ty::Weak {
                pointee: Box::new(pointee.clone()),
            });
        }
        Instr::ArcNew => {
            let ty = pop_one(stack, location)?;
            super::types::check_value(&module.types, &ty, location)?;
            stack.push(Ty::Arc {
                pointee: Box::new(ty),
            });
        }
        Instr::ArcData | Instr::Downgrade | Instr::Upgrade => {
            let source = pop_one(stack, location)?;
            let result = match (instr, &source) {
                (Instr::ArcData, Ty::Arc { pointee }) => Ty::Pointer {
                    pointee: pointee.clone(),
                },
                (Instr::Downgrade, Ty::Arc { pointee }) => Ty::Weak {
                    pointee: pointee.clone(),
                },
                (Instr::Upgrade, Ty::Weak { pointee }) => Ty::union_of([
                    Ty::Arc {
                        pointee: pointee.clone(),
                    },
                    Ty::None,
                ]),
                _ => {
                    return Err(location.error(VerifyErrorKind::TypeMismatch {
                        expected: Ty::Arc {
                            pointee: Box::new(Ty::Unit),
                        },
                        found: source,
                    }));
                }
            };
            stack.push(result);
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
            let from = if let Ty::Pointer { pointee } = from {
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
        Instr::Shader { function, stage } => {
            let target = module.functions.get(function.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidFunction {
                    function: function.index(),
                })
            })?;
            if target.foreign.is_some()
                || module
                    .shaders
                    .get(function)
                    .is_none_or(|entry| entry.stage.as_ref() != stage.as_ref() || !entry.embedded)
            {
                return Err(location.error(VerifyErrorKind::InvalidShader));
            }
            stack.push(Ty::shader());
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
        Instr::LocalAddress { local } => {
            let local = function.locals.get(local.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidLocal {
                    local: local.index(),
                })
            })?;
            stack.push(Ty::Pointer {
                pointee: Box::new(local.ty.clone()),
            });
        }
        Instr::AccessStatic { index } => {
            let source = pop_one(stack, location)?;
            stack.push(project_static(&module.types, source, *index, location)?);
        }
        Instr::AccessDynamic => {
            let index = pop_one(stack, location)?;
            if !is_integer(&module.types, &index, location)? {
                return Err(location.error(VerifyErrorKind::ExpectedInteger { found: index }));
            }
            let source = pop_one(stack, location)?;
            stack.push(project_dynamic(&module.types, source, location)?);
        }
        Instr::Load | Instr::TransferLoad => {
            let address = pop_one(stack, location)?;
            let shape = shape(&module.types, address.clone(), location)?;
            let Ty::Pointer { pointee } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            stack.push(*pointee);
        }
        Instr::Store | Instr::Replace => {
            let value = pop_one(stack, location)?;
            let address = pop_one(stack, location)?;
            let shape = shape(&module.types, address.clone(), location)?;
            let Ty::Pointer { pointee } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            expect_type(*pointee, value.clone(), location)?;
            stack.push(value);
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
        Instr::Call => {
            let arg = pop_one(stack, location)?;
            let callee = pop_one(stack, location)?;
            let shape = shape(&module.types, callee.clone(), location)?;
            let Ty::Function { param, result } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedFunction { found: callee }));
            };
            expect_type(*param, arg, location)?;
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
            let signature = crate::ir::TyperContext::from_definitions(module.types.clone())
                .type_builtin_call(name, params)
                .map_err(|error| {
                    location.error(
                        if error.kind == crate::ir::TypeErrorKind::PointerArithmetic {
                            VerifyErrorKind::PointerArithmetic
                        } else {
                            VerifyErrorKind::InvalidBuiltin(error)
                        },
                    )
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
        Value::Bytes { .. } => Ty::byte_span(),
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
        Ty::Pointer { pointee } => Ok(Ty::Pointer {
            pointee: Box::new(project_static(table, *pointee, index, location)?),
        }),
        Ty::Defined { .. } => {
            project_static(table, shape(table, source, location)?, index, location)
        }
        span @ Ty::Span { .. } => {
            project_static(table, span.span_record().unwrap(), index, location)
        }
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
        Ty::Pointer { pointee } => match shape(table, *pointee, location)? {
            Ty::Array { element, .. } => Ok(Ty::Pointer { pointee: element }),
            found => Err(location.error(VerifyErrorKind::ExpectedArray { found })),
        },
        Ty::Defined { .. } => project_dynamic(table, shape(table, source, location)?, location),
        Ty::Span { element } => Ok(Ty::Pointer { pointee: element }),
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

fn pop(stack: &mut Vec<Ty>, count: usize, location: Location) -> Result<Vec<Ty>, VerifyError> {
    if stack.len() < count {
        return Err(location.error(VerifyErrorKind::StackUnderflow {
            needed: count,
            available: stack.len(),
        }));
    }
    Ok(stack.split_off(stack.len() - count))
}
