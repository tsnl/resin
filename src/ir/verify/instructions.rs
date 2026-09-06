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
        Instr::Shader { function, stage } => {
            let target = module.functions.get(function.index()).ok_or_else(|| {
                location.error(VerifyErrorKind::InvalidFunction {
                    function: function.index(),
                })
            })?;
            if target.foreign.is_some()
                || !matches!(stage.as_ref(), "compute" | "vertex" | "fragment")
            {
                return Err(location.error(VerifyErrorKind::InvalidShader));
            }
            stack.push(Ty::shader());
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
        Instr::Load => {
            let address = pop_one(stack, location)?;
            let shape = shape(&module.types, address.clone(), location)?;
            let Ty::Pointer { pointee } = shape else {
                return Err(location.error(VerifyErrorKind::ExpectedPointer { found: address }));
            };
            stack.push(*pointee);
        }
        Instr::Store => {
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
        Instr::CallBuiltin { params, result, .. } => {
            for param in params {
                check_type(&module.types, param, location)?;
            }
            check_type(&module.types, result, location)?;
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
        Value::Unit => Ty::Unit,
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
