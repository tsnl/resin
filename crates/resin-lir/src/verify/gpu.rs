//! GPU operations keep allocation ownership separate from ordinary address values.
use super::error::Location;
use super::instructions::pop;
use super::rules::{check_type, expect_type};
use crate::{Instr, Module, VerifyError, VerifyErrorKind};
use resin_types::prelude::*;

pub(super) fn check(
    module: &Module,
    instr: &Instr,
    stack: &mut Vec<Ty>,
    location: Location,
) -> Result<(), VerifyError> {
    let args = pop(stack, super::stack_effect(instr).pops, location)?;
    let invalid = || location.error(VerifyErrorKind::InvalidGpuOperation);
    let result = match instr {
        Instr::GpuNew { allocator, element } | Instr::GpuAllocate { allocator, element } => {
            check_type(&module.types, element, location)?;
            if !element.gpu_element(&module.types) {
                return Err(location.error(VerifyErrorKind::UnsupportedGpuElement {
                    ty: element.clone(),
                }));
            }
            let expected = if matches!(instr, Instr::GpuNew { .. }) {
                element.clone()
            } else {
                Ty::UInt64
            };
            expect_type(expected, args[1].clone(), location)?;
            let error = allocation_error(module, *allocator, &args[0], location)?;
            let value = if matches!(instr, Instr::GpuNew { .. }) {
                Ty::GpuPointer {
                    pointee: Box::new(element.clone()),
                }
            } else {
                Ty::GpuSpan {
                    element: Box::new(element.clone()),
                }
            };
            Ty::Result {
                value: Box::new(value),
                error: Box::new(error),
            }
        }
        Instr::GpuAllocateNative => {
            if !matches!(&args[0], Ty::Pointer { .. }) || !matches!(&args[1], Ty::Arc { .. }) {
                return Err(invalid());
            }
            for (expected, found) in [Ty::UInt64, Ty::UInt64, Ty::Int32]
                .into_iter()
                .zip(&args[2..])
            {
                expect_type(expected, found.clone(), location)?;
            }
            Ty::Record {
                fields: vec![
                    RecordField {
                        name: "value".into(),
                        ty: Ty::union_of([
                            Ty::GpuPointer {
                                pointee: Box::new(Ty::UInt8),
                            },
                            Ty::None,
                        ]),
                    },
                    RecordField {
                        name: "status".into(),
                        ty: Ty::Int32,
                    },
                ],
            }
        }
        Instr::GpuReadOnly | Instr::GpuWriteOnly => {
            if !matches!(&args[0], Ty::GpuPointer { .. } | Ty::GpuSpan { .. }) {
                return Err(invalid());
            }
            args[0].clone()
        }
        Instr::GpuSlice => {
            let element = match &args[0] {
                Ty::GpuSpan { element } | Ty::GpuPointer { pointee: element } => element.clone(),
                _ => return Err(invalid()),
            };
            expect_type(Ty::UInt64, args[1].clone(), location)?;
            expect_type(Ty::UInt64, args[2].clone(), location)?;
            Ty::GpuSpan { element }
        }
        Instr::GpuCopyTo => {
            let Ty::GpuSpan { element } = &args[0] else {
                return Err(invalid());
            };
            if !element.gpu_element(&module.types) {
                return Err(invalid());
            }
            expect_type(
                Ty::Span {
                    element: element.clone(),
                },
                args[1].clone(),
                location,
            )?;
            Ty::Unit
        }
        Instr::GpuArgumentsDispatch | Instr::GpuArgumentsDraw => {
            expect_type(Ty::GpuArguments, args[0].clone(), location)?;
            expect_type(byte_pointer(), args[1].clone(), location)?;
            for arg in &args[2..] {
                expect_type(Ty::UInt32, arg.clone(), location)?;
            }
            Ty::Int32
        }
        Instr::GpuCopyImage => {
            expect_type(
                Ty::GpuSpan {
                    element: Box::new(Ty::UInt8),
                },
                args[0].clone(),
                location,
            )?;
            expect_type(byte_pointer(), args[1].clone(), location)?;
            expect_type(byte_pointer(), args[2].clone(), location)?;
            Ty::Int32
        }
        _ => unreachable!("GPU instruction dispatch"),
    };
    stack.push(result);
    Ok(())
}

fn byte_pointer() -> Ty {
    Ty::Pointer {
        pointee: Box::new(Ty::UInt8),
    }
}

pub(super) fn allocation_error(
    module: &Module,
    allocator: FunctionId,
    gpu: &Ty,
    location: Location,
) -> Result<Ty, VerifyError> {
    let invalid = || location.error(VerifyErrorKind::InvalidGpuOperation);
    let function = module
        .functions
        .get(allocator.index())
        .ok_or_else(invalid)?;
    let Some(parameter) = function.locals.first() else {
        return Err(invalid());
    };
    expect_type(
        Ty::parameter(&[gpu.clone(), Ty::UInt64, Ty::UInt64, Ty::Int32]),
        parameter.ty.clone(),
        location,
    )?;
    let Ty::Result { value, error } = &function.result else {
        return Err(invalid());
    };
    expect_type(
        Ty::GpuPointer {
            pointee: Box::new(Ty::UInt8),
        },
        *value.clone(),
        location,
    )?;
    Ok(*error.clone())
}
