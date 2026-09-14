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
        Instr::GpuElementLayout { element } => {
            gpu_element(module, element, location)?;
            Ty::Record {
                fields: vec![
                    RecordField {
                        name: "size".into(),
                        ty: Ty::UInt64,
                    },
                    RecordField {
                        name: "alignment".into(),
                        ty: Ty::UInt64,
                    },
                ],
            }
        }
        Instr::GpuViewAllocate => {
            if !matches!(&args[0], Ty::Pointer { .. }) || !matches!(&args[1], Ty::StrongOwner) {
                return Err(invalid());
            }
            super::rules::expect_types(&[Ty::UInt64, Ty::UInt64, Ty::Int32], &args[2..], location)?;
            Ty::Record {
                fields: vec![
                    RecordField {
                        name: "value".into(),
                        ty: Ty::union_of([Ty::GpuView, Ty::None]),
                    },
                    RecordField {
                        name: "status".into(),
                        ty: Ty::Int32,
                    },
                ],
            }
        }
        Instr::GpuViewOffset => {
            super::rules::expect_types(
                &[Ty::GpuView, Ty::UInt64, Ty::UInt64, Ty::UInt64],
                &args,
                location,
            )?;
            Ty::GpuView
        }
        Instr::GpuViewRestrict => {
            super::rules::expect_types(&[Ty::GpuView, Ty::UInt32], &args, location)?;
            Ty::GpuView
        }
        Instr::GpuViewLoad { element } => {
            expect_type(Ty::GpuView, args[0].clone(), location)?;
            gpu_element(module, element, location)?;
            element.clone()
        }
        Instr::GpuViewStore | Instr::GpuViewReplace => {
            expect_type(Ty::GpuView, args[0].clone(), location)?;
            gpu_element(module, &args[1], location)?;
            if matches!(instr, Instr::GpuViewStore) {
                Ty::Unit
            } else {
                args[1].clone()
            }
        }
        Instr::GpuViewCopyTo => {
            expect_type(Ty::GpuView, args[0].clone(), location)?;
            expect_type(Ty::UInt64, args[1].clone(), location)?;
            expect_type(Ty::UInt64, args[3].clone(), location)?;
            let Ty::Pointer { pointee } = &args[2] else {
                return Err(invalid());
            };
            gpu_element(module, pointee, location)?;
            Ty::Unit
        }
        Instr::GpuViewCopyImage => {
            super::rules::expect_types(
                &[Ty::GpuView, Ty::UInt64, byte_pointer(), byte_pointer()],
                &args,
                location,
            )?;
            Ty::Int32
        }

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
            if !matches!(&args[0], Ty::Pointer { .. }) || !matches!(&args[1], Ty::StrongOwner) {
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
                Ty::pointer_length(*element.clone()),
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
    let parameters = function
        .locals
        .get(..function.parameter_count)
        .ok_or_else(invalid)?;
    super::rules::expect_types(
        &[gpu.clone(), Ty::UInt64, Ty::UInt64, Ty::Int32],
        &parameters
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
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

fn gpu_element(module: &Module, ty: &Ty, location: Location) -> Result<(), VerifyError> {
    check_type(&module.types, ty, location)?;
    if ty.gpu_element(&module.types) {
        Ok(())
    } else {
        Err(location.error(VerifyErrorKind::UnsupportedGpuElement { ty: ty.clone() }))
    }
}
