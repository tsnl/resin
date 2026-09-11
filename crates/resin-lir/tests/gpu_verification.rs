use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator, VerifyErrorKind};
use resin_types::prelude::*;

fn module(parameter: Ty, result: Ty, operations: Vec<Instr>) -> Module {
    let mut instrs = vec![Instr::TakeLocal {
        local: LocalId::from_index(0),
    }];
    instrs.extend(operations);
    Module {
        functions: vec![Function {
            name: None,
            foreign: None,
            result,
            locals: vec![Local {
                name: None,
                ty: parameter,
            }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: None,
                instrs,
                terminator: Terminator::Return,
            }],
        }],
        ..Default::default()
    }
}

fn gpu(ty: Ty) -> Ty {
    Ty::GpuPointer {
        pointee: Box::new(ty),
    }
}

#[test]
fn gpu_field_addresses_preserve_owner_type_and_cannot_be_pointer_cast() {
    let element = Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty: Ty::UInt32,
        }],
    };
    let mut module = module(
        gpu(element),
        gpu(Ty::UInt32),
        vec![Instr::AccessStatic { index: 0 }],
    );
    resin_lir::verify(&module).unwrap();
    module.functions[0].result = Ty::UInt64;
    module.functions[0].blocks[0]
        .instrs
        .push(Instr::PointerCast { ty: Ty::UInt64 });
    assert!(matches!(
        resin_lir::verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidPointerCast { .. }
    ));
}

#[test]
fn gpu_pointer_slicing_produces_an_owning_span() {
    let span = Ty::GpuSpan {
        element: Box::new(Ty::Int64),
    };
    let module = module(
        gpu(Ty::Int64),
        span,
        vec![
            Instr::Push {
                value: Value::UInt64 { value: 1 },
            },
            Instr::Push {
                value: Value::UInt64 { value: 2 },
            },
            Instr::GpuSlice,
        ],
    );
    resin_lir::verify(&module).unwrap();
}

#[test]
fn gpu_index_returns_owner_and_load_returns_plain_element() {
    let module = module(
        Ty::GpuSpan {
            element: Box::new(Ty::UInt32),
        },
        Ty::UInt32,
        vec![
            Instr::Push {
                value: Value::UInt64 { value: 1 },
            },
            Instr::AccessDynamic,
            Instr::Load,
        ],
    );
    resin_lir::verify(&module).unwrap();
}

#[test]
fn managed_gpu_elements_and_permission_changes_on_raw_pointers_are_rejected() {
    let managed = gpu(Ty::Arc {
        pointee: Box::new(Ty::UInt32),
    });
    let module = module(managed.clone(), managed, vec![]);
    assert!(matches!(
        resin_lir::verify(&module).unwrap_err().kind,
        VerifyErrorKind::UnsupportedGpuElement { .. }
    ));
    let raw = Ty::Pointer {
        pointee: Box::new(Ty::UInt32),
    };
    let module = self::module(raw.clone(), raw, vec![Instr::GpuReadOnly]);
    assert_eq!(
        resin_lir::verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidGpuOperation
    );
}
