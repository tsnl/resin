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
            profile: resin_lir::Profile::Host,
            foreign: None,
            result,
            parameter_count: 1,
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

fn uint(value: u64) -> Instr {
    Instr::Push {
        value: Value::UInt64 { value },
    }
}

#[test]
fn opaque_gpu_offset_preserves_ownership_and_rejects_pointer_casts() {
    let mut module = module(
        Ty::GpuView,
        Ty::GpuView,
        vec![uint(4), uint(4), uint(4), Instr::GpuViewOffset],
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
fn gpu_range_and_index_preserve_the_opaque_view_until_explicit_load() {
    let range = module(
        Ty::GpuView,
        Ty::GpuView,
        vec![
            uint(4),
            uint(1),
            uint(2),
            Instr::GpuViewRange { element: Ty::Int64 },
        ],
    );
    resin_lir::verify(&range).unwrap();
    let indexed = module(
        Ty::GpuView,
        Ty::UInt32,
        vec![
            uint(4),
            uint(1),
            Instr::GpuViewIndex {
                element: Ty::UInt32,
            },
            Instr::GpuViewLoad {
                element: Ty::UInt32,
            },
        ],
    );
    resin_lir::verify(&indexed).unwrap();
}

#[test]
fn gpu_load_and_index_reject_managed_element_types() {
    for (result, operations) in [
        (
            Ty::StrongOwner,
            vec![Instr::GpuViewLoad {
                element: Ty::StrongOwner,
            }],
        ),
        (
            Ty::GpuView,
            vec![
                uint(4),
                uint(1),
                Instr::GpuViewIndex {
                    element: Ty::StrongOwner,
                },
            ],
        ),
    ] {
        let module = module(Ty::GpuView, result, operations);
        assert!(matches!(
            resin_lir::verify(&module).unwrap_err().kind,
            VerifyErrorKind::UnsupportedGpuElement { .. }
        ));
    }
}

#[test]
fn opaque_gpu_views_do_not_support_ordinary_load_or_raw_pointer_permissions() {
    let ordinary = module(Ty::GpuView, Ty::UInt32, vec![Instr::Load]);
    assert!(resin_lir::verify(&ordinary).is_err());
    let raw = Ty::Pointer {
        pointee: Box::new(Ty::UInt32),
    };
    let restricted = module(
        raw,
        Ty::GpuView,
        vec![
            Instr::Push {
                value: Value::UInt32 { value: 1 },
            },
            Instr::GpuViewRestrict,
        ],
    );
    assert!(resin_lir::verify(&restricted).is_err());
}
