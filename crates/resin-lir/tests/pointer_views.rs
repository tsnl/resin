//! Primitive boundaries remain checked even when a caller constructs LIR directly.
use resin_lir::{
    BasicBlock, BlockId, Function, Instr, Local, Module, Profile, Terminator, VerifiedModule,
    VerifyErrorKind,
};
use resin_types::{LocalId, Ty};

fn pointer(element: Ty) -> Ty {
    Ty::Pointer {
        pointee: Box::new(element),
    }
}

fn module(params: &[Ty], result: Ty, operation: Instr) -> Module {
    let mut instrs = (0..params.len())
        .map(|index| Instr::TakeLocal {
            local: LocalId::from_index(index),
        })
        .collect::<Vec<_>>();
    instrs.push(operation);
    Module {
        functions: vec![Function {
            name: None,
            profile: Profile::Host,
            foreign: None,
            parameter_count: params.len(),
            locals: params
                .iter()
                .map(|ty| Local {
                    name: None,
                    ty: ty.clone(),
                })
                .collect(),
            result,
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

#[test]
fn indexing_and_ranges_require_a_pointer_and_unsigned_bounds() {
    for (operation, bounds) in [(Instr::PointerIndex, 2), (Instr::PointerRange, 3)] {
        let result = pointer(Ty::Int32);
        let mut params = vec![result.clone()];
        params.extend(vec![Ty::UInt64; bounds]);
        VerifiedModule::new(module(&params, result.clone(), operation.clone())).unwrap();
        for index in 0..params.len() {
            let mut invalid = params.clone();
            invalid[index] = Ty::Int32;
            let error = VerifiedModule::new(module(&invalid, result.clone(), operation.clone()))
                .err()
                .unwrap();
            if index == 0 {
                assert_eq!(
                    error.kind,
                    VerifyErrorKind::ExpectedPointer { found: Ty::Int32 }
                );
            }
        }
        assert!(VerifiedModule::new(module(&params, pointer(Ty::UInt32), operation)).is_err());
    }
}

#[test]
fn byte_views_require_numeric_elements_and_an_unsigned_count() {
    for element in [Ty::UInt8, Ty::Int32, Ty::UInt64, Ty::Float64] {
        let params = [pointer(element), Ty::UInt64];
        VerifiedModule::new(module(&params, Ty::byte_span(), Instr::PointerBytes)).unwrap();
    }
    for params in [
        [Ty::UInt64, Ty::UInt64],
        [pointer(Ty::Bool), Ty::UInt64],
        [pointer(pointer(Ty::UInt32)), Ty::UInt64],
        [pointer(Ty::UInt32), Ty::Int32],
    ] {
        assert!(
            VerifiedModule::new(module(&params, Ty::byte_span(), Instr::PointerBytes)).is_err()
        );
    }
}

#[test]
fn byte_and_range_checks_stay_host_only_while_pointer_indexing_is_shader_legal() {
    for (operation, bounds, result, accepted) in [
        (Instr::PointerIndex, 2, pointer(Ty::UInt32), true),
        (Instr::PointerRange, 3, pointer(Ty::UInt32), false),
        (Instr::PointerBytes, 1, Ty::byte_span(), false),
    ] {
        let mut params = vec![pointer(Ty::UInt32)];
        params.extend(vec![Ty::UInt64; bounds]);
        let mut program = module(&params, result, operation);
        program.functions[0].profile = Profile::Shader;
        assert_eq!(VerifiedModule::new(program).is_ok(), accepted);
    }
}
