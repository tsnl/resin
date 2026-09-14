use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Profile, Terminator};
use resin_types::{Ty, Value};

fn module(result: Ty, instrs: Vec<Instr>) -> Module {
    Module {
        functions: vec![Function {
            name: None,
            profile: Profile::Host,
            foreign: None,
            result,
            parameter_count: 0,
            locals: vec![Local {
                name: None,
                ty: Ty::Unit,
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

#[test]
fn allocation_checks_initialized_element_types_and_opaque_result() {
    let instrs = vec![
        Instr::Push {
            value: Value::UInt64 { value: 2 },
        },
        Instr::Push {
            value: Value::Int32 { value: 7 },
        },
        Instr::OwnerAllocate { element: Ty::Int32 },
    ];
    let result = Ty::union_of([Ty::None, Ty::StrongOwner]);
    let mut program = module(result, instrs);
    resin_lir::verify(&program).unwrap();
    program.functions[0].blocks[0].instrs[2] = Instr::OwnerAllocate { element: Ty::Bool };
    assert!(resin_lir::verify(&program).is_err());
    program.functions[0].blocks[0].instrs[2] = Instr::OwnerAllocate { element: Ty::Int32 };
    program.functions[0].result = Ty::WeakOwner;
    assert!(resin_lir::verify(&program).is_err());
}

#[test]
fn expired_handles_never_authorize_payload_access() {
    resin_lir::verify(&module(Ty::WeakOwner, vec![Instr::WeakEmpty])).unwrap();
    for instruction in [
        Instr::OwnerData { pointee: Ty::Int32 },
        Instr::OwnerLength,
        Instr::OwnerDowngrade,
        Instr::OwnerUpgrade,
    ] {
        // Operations borrow typed handle storage; even a live value is not its address.
        assert!(resin_lir::verify(&module(Ty::Unit, vec![Instr::WeakEmpty, instruction])).is_err());
    }
}
