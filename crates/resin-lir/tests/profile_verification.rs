//! Direct LIR clients must satisfy the same target rules as source specialization.
use resin_lir::{
    BasicBlock, BlockId, Function, Instr, Local, Module, Profile, Terminator, VerifyErrorKind,
};
use resin_types::{LocalId, Ty, Value};

fn shader(parameter: Ty, instructions: Vec<Instr>) -> Module {
    Module {
        functions: vec![Function {
            name: None,
            profile: Profile::Shader,
            foreign: None,
            result: Ty::Unit,
            parameter_count: 1,
            locals: vec![Local {
                name: None,
                ty: parameter,
            }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: None,
                instrs: instructions,
                terminator: Terminator::Return,
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn opaque_addresses_do_not_authorize_copying_managed_payloads() {
    let parameter = Ty::Pointer {
        pointee: Box::new(Ty::ArcPtr {
            pointee: Box::new(Ty::UInt32),
        }),
    };
    let mut instructions = vec![
        Instr::TakeLocal {
            local: LocalId::from_index(0),
        },
        Instr::Discard,
        Instr::Push { value: Value::Unit },
    ];
    resin_lir::verify(&shader(parameter.clone(), instructions.clone())).unwrap();
    instructions.insert(1, Instr::Load);
    assert!(matches!(
        resin_lir::verify(&shader(parameter, instructions))
            .unwrap_err()
            .kind,
        VerifyErrorKind::UnsupportedProfile {
            profile: Profile::Shader,
            ..
        }
    ));
}

#[test]
fn discarded_unsupported_literals_cannot_bypass_profile_certification() {
    let module = shader(
        Ty::Unit,
        vec![
            Instr::Push {
                value: Value::Int8 { value: 1 },
            },
            Instr::Discard,
            Instr::Push { value: Value::Unit },
        ],
    );
    assert!(matches!(
        resin_lir::verify(&module).unwrap_err().kind,
        VerifyErrorKind::UnsupportedProfile {
            profile: Profile::Shader,
            ..
        }
    ));
}
