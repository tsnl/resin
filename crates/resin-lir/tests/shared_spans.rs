use resin_lir::{
    BasicBlock, BlockId, Function, Instr, Local, Module, Profile, Terminator, VerifiedModule,
};
use resin_types::prelude::*;

fn module(param: Ty, result: Ty, instrs: Vec<Instr>) -> Module {
    Module {
        functions: vec![Function {
            name: None,
            profile: Profile::Host,
            foreign: None,
            result,
            locals: vec![Local {
                name: None,
                ty: param,
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

fn span(element: Ty) -> Ty {
    Ty::Span {
        element: Box::new(element),
    }
}

fn owner(element: Ty) -> Ty {
    Ty::ArcSpan {
        element: Box::new(element),
    }
}

fn parameter() -> Vec<Instr> {
    vec![
        Instr::LocalAddress {
            local: LocalId::from_index(0),
        },
        Instr::Load,
    ]
}

#[test]
fn allocation_and_weak_upgrade_preserve_the_sequence_family() {
    let result = Ty::union_of([owner(Ty::Int32), Ty::None]);
    let program = module(
        Ty::Unit,
        result.clone(),
        vec![
            Instr::Push {
                value: Value::UInt64 { value: 2 },
            },
            Instr::Push {
                value: Value::Int32 { value: 42 },
            },
            Instr::ArcSpanTryNew { element: Ty::Int32 },
            Instr::ExcludeNone,
            Instr::Downgrade,
            Instr::Upgrade,
        ],
    );
    VerifiedModule::new(program).unwrap();

    let mut program = module(
        Ty::Unit,
        result,
        vec![
            Instr::WeakEmpty {
                ty: Ty::WeakSpan {
                    element: Box::new(Ty::Int32),
                },
            },
            Instr::Upgrade,
        ],
    );
    VerifiedModule::new(program.clone()).unwrap();
    program.functions[0].result = Ty::union_of([
        Ty::ArcPtr {
            pointee: Box::new(Ty::Int32),
        },
        Ty::None,
    ]);
    assert!(VerifiedModule::new(program).is_err());
}

#[test]
fn allocation_requires_an_unsigned_count_and_matching_initializer() {
    for (count, initial) in [
        (Value::Int32 { value: 2 }, Value::Int32 { value: 42 }),
        (Value::UInt64 { value: 2 }, Value::Bool { value: true }),
    ] {
        let program = module(
            Ty::Unit,
            Ty::union_of([owner(Ty::Int32), Ty::None]),
            vec![
                Instr::Push { value: count },
                Instr::Push { value: initial },
                Instr::ArcSpanTryNew { element: Ty::Int32 },
            ],
        );
        assert!(VerifiedModule::new(program).is_err());
    }
}

#[test]
fn span_data_requires_a_sequence_owner_and_weak_empty_requires_a_weak_type() {
    for source in [
        Ty::ArcPtr {
            pointee: Box::new(Ty::Int32),
        },
        Ty::Int32,
    ] {
        let mut instrs = parameter();
        instrs.push(Instr::ArcSpanData);
        assert!(VerifiedModule::new(module(source, span(Ty::Int32), instrs)).is_err());
    }
    assert!(
        VerifiedModule::new(module(
            Ty::Unit,
            Ty::Unit,
            vec![Instr::WeakEmpty { ty: Ty::Unit },]
        ))
        .is_err()
    );
}

#[test]
fn byte_views_reject_elements_with_ownership() {
    let mut instrs = parameter();
    instrs.push(Instr::SpanBytes);
    VerifiedModule::new(module(span(Ty::UInt32), span(Ty::UInt8), instrs.clone())).unwrap();
    assert!(
        VerifiedModule::new(module(
            span(Ty::ArcPtr {
                pointee: Box::new(Ty::Int32)
            }),
            span(Ty::UInt8),
            instrs,
        ))
        .is_err()
    );
}

#[test]
fn slices_require_a_borrowed_span_and_unsigned_bounds() {
    let mut instrs = parameter();
    instrs.extend([
        Instr::Push {
            value: Value::UInt64 { value: 0 },
        },
        Instr::Push {
            value: Value::UInt64 { value: 1 },
        },
        Instr::SpanSlice,
    ]);
    VerifiedModule::new(module(span(Ty::Int32), span(Ty::Int32), instrs.clone())).unwrap();
    assert!(
        VerifiedModule::new(module(owner(Ty::Int32), span(Ty::Int32), instrs.clone())).is_err()
    );
    instrs[2] = Instr::Push {
        value: Value::Int32 { value: 0 },
    };
    assert!(VerifiedModule::new(module(span(Ty::Int32), span(Ty::Int32), instrs)).is_err());
}

#[test]
fn host_allocation_checks_the_error_factory_signature_and_profile() {
    let error = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let mut program = module(
        Ty::Unit,
        Ty::Result {
            value: Box::new(owner(Ty::Int32)),
            error: Box::new(error.clone()),
        },
        vec![
            Instr::Push {
                value: Value::UInt64 { value: 2 },
            },
            Instr::Push {
                value: Value::Int32 { value: 42 },
            },
            Instr::HostAllocate {
                error: FunctionId::from_index(1),
                element: Ty::Int32,
            },
        ],
    );
    program.types = vec![TypeDef::new("OutOfMemory", Ty::Record { fields: vec![] })].into();
    let factory = module(
        Ty::Unit,
        error.clone(),
        vec![
            Instr::MakeRecord { fields: vec![] },
            Instr::Ascribe { ty: error },
        ],
    );
    program.functions.push(factory.functions[0].clone());
    VerifiedModule::new(program.clone()).unwrap();
    let mut invalid_error = program.clone();
    invalid_error.functions[0].result = Ty::Unit;
    invalid_error.functions[0].blocks[0]
        .instrs
        .extend([Instr::Discard, Instr::Push { value: Value::Unit }]);
    invalid_error.functions[1].result = Ty::UInt32;
    invalid_error.functions[1].blocks[0].instrs = vec![Instr::Push {
        value: Value::UInt32 { value: 1 },
    }];
    assert!(VerifiedModule::new(invalid_error).is_err());
    program.functions[1].locals[0].ty = Ty::Int32;
    assert!(VerifiedModule::new(program.clone()).is_err());
    program.functions[1].locals[0].ty = Ty::Unit;
    program.functions[1].profile = Profile::Shader;
    assert!(VerifiedModule::new(program).is_err());
}
