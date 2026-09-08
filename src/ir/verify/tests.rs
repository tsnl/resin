use super::*;
use crate::ir::{BasicBlock, Local, LocalId, TypeDef};
use crate::ir::{BlockId, Function, Instr, Module, Terminator, Ty, TypeId, Value};

#[test]
fn ascribe_wraps_a_representation() {
    let meters = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        foreign: None,
        name: None,
        result: meters.clone(),
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::Push {
                    value: Value::Int32 { value: 3 },
                },
                Instr::MakeRecord {
                    fields: vec!["value".into()],
                },
                Instr::Ascribe { ty: meters },
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![TypeDef::new("Meters", record())].into(),
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn ascribe_unwraps_one_nominal_layer() {
    let meters = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let function = Function {
        foreign: None,
        name: None,
        result: record(),
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::Push {
                    value: Value::Int32 { value: 3 },
                },
                Instr::MakeRecord {
                    fields: vec!["value".into()],
                },
                Instr::Ascribe { ty: meters },
                Instr::Ascribe { ty: record() },
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![TypeDef::new("Meters", record())].into(),
        functions: vec![function],
    })
    .unwrap();
}

fn record() -> Ty {
    Ty::Record {
        fields: vec![crate::ir::RecordField {
            name: "value".into(),
            ty: Ty::Int32,
        }],
    }
}

#[test]
fn chained_assignment_preserves_the_value() {
    let function = Function {
        foreign: None,
        name: None,
        result: Ty::Int32,
        locals: vec![
            Local {
                name: None,
                ty: Ty::Int32,
            },
            Local {
                name: None,
                ty: Ty::Int32,
            },
        ],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::LocalAddress {
                    local: LocalId::from_index(0),
                },
                Instr::LocalAddress {
                    local: LocalId::from_index(1),
                },
                Instr::Push {
                    value: Value::Int32 { value: 1 },
                },
                Instr::Store,
                Instr::Store,
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![].into(),
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn conflicting_join_stacks_are_rejected() {
    let function = Function {
        foreign: None,
        name: None,
        result: Ty::Int32,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Bool { value: true },
                }],
                terminator: Terminator::Branch {
                    then: BlockId::from_index(1),
                    els: BlockId::from_index(2),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Int32 { value: 1 },
                }],
                terminator: Terminator::Break {
                    target: BlockId::from_index(3),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Float32 { value: 1.0 },
                }],
                terminator: Terminator::Break {
                    target: BlockId::from_index(3),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Return,
            },
        ],
    };

    let error = verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![].into(),
        functions: vec![function],
    })
    .unwrap_err();

    assert!(matches!(
        error.kind,
        VerifyErrorKind::ConflictingBasicBlockStack { .. }
    ));
}

#[test]
fn indirect_calls_use_the_callee_on_the_stack() {
    let target = Function {
        foreign: None,
        name: None,
        result: Ty::Int32,
        locals: vec![Local {
            name: None,
            ty: Ty::Int32,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::LocalAddress {
                    local: LocalId::from_index(0),
                },
                Instr::Load,
            ],
            terminator: Terminator::Return,
        }],
    };
    let caller = Function {
        foreign: None,
        name: None,
        result: Ty::Int32,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::Function {
                    function: FunctionId::from_index(0),
                },
                Instr::Push {
                    value: Value::Int32 { value: 4 },
                },
                Instr::Call,
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![].into(),
        functions: vec![target, caller],
    })
    .unwrap();
}

#[test]
fn loop_backedges_must_match_the_header_stack() {
    let function = Function {
        foreign: None,
        name: None,
        result: Ty::Unit,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Break {
                    target: BlockId::from_index(1),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Bool { value: true },
                }],
                terminator: Terminator::Branch {
                    then: BlockId::from_index(2),
                    els: BlockId::from_index(3),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Break {
                    target: BlockId::from_index(1),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push { value: Value::Unit }],
                terminator: Terminator::Return,
            },
        ],
    };

    verify(&Module {
        shaders: Default::default(),
        origins: Default::default(),
        entries: Default::default(),
        types: vec![].into(),
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn all_functions_require_parameter_local_zero() {
    for foreign in [
        None,
        Some(crate::ir::Foreign {
            header: "test.h".into(),
            params: vec![],
        }),
    ] {
        let mut function = Function {
            name: Some("f".into()),
            foreign,
            result: Ty::Unit,
            locals: vec![],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: None,
                instrs: vec![Instr::Push { value: Value::Unit }],
                terminator: Terminator::Return,
            }],
        };
        if function.foreign.is_some() {
            function.blocks.clear();
        }
        assert_eq!(function.ty(), None);
        let mut module = Module {
            functions: vec![function],
            ..Default::default()
        };
        assert!(matches!(
            verify(&module).unwrap_err().kind,
            VerifyErrorKind::InvalidLocal { local: 0 }
        ));
        module.functions[0].locals.push(Local {
            name: None,
            ty: Ty::Unit,
        });
        verify(&module).unwrap();
        assert_eq!(
            module.functions[0].ty(),
            Some(Ty::Function {
                param: Box::new(Ty::Unit),
                result: Box::new(Ty::Unit)
            })
        );
    }
}

#[test]
fn destruction_hooks_reference_a_function_with_the_nominal_pointer_signature() {
    let mut definition = TypeDef::new("Resource", record());
    if let crate::ir::TypeDef::Nominal { drop, .. } = &mut definition {
        *drop = Some(FunctionId::from_index(0));
    }
    let mut module = Module {
        types: vec![definition],
        ..Default::default()
    };
    assert_eq!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidDropHook
    );
    module.functions.push(Function {
        name: None,
        foreign: None,
        result: Ty::Unit,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![Instr::Push { value: Value::Unit }],
            terminator: Terminator::Return,
        }],
    });
    assert_eq!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidDropHook
    );
    module.functions[0].locals[0].ty = Ty::Pointer {
        pointee: Box::new(Ty::Defined {
            definition: TypeId::from_index(0),
        }),
    };
    verify(&module).unwrap();
}
