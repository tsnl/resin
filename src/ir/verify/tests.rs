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
        param: LocalId::from_index(0),
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
                Instr::Ascribe { ty: meters },
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        entries: Default::default(),
        types: vec![TypeDef::new("Meters", Ty::Int32)],
        globals: vec![],
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
        param: LocalId::from_index(0),
        result: Ty::Int32,
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
                Instr::Ascribe { ty: meters },
                Instr::Ascribe { ty: Ty::Int32 },
            ],
            terminator: Terminator::Return,
        }],
    };

    verify(&Module {
        entries: Default::default(),
        types: vec![TypeDef::new("Meters", Ty::Int32)],
        globals: vec![],
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn chained_assignment_preserves_the_value() {
    let function = Function {
        foreign: None,
        name: None,
        param: LocalId::from_index(0),
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
        entries: Default::default(),
        types: vec![],
        globals: vec![],
        functions: vec![function],
    })
    .unwrap();
}

#[test]
fn conflicting_join_stacks_are_rejected() {
    let function = Function {
        foreign: None,
        name: None,
        param: LocalId::from_index(0),
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
        entries: Default::default(),
        types: vec![],
        globals: vec![],
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
        param: LocalId::from_index(0),
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
        param: LocalId::from_index(0),
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
        entries: Default::default(),
        types: vec![],
        globals: vec![],
        functions: vec![target, caller],
    })
    .unwrap();
}

#[test]
fn loop_backedges_must_match_the_header_stack() {
    let function = Function {
        foreign: None,
        name: None,
        param: LocalId::from_index(0),
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
        entries: Default::default(),
        types: vec![],
        globals: vec![],
        functions: vec![function],
    })
    .unwrap();
}
