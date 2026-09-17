use crate::{BasicBlock, Local};
use crate::{BlockId, Function, Instr, Module, Terminator};
use crate::{VerifyErrorKind, verify};
use resin_types::prelude::*;

#[test]
fn ascribe_wraps_and_unwraps_a_nominal_representation() {
    let meters = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    for result in [meters.clone(), record()] {
        let mut instrs = vec![
            Instr::Push {
                value: Value::Int32 { value: 3 },
            },
            Instr::MakeRecord {
                fields: vec!["value".into()],
            },
            Instr::Ascribe { ty: meters.clone() },
        ];
        if result != meters {
            instrs.push(Instr::Ascribe { ty: result.clone() });
        }
        verify(&Module {
            types: vec![TypeDef::new("Meters", record())].into(),
            functions: vec![Function {
                profile: crate::Profile::Host,
                foreign: None,
                name: None,
                result,
                parameter_count: 1,
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
        })
        .unwrap();
    }
}

fn record() -> Ty {
    Ty::Record {
        fields: vec![RecordField {
            name: "value".into(),
            ty: Ty::Int32,
        }],
    }
}

#[test]
fn store_returns_unit_so_numeric_assignment_cannot_be_chained() {
    let function = Function {
        profile: crate::Profile::Host,
        foreign: None,
        name: None,
        result: Ty::Int32,
        parameter_count: 1,
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
                Instr::LocalRef {
                    local: LocalId::from_index(0),
                },
                Instr::LocalRef {
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

    let error = verify(&Module {
        functions: vec![function],
        ..Default::default()
    })
    .unwrap_err();
    assert!(matches!(
        error.kind,
        VerifyErrorKind::TypeMismatch {
            expected: Ty::Int32,
            found: Ty::Unit
        }
    ));
}

#[test]
fn conflicting_join_stacks_are_rejected() {
    let function = Function {
        profile: crate::Profile::Host,
        foreign: None,
        name: None,
        result: Ty::Int32,
        parameter_count: 1,
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
                terminator: Terminator::If {
                    then: BlockId::from_index(1),
                    els: BlockId::from_index(2),
                    next: Some(BlockId::from_index(3)),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Int32 { value: 1 },
                }],
                terminator: Terminator::Merge,
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Float32 { value: 1.0 },
                }],
                terminator: Terminator::Merge,
            },
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Return,
            },
        ],
    };

    let error = verify(&Module {
        functions: vec![function],
        ..Default::default()
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
        profile: crate::Profile::Host,
        foreign: None,
        name: None,
        result: Ty::Int32,
        parameter_count: 1,
        locals: vec![Local {
            name: None,
            ty: Ty::Int32,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs: vec![
                Instr::LocalRef {
                    local: LocalId::from_index(0),
                },
                Instr::Load,
            ],
            terminator: Terminator::Return,
        }],
    };
    let caller = Function {
        profile: crate::Profile::Host,
        foreign: None,
        name: None,
        result: Ty::Int32,
        parameter_count: 1,
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
                Instr::Call { arguments: 1 },
            ],
            terminator: Terminator::Return,
        }],
    };

    let mut module = Module {
        functions: vec![target, caller],
        ..Default::default()
    };
    verify(&module).unwrap();
    for arguments in [0, 2, usize::MAX] {
        module.functions[1].blocks[0].instrs[2] = Instr::Call { arguments };
        assert!(verify(&module).is_err(), "argument count {arguments}");
    }
}

#[test]
fn loop_body_preserves_the_condition_stack() {
    let function = Function {
        profile: crate::Profile::Host,
        foreign: None,
        name: None,
        result: Ty::Unit,
        parameter_count: 1,
        locals: vec![Local {
            name: None,
            ty: Ty::Unit,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Loop {
                    condition: BlockId::from_index(1),
                    body: BlockId::from_index(2),
                    next: Some(BlockId::from_index(3)),
                },
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push {
                    value: Value::Bool { value: true },
                }],
                terminator: Terminator::LoopTest,
            },
            BasicBlock {
                name: None,
                instrs: vec![],
                terminator: Terminator::Continue,
            },
            BasicBlock {
                name: None,
                instrs: vec![Instr::Push { value: Value::Unit }],
                terminator: Terminator::Return,
            },
        ],
    };

    verify(&Module {
        functions: vec![function],
        ..Default::default()
    })
    .unwrap();
}

#[test]
fn parameter_counts_must_fit_the_declared_locals() {
    for foreign in [
        None,
        Some(crate::Foreign {
            header: crate::ForeignHeader {
                source: resin_source::SourceId::new("native.resin"),
                spelling: "test.h".into(),
            },
            params: vec![],
        }),
    ] {
        let mut function = Function {
            profile: crate::Profile::Host,
            name: Some("f".into()),
            foreign,
            result: Ty::Unit,
            parameter_count: 1,
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
        module.functions[0].parameter_count = 0;
        verify(&module).unwrap();
        assert_eq!(
            module.functions[0].ty(),
            Some(Ty::Function {
                params: vec![],
                result: Box::new(Ty::Unit)
            })
        );
    }
}

fn builtin_module(name: &str, params: &[Ty], result: Ty) -> Module {
    let mut instrs: Vec<_> = params
        .iter()
        .enumerate()
        .flat_map(|(index, _)| {
            [
                Instr::LocalRef {
                    local: LocalId::from_index(index + 1),
                },
                Instr::Load,
            ]
        })
        .collect();
    instrs.push(Instr::CallBuiltin {
        name: name.into(),
        params: params.to_vec(),
        result: result.clone(),
    });
    Module {
        functions: vec![Function {
            profile: crate::Profile::Host,
            foreign: None,
            name: None,
            result,
            parameter_count: 1,
            locals: std::iter::once(Ty::Unit)
                .chain(params.iter().cloned())
                .map(|ty| Local { name: None, ty })
                .collect(),
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
fn builtin_instructions_check_names_arity_and_operand_rules() {
    for (name, params, result) in [
        ("unknown", vec![], Ty::Unit),
        ("==", vec![Ty::Int32], Ty::Bool),
        ("+", vec![], Ty::Int32),
        ("+", vec![Ty::Int32, Ty::Float64], Ty::Int32),
        ("&&", vec![Ty::Int32, Ty::Int32], Ty::Bool),
        ("print", vec![Ty::Int32], Ty::Unit),
    ] {
        assert!(
            matches!(
                verify(&builtin_module(name, &params, result))
                    .unwrap_err()
                    .kind,
                VerifyErrorKind::InvalidBuiltin(_)
            ),
            "{name} {params:?}"
        );
    }
}

#[test]
fn builtin_results_are_derived_from_the_operation() {
    for (name, params, result, expected) in [
        ("==", vec![Ty::Int32, Ty::Int32], Ty::Int32, Ty::Bool),
        ("+", vec![Ty::Int32, Ty::Int32], Ty::Bool, Ty::Int32),
        ("!", vec![Ty::Bool], Ty::Unit, Ty::Bool),
    ] {
        assert_eq!(
            verify(&builtin_module(name, &params, result.clone()))
                .unwrap_err()
                .kind,
            VerifyErrorKind::TypeMismatch {
                expected,
                found: result
            }
        );
    }
}

#[test]
fn builtin_pointer_comparison_is_valid_but_arithmetic_is_not() {
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Int32),
    };
    verify(&builtin_module(
        "==",
        &[pointer.clone(), pointer.clone()],
        Ty::Bool,
    ))
    .unwrap();
    assert_eq!(
        verify(&builtin_module(
            "+",
            &[pointer.clone(), pointer.clone()],
            pointer
        ))
        .unwrap_err()
        .kind,
        VerifyErrorKind::PointerArithmetic
    );
}

#[test]
fn builtin_verification_preserves_arithmetic_without_numeric_traits() {
    verify(&builtin_module("+", &[record(), record()], record())).unwrap();
}

#[test]
fn ascription_cannot_stand_in_for_cast_or_widen_instructions() {
    let first = TypeId::from_index(0);
    let second = TypeId::from_index(1);
    for (from, to) in [
        (Ty::Int32, Ty::Float64),
        (
            Ty::Defined { definition: first },
            Ty::union([first, second]),
        ),
        (
            Ty::UInt64,
            Ty::Pointer {
                pointee: Box::new(Ty::Int32),
            },
        ),
        (
            Ty::Defined { definition: first },
            Ty::Defined { definition: second },
        ),
    ] {
        let function = Function {
            profile: crate::Profile::Host,
            foreign: None,
            name: None,
            result: to.clone(),
            parameter_count: 1,
            locals: vec![Local {
                name: None,
                ty: from.clone(),
            }],
            entry: BlockId::from_index(0),
            blocks: vec![BasicBlock {
                name: None,
                instrs: vec![
                    Instr::LocalRef {
                        local: LocalId::from_index(0),
                    },
                    Instr::Load,
                    Instr::Ascribe { ty: to.clone() },
                ],
                terminator: Terminator::Return,
            }],
        };
        let error = verify(&Module {
            types: vec![
                TypeDef::new("First", record()),
                TypeDef::new("Second", record()),
            ]
            .into(),
            functions: vec![function],
            ..Default::default()
        })
        .unwrap_err();
        assert_eq!(
            error.kind,
            VerifyErrorKind::TypeMismatch {
                expected: to,
                found: from
            }
        );
    }
}

#[test]
fn destruction_hooks_require_the_nominal_reference_signature() {
    let mut definition = TypeDef::new("Resource", record());
    if let TypeDef::Nominal { drop, .. } = &mut definition {
        *drop = Some(FunctionId::from_index(0));
    }
    let mut module = Module {
        types: vec![definition].into(),
        ..Default::default()
    };
    assert_eq!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidDropHook
    );
    module.functions.push(Function {
        name: None,
        profile: crate::Profile::Host,
        foreign: None,
        result: Ty::Unit,
        parameter_count: 1,
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
    module.functions[0].locals[0].ty = Ty::Reference {
        mutable: true,
        referent: Box::new(Ty::Defined {
            definition: TypeId::from_index(0),
        }),
    };
    verify(&module).unwrap();
}

#[test]
fn loads_copy_plain_structs_but_cannot_duplicate_custom_owners() {
    let nominal = Ty::Defined {
        definition: TypeId::from_index(0),
    };
    let mut module = expression_module(
        nominal.clone(),
        nominal.clone(),
        vec![
            Instr::LocalRef {
                local: LocalId::from_index(0),
            },
            Instr::Load,
        ],
    );
    module.types = vec![TypeDef::new("Resource", record())].into();
    verify(&module).unwrap();
    let hook = expression_module(
        Ty::Reference {
            mutable: true,
            referent: Box::new(nominal.clone()),
        },
        Ty::Unit,
        vec![Instr::Push { value: Value::Unit }],
    )
    .functions
    .remove(0);
    module.functions.push(hook);
    let mut definition = module.types[0].clone();
    if let TypeDef::Nominal { drop, .. } = &mut definition {
        *drop = Some(FunctionId::from_index(1));
    }
    module.types = vec![definition].into();
    assert_eq!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidCopy { ty: nominal }
    );
    module.functions[0].blocks[0].instrs = vec![Instr::TakeLocal {
        local: LocalId::from_index(0),
    }];
    verify(&module).unwrap();
}

#[test]
fn string_immediates_have_a_distinct_type() {
    let literal = Instr::Push {
        value: Value::Str {
            value: b"a\0b".as_slice().into(),
        },
    };
    verify(&expression_module(Ty::Unit, Ty::Str, vec![literal.clone()])).unwrap();
    assert!(verify(&expression_module(Ty::Unit, Ty::byte_span(), vec![literal])).is_err());
}

#[test]
fn string_views_preserve_fields_and_byte_indexing() {
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::UInt8),
    };
    for (instruction, result) in [
        (Instr::AccessStatic { index: 0 }, pointer.clone()),
        (Instr::AccessStatic { index: 1 }, Ty::UInt64),
        (
            Instr::Ascribe {
                ty: Ty::Str.view_record().unwrap(),
            },
            Ty::Str.view_record().unwrap(),
        ),
    ] {
        verify(&parameter_expression(Ty::Str, result, vec![instruction])).unwrap();
    }
    let index = Instr::Push {
        value: Value::UInt64 { value: 0 },
    };
    verify(&parameter_expression(
        Ty::Str,
        pointer,
        vec![index, Instr::AccessDynamic],
    ))
    .unwrap();
}

#[test]
fn byte_views_cannot_be_ascribed_as_strings() {
    for source in [Ty::byte_span(), Ty::Str.view_record().unwrap()] {
        let module = parameter_expression(
            source.clone(),
            Ty::Str,
            vec![Instr::Ascribe { ty: Ty::Str }],
        );
        assert_eq!(
            verify(&module).unwrap_err().kind,
            VerifyErrorKind::TypeMismatch {
                expected: Ty::Str,
                found: source,
            }
        );
    }
}

fn parameter_expression(param: Ty, result: Ty, instructions: Vec<Instr>) -> Module {
    let mut body = vec![
        Instr::LocalRef {
            local: LocalId::from_index(0),
        },
        Instr::Load,
    ];
    body.extend(instructions);
    expression_module(param, result, body)
}

fn expression_module(param: Ty, result: Ty, instrs: Vec<Instr>) -> Module {
    Module {
        functions: vec![Function {
            profile: crate::Profile::Host,
            foreign: None,
            name: None,
            result,
            parameter_count: 1,
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

#[test]
fn explicit_loop_exits_require_an_enclosing_body() {
    for exit in [Terminator::Break, Terminator::NextIteration] {
        let mut module = expression_module(Ty::Unit, Ty::Unit, vec![]);
        module.functions[0].blocks[0].terminator = exit;
        assert_eq!(
            verify(&module).unwrap_err().kind,
            VerifyErrorKind::UnexpectedLoopExit
        );
    }
}

#[test]
fn local_references_cannot_be_retyped_returned_or_passed_as_pointers() {
    let reference = Ty::Reference {
        mutable: true,
        referent: Box::new(Ty::Int32),
    };
    let pointer = Ty::Pointer {
        pointee: Box::new(Ty::Int32),
    };
    let function = |result: Ty, instrs| Function {
        name: None,
        profile: crate::Profile::Host,
        foreign: None,
        result,
        parameter_count: 1,
        locals: vec![Local {
            name: None,
            ty: Ty::Int32,
        }],
        entry: BlockId::from_index(0),
        blocks: vec![BasicBlock {
            name: None,
            instrs,
            terminator: Terminator::Return,
        }],
    };
    let local = Instr::LocalRef {
        local: LocalId::from_index(0),
    };
    for conversion in [
        Instr::PointerCast {
            ty: pointer.clone(),
        },
        Instr::Ascribe {
            ty: pointer.clone(),
        },
        Instr::Widen {
            ty: pointer.clone(),
        },
    ] {
        let module = Module {
            functions: vec![function(pointer.clone(), vec![local.clone(), conversion])],
            ..Default::default()
        };
        assert!(verify(&module).is_err());
    }
    let mut module = Module {
        functions: vec![function(reference, vec![local.clone()])],
        ..Default::default()
    };
    verify(&module).unwrap();
    module.functions[0].result = pointer.clone();
    assert!(matches!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::InvalidReturnStack { .. }
    ));
    let mut callee = function(
        Ty::Int32,
        vec![Instr::Push {
            value: Value::Int32 { value: 42 },
        }],
    );
    callee.locals[0].ty = pointer;
    module.functions = vec![
        function(
            Ty::Int32,
            vec![
                Instr::Function {
                    function: FunctionId::from_index(1),
                },
                local,
                Instr::Call { arguments: 1 },
            ],
        ),
        callee,
    ];
    assert!(matches!(
        verify(&module).unwrap_err().kind,
        VerifyErrorKind::TypeMismatch { .. }
    ));
}

#[test]
fn readonly_references_cannot_store_or_regain_write_permission() {
    let local = Instr::LocalRef {
        local: LocalId::from_index(0),
    };
    let mut module = expression_module(
        Ty::Int32,
        Ty::Unit,
        vec![
            local.clone(),
            Instr::ReadOnly,
            Instr::Push {
                value: Value::Int32 { value: 42 },
            },
            Instr::Store,
        ],
    );
    assert!(verify(&module).is_err());
    module.functions[0].blocks[0].instrs = vec![
        local.clone(),
        Instr::ReadOnly,
        Instr::Ascribe {
            ty: Ty::Reference {
                mutable: true,
                referent: Box::new(Ty::Int32),
            },
        },
        Instr::Discard,
        Instr::Push { value: Value::Unit },
    ];
    assert!(verify(&module).is_err());
    module.functions[0].blocks[0].instrs = vec![local, Instr::ReadOnly, Instr::Load];
    module.functions[0].result = Ty::Int32;
    verify(&module).unwrap();
}
