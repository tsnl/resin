use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator};
use resin_lir::{VerifiedModule, VerifyErrorKind};
use resin_types::prelude::*;

fn module() -> Module {
    Module {
        functions: vec![Function {
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
        }],
        ..Default::default()
    }
}

#[test]
fn editing_consumes_the_certificate_and_requires_reverification() {
    let checked = VerifiedModule::new(module()).unwrap();
    let mut editable = checked.into_module();
    editable.functions[0].locals.clear();
    let error = VerifiedModule::new(editable)
        .err()
        .expect("invalid parameter slot");
    assert_eq!(error.kind, VerifyErrorKind::InvalidLocal { local: 0 });
}

#[test]
fn canonical_type_table_agrees_with_the_certified_analysis() {
    let checked = VerifiedModule::new(module()).unwrap();
    let view = checked.view();
    assert_eq!(view.module().types, view.analysis().types);
    assert_eq!(view.analysis().functions[0].results, [vec![Some(Ty::Unit)]]);
}

#[test]
fn certified_operands_follow_block_indices_and_keep_instructions_without_results() {
    let mut module = module();
    let parameter = Ty::Record {
        fields: vec![
            RecordField {
                name: "_0".into(),
                ty: Ty::Int32,
            },
            RecordField {
                name: "_1".into(),
                ty: Ty::Bool,
            },
        ],
    };
    module.functions[0].locals[0].ty = parameter;
    let callee = module.functions[0].ty().unwrap();
    let mut caller = module.functions[0].clone();
    caller.locals = vec![
        Local {
            name: None,
            ty: Ty::Unit,
        },
        Local {
            name: None,
            ty: Ty::Array {
                element: Box::new(Ty::Int32),
                length: 3,
            },
        },
    ];
    caller.entry = BlockId::from_index(1);
    caller.blocks = vec![
        BasicBlock {
            name: None,
            instrs: vec![
                Instr::MakeRecord {
                    fields: vec!["_0".into(), "_1".into()],
                },
                Instr::Call,
                Instr::Discard,
                Instr::Push {
                    value: Value::Int32 { value: 10 },
                },
                Instr::Push {
                    value: Value::Int32 { value: 20 },
                },
                Instr::Push {
                    value: Value::Int32 { value: 30 },
                },
                Instr::MakeArray {
                    elements: 3,
                    element: Ty::Int32,
                },
                Instr::SetLocal {
                    local: LocalId::from_index(1),
                },
                Instr::Push { value: Value::Unit },
            ],
            terminator: Terminator::Return,
        },
        BasicBlock {
            name: None,
            instrs: vec![
                Instr::Function {
                    function: FunctionId::from_index(0),
                },
                Instr::Push {
                    value: Value::Int32 { value: 42 },
                },
                Instr::Push {
                    value: Value::Bool { value: true },
                },
                Instr::Push {
                    value: Value::Bool { value: false },
                },
            ],
            terminator: Terminator::If {
                then: BlockId::from_index(2),
                els: BlockId::from_index(3),
                next: Some(BlockId::from_index(0)),
            },
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Terminator::Yield,
        },
        BasicBlock {
            name: None,
            instrs: vec![],
            terminator: Terminator::Yield,
        },
    ];
    module.functions.push(caller);
    let checked = VerifiedModule::new(module).unwrap();
    let analysis = &checked.view().analysis().functions[1];
    let carried = vec![callee, Ty::Int32, Ty::Bool];
    assert_eq!(
        analysis.inputs,
        [carried.clone(), vec![], carried.clone(), carried]
    );
    for (block, counts) in [
        vec![2, 2, 1, 0, 0, 0, 3, 1, 0],
        vec![0, 0, 0, 0],
        vec![],
        vec![],
    ]
    .iter()
    .enumerate()
    {
        assert_eq!(analysis.results[block].len(), counts.len());
        for (instruction, expected) in counts.iter().enumerate() {
            assert_eq!(
                analysis.operand_count(BlockId::from_index(block), instruction),
                *expected
            );
        }
    }
    assert_eq!(analysis.results[0][1], Some(Ty::Unit));
    assert_eq!(analysis.results[0][2], None);
    assert_eq!(analysis.results[0][7], None);
}
