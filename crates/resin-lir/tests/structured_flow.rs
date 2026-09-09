use resin_lir::{BasicBlock, BlockId, Function, Local, Module, Terminator, VerifyErrorKind};
use resin_lir::{Instr, VerifiedModule, format_module};
use resin_types::prelude::*;

fn block(values: Vec<Value>, terminator: Terminator) -> BasicBlock {
    BasicBlock {
        name: None,
        instrs: values
            .into_iter()
            .map(|value| Instr::Push { value })
            .collect(),
        terminator,
    }
}

fn module(blocks: Vec<BasicBlock>) -> Module {
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
            blocks,
        }],
        ..Default::default()
    }
}

fn branch(next: Option<usize>) -> Terminator {
    Terminator::If {
        then: BlockId::from_index(1),
        els: BlockId::from_index(2),
        next: next.map(BlockId::from_index),
    }
}

fn loop_module() -> Module {
    module(vec![
        block(
            vec![],
            Terminator::Loop {
                condition: BlockId::from_index(1),
                body: BlockId::from_index(2),
                next: Some(BlockId::from_index(3)),
            },
        ),
        block(vec![Value::Bool { value: false }], Terminator::Yield),
        block(vec![], Terminator::Yield),
        block(vec![Value::Unit], Terminator::Return),
    ])
}

fn error(module: Module) -> VerifyErrorKind {
    VerifiedModule::new(module)
        .err()
        .expect("invalid structured LIR")
        .kind
}

#[test]
fn blocks_have_exactly_one_owner_and_cannot_form_cycles() {
    for reused in [0, 1] {
        let mut m = module(vec![
            block(vec![Value::Bool { value: true }], branch(None)),
            block(vec![Value::Unit], Terminator::Return),
        ]);
        m.functions[0].blocks[0].terminator = Terminator::If {
            then: BlockId::from_index(1),
            els: BlockId::from_index(reused),
            next: None,
        };
        assert!(format_module(&m).contains("reused-block"));
        assert_eq!(error(m), VerifyErrorKind::ReusedBasicBlock);
    }
    let m = module(vec![
        block(vec![Value::Unit], Terminator::Return),
        block(vec![Value::Unit], Terminator::Return),
    ]);
    assert!(format_module(&m).contains("unreachable"));
    assert_eq!(error(m), VerifyErrorKind::UnreachableBasicBlock);
}

#[test]
fn invalid_child_ids_are_rejected() {
    let m = module(vec![block(vec![Value::Bool { value: true }], branch(None))]);
    assert!(format_module(&m).contains("invalid-block"));
    assert_eq!(
        error(m),
        VerifyErrorKind::InvalidBasicBlock { basic_block: 1 }
    );
}

#[test]
fn function_paths_must_return_instead_of_yielding() {
    for m in [
        module(vec![block(vec![Value::Unit], Terminator::Yield)]),
        module(vec![
            block(vec![Value::Bool { value: true }], branch(None)),
            block(vec![Value::Unit], Terminator::Return),
            block(vec![Value::Unit], Terminator::Yield),
        ]),
    ] {
        assert_eq!(error(m), VerifyErrorKind::UnexpectedYield);
    }
}

#[test]
fn returning_arms_do_not_contribute_operands_to_the_continuation() {
    let m = module(vec![
        block(vec![Value::Bool { value: true }], branch(Some(3))),
        block(vec![Value::Unit], Terminator::Return),
        block(vec![], Terminator::Yield),
        block(vec![Value::Unit], Terminator::Return),
    ]);
    VerifiedModule::new(m.clone()).unwrap();
    let mut both_return = m;
    both_return.functions[0].blocks[2] = block(vec![Value::Unit], Terminator::Return);
    assert_eq!(error(both_return.clone()), VerifyErrorKind::MissingYield);
    both_return.functions[0].blocks[0].terminator = branch(None);
    both_return.functions[0].blocks.pop();
    VerifiedModule::new(both_return).unwrap();
}

#[test]
fn loop_condition_and_body_must_preserve_carried_operand_types() {
    VerifiedModule::new(loop_module()).unwrap();
    for id in [1, 2] {
        let mut m = loop_module();
        m.functions[0].blocks[id].instrs.insert(
            0,
            Instr::Push {
                value: Value::Int32 { value: 7 },
            },
        );
        assert!(matches!(
            error(m),
            VerifyErrorKind::ConflictingBasicBlockStack { .. }
        ));
    }
    let mut m = loop_module();
    m.functions[0].blocks[1] = block(vec![Value::Int32 { value: 1 }], Terminator::Yield);
    assert_eq!(
        error(m),
        VerifyErrorKind::TypeMismatch {
            expected: Ty::Bool,
            found: Ty::Int32
        }
    );
}

#[test]
fn loop_body_may_return_and_condition_needs_a_yielding_path() {
    let mut m = loop_module();
    m.functions[0].blocks[2] = block(vec![Value::Unit], Terminator::Return);
    VerifiedModule::new(m.clone()).unwrap();
    m.functions[0].blocks[1] = block(vec![Value::Unit], Terminator::Return);
    assert_eq!(error(m), VerifyErrorKind::MissingYield);
}

#[test]
fn long_sequences_do_not_consume_nesting_depth() {
    let mut blocks = Vec::new();
    for i in 0..1024 {
        let start = i * 3;
        blocks.push(block(
            vec![Value::Bool { value: true }],
            Terminator::If {
                then: BlockId::from_index(start + 1),
                els: BlockId::from_index(start + 2),
                next: Some(BlockId::from_index(start + 3)),
            },
        ));
        blocks.push(block(vec![], Terminator::Yield));
        blocks.push(block(vec![], Terminator::Yield));
    }
    blocks.push(block(vec![Value::Unit], Terminator::Return));
    let checked = VerifiedModule::new(module(blocks)).unwrap();
    let output = format_module(checked.view().module());
    assert_eq!(output.matches("(if ").count(), 1024);
    assert!(
        output.len() < 200_000,
        "continuations must not increase indentation"
    );
}
