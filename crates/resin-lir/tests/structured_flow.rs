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
        block(vec![Value::Bool { value: false }], Terminator::LoopTest),
        block(vec![], Terminator::Continue),
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
fn function_paths_require_an_explicit_return() {
    for (terminator, expected) in [
        (Terminator::Merge, VerifyErrorKind::UnexpectedMerge),
        (Terminator::LoopTest, VerifyErrorKind::UnexpectedLoopTest),
        (Terminator::Continue, VerifyErrorKind::UnexpectedContinue),
    ] {
        assert_eq!(
            error(module(vec![block(vec![Value::Unit], terminator)])),
            expected
        );
    }
    let m = module(vec![
        block(vec![Value::Bool { value: true }], branch(None)),
        block(vec![Value::Unit], Terminator::Return),
        block(vec![Value::Unit], Terminator::Merge),
    ]);
    assert_eq!(error(m), VerifyErrorKind::MissingRegionContinuation);
}

#[test]
fn region_exit_variants_are_not_interchangeable() {
    for (terminator, expected) in [
        (Terminator::LoopTest, VerifyErrorKind::UnexpectedLoopTest),
        (Terminator::Continue, VerifyErrorKind::UnexpectedContinue),
    ] {
        let m = module(vec![
            block(vec![Value::Bool { value: true }], branch(Some(3))),
            block(vec![], terminator),
            block(vec![], Terminator::Merge),
            block(vec![Value::Unit], Terminator::Return),
        ]);
        assert_eq!(error(m), expected);
    }
    for (id, terminator, expected) in [
        (1, Terminator::Merge, VerifyErrorKind::UnexpectedMerge),
        (1, Terminator::Continue, VerifyErrorKind::UnexpectedContinue),
        (2, Terminator::Merge, VerifyErrorKind::UnexpectedMerge),
        (2, Terminator::LoopTest, VerifyErrorKind::UnexpectedLoopTest),
    ] {
        let mut m = loop_module();
        m.functions[0].blocks[id].terminator = terminator;
        assert_eq!(error(m), expected);
    }
}

#[test]
fn nested_condition_branches_merge_before_testing_the_loop() {
    let mut m = loop_module();
    m.functions[0].blocks[1] = block(
        vec![Value::Bool { value: true }],
        Terminator::If {
            then: BlockId::from_index(4),
            els: BlockId::from_index(5),
            next: Some(BlockId::from_index(6)),
        },
    );
    m.functions[0].blocks.extend([
        block(vec![Value::Bool { value: false }], Terminator::Merge),
        block(vec![Value::Unit], Terminator::Return),
        block(vec![], Terminator::LoopTest),
    ]);
    VerifiedModule::new(m.clone()).unwrap();
    let mut direct_test = m.clone();
    direct_test.functions[0].blocks[4].terminator = Terminator::LoopTest;
    assert_eq!(error(direct_test), VerifyErrorKind::UnexpectedLoopTest);
    let Terminator::If { next, .. } = &mut m.functions[0].blocks[1].terminator else {
        unreachable!()
    };
    *next = None;
    m.functions[0].blocks.pop();
    assert_eq!(error(m), VerifyErrorKind::MissingRegionContinuation);
}

#[test]
fn nested_body_branches_merge_before_continuing_the_loop() {
    let mut m = loop_module();
    m.functions[0].blocks[2] = block(
        vec![Value::Bool { value: true }],
        Terminator::If {
            then: BlockId::from_index(4),
            els: BlockId::from_index(5),
            next: Some(BlockId::from_index(6)),
        },
    );
    m.functions[0].blocks.extend([
        block(vec![], Terminator::Merge),
        block(vec![], Terminator::Merge),
        block(vec![], Terminator::Continue),
    ]);
    VerifiedModule::new(m.clone()).unwrap();
    let mut direct_continue = m.clone();
    direct_continue.functions[0].blocks[4].terminator = Terminator::Continue;
    assert_eq!(error(direct_continue), VerifyErrorKind::UnexpectedContinue);
    let Terminator::If { next, .. } = &mut m.functions[0].blocks[2].terminator else {
        unreachable!()
    };
    *next = None;
    m.functions[0].blocks.pop();
    assert_eq!(error(m), VerifyErrorKind::MissingRegionContinuation);
}

#[test]
fn nested_loops_need_an_explicit_enclosing_region_exit() {
    for (id, values, terminator) in [
        (1, vec![Value::Bool { value: false }], Terminator::LoopTest),
        (2, vec![], Terminator::Continue),
    ] {
        let mut m = loop_module();
        m.functions[0].blocks[id] = block(
            vec![],
            Terminator::Loop {
                condition: BlockId::from_index(4),
                body: BlockId::from_index(5),
                next: Some(BlockId::from_index(6)),
            },
        );
        m.functions[0].blocks.extend([
            block(vec![Value::Bool { value: false }], Terminator::LoopTest),
            block(vec![], Terminator::Continue),
            block(values, terminator),
        ]);
        VerifiedModule::new(m.clone()).unwrap();
        let Terminator::Loop { next, .. } = &mut m.functions[0].blocks[id].terminator else {
            unreachable!()
        };
        *next = None;
        m.functions[0].blocks.pop();
        assert_eq!(error(m), VerifyErrorKind::MissingRegionContinuation);
    }
}

#[test]
fn tail_selections_can_forward_merge_values_to_an_outer_selection() {
    let m = module(vec![
        block(vec![Value::Bool { value: true }], branch(Some(3))),
        block(
            vec![Value::Bool { value: false }],
            Terminator::If {
                then: BlockId::from_index(4),
                els: BlockId::from_index(5),
                next: None,
            },
        ),
        block(vec![Value::Unit], Terminator::Merge),
        block(vec![], Terminator::Return),
        block(vec![Value::Unit], Terminator::Merge),
        block(vec![Value::Unit], Terminator::Merge),
    ]);
    VerifiedModule::new(m).unwrap();
}

#[test]
fn returning_arms_do_not_contribute_operands_to_the_continuation() {
    let m = module(vec![
        block(vec![Value::Bool { value: true }], branch(Some(3))),
        block(vec![Value::Unit], Terminator::Return),
        block(vec![], Terminator::Merge),
        block(vec![Value::Unit], Terminator::Return),
    ]);
    VerifiedModule::new(m.clone()).unwrap();
    let mut both_return = m;
    both_return.functions[0].blocks[2] = block(vec![Value::Unit], Terminator::Return);
    assert_eq!(
        error(both_return.clone()),
        VerifyErrorKind::MissingRegionResult
    );
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
    m.functions[0].blocks[1] = block(vec![Value::Int32 { value: 1 }], Terminator::LoopTest);
    assert_eq!(
        error(m),
        VerifyErrorKind::TypeMismatch {
            expected: Ty::Bool,
            found: Ty::Int32
        }
    );
}

#[test]
fn loop_body_may_return_and_condition_needs_a_test_path() {
    let mut m = loop_module();
    m.functions[0].blocks[2] = block(vec![Value::Unit], Terminator::Return);
    VerifiedModule::new(m.clone()).unwrap();
    m.functions[0].blocks[1] = block(vec![Value::Unit], Terminator::Return);
    assert_eq!(error(m), VerifyErrorKind::MissingLoopTest);
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
        blocks.push(block(vec![], Terminator::Merge));
        blocks.push(block(vec![], Terminator::Merge));
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
