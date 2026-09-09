use resin_common::prelude::*;
use resin_lir::{BasicBlock, BlockId, Function, Instr, Local, Module, Terminator};
use resin_lir_verifier::{VerifiedModule, VerifyErrorKind};

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
