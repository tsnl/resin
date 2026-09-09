use super::{CheckedFile, Scopes};
use crate::lower::{
    Generator,
    typed::{StatementKind, TermKind},
};
use resin_common::prelude::*;

fn check(source: &str, generator: &mut Generator) -> CheckedFile {
    let file =
        resin_ast::generate(&resin_cst::Document::reparse(source.to_string(), None)).unwrap();
    let mut scopes = Scopes::new();
    assert!(
        scopes
            .prepare(&file, &mut generator.typer, generator.source_module)
            .is_empty()
    );
    super::file(
        &file,
        &mut generator.typer,
        scopes,
        generator.source_module,
        Default::default(),
    )
}

#[test]
fn checking_resolves_types_in_earlier_expressions_and_annotations() {
    let mut generator = Generator::new();
    let checked = check(
        "def narrow(n: int) -> int = { n };\n\
         def value() -> _ = {\n\
             var item = 42;\n\
             var pointer: Ptr<_>;\n\
             pointer := &item;\n\
             narrow(pointer.*)\n\
         };",
        &mut generator,
    );
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    assert!(generator.module.functions.is_empty());
    assert_eq!(checked.signatures["value"].result.ty, Ty::Int32);
    let TermKind::Block { stmts, .. } = &checked.bodies["value"].kind else {
        panic!()
    };
    let StatementKind::Define { init, .. } = &stmts[0].kind else {
        panic!()
    };
    assert_eq!(init.ty, Ty::Int32); // The later call constrained the earlier literal.
    let StatementKind::Declare { ty, .. } = &stmts[1].kind else {
        panic!()
    };
    assert_eq!(
        ty.ty,
        Ty::Pointer {
            pointee: Box::new(Ty::Int32)
        }
    );
}

#[test]
fn recovery_keeps_signatures_and_healthy_trees_without_lowering_failed_bodies() {
    let checked = check(
        "def bad() -> int = { missing() }; def healthy() -> _ = { 42_i };",
        &mut Generator::new(),
    );
    assert!(!checked.errors.is_empty());
    assert_eq!(checked.signatures["bad"].result.ty, Ty::Int32);
    assert!(!checked.bodies.contains_key("bad"));
    assert_eq!(checked.bodies["healthy"].ty, Ty::Int32);
}
