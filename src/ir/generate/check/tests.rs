use super::{CheckedFile, Scopes};
use crate::{
    ast::{AstGen, Ident, Span},
    ir::{
        Ty,
        generate::{
            Generator,
            typed::{StatementKind, TermKind},
        },
    },
};

fn check(source: &str, generator: &mut Generator) -> CheckedFile {
    let mut parser = tree_sitter::Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .unwrap();
    let tree = parser.parse(source, None).unwrap();
    let file = AstGen::new(source)
        .gen_source_file(tree.root_node())
        .unwrap();
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
fn lowering_uses_concrete_trees_after_source_and_solver_are_dropped() {
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

    // Only the concrete checked product and declaration context cross this boundary.
    generator.environment.context = checked.context;
    let name = |name: &std::sync::Arc<str>| Ident {
        val: name.clone(),
        span: Span { start: 0, end: 0 },
    };
    for (key, signature) in &checked.signatures {
        generator.declare_function(&name(key), signature).unwrap();
    }
    for (key, body) in &checked.bodies {
        let environment = generator.environment.clone();
        generator
            .gen_function(&name(key), &checked.signatures[key], body)
            .unwrap();
        generator.environment = environment;
    }
    let verified = generator.finish().unwrap();
    let function = verified
        .view()
        .module()
        .functions
        .iter()
        .find(|function| function.name.as_deref() == Some("value"))
        .unwrap();
    assert_eq!(function.result, Ty::Int32);
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
