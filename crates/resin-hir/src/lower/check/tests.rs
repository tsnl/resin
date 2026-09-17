use super::{CheckedFile, Scopes};
use crate::lower::Generator;
use crate::{Statement, TermKind};

fn check(source: &str, generator: &mut Generator) -> CheckedFile {
    let file = crate::lower::test_source(source).file;
    let mut scopes = Scopes::new();
    assert!(
        scopes
            .prepare(&file, &mut generator.typer)
            .errors
            .is_empty()
    );
    super::file(&file, generator, scopes, Default::default())
}

#[test]
fn checking_resolves_types_in_earlier_expressions_and_annotations() {
    let mut generator = Generator::new();
    let checked = check(
        "fn narrow(n: i32) -> i32  { n }\nfn value() -> _  {\nlet mut item = 42;\nlet mut copied: _;\ncopied = item;\nnarrow(copied)\n}",
        &mut generator,
    );
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    assert!(generator.functions.iter().all(|function| {
        function
            .as_ref()
            .is_some_and(|function| function.body.is_none())
    }));
    let value = checked.context.lookup("value", false).unwrap();
    assert_eq!(checked.signatures[&value].result.ty, crate::Type::Int32);
    let TermKind::Block { stmts, .. } = &checked.bodies[&value].kind else {
        panic!()
    };
    let Statement::Define { init, .. } = &stmts[0] else {
        panic!()
    };
    assert_eq!(init.ty, crate::Type::Int32); // The later call constrained the earlier literal.
    let Statement::Declare { ty, .. } = &stmts[1] else {
        panic!()
    };
    assert_eq!(ty.ty, crate::Type::Int32);
}

#[test]
fn recovery_keeps_signatures_and_healthy_trees_without_lowering_failed_bodies() {
    let checked = check(
        "fn bad() -> i32  { missing() } fn healthy() -> _  { i32(42) }",
        &mut Generator::new(),
    );
    assert!(!checked.errors.is_empty());
    let bad = checked.context.lookup("bad", false).unwrap();
    let healthy = checked.context.lookup("healthy", false).unwrap();
    assert_eq!(checked.signatures[&bad].result.ty, crate::Type::Int32);
    assert!(!checked.bodies.contains_key(&bad));
    assert_eq!(checked.bodies[&healthy].ty, crate::Type::Int32);
}

#[test]
fn recursive_groups_follow_dependencies() {
    assert_eq!(
        super::groups(&[vec![1], vec![0, 2], vec![]]),
        vec![vec![2], vec![1, 0]]
    );
}

#[test]
fn completed_bodies_keep_shadowed_references_after_discarding_construction_state() {
    let mut generator = Generator::new();
    let mut checked = check(
        "fn target(n: i32) -> i32  { n }\nfn caller(target: i32) -> i32  {\nlet mut outer = target;\n{ let mut target = outer + 1; target } + target\n}\nfn invoke() -> i32  { target(7) }",
        &mut generator,
    );
    assert!(checked.errors.is_empty(), "{:?}", checked.errors);
    // Assembly only consumes completed HIR, even after construction state is gone.
    generator.typer = Default::default();
    checked.context = Scopes::new().finish();
    generator.define_functions(checked);
    assert!(generator.errors.is_empty(), "{:?}", generator.errors);
    let module = generator.finish();

    let caller = &module.functions[1];
    let parameter = caller.signature.params[0].binding.unwrap();
    let crate::TermKind::Block { stmts, tail } = &caller.body.as_ref().unwrap().kind else {
        panic!()
    };
    let crate::Statement::Define { init, .. } = &stmts[0] else {
        panic!()
    };
    assert!(matches!(init.kind, crate::TermKind::Local { binding, .. } if binding == parameter));
    let crate::TermKind::Builtin { args, .. } = &tail.kind else {
        panic!()
    };
    assert!(matches!(args[1].kind, crate::TermKind::Local { binding, .. } if binding == parameter));
    let crate::TermKind::Block { stmts, tail } = &args[0].kind else {
        panic!()
    };
    let crate::Statement::Define { binding, .. } = &stmts[0] else {
        panic!()
    };
    assert_ne!(*binding, parameter);
    assert!(
        matches!(tail.kind, crate::TermKind::Local { binding: local, .. } if local == *binding)
    );

    let crate::TermKind::Block { tail, .. } = &module.functions[2].body.as_ref().unwrap().kind
    else {
        panic!()
    };
    let crate::TermKind::Call { func, .. } = &tail.kind else {
        panic!()
    };
    assert!(
        matches!(func.kind, crate::TermKind::Function { function, .. } if function.index() == 0)
    );
}

#[test]
fn declaration_identities_do_not_depend_on_unique_source_spans() {
    use resin_source::prelude::*;
    let mut file = crate::lower::test_source(
        "fn integer(value: i32) -> _  { value }\nfn boolean(value: bool) -> _  { value }",
    )
    .file;
    for stmt in &mut file.stmts {
        let resin_ast::StmtKind::Function {
            name, params, body, ..
        } = &mut stmt.val
        else {
            panic!()
        };
        name.span = Span { start: 0, end: 0 };
        params[0].0.name.span = name.span;
        body.span = name.span;
        let resin_ast::TermKind::Block { tail, .. } = &mut body.val else {
            panic!()
        };
        tail.span = name.span;
        let resin_ast::TermKind::Var { name: reference } = &mut tail.val else {
            panic!()
        };
        reference.span = name.span;
    }
    let mut generator = Generator::new();
    generator.generate_file(&file, Scopes::new());
    assert!(generator.errors.is_empty(), "{:?}", generator.errors);
    let functions = &generator.finish().functions;
    assert_eq!(functions[0].signature.result.ty, crate::Type::Int32);
    assert_eq!(functions[1].signature.result.ty, crate::Type::Bool);
    assert_ne!(
        functions[0].signature.params[0].binding,
        functions[1].signature.params[0].binding
    );
    for function in functions {
        let crate::TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
            panic!()
        };
        assert!(
            matches!(tail.kind, crate::TermKind::Local { binding, .. } if Some(binding) == function.signature.params[0].binding)
        );
    }
}
