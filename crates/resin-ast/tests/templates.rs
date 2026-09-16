mod common;

use resin_ast::{SourceFile, StmtKind, TermKind, TypeKind};

fn parse(text: &str) -> SourceFile {
    common::parse(text).file
}

#[test]
fn type_and_function_binders_remain_separate_from_weak_variables() {
    let source = "struct Pair<T> { first: T,  }\nfn map<T, U>(self: Pair<T>, value: U) -> Pair<_>  { Pair<U> { first = value } }\n type Shared<T> = ArcPtr<Pair<T>>;";
    let file = parse(source);
    let StmtKind::Struct {
        type_params, body, ..
    } = &file.stmts[0].val
    else {
        panic!()
    };
    assert_eq!(
        type_params
            .iter()
            .map(|p| p.val.as_ref())
            .collect::<Vec<_>>(),
        ["T"]
    );
    assert!(matches!(&body.val, TypeKind::Record { fields } if fields.len() == 1));
    let StmtKind::Function {
        type_params,
        name,
        result,
        ..
    } = &file.stmts[1].val
    else {
        panic!()
    };
    assert_eq!(name.val.as_ref(), "map");
    assert_eq!(
        type_params
            .iter()
            .map(|p| p.val.as_ref())
            .collect::<Vec<_>>(),
        ["T", "U"]
    );
    let TypeKind::App { head, args } = &result.val else {
        panic!()
    };
    assert_eq!(head.val.as_ref(), "Pair");
    assert!(matches!(args[0].val, TypeKind::Infer));
    assert_eq!(&source[args[0].span.start..args[0].span.end], "_");
    let StmtKind::DefineType {
        type_params, init, ..
    } = &file.stmts[2].val
    else {
        panic!()
    };
    assert_eq!(type_params.len(), 1);
    assert!(matches!(&init.val, TypeKind::App { head, .. } if head.val.as_ref() == "ArcPtr"));
    let printed = resin_ast::format_source(&file);
    assert!(printed.contains("(template Pair T)"), "{printed}");
    assert!(printed.contains("(template map T U)"), "{printed}");
}

#[test]
fn function_applications_and_method_arguments_have_explicit_ast_forms() {
    let file = parse(
        "fn example()  { let mut f = identity::<Ptr<int>>; f(1); gpu:alloc::<Pair<int, long>>(4); create::<int, long, ubyte>(7); }",
    );
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!()
    };
    let TermKind::Block { stmts, .. } = &body.val else {
        panic!()
    };
    let StmtKind::Define { init, .. } = &stmts[0].val else {
        panic!()
    };
    let TermKind::TypeApply { function, args } = &init.val else {
        panic!("{init:?}")
    };
    assert!(matches!(&function.val, TermKind::Var { name } if name.val.as_ref() == "identity"));
    assert!(matches!(&args[0].val, TypeKind::App { head, .. } if head.val.as_ref() == "Ptr"));
    for statement in &stmts[2..3] {
        let StmtKind::Expr { term } = &statement.val else {
            panic!()
        };
        let TermKind::MethodCall { type_args, .. } = &term.val else {
            panic!("{term:?}")
        };
        assert_eq!(type_args.len(), 1);
    }
}

#[test]
fn type_arguments_do_not_consume_comparisons_or_shifts() {
    parse(
        "fn f<T>(value: T) -> T  { value } fn main()  { let mut n = f::<int>(1); let mut less = n < 2; let mut more = n > 0; let mut shifted = n >> 1; }",
    );
}

#[test]
fn template_lists_require_named_parameters_and_nonempty_arguments() {
    for source in [
        "fn f<_>(x: int) -> int  { x }",
        "fn f<>()  {}",
        "fn f()  { identity::<>(1); }",
        "extern { \"test.h\": { fn native<T>(x: T) -> T; } };",
        "type Empty<> = int;",
    ] {
        assert!(!common::parse(source).errors.is_empty(), "{source}");
    }
}

#[test]
fn free_function_references_keep_explicit_arguments_without_a_call() {
    let source = "fn example()  { let mut first = select::<int, ulong>; let mut second = create::<Ptr<int>>; select::<int, ulong>(7, 42); (select::<int, ulong>)(7, 42); }";
    let file = parse(source);
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!("function")
    };
    let TermKind::Block { stmts, .. } = &body.val else {
        panic!("block")
    };
    for (statement, (expected, count)) in stmts[..2].iter().zip([("select", 2), ("create", 1)]) {
        let StmtKind::Define { init, .. } = &statement.val else {
            panic!("binding")
        };
        let TermKind::TypeApply { function, args } = &init.val else {
            panic!("method application: {init:?}")
        };
        assert!(matches!(function.val, TermKind::Var { .. }));
        assert_eq!(args.len(), count);
        assert_eq!(&source[function.span.start..function.span.end], expected);
        assert!(source[init.span.start..init.span.end].ends_with('>'));
    }
    let StmtKind::Expr { term: direct } = &stmts[2].val else {
        panic!("call")
    };
    assert!(matches!(&direct.val, TermKind::Call { .. }));
    let StmtKind::Expr { term: indirect } = &stmts[3].val else {
        panic!("call")
    };
    let TermKind::Call { func, .. } = &indirect.val else {
        panic!("ordinary call")
    };
    assert!(matches!(&func.val, TermKind::TypeApply { args, .. } if args.len() == 2));
}
