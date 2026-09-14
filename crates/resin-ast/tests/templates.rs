use resin_ast::{SourceFile, StmtKind, TermKind, TypeKind};
use resin_cst::Document;

fn parse(text: &str) -> SourceFile {
    resin_ast::generate(&Document::reparse(text.into(), None)).unwrap()
}

#[test]
fn type_and_method_binders_remain_separate_from_weak_variables() {
    let source = "struct Pair<T> { first: T, def map<U>(self: Pair<T>, value: U) -> Pair<_> = { Pair<U> { first = value } }; }; type Shared<T> = ArcPtr<Pair<T>>;";
    let file = parse(source);
    let StmtKind::Struct {
        type_params,
        body,
        methods,
        ..
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
    } = &methods[0].val
    else {
        panic!()
    };
    assert_eq!(name.val.as_ref(), "Pair.map");
    assert_eq!(
        type_params
            .iter()
            .map(|p| p.val.as_ref())
            .collect::<Vec<_>>(),
        ["U"]
    );
    let TypeKind::App { head, args } = &result.val else {
        panic!()
    };
    assert_eq!(head.val.as_ref(), "Pair");
    assert!(matches!(args[0].val, TypeKind::Infer));
    assert_eq!(&source[args[0].span.start..args[0].span.end], "_");
    let StmtKind::DefineType {
        type_params, init, ..
    } = &file.stmts[1].val
    else {
        panic!()
    };
    assert_eq!(type_params.len(), 1);
    assert!(matches!(&init.val, TypeKind::App { head, .. } if head.val.as_ref() == "ArcPtr"));
    let printed = resin_ast::format_source(&file);
    assert!(printed.contains("(template Pair T)"), "{printed}");
    assert!(printed.contains("(template Pair.map U)"), "{printed}");
}

#[test]
fn function_applications_and_method_arguments_have_explicit_ast_forms() {
    let file = parse(
        "def example() = { var f = identity::<Ptr<int>>; f(1); gpu.alloc::<Pair<int, long>>(4); Pair<int, long>.create::<ubyte>(7); };",
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
    for statement in &stmts[2..] {
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
        "def f<T>(value: T) -> T = { value }; def main() = { var n = f::<int>(1); var less = n < 2; var more = n > 0; var shifted = n >> 1; };",
    );
}

#[test]
fn template_lists_require_named_parameters_and_nonempty_arguments() {
    for source in [
        "def f<_>(x: int) -> int = { x };",
        "def f<>() = {};",
        "def f() = { identity::<>(1); };",
        "extern \"test.h\" def native<T>(x: T) -> T;",
        "type Empty<> = int;",
    ] {
        assert!(
            resin_ast::generate(&Document::reparse(source.into(), None)).is_err(),
            "{source}"
        );
    }
}

#[test]
fn associated_method_references_keep_explicit_arguments_without_a_call() {
    let source = "def example() = { var first = Cell<int>.select::<ulong>; var second = Factory.create::<Ptr<int>>; Cell<int>.select::<ulong>(7, 42); (Cell<int>.select::<ulong>)(7, 42); };";
    let file = parse(source);
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!("function")
    };
    let TermKind::Block { stmts, .. } = &body.val else {
        panic!("block")
    };
    for (statement, expected) in stmts[..2]
        .iter()
        .zip(["Cell<int>.select", "Factory.create"])
    {
        let StmtKind::Define { init, .. } = &statement.val else {
            panic!("binding")
        };
        let TermKind::TypeApply { function, args } = &init.val else {
            panic!("method application: {init:?}")
        };
        assert!(matches!(function.val, TermKind::Field { .. }));
        assert_eq!(args.len(), 1);
        assert_eq!(&source[function.span.start..function.span.end], expected);
        assert!(source[init.span.start..init.span.end].ends_with('>'));
    }
    let StmtKind::Expr { term: direct } = &stmts[2].val else {
        panic!("call")
    };
    assert!(matches!(&direct.val, TermKind::MethodCall { type_args, .. } if type_args.len() == 1));
    let StmtKind::Expr { term: indirect } = &stmts[3].val else {
        panic!("call")
    };
    let TermKind::Call { func, .. } = &indirect.val else {
        panic!("ordinary call")
    };
    assert!(matches!(&func.val, TermKind::TypeApply { args, .. } if args.len() == 1));
}
