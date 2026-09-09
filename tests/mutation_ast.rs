use resin_ast::{SourceFile, StmtKind, Term, TermKind, format_source};
use resin_source::prelude::*;

fn parse(src: &str) -> SourceFile {
    resin_ast::generate(&resin_cst::Document::reparse(src.to_string(), None))
        .unwrap_or_else(|err| panic!("{err}"))
}

fn expect_var(term: &Term, expected: &str) {
    let TermKind::Var { name } = &term.val else {
        panic!("expected variable, got {:?}", term.val);
    };
    assert_eq!(name.val.as_ref(), expected);
}

fn expect_deref_var(term: &Term, expected: &str) {
    let TermKind::Deref { pointer } = &term.val else {
        panic!("expected pointer dereference, got {:?}", term.val);
    };
    expect_var(pointer, expected);
}

fn first_statement(file: &SourceFile) -> &resin_ast::Stmt {
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!("expected function");
    };
    let TermKind::Block { stmts, .. } = &body.val else {
        panic!("expected function body");
    };
    &stmts[0]
}

#[test]
fn omitted_function_results_lower_to_unit() {
    let source =
        "def implicit() = {}; def explicit() -> () = {}; extern \"header.h\" def foreign();";
    let file = parse(source);
    for (index, statement) in file.stmts.iter().enumerate() {
        let (name, result) = match &statement.val {
            StmtKind::Function { name, result, .. }
            | StmtKind::ForeignFunction { name, result, .. } => (name, result),
            _ => panic!("expected function"),
        };
        assert!(matches!(result.val, resin_ast::TypeKind::Unit));
        assert_eq!(
            &source[result.span.start..result.span.end],
            if index == 1 { "()" } else { name.val.as_ref() }
        );
    }
}

#[test]
fn definition_keywords_preserve_statement_and_field_spans() {
    let source = r#"
        struct Pair { first: int, second: int };
        def make(seed: int) -> Pair = {
            var first = seed;
            var second: int;
            second := seed + 1;
            Pair { first = first, second = { var next = second; next } }
        };
    "#;
    let file = parse(source);
    let text = |span: Span| &source[span.start..span.end];
    let definition = &file.stmts[0];
    assert!(matches!(definition.val, StmtKind::Struct { .. }));
    assert_eq!(
        text(definition.span),
        "struct Pair { first: int, second: int };"
    );
    let function = &file.stmts[1];
    let StmtKind::Function {
        name, params, body, ..
    } = &function.val
    else {
        panic!("expected function");
    };
    assert!(text(function.span).starts_with("def make("));
    assert_eq!(text(name.span), "make");
    assert_eq!(text(params[0].0.span), "seed");
    let TermKind::Block { stmts, tail } = &body.val else {
        panic!("expected function body");
    };
    assert!(matches!(stmts[0].val, StmtKind::Define { .. }));
    assert_eq!(text(stmts[0].span), "var first = seed;");
    assert!(matches!(stmts[1].val, StmtKind::Declare { .. }));
    assert_eq!(text(stmts[1].span), "var second: int;");
    assert!(matches!(stmts[2].val, StmtKind::Expr { .. }));
    let TermKind::Call { arg, .. } = &tail.val else {
        panic!("expected nominal conversion");
    };
    let TermKind::Record { fields } = &arg.val else {
        panic!("expected record initializer");
    };
    assert_eq!(text(fields[0].0.span), "first");
    expect_var(&fields[0].1, "first");
    let TermKind::Block { stmts, tail } = &fields[1].1.val else {
        panic!("expected block-valued field");
    };
    assert_eq!(text(stmts[0].span), "var next = second;");
    expect_var(tail, "next");
}

#[test]
fn while_has_a_condition_and_a_scoped_body() {
    let file = parse("def main() -> () = { while (ready) { count := count + 1; }; };");
    let StmtKind::Expr { term } = &first_statement(&file).val else {
        panic!("expected expression statement");
    };
    let TermKind::While { cond, body } = &term.val else {
        panic!("expected while expression");
    };
    expect_var(cond, "ready");
    let TermKind::Block { stmts, tail } = &body.val else {
        panic!("expected loop body");
    };
    assert_eq!(stmts.len(), 1);
    assert!(matches!(tail.val, TermKind::Unit));
    assert!(format_source(&file).contains("(while"));

    parse("def main() -> () = { while (ready) {}; };");
    parse("def f () -> () = { while (ready) { while (ready) {}; } };");
    parse("def main() -> () = { var value = while (ready) { 42 }; };");
}

#[test]
fn assignment_is_right_associative_and_deref_is_explicit() {
    let file = parse("def main() -> () = { p.* := q.* := 1; };");
    let StmtKind::Expr { term } = &first_statement(&file).val else {
        panic!("expected expression statement");
    };
    let TermKind::Assign { place, value } = &term.val else {
        panic!("expected assignment, got {:?}", term.val);
    };
    expect_deref_var(place, "p");

    let TermKind::Assign { place, value } = &value.val else {
        panic!("expected right-nested assignment, got {:?}", value.val);
    };
    expect_deref_var(place, "q");
    assert!(matches!(value.val, TermKind::Num { .. }));

    let rendered = format_source(&file);
    assert!(rendered.contains("(expr"));
    assert_eq!(rendered.matches("(assign").count(), 2);
    assert_eq!(rendered.matches("(deref").count(), 2);
}

#[test]
fn assignment_can_be_sequenced_in_a_block() {
    let file = parse("def f (p: Ptr<int>) -> int = { p.* := 1; p.* };");
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!("expected function definition");
    };
    let TermKind::Block { stmts, tail } = &body.val else {
        panic!("expected block, got {:?}", body.val);
    };
    assert!(matches!(stmts[0].val, StmtKind::Expr { .. }));
    expect_deref_var(tail, "p");
}

#[test]
fn dereference_composes_with_other_postfix_operations() {
    let file = parse("def main() -> () = { p.next.* := 1; };");
    let StmtKind::Expr { term } = &first_statement(&file).val else {
        panic!("expected expression statement");
    };
    let TermKind::Assign { place, .. } = &term.val else {
        panic!("expected assignment, got {:?}", term.val);
    };
    let TermKind::Deref { pointer } = &place.val else {
        panic!("expected dereference, got {:?}", place.val);
    };
    let TermKind::Field { base, name } = &pointer.val else {
        panic!("expected field access, got {:?}", pointer.val);
    };
    expect_var(base, "p");
    assert_eq!(name.val.as_ref(), "next");

    let rendered = format_source(&file);
    assert!(rendered.contains("(field"));
}
