use resin::ast::{SourceFile, StmtKind, Term, TermKind, generate::AstGen, print};
use tree_sitter::Parser;

fn parse(src: &str) -> SourceFile {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_resin::LANGUAGE.into())
        .expect("failed to load Resin grammar");
    let tree = parser.parse(src, None).expect("parser returned no tree");
    AstGen::new(src)
        .gen_source_file(tree.root_node())
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

#[test]
fn assignment_is_right_associative_and_deref_is_explicit() {
    let file = parse("p.* := q.* := 1;");
    let StmtKind::Expr { term } = &file.stmts[0].val else {
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

    let rendered = print::format_source(&file);
    assert!(rendered.contains("(expr"));
    assert_eq!(rendered.matches("(assign").count(), 2);
    assert_eq!(rendered.matches("(deref").count(), 2);
}

#[test]
fn assignment_can_be_sequenced_in_a_block() {
    let file = parse("f = (p: Ptr (int)) => { p.* := 1; p.* };");
    let StmtKind::Define { init, .. } = &file.stmts[0].val else {
        panic!("expected definition");
    };
    let TermKind::Lambda { body, .. } = &init.val else {
        panic!("expected lambda, got {:?}", init.val);
    };
    let TermKind::Block { stmts, tail } = &body.val else {
        panic!("expected block, got {:?}", body.val);
    };
    assert!(matches!(stmts[0].val, StmtKind::Expr { .. }));
    expect_deref_var(tail, "p");
}

#[test]
fn dereference_composes_with_other_postfix_operations() {
    let file = parse("p.next.* := 1;");
    let StmtKind::Expr { term } = &file.stmts[0].val else {
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

    let rendered = print::format_source(&file);
    assert!(rendered.contains("(field"));
}
