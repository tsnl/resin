use resin::{
    ast::{AstGen, StmtKind, TermKind},
    ir,
};

mod support;
use support::{module, parse};

fn rejects(source: &str, message: &str) {
    let error = ir::generate(&parse(source)).unwrap_err().to_string();
    assert!(
        error
            .to_ascii_lowercase()
            .contains(&message.to_ascii_lowercase()),
        "{source}\n{error}"
    );
}

#[test]
fn defer_is_only_a_prefix_statement() {
    let file = parse("def main() = { defer 1 + 2; };");
    let StmtKind::Function { body, .. } = &file.stmts[0].val else {
        panic!()
    };
    let TermKind::Block { stmts, .. } = &body.val else {
        panic!()
    };
    let StmtKind::Defer { body } = &stmts[0].val else {
        panic!()
    };
    assert!(matches!(&body.val, TermKind::Builtin { name, .. } if name.as_ref() == "+"));
    assert!(resin::ast::print::format_source(&file).contains("defer"));
    for source in [
        "defer {};",
        "def main() = { defer {} };",
        "def main() = { var defer = 1; };",
        "def main() = { var x = defer 42; };",
        "def main() = { print(defer 42); };",
        "def main() = { 1 + defer 42; };",
        "def main() = { defer defer 42; };",
        "def main() = { defer 42 };",
        "def main() = { defer var n = 42; };",
    ] {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_resin::LANGUAGE.into())
            .unwrap();
        let tree = parser.parse(source, None).unwrap();
        assert!(
            AstGen::new(source)
                .gen_source_file(tree.root_node())
                .is_err(),
            "{source}"
        );
    }
}

#[test]
fn deferred_expressions_discard_their_values() {
    module(
        r#"
        struct Pair { first: int, second: int };
        def answer() -> int = { 42 };
        def f() = {
            var n = 0;
            var p = &n;
            defer answer();
            defer n := n + 1;
            defer -n + 42;
            defer 1 == 1 && 2 == 2;
            defer { 42 };
            defer { var local = answer(); };
            defer if (n == 0) { answer() } else { n };
            defer while (n < 3) { n := n + 1; };
            defer [1, 2];
            defer (1, 2);
            defer { first = 1, second = 2 };
            defer Pair { first = 1, second = 2 };
            defer Pair { first = 1, second = 2 }.first;
            defer p.*;
            defer &n;
        };
    "#,
    );
}

#[test]
fn deferred_expressions_cannot_propagate() {
    rejects(
        "struct E {}; def f(r: Result<(), E>) -> Result<(), E> = { defer r?; ok(()) };",
        "not allowed in a deferred expression",
    );
    rejects(
        "struct E {}; def f(r: Result<(), E>) -> Result<(), E> = { defer { r?; }; ok(()) };",
        "not allowed in a deferred expression",
    );
    rejects(
        "struct E {}; def f(r: Result<(), E>) -> Result<(), E> = { defer { defer { r?; }; }; ok(()) };",
        "not allowed in a deferred expression",
    );
    module(
        "struct E {}; def f(r: Result<(), E>) = { defer { match (r) { ok(n) => {}, err(e) => {} }; }; };",
    );
    module(
        "struct E {}; def f(r: Result<int, E>) = { defer match (r) { ok(n) => { n }, err(e) => { 42 } }; };",
    );
}

#[test]
fn statement_only_chain_expressions_yield_unit() {
    module(
        "def f() = { var result = { defer 42; }; var empty = {}; if (1 == 1) { defer 1; } else { defer 2; }; result };",
    );
    rejects("def f() -> int = { { defer 42; } };", "incompatible");
}

#[test]
fn deferred_names_are_bound_at_registration() {
    rejects(
        "def f() = { defer { print(\"{0}\", (later,)); }; var later = 1; };",
        "unbound",
    );
    rejects(
        "def f() = { defer { var inner = 1; }; print(\"{0}\", (inner,)); };",
        "unbound",
    );
    module("def f() = { var n = 1; defer { print(\"{0}\", (n,)); }; n := 2; };");
}

#[test]
fn deferred_reads_require_initialization_on_every_exit() {
    module("def f() = { var n: int; defer { print(\"{0}\", (n,)); }; n := 42; };");
    rejects(
        "def f() = { var n: int; defer { print(\"{0}\", (n,)); }; };",
        "uninitialized",
    );
    rejects(
        "struct E {}; def f(r: Result<(), E>) -> Result<(), E> = { var n: int; defer { print(\"{0}\", (n,)); }; r?; n := 42; ok(()) };",
        "uninitialized",
    );
    rejects(
        "def f(b: bool) = { var n: int; defer { print(\"{0}\", (n,)); }; if (b) { n := 42; () } else { () }; };",
        "uninitialized",
    );
}

#[test]
fn cleanup_initialization_follows_execution_order() {
    module("def f() -> int = { var n: int; { defer { n := 42; }; () }; n };");
    module("def f() = { var n: int; defer { print(\"{0}\", (n,)); }; defer { n := 42; }; };");
    rejects(
        "def f() = { var n: int; defer { n := 42; }; defer { print(\"{0}\", (n,)); }; };",
        "uninitialized",
    );
    rejects(
        "def f() -> int = { var n: int; defer { n := 42; }; n };",
        "uninitialized",
    );
    rejects(
        "def f(b: bool) -> int = { var n: int; if (b) { defer { n := 42; }; () } else { () }; n };",
        "uninitialized",
    );
    rejects(
        "struct E {}; def f(r: Result<(), E>) -> Result<int, E> = { var n: int; defer { n := 42; }; r?; ok(n) };",
        "uninitialized",
    );
}

#[test]
fn deferred_inference_and_local_structs_are_reused_at_each_exit() {
    let m = module(
        "struct E {}; def f(r: Result<(), E>) -> Result<(), _> = { defer { struct Local { n: int }; type Alias = Local; var x: _; x := Alias { n = 42 }; print(\"{0}\", (x.n,)); }; r?; r?; ok(()) };",
    );
    assert_eq!(m.types.len(), 2);
}
