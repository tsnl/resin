use resin::{
    ast::{self, StmtKind, TypeKind},
    ir::{self, Ty},
};

mod support;
use support::{module, parse};

fn result(source: &str, name: &str) -> Ty {
    module(source)
        .functions
        .into_iter()
        .find(|f| f.name.as_deref() == Some(name))
        .unwrap()
        .result
}

fn rejects(source: &str, message: &str) {
    let error = ir::generate(&parse(source)).unwrap_err().to_string();
    assert!(error.contains(message), "{source}\n{error}");
}

#[test]
fn explicit_holes_are_not_editor_recovery_holes() {
    let file = parse("def answer() -> _ = { 42 };");
    let StmtKind::Function { result, .. } = &file.stmts[0].val else {
        panic!()
    };
    assert!(matches!(result.val, TypeKind::Infer));
    assert!(ast::print::format_source(&file).contains("infer-type"));
    assert_eq!(
        self::result("def answer() -> _ = { 42 };", "answer"),
        Ty::Int32
    );
}

#[test]
fn holes_compose_inside_pointers_spans_records_and_functions() {
    let m = module(
        "\
        def pointer(p: Ptr<Ptr<int>>) -> Ptr<Ptr<_>> = { p };\
        def span(p: Span<Ptr<int>>) -> Span<Ptr<_>> = { p };\
        def plus(n: int) -> int = { n + 1 };\
        def function() -> (int) -> _ = { plus };\
        def record(p: Ptr<int>) -> { pointer: Ptr<_>, number: _, flag: _ } = {\
            { flag = 1 == 1, number = 7, pointer = p }\
        };",
    );
    for f in &m.functions[..2] {
        assert_eq!(f.result, f.locals[f.param.index()].ty);
    }
    assert_eq!(
        m.functions[3].result,
        Ty::Function {
            param: Box::new(Ty::Int32),
            result: Box::new(Ty::Int32)
        }
    );
    let Ty::Record { fields } = &m.functions[4].result else {
        panic!()
    };
    assert_eq!(fields[1].ty, Ty::Int32);
    assert_eq!(fields[2].ty, Ty::Bool);
}

#[test]
fn later_assignments_resolve_local_holes() {
    assert_eq!(
        result(
            "def answer() -> _ = { var n: _; var p: Ptr<_>; p := &n; n := 42; p.* };",
            "answer"
        ),
        Ty::Int32
    );
    assert_eq!(
        result(
            "def answer() -> _ = { var n = 42; var p = Ptr<_>(&n); p.* };",
            "answer"
        ),
        Ty::Int32
    );
}

#[test]
fn locally_inferred_function_types_are_monomorphic() {
    assert_eq!(
        result(
            "def next(n: int) -> int = { n + 1 }; def answer() -> _ = { var f: (_) -> _; f := next; f(41) };",
            "answer"
        ),
        Ty::Int32
    );
    rejects(
        "def next(n: int) -> int = { n + 1 }; def answer() -> _ = { var f: (_) -> _; f := next; f(1 == 1) };",
        "incompatible",
    );
}

#[test]
fn shadowed_function_names_do_not_create_inference_dependencies() {
    let source = "def first() -> _ = { var second = 42; second }; def second() -> _ = { first() };";
    assert_eq!(result(source, "first"), Ty::Int32);
    assert_eq!(result(source, "second"), Ty::Int32);
}

#[test]
fn casts_do_not_choose_an_unrelated_nominal_type_for_a_hole() {
    rejects(
        "type One = int; type Two = int; def value() -> _ = { var v: _; v := One(1); v := Two(2); v };",
        "incompatible",
    );
    assert_eq!(
        result(
            "type One = int; def value() -> _ = { var v = One(1); _(v) };",
            "value"
        ),
        Ty::Defined {
            definition: ir::TypeId::from_index(0)
        }
    );
}

#[test]
fn numeric_choices_are_delayed_until_context_is_known() {
    assert_eq!(
        result(
            "def wide() -> _ = { var n = 1; var p: Ptr<ulong>; p := &n; n };",
            "wide"
        ),
        Ty::UInt64
    );
    assert_eq!(
        result(
            "def wide() -> _ = { var n: _; n := ulong(4294967297); n };",
            "wide"
        ),
        Ty::UInt64
    );
    assert_eq!(
        result("def small() -> _ = { float32(1.5) };", "small"),
        Ty::Float32
    );
    assert_eq!(
        result(
            "def value() -> _ = { var n: long; n := -9223372036854775808; n };",
            "value"
        ),
        Ty::Int64
    );
    rejects(
        "def value() -> _ = { 1 }; def use() -> long = { value() };",
        "TypeMismatch",
    );
}

#[test]
fn recursive_groups_infer_from_bodies_not_callers() {
    let source = "def odd(n: int) -> _ = { if (n == 0) { 0 } else { even(n - 1) } }; def even(n: int) -> _ = { odd(n) };";
    assert_eq!(result(source, "odd"), Ty::Int32);
    assert_eq!(result(source, "even"), Ty::Int32);
    rejects(
        "def looped() -> _ = { looped() }; def main() -> int = { looped() };",
        "cannot infer",
    );
    rejects(
        "def a() -> _ = { b() }; def b() -> _ = { a() };",
        "cannot infer",
    );
}

#[test]
fn dependency_order_does_not_depend_on_source_order() {
    for source in [
        "def first() -> _ = { second() }; def second() -> _ = { long(7) };",
        "def second() -> _ = { long(7) }; def first() -> _ = { second() };",
    ] {
        assert_eq!(result(source, "first"), Ty::Int64);
    }
}

#[test]
fn nominal_identity_and_local_type_definitions_survive_inference() {
    let m = module("type Meters = int; def make() -> _ = { Meters(42) };");
    assert!(matches!(m.functions[0].result, Ty::Defined { .. }));
    let m = module(
        "def main() -> _ = { type Meters = int; var distance = Meters(42); int(distance) };",
    );
    assert_eq!(m.types.len(), 1);
    assert_eq!(m.functions[0].result, Ty::Int32);
}

#[test]
fn ambiguous_infinite_and_forbidden_holes_are_diagnostics() {
    for source in [
        "def main() = { var p: Ptr<_>; };",
        "def main() = { var p = Ptr<_>(ulong(0)); };",
    ] {
        rejects(source, "cannot infer");
    }
    rejects("def main() = { var p: _; p := &p; };", "infinite type");
    for source in [
        "def f(x: _) = {};",
        "def f(x: Ptr<_>) = {};",
        "type Foo = Ptr<_>;",
        "type Foo = { value: _ };",
        "extern \"api.h\" def f() -> _;",
        "def main() -> _ = { type Foo = _; () };",
    ] {
        rejects(
            source,
            "only allowed in local annotations and function results",
        );
    }
}

#[test]
fn inference_preserves_unit_defaults_and_initialization_checks() {
    rejects("def main() = { var n: _; n := 1; n };", "incompatible");
    rejects(
        "def main() -> _ = { var n: _; print(\"{0}\", (n,)); n := 42; n };",
        "UninitializedValue",
    );
}

#[test]
fn distinct_nodes_with_identical_spans_do_not_share_inference_variables() {
    let mut file = parse("def answer() -> (_, _) = { (1, 1 == 1) };");
    let StmtKind::Function { result, .. } = &mut file.stmts[0].val else {
        panic!()
    };
    let TypeKind::Record { fields } = &mut result.val else {
        panic!()
    };
    fields[1].1.span = fields[0].1.span;
    let m = ir::generate(&file).unwrap();
    let Ty::Record { fields } = &m.functions[0].result else {
        panic!()
    };
    assert_eq!(fields[0].ty, Ty::Int32);
    assert_eq!(fields[1].ty, Ty::Bool);
}
