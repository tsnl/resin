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
        assert_eq!(f.result, f.locals[0].ty);
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
        "TypeMismatch",
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
        "struct One { value: int }; struct Two { value: int }; def value() -> _ = { var v: _; v := One { value = 1 }; v := Two { value = 2 }; v };",
        "TypeMismatch",
    );
    assert_eq!(
        result(
            "struct One { value: int }; def value() -> _ = { var v = One { value = 1 }; _(v) };",
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
    let m = module("struct Meters { value: int }; def make() -> _ = { Meters { value = 42 } };");
    assert!(matches!(m.functions[0].result, Ty::Defined { .. }));
    let m = module(
        "def main() -> _ = { struct Meters { value: int }; var distance = Meters { value = 42 }; distance.value };",
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
        "struct Foo { value: _ };",
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
    rejects("def main() = { var n: _; n := 1; n };", "TypeMismatch");
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

#[test]
fn span_construction_and_indexing_infer_element_and_pointer_types() {
    assert_eq!(
        result(
            "def get(p: Ptr<int>) -> _ = { var s = Span<_> { data = p, length = ulong(1) }; s(0) };",
            "get"
        ),
        Ty::Pointer {
            pointee: Box::new(Ty::Int32)
        }
    );
    assert_eq!(
        result("def get() -> _ = { var xs = [1, 2]; xs(1).* };", "get"),
        Ty::Int32
    );
    assert_eq!(
        result(
            "@compute_shader def kernel(i: uint, output: Ptr<uint>) -> _ = { output.* := { i }; }; def artifact() -> _ = { kernel.spirv };",
            "artifact"
        ),
        Ty::shader()
    );
}

#[test]
fn numeric_suffixes_select_exact_types_with_or_without_inference() {
    for (literal, ty) in [
        ("-128b", Ty::Int8),
        ("255B", Ty::UInt8),
        ("-32768h", Ty::Int16),
        ("65535H", Ty::UInt16),
        ("-2147483648i", Ty::Int32),
        ("4294967295I", Ty::UInt32),
        ("-9223372036854775808l", Ty::Int64),
        ("18446744073709551615L", Ty::UInt64),
        ("1.25f", Ty::Float32),
        ("1e2d", Ty::Float64),
        ("42f", Ty::Float32),
        ("42d", Ty::Float64),
        ("0xFFFF_FFFFL", Ty::UInt64),
        ("0x7fffl", Ty::Int64),
        ("0xff", Ty::Int32),
        ("0xAB", Ty::Int32),
        ("0xdead", Ty::Int32),
    ] {
        assert_eq!(
            result(&format!("def value() -> _ = {{ {literal} }};"), "value"),
            ty
        );
        let m = module(&format!("def main() = {{ var n = {literal}; }};"));
        assert!(
            m.functions[0]
                .locals
                .iter()
                .any(|local| local.name.as_deref() == Some("n") && local.ty == ty),
            "{literal}"
        );
    }
    for source in [
        "def value() -> long = { 42L };",
        "def value() -> _ = { var n: long; n := 42L; n };",
        "def value() -> _ = { 42L + 1l };",
        "def value() -> float64 = { 1.5f };",
    ] {
        assert!(ir::generate(&parse(source)).is_err(), "{source}");
    }
}

#[test]
fn suffixed_literals_reject_overflow_and_invalid_integer_forms() {
    for literal in [
        "128b",
        "256B",
        "32768h",
        "65536H",
        "2147483648i",
        "4294967296I",
        "9223372036854775808l",
        "18446744073709551616L",
        "-1L",
        "-129b",
        "1.5L",
        "1e3I",
        "1e50f",
        "1e400d",
    ] {
        for annotation in ["", " -> _"] {
            let source = if annotation.is_empty() {
                format!("def main() = {{ var n = {literal}; }};")
            } else {
                format!("def main(){annotation} = {{ {literal} }};")
            };
            rejects(&source, "literal");
        }
    }
}

#[test]
fn one_armed_if_infers_unit_and_requires_a_unit_body() {
    assert_eq!(result("def f() -> _ = { if (1 == 1) {} };", "f"), Ty::Unit);
    rejects("def f() -> _ = { if (1 == 1) { 42 } };", "TypeMismatch");
}

#[test]
fn checking_does_not_depend_on_inference_trigger_syntax() {
    for marker in ["", "defer ();", "var unused: _; unused := 1;"] {
        let source = format!(
            "def consume(p: Ptr<ubyte>) = {{}}; def main() = {{ {marker} var value = 0; consume(&value); }};"
        );
        let module = ir::generate(&parse(&source)).unwrap();
        let value = module
            .functions
            .iter()
            .flat_map(|f| &f.locals)
            .find(|l| l.name.as_deref() == Some("value"))
            .unwrap();
        assert_eq!(value.ty, Ty::UInt8, "{source}");

        let source = format!(
            "def consume(p: Ptr<ubyte>) = {{}}; def main() = {{ {marker} var value = 0i; consume(&value); }};"
        );
        let error = ir::generate(&parse(&source)).unwrap_err();
        assert!(
            matches!(
                error.kind,
                ir::GenerateErrorKind::Type(ir::TypeErrorKind::TypeMismatch { .. })
            ),
            "{source}: {error}"
        );

        let source = format!(
            "def main() -> int = {{ {marker} var values = [10, 20]; var data = Span<int> {{ data = Ptr<int>(&values), length = 2L }}; data(1).* }};"
        );
        ir::generate(&parse(&source)).unwrap();
    }
}

#[test]
fn pointer_reinterpretation_does_not_narrow_source_storage() {
    for bindings in [
        "var n = 300; var bytes = Ptr<ubyte>(&n);",
        "var n: _; n := 300; var bytes = Ptr<ubyte>(&n);",
        "var n: int; n := 300; var source: Ptr<int>; source := &n; var bytes = Ptr<ubyte>(source);",
    ] {
        for marker in ["", "defer ();"] {
            let source = format!("export {{ main }}; def main() = {{ {bindings} {marker} }};");
            let m = module(&source);
            let n = m
                .functions
                .iter()
                .flat_map(|f| &f.locals)
                .find(|l| l.name.as_deref() == Some("n"))
                .unwrap();
            assert_eq!(n.ty, Ty::Int32, "{source}");
            let bytes = m
                .functions
                .iter()
                .flat_map(|f| &f.locals)
                .find(|l| l.name.as_deref() == Some("bytes"))
                .unwrap();
            assert_eq!(
                bytes.ty,
                Ty::Pointer {
                    pointee: Box::new(Ty::UInt8)
                },
                "{source}"
            );
        }
    }
}
