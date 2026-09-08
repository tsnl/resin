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
        Ty::Int64
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
    assert_eq!(fields[1].ty, Ty::Int64);
    assert_eq!(fields[2].ty, Ty::Bool);
}

#[test]
fn later_assignments_resolve_local_holes() {
    assert_eq!(
        result(
            "def answer() -> _ = { var n: _; var p: Ptr<_>; p := &n; n := 42; p.* };",
            "answer"
        ),
        Ty::Int64
    );
    assert_eq!(
        result(
            "def answer() -> _ = { var n = 42; var p = Ptr<_>(&n); p.* };",
            "answer"
        ),
        Ty::Int64
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
    assert_eq!(result(source, "first"), Ty::Int64);
    assert_eq!(result(source, "second"), Ty::Int64);
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
            definition: ir::TypeId::from_index(1)
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
        "def value() -> _ = { 1 }; def use() -> int = { value() };",
        "TypeMismatch",
    );
}

#[test]
fn recursive_groups_infer_from_bodies_not_callers() {
    let source = "def odd(n: int) -> _ = { if (n == 0) { 0 } else { even(n - 1) } }; def even(n: int) -> _ = { odd(n) };";
    assert_eq!(result(source, "odd"), Ty::Int64);
    assert_eq!(result(source, "even"), Ty::Int64);
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
fn nominal_identity_and_local_type_definitions_survive_inference() {
    let m = module("struct Meters { value: int }; def make() -> _ = { Meters { value = 42 } };");
    assert!(matches!(m.functions[0].result, Ty::Defined { .. }));
    let m = module(
        "def main() -> _ = { struct Meters { value: int }; var distance = Meters { value = 42 }; distance.value };",
    );
    assert_eq!(m.types.iter().filter(|d| d.name().is_some()).count(), 2);
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
        "def main() -> _ = { var n: _; print(fmt(\"{0}\", (n,))); n := 42; n };",
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
    assert_eq!(fields[0].ty, Ty::Int64);
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
        Ty::Int64
    );
    assert_eq!(
        result(
            "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) -> _ = { var i = uint(invocation); output.* := { i }; }; def artifact() -> _ = { kernel.spirv };",
            "artifact"
        ),
        Ty::shader()
    );
}

#[test]
fn numeric_suffixes_select_exact_types() {
    for (literal, ty) in [
        ("-128_b", Ty::Int8),
        ("255_ub", Ty::UInt8),
        ("-32768_h", Ty::Int16),
        ("65535_uh", Ty::UInt16),
        ("-2147483648_i", Ty::Int32),
        ("4294967295_ui", Ty::UInt32),
        ("-9223372036854775808_l", Ty::Int64),
        ("18446744073709551615_ul", Ty::UInt64),
        ("1.25_f", Ty::Float32),
        ("1e2_d", Ty::Float64),
        ("42_f", Ty::Float32),
        ("42_d", Ty::Float64),
        ("0xFFFF_FFFF_ul", Ty::UInt64),
        ("0x7fff_l", Ty::Int64),
        ("0xff", Ty::Int64),
        ("0xAB", Ty::Int64),
        ("0xdead", Ty::Int64),
    ] {
        assert_eq!(
            result(&format!("def value() -> _ = {{ {literal} }};"), "value"),
            ty
        );
    }
    for source in [
        "def value() -> long = { 42_ul };",
        "def value() -> _ = { var n: long; n := 42_ul; n };",
        "def value() -> _ = { 42_ul + 1_l };",
        "def value() -> float64 = { 1.5_f };",
    ] {
        assert!(ir::generate(&parse(source)).is_err(), "{source}");
    }
}

#[test]
fn suffixed_literals_reject_overflow_and_invalid_integer_forms() {
    for literal in [
        "128_b",
        "256_ub",
        "32768_h",
        "65536_uh",
        "2147483648_i",
        "4294967296_ui",
        "9223372036854775808_l",
        "18446744073709551616_ul",
        "-1_ul",
        "-129_b",
        "1.5_ul",
        "1e3_ui",
        "1e50_f",
        "1e400_d",
    ] {
        rejects(&format!("def value() -> _ = {{ {literal} }};"), "literal");
    }
}

#[test]
fn else_if_chains_infer_results_and_require_unit_without_a_final_else() {
    assert_eq!(
        result(
            "def f() -> _ = { if (1 == 0) { 1 } else if (1 == 1) { 2 } else { 3 } };",
            "f"
        ),
        Ty::Int64,
    );
    assert_eq!(
        result(
            "def f() -> _ = { if (1 == 0) {} else if (1 == 1) {} };",
            "f"
        ),
        Ty::Unit,
    );
    rejects(
        "def f() -> _ = { if (1 == 0) { 1 } else if (1 == 1) { 2 } };",
        "TypeMismatch",
    );
}

#[test]
fn one_armed_if_infers_unit_and_requires_a_unit_body() {
    assert_eq!(result("def f() -> _ = { if (1 == 1) {} };", "f"), Ty::Unit);
    rejects("def f() -> _ = { if (1 == 1) { 42 } };", "TypeMismatch");
}

#[test]
fn checking_does_not_depend_on_inference_trigger_syntax() {
    for marker in ["", "{ var unused = (); };", "var unused: _; unused := 1;"] {
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
            "def consume(p: Ptr<ubyte>) = {{}}; def main() = {{ {marker} var value = 0_i; consume(&value); }};"
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
            "def main() -> int = {{ {marker} var values = [10, 20]; var data = Span<int> {{ data = Ptr<int>(&values), length = 2_ul }}; data.at(1).* }};"
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
        for marker in ["", "{ var unused = (); };"] {
            let source = format!("export {{ main }}; def main() = {{ {bindings} {marker} }};");
            let m = module(&source);
            let n = m
                .functions
                .iter()
                .flat_map(|f| &f.locals)
                .find(|l| l.name.as_deref() == Some("n"))
                .unwrap();
            assert_eq!(
                n.ty,
                if bindings.contains("var n: int") {
                    Ty::Int32
                } else {
                    Ty::Int64
                },
                "{source}"
            );
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

#[test]
fn shared_layout_queries_reject_unsupported_or_unresolved_types() {
    for operand in ["bool", "()", "[1_ub, 2_ub]"] {
        rejects(
            &format!("def main() = {{ size_of({operand}); }};"),
            "no shared host/device layout",
        );
    }
    rejects(
        "def main() = { size_of(Ptr<_>); };",
        "holes are only allowed",
    );
}

#[test]
fn numeric_conversions_do_not_choose_source_storage_types() {
    let m = module("def main() = { var n = 300; var byte = ubyte(n); };");
    assert_eq!(
        m.functions[0]
            .locals
            .iter()
            .find(|l| l.name.as_deref() == Some("n"))
            .unwrap()
            .ty,
        Ty::Int64
    );
    assert!(
        m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, ir::Instr::NumericCast { ty: Ty::UInt8 }))
    );
}

#[test]
fn never_elimination_requires_an_empty_input_and_resolved_context() {
    assert_eq!(
        result(
            "def impossible(n: Never) -> int = { absurd(n) };",
            "impossible"
        ),
        Ty::Int32
    );
    rejects("def bad(n: int) -> int = { absurd(n) };", "TypeMismatch");
    rejects("def ambiguous(n: Never) -> _ = { absurd(n) };", "infer");
}

#[test]
fn layout_operands_check_nested_declarations_without_emitting_them() {
    let m = module(
        "def effect() -> int = { 42 }; def measure() -> _ = { size_of({ struct Local { n: int }; effect(); var value = Local { n = effect() }; value }) };",
    );
    assert_eq!(
        m.types
            .iter()
            .filter(|definition| definition
                .name()
                .is_some_and(|name| name.as_ref() == "Local"))
            .count(),
        1
    );
    let measure = m
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("measure"))
        .unwrap();
    assert_eq!(measure.result, Ty::UInt64);
    assert!(
        measure
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .all(|instr| !matches!(instr, ir::Instr::Call | ir::Instr::MakeRecord { .. }))
    );

    rejects(
        "def measure() -> _ = { size_of({ var value: int; value := 1 == 1; value }) };",
        "TypeMismatch",
    );
}

#[test]
fn layout_operands_do_not_read_or_initialize_runtime_locals() {
    assert_eq!(
        result(
            "def measure() -> _ = { var n: int; size_of(n) };",
            "measure"
        ),
        Ty::UInt64
    );
    rejects(
        "def f() -> int = { var n: int; size_of(n := 42); n };",
        "UninitializedValue",
    );
}

#[test]
fn nested_record_annotations_reject_duplicate_field_names() {
    for source in [
        "def f(x: { n: int, n: int }) = {};",
        "def f() = { var x: Ptr<{ n: int, n: int }>; };",
        "def f() = { struct Local { n: int, n: int }; };",
        "def f() = { type Local = { n: int, n: int }; };",
        "def f() = { { var x: { n: int, n: int }; }; };",
    ] {
        rejects(source, "DuplicateField");
    }
}

#[test]
fn numeric_defaults_stay_with_their_dependency_group() {
    let plain = "def plain() -> _ = { 1 };";
    let wide = "def wide() -> _ = { var n = 1; var p: Ptr<ulong>; p := &n; n };";
    let fraction = "def fraction() -> _ = { var n = 1.5; var p: Ptr<float32>; p := &n; n };";
    let caller = "def caller() -> _ = { wide() };";
    for declarations in [
        [plain, wide, fraction, caller],
        [caller, fraction, plain, wide],
        [fraction, caller, wide, plain],
    ] {
        let source = declarations.join(" ");
        let m = module(&source);
        for (name, expected) in [
            ("plain", Ty::Int64),
            ("wide", Ty::UInt64),
            ("fraction", Ty::Float32),
            ("caller", Ty::UInt64),
        ] {
            assert_eq!(
                m.functions
                    .iter()
                    .find(|f| f.name.as_deref() == Some(name))
                    .unwrap()
                    .result,
                expected,
                "{source}"
            );
        }
    }
}

#[test]
fn numeric_suffixes_ignore_case_and_accept_optional_separators() {
    for (digits, suffix, ty) in [
        ("127", "b", Ty::Int8),
        ("255", "ub", Ty::UInt8),
        ("32767", "h", Ty::Int16),
        ("65535", "uh", Ty::UInt16),
        ("2147483647", "i", Ty::Int32),
        ("4294967295", "ui", Ty::UInt32),
        ("9223372036854775807", "l", Ty::Int64),
        ("18446744073709551615", "ul", Ty::UInt64),
        ("1.25", "f", Ty::Float32),
        ("1e2", "d", Ty::Float64),
        ("0xFF", "ub", Ty::UInt8),
        ("0x7FFF", "h", Ty::Int16),
        ("0xFFFF", "uh", Ty::UInt16),
        ("0x7FFF_FFFF", "i", Ty::Int32),
        ("0xFFFF_FFFF", "ui", Ty::UInt32),
        ("0x7FFF_FFFF_FFFF_FFFF", "l", Ty::Int64),
        ("0xFFFF_FFFF_FFFF_FFFF", "ul", Ty::UInt64),
    ] {
        for mask in 0..(1 << suffix.len()) {
            let suffix: String = suffix
                .chars()
                .enumerate()
                .map(|(i, c)| {
                    if mask & (1 << i) != 0 {
                        c.to_ascii_uppercase()
                    } else {
                        c
                    }
                })
                .collect();
            for separator in ["", "_"] {
                let literal = format!("{digits}{separator}{suffix}");
                assert_eq!(
                    result(&format!("def value() -> _ = {{ {literal} }};"), "value"),
                    ty,
                    "{literal}"
                );
            }
        }
    }
    for (literal, ty) in [
        ("0x7f_b", Ty::Int8),
        ("0X7F_B", Ty::Int8),
        ("-0x80_b", Ty::Int8),
        ("0xAB", Ty::Int64),
        ("0x1_d", Ty::Int64),
        ("0x1_F", Ty::Int64),
        ("-0xdead", Ty::Int64),
        ("-0XFE", Ty::Int64),
    ] {
        assert_eq!(
            result(&format!("def value() -> _ = {{ {literal} }};"), "value"),
            ty,
            "{literal}"
        );
    }
}

#[test]
fn unsuffixed_numbers_infer_from_uses_and_fall_back_to_64_bits() {
    for (expression, ty) in [
        ("42", Ty::Int64),
        ("2147483648", Ty::Int64),
        ("9223372036854775807", Ty::Int64),
        ("-9223372036854775808", Ty::Int64),
        ("1.25", Ty::Float64),
        ("1e3", Ty::Float64),
        ("1 + 2", Ty::Int64),
        ("1 + 2.5", Ty::Float64),
        ("1.5 + 2", Ty::Float64),
    ] {
        for body in [expression.to_owned(), format!("var n = {expression}; n")] {
            assert_eq!(
                result(&format!("def value() -> _ = {{ {body} }};"), "value"),
                ty,
                "{body}"
            );
        }
    }
    for (name, ty) in [
        ("sbyte", Ty::Int8),
        ("ubyte", Ty::UInt8),
        ("short", Ty::Int16),
        ("ushort", Ty::UInt16),
        ("int", Ty::Int32),
        ("uint", Ty::UInt32),
        ("long", Ty::Int64),
        ("ulong", Ty::UInt64),
        ("float32", Ty::Float32),
        ("float64", Ty::Float64),
    ] {
        let mut literals = vec!["42"];
        if ty == Ty::Float32 || ty == Ty::Float64 {
            literals.extend(["1.5", "1e2"]);
        }
        for literal in literals {
            for body in [
                format!("var n = {literal}; var p: Ptr<{name}>; p := &n; n"),
                format!("var n = {literal}; take(n)"),
                format!("var n = {literal}; n + take(1)"),
                format!("var n = {literal}; take(1) + n"),
            ] {
                let source = format!(
                    "def take(n: {name}) -> {name} = {{ n }}; def value() -> _ = {{ {body} }};"
                );
                assert_eq!(result(&source, "value"), ty, "{source}");
            }
        }
    }
    for literal in ["9223372036854775808", "-9223372036854775809", "1e400"] {
        rejects(&format!("def value() -> _ = {{ {literal} }};"), "literal");
    }
    rejects("def value() -> ubyte = { 256 };", "literal");
    rejects("def value() -> float32 = { 1e50 };", "literal");
    rejects("def value() -> int = { 1.5 };", "TypeMismatch");
}
