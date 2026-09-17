use resin_ast::{StmtKind, TypeKind};
use resin_hir::GenerateErrorKind;
use resin_types::prelude::*;
use support::pipeline;

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
    let error = pipeline::generate(&parse(source)).unwrap_err().to_string();
    assert!(error.contains(message), "{source}\n{error}");
}

#[test]
fn explicit_holes_are_not_editor_recovery_holes() {
    let file = parse("fn answer() -> _  { 42 }");
    let StmtKind::Function { result, .. } = &file.stmts[0].val else {
        panic!()
    };
    assert!(matches!(result.val, TypeKind::Infer));
    assert!(resin_ast::format_source(&file).contains("infer-type"));
    assert_eq!(
        self::result("fn answer() -> _  { 42 }", "answer"),
        Ty::Int64
    );
}

#[test]
fn holes_compose_inside_pointers_spans_records_and_functions() {
    let m = module(
        r#"struct FieldsPointerNumberFlag<T0, T1, T2> { pointer: T0, number: T1, flag: T2, }
        struct Span<T> { data: Ptr<T>, length: ulong, }
        fn pointer(p: Ptr<Ptr<int>>) -> Ptr<Ptr<_>>  { p }
        fn span(p: Span<Ptr<int>>) -> Span<Ptr<_>>  { p }
        fn plus(n: int) -> int  { n + 1 }
        fn function() -> (int) -> _  { plus }
        fn record(p: Ptr<int>) -> FieldsPointerNumberFlag<Ptr<_>, _, _>  {
            FieldsPointerNumberFlag<_, _, _> { flag = true, number = 7, pointer = p }
        }
        "#,
    );
    for f in &m.functions[..2] {
        assert_eq!(f.result, f.locals[0].ty);
    }
    assert_eq!(
        m.functions[3].result,
        Ty::Function {
            params: vec![Ty::Int32],
            result: Box::new(Ty::Int32)
        }
    );
    let Ty::Defined { definition } = &m.functions[4].result else {
        panic!()
    };
    let Ty::Record { fields } = m.types[definition.index()].body().unwrap() else {
        panic!()
    };
    assert_eq!(fields[1].ty, Ty::Int64);
    assert_eq!(fields[2].ty, Ty::Bool);
}

#[test]
fn later_assignments_resolve_local_holes() {
    assert_eq!(
        result(
            "fn answer() -> _  { let mut n: _; let mut copied: _; n = 42; copied = n; copied }",
            "answer"
        ),
        Ty::Int64
    );
    assert_eq!(
        result(
            "fn answer() -> _  { let mut n = 42; let reference: Ref<_> = n; reference }",
            "answer"
        ),
        Ty::Int64
    );
}

#[test]
fn locally_inferred_function_types_are_monomorphic() {
    assert_eq!(
        result(
            "fn next(n: int) -> int  { n + 1 } fn answer() -> _  { let mut f: (_) -> _; f = next; f(41) }",
            "answer"
        ),
        Ty::Int32
    );
    rejects(
        "fn next(n: int) -> int  { n + 1 } fn answer() -> _  { let mut f: (_) -> _; f = next; f(1 == 1) }",
        "TypeMismatch",
    );
}

#[test]
fn shadowed_function_names_do_not_create_inference_dependencies() {
    let source = "fn first() -> _  { let mut second = 42; second } fn second() -> _  { first() }";
    assert_eq!(result(source, "first"), Ty::Int64);
    assert_eq!(result(source, "second"), Ty::Int64);
}

#[test]
fn casts_do_not_choose_an_unrelated_nominal_type_for_a_hole() {
    rejects(
        "struct One { value: int, } struct Two { value: int, } fn value() -> _  { let mut v: _; v = One { value = 1 }; v = Two { value = 2 }; v }",
        "TypeMismatch",
    );
    let m = module(
        "struct One { value: int, } fn value() -> _  { let mut v = One { value = 1 }; _(v) }",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Defined {
            definition: pipeline::nominal(&m, "One")
        }
    );
}

#[test]
fn numeric_choices_are_delayed_until_context_is_known() {
    assert_eq!(
        result(
            "fn wide() -> _  { let mut n = 1; let p: Ref<ulong> = n; n }",
            "wide"
        ),
        Ty::UInt64
    );
    assert_eq!(
        result(
            "fn wide() -> _  { let mut n: _; n = ulong(4294967297); n }",
            "wide"
        ),
        Ty::UInt64
    );
    assert_eq!(
        result("fn small() -> _  { float32(1.5) }", "small"),
        Ty::Float32
    );
    assert_eq!(
        result(
            "fn value() -> _  { let mut n: long; n = -9223372036854775808; n }",
            "value"
        ),
        Ty::Int64
    );
    rejects(
        "fn value() -> _  { 1 } fn use() -> int  { value() }",
        "TypeMismatch",
    );
}

#[test]
fn recursive_groups_infer_from_bodies_not_callers() {
    let source = "fn odd(n: int) -> _  { if (n == 0) { 0 } else { even(n - 1) } } fn even(n: int) -> _  { odd(n) }";
    assert_eq!(result(source, "odd"), Ty::Int64);
    assert_eq!(result(source, "even"), Ty::Int64);
    rejects(
        "fn looped() -> _  { looped() } fn main() -> int  { looped() }",
        "cannot infer",
    );
    rejects("fn a() -> _  { b() } fn b() -> _  { a() }", "cannot infer");
}

#[test]
fn nominal_identity_and_local_type_definitions_survive_inference() {
    let m = module("struct Meters { value: int, } fn make() -> _  { Meters { value = 42 } }");
    assert!(matches!(m.functions[0].result, Ty::Defined { .. }));
    let m = module(
        "fn main() -> _  { struct Meters { value: int, } let mut distance = Meters { value = 42 }; distance.value }",
    );
    assert_eq!(m.types.iter().filter(|d| d.name().is_some()).count(), 1);
    assert_eq!(m.functions[0].result, Ty::Int32);
}

#[test]
fn ambiguous_infinite_and_forbidden_holes_are_diagnostics() {
    for source in [
        "fn main()  { let mut p: Ptr<_>; }",
        "fn main()  { let mut p = Ptr<_>(ulong(0)); }",
    ] {
        rejects(source, "cannot infer");
    }
    rejects("fn main()  { let mut p: _; p = &p; }", "infinite type");
    for source in [
        "fn f(x: _)  {}",
        "fn f(x: Ptr<_>)  {}",
        "type Foo = Ptr<_>;",
        "struct Foo { value: _, }",
        "extern { \"api.h\": { fn f() -> _; } };",
        "fn main() -> _  { type Foo = _; () }",
    ] {
        rejects(
            source,
            "only allowed in local annotations and function results",
        );
    }
}

#[test]
fn inference_preserves_unit_defaults_and_initialization_checks() {
    rejects("fn main()  { let mut n: _; n = 1; n }", "TypeMismatch");
    rejects(
        "fn consume(n: long)  {} fn main() -> _  { let mut n: _; consume(n); n = 42; n }",
        "UninitializedValue",
    );
}

#[test]
fn distinct_nodes_with_identical_spans_do_not_share_inference_variables() {
    let mut file = parse("fn answer() -> (_, _)  { (1, 1 == 1) }");
    let StmtKind::Function { result, .. } = &mut file.stmts[0].val else {
        panic!()
    };
    let TypeKind::Record { fields } = &mut result.val else {
        panic!()
    };
    fields[1].1.span = fields[0].1.span;
    let m = pipeline::generate(&file).unwrap();
    let Ty::Record { fields } = &m.functions[0].result else {
        panic!()
    };
    assert_eq!(fields[0].ty, Ty::Int64);
    assert_eq!(fields[1].ty, Ty::Bool);
}

#[test]
fn span_construction_and_indexing_infer_element_value_types() {
    assert_eq!(
        result(
            "import { \"$/span.resin\" }; fn get(p: Ptr<int>) -> _  { let mut s = Span<_> { data = p, length = ulong(1) }; s:at(0) }",
            "get"
        ),
        Ty::Int32
    );
    assert_eq!(
        result("fn get() -> _  { let mut xs = [1, 2]; xs(1) }", "get"),
        Ty::Int64
    );
    assert_eq!(
        result(
            "@compute_shader fn kernel(invocation: ulong, output: Ptr<uint>) -> _  { let mut i = uint(invocation); output.* = { i }; } fn reference() -> _  { kernel }",
            "reference"
        ),
        Ty::Function {
            params: vec![
                Ty::UInt64,
                Ty::Pointer {
                    pointee: Box::new(Ty::UInt32)
                }
            ],
            result: Box::new(Ty::Unit)
        }
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
            result(&format!("fn value() -> _  {{ {literal} }}"), "value"),
            ty
        );
    }
    for source in [
        "fn value() -> long  { 42_ul }",
        "fn value() -> _  { let mut n: long; n = 42_ul; n }",
        "fn value() -> _  { 42_ul + 1_l }",
        "fn value() -> float64  { 1.5_f }",
    ] {
        assert!(pipeline::generate(&parse(source)).is_err(), "{source}");
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
        rejects(&format!("fn value() -> _  {{ {literal} }}"), "literal");
    }
}

#[test]
fn else_if_chains_infer_results_and_require_unit_without_a_final_else() {
    assert_eq!(
        result(
            "fn f() -> _  { if (1 == 0) { 1 } else if (1 == 1) { 2 } else { 3 } }",
            "f"
        ),
        Ty::Int64,
    );
    assert_eq!(
        result("fn f() -> _  { if (1 == 0) {} else if (1 == 1) {} }", "f"),
        Ty::Unit,
    );
    rejects(
        "fn f() -> _  { if (1 == 0) { 1 } else if (1 == 1) { 2 } }",
        "TypeMismatch",
    );
}

#[test]
fn one_armed_if_infers_unit_and_requires_a_unit_body() {
    assert_eq!(result("fn f() -> _  { if (1 == 1) {} }", "f"), Ty::Unit);
    rejects("fn f() -> _  { if (1 == 1) { 42 } }", "TypeMismatch");
}

#[test]
fn checking_does_not_depend_on_inference_trigger_syntax() {
    for marker in [
        "",
        "{ let mut unused = (); };",
        "let mut unused: _; unused = 1;",
    ] {
        let source = format!(
            "fn consume(p: Ref<ubyte>)  {{}} fn main()  {{ {marker} let mut value = 0; consume(value); }}"
        );
        let module = pipeline::generate(&parse(&source)).unwrap();
        let value = module
            .functions
            .iter()
            .flat_map(|f| &f.locals)
            .find(|l| l.name.as_deref() == Some("value"))
            .unwrap();
        assert_eq!(value.ty, Ty::UInt8, "{source}");

        let source = format!(
            "fn consume(p: Ref<ubyte>)  {{}} fn main()  {{ {marker} let mut value = 0_i; consume(value); }}"
        );
        let error = pipeline::generate(&parse(&source)).unwrap_err();
        assert!(
            matches!(
                error.kind,
                GenerateErrorKind::Type {
                    kind: TypeErrorKind::TypeMismatch { .. }
                }
            ),
            "{source}: {error}"
        );

        let source = format!(
            "import {{ \"$/span.resin\" }}; fn main() -> int  {{ {marker} let mut values = [10, 20]; let data: Ref<_> = values; data:at(1) }}"
        );
        pipeline::source_module(&source).unwrap();
    }
}

#[test]
fn pointer_reinterpretation_does_not_narrow_source_storage() {
    for bindings in [
        "let mut n = Ptr<long>(300_ul); let mut bytes = Ptr<ubyte>(n);",
        "let mut n: _; n = Ptr<long>(300_ul); let mut bytes = Ptr<ubyte>(n);",
        "let mut n: Ptr<int>; n = Ptr<int>(300_ul); let mut source: Ptr<int>; source = n; let mut bytes = Ptr<ubyte>(source);",
    ] {
        for marker in ["", "{ let mut unused = (); };"] {
            let source = format!("export {{ main }}; fn main()  {{ {bindings} {marker} }}");
            let m = module(&source);
            let n = m
                .functions
                .iter()
                .flat_map(|f| &f.locals)
                .find(|l| l.name.as_deref() == Some("n"))
                .unwrap();
            assert_eq!(
                n.ty,
                Ty::Pointer {
                    pointee: Box::new(if bindings.contains("let mut n: Ptr<int>") {
                        Ty::Int32
                    } else {
                        Ty::Int64
                    })
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
    for operand in ["bool", "()"] {
        rejects(
            &format!("fn main()  {{ size_of({operand}); }}"),
            "no shared host/device layout",
        );
    }
    rejects("fn main()  { size_of(Ptr<_>); }", "holes are only allowed");
}

#[test]
fn numeric_conversions_do_not_choose_source_storage_types() {
    let m = module("fn main()  { let mut n = 300; let mut byte = ubyte(n); }");
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
            .any(|i| matches!(i, resin_lir::Instr::NumericCast { ty: Ty::UInt8 }))
    );
}

#[test]
fn never_elimination_requires_an_empty_input_and_resolved_context() {
    assert_eq!(
        result(
            "fn impossible(n: Never) -> int  { absurd(n) }",
            "impossible"
        ),
        Ty::Int32
    );
    rejects("fn bad(n: int) -> int  { absurd(n) }", "TypeMismatch");
    rejects("fn ambiguous(n: Never) -> _  { absurd(n) }", "infer");
}

#[test]
fn layout_operands_check_nested_declarations_without_emitting_them() {
    let m = module(
        "fn effect() -> int  { 42 } fn measure() -> _  { size_of({ struct Local { n: int, } effect(); let mut value = Local { n = effect() }; value }) }",
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
            .all(|instr| !matches!(
                instr,
                resin_lir::Instr::Call { .. } | resin_lir::Instr::MakeRecord { .. }
            ))
    );

    rejects(
        "fn measure() -> _  { size_of({ let mut value: int; value = 1 == 1; value }) }",
        "TypeMismatch",
    );
}

#[test]
fn layout_operands_do_not_read_or_initialize_runtime_locals() {
    assert_eq!(
        result(
            "fn measure() -> _  { let mut n: int; size_of(n) }",
            "measure"
        ),
        Ty::UInt64
    );
    rejects(
        "fn f() -> int  { let mut n: int; size_of({ n = 42; n }); n }",
        "UninitializedValue",
    );
}

#[test]
fn nested_record_annotations_reject_duplicate_field_names() {
    for source in [
        "struct FieldsNN<T0, T1> { n: T0, n: T1, }\nfn f(x: FieldsNN<int, int>)  {}",
        "struct FieldsNN<T0, T1> { n: T0, n: T1, }\nfn f()  { let mut x: Ptr<FieldsNN<int, int>>; }",
        "fn f()  { struct Local { n: int, n: int, } }",
        "struct FieldsNN<T0, T1> { n: T0, n: T1, }\nfn f()  { type Local = FieldsNN<int, int>; }",
        "struct FieldsNN<T0, T1> { n: T0, n: T1, }\nfn f()  { { let mut x: FieldsNN<int, int>; }; }",
    ] {
        rejects(source, "DuplicateField");
    }
}

#[test]
fn numeric_defaults_stay_with_their_dependency_group() {
    let plain = "fn plain() -> _  { 1 }";
    let wide = "fn wide() -> _  { let mut n = 1; let p: Ref<ulong> = n; n }";
    let fraction = "fn fraction() -> _  { let mut n = 1.5; let p: Ref<float32> = n; n }";
    let caller = "fn caller() -> _  { wide() }";
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
                    result(&format!("fn value() -> _  {{ {literal} }}"), "value"),
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
            result(&format!("fn value() -> _  {{ {literal} }}"), "value"),
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
        for body in [
            expression.to_owned(),
            format!("let mut n = {expression}; n"),
        ] {
            assert_eq!(
                result(&format!("fn value() -> _  {{ {body} }}"), "value"),
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
                format!("let mut n = {literal}; let p: Ref<{name}> = n; n"),
                format!("let mut n = {literal}; take(n)"),
                format!("let mut n = {literal}; n + take(1)"),
                format!("let mut n = {literal}; take(1) + n"),
            ] {
                let source =
                    format!("fn take(n: {name}) -> {name} {{ n }} fn value() -> _  {{ {body} }}");
                assert_eq!(result(&source, "value"), ty, "{source}");
            }
        }
    }
    for literal in ["9223372036854775808", "-9223372036854775809", "1e400"] {
        rejects(&format!("fn value() -> _  {{ {literal} }}"), "literal");
    }
    rejects("fn value() -> ubyte  { 256 }", "literal");
    rejects("fn value() -> float32  { 1e50 }", "literal");
    rejects("fn value() -> int  { 1.5 }", "TypeMismatch");
}
