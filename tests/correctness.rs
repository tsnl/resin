#[allow(dead_code)]
mod support;

use resin_ast::SourceFile;
use resin_hir::GenerateErrorKind;
use resin_lir::Instr;
use resin_types::prelude::*;
use support::pipeline;

fn parse(src: &str) -> SourceFile {
    let parsed = support::frontend::ast(&support::frontend::cst(src, None));
    assert!(parsed.errors.is_empty(), "{src}\n{:?}", parsed.errors);
    parsed.file
}

fn compile(src: &str) -> resin_lir::Module {
    pipeline::generate(&parse(src)).unwrap_or_else(|err| panic!("{src}\n{err}"))
}

#[test]
fn only_parenthesized_lists_apply_functions() {
    for call in [
        "f [1, 2]",
        "f { value = 1 }",
        "f { 1 }",
        "f {}",
        "f 1",
        "x.method [1]",
    ] {
        let source = format!("fn main()  {{ {call}; }}");
        let document = support::frontend::cst(source.clone(), None);
        assert!(
            !support::frontend::ast(&document).errors.is_empty(),
            "{source}"
        );
        assert!(resin_cst::format_source(&source).is_none(), "{source}");
    }
    for call in [
        "f()",
        "f(())",
        "f(1,)",
        "f((1,))",
        "f((1, 2))",
        "f([1, 2])",
        "f({ 1 })",
        "f(1)(2)",
    ] {
        parse(&format!("fn main()  {{ {call}; }}"));
    }
}

#[test]
fn zero_and_multiple_argument_calls_have_distinct_parameter_counts() {
    let module = compile(
        "export { main }; fn f () -> i32  { 1 } fn add (a: i32, b: i32) -> i32  { a + b } fn main() -> ()  { let mut pair = (1, 2); let mut x = f(); let mut y = add(pair.0, pair.1); let mut z = add(3, 4); }",
    );
    let f = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    assert_eq!(f.parameter_count, 0);
    let add = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("add"))
        .unwrap();
    assert_eq!(add.parameter_count, 2);
    assert_eq!(
        add.locals[..2]
            .iter()
            .map(|local| local.ty.clone())
            .collect::<Vec<_>>(),
        [Ty::Int32, Ty::Int32]
    );
    let instructions = &module.functions[module.entries["main"].index()].blocks[0].instrs;
    assert_eq!(
        instructions
            .iter()
            .filter(|i| matches!(i, Instr::Call { .. }))
            .count(),
        3
    );
    assert!(
        instructions
            .iter()
            .any(|i| matches!(i, Instr::Push { value: Value::Unit }))
    );
}

#[test]
fn function_types_accept_unit_tuples_and_higher_order_calls() {
    compile(
        "struct Span<T> { data: Ptr<T>, length: u64, } type P = Ptr<()>; type S = Span<(i32, i32)>; fn f (p: P) -> P  { p }",
    );
    compile(
        "export { main }; type F = () -> i32; fn one () -> i32  { 1 } fn main() -> ()  { let mut f = F(one); let mut x = f(); }",
    );
    compile(
        "export { main }; type Add = (i32, i32) -> i32; fn add (a: i32, b: i32) -> i32  { a + b } fn main() -> ()  { let mut f = Add(add); let mut p = (1, 2); let mut x = f(p.0, p.1); }",
    );
    compile(
        "export { main }; fn apply (f: (i32, i32) -> i32, p: (i32, i32)) -> i32  { f(p.0, p.1) } fn add (a: i32, b: i32) -> i32  { a + b } fn main() -> ()  { let mut x = apply(add, (1, 2)); }",
    );
    compile(
        "export { main }; fn identity (p: (i32, i32)) -> (i32, i32)  { p } fn main() -> ()  { let mut x = identity((1, 2)); }",
    );
    compile(
        "export { main }; type Unit = (); fn f (u: Unit) -> ()  { () } fn main() -> ()  { let mut x = Unit(()); let mut y = f(x); }",
    );
    for (annotation, declaration) in [
        ("() -> i32", "fn f(value: ()) -> i32  { 0 }"),
        ("(i32, i32) -> i32", "fn f(value: (i32, i32)) -> i32  { 0 }"),
    ] {
        let source =
            format!("{declaration} fn main()  {{ let mut value: {annotation}; value = f; }}");
        assert!(pipeline::generate(&parse(&source)).is_err(), "{source}");
    }
}

#[test]
fn type_formers_take_types_between_angle_brackets() {
    let m = compile(
        "export { main }; struct FieldsValueNext<T0, T1> { value: T0, next: T1, }\nstruct Span<T> { data: Ptr<T>, length: u64, } type Pointer = Ptr<Ptr<i32>>; type View = Span<FieldsValueNext<i32, Pointer>>; type Callback = Ptr<(i32) -> i32>; type UnitPointer = Ptr<()>; fn identity(p: Pointer) -> Pointer  { p } fn main() -> ()  { let mut p = Ptr<i32>(u64(0)); }",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Pointer {
            pointee: Box::new(Ty::Pointer {
                pointee: Box::new(Ty::Int32)
            })
        }
    );
    compile(
        "export { main }; fn f(x: i32) -> i32  { x } struct Record { value: i32, } fn main() -> ()  { let mut a = f(1); let mut b = f(2); let mut r = Record { value = 3 }; }",
    );
    for source in [
        "type P = Ptr(i32);",
        "type P = Ptr i32;",
        "type P = Ptr<1>;",
        "type P = Ptr<>;",
        "type P = Ptr<i32, i32>;",
    ] {
        assert!(
            !support::frontend::ast(&support::frontend::cst(source, None))
                .errors
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn typechecking_rejects_incorrect_argument_counts() {
    for src in [
        "export { main }; fn f () -> i32  { 1 } fn main() -> ()  { let mut x = f(1); }",
        "export { main }; fn f (a: i32, b: i32) -> i32  { a } fn main() -> ()  { let mut x = f(1); }",
        "export { main }; fn f (a: i32) -> i32  { a } fn main() -> ()  { let mut x = f(1, 2); }",
        "fn f()  {} fn main()  { f(()); }",
        "fn f(value: ())  {} fn main()  { f(); }",
        "fn f(value: (i32, i32))  {} fn main()  { f(1, 2); }",
        "fn f(a: i32, b: i32)  {} fn main()  { f((1, 2)); }",
    ] {
        assert!(
            matches!(
                pipeline::generate(&parse(src)).unwrap_err().kind,
                GenerateErrorKind::Inference { .. }
            ),
            "{src}"
        );
    }
}

#[test]
fn nominal_conversion_requires_an_explicit_ascription() {
    for src in [
        "export { main }; struct Meters { value: i32, } fn f (m: Meters) -> i32  { m.value } fn main() -> ()  { let mut x = 1; let mut y = f(x); }",
        "export { main }; struct Meters { value: i32, } fn f (m: Meters) -> i32  { m.value } fn main() -> ()  { let mut y = f(1); }",
        "export { main }; struct Meters { value: i32, } fn f (x: i32) -> i32  { x } fn main() -> ()  { let mut m = Meters { value = 1 }; let mut y = f(m); }",
        "export { main }; struct Meters { value: i32, } fn main() -> ()  { let mut m = Meters { value = 1 }; let mut n = 2; m = n; }",
        "export { main }; struct Meters { value: i32, } fn main() -> ()  { let mut m = Meters { value = 1 }; let mut n = 2; n = m; }",
        "export { main }; struct Meters { value: i32, } struct R { value: Meters, } fn main() -> ()  { let mut x = R { value = 1 }; }",
        "export { main }; struct Meters { value: i32, } fn f (m: Meters, x: i32) -> i32  { x } fn main() -> ()  { let mut y = f(1, 2); }",
    ] {
        assert!(
            matches!(
                pipeline::generate(&parse(src)).unwrap_err().kind,
                GenerateErrorKind::Type {
                    kind: TypeErrorKind::TypeMismatch { .. }
                }
            ),
            "{src}"
        );
    }
    compile(
        "export { main }; struct Meters { value: i32, } fn f (m: Meters) -> i32  { m.value } fn main() -> ()  { let mut x = 1; let mut y = f(Meters { value = x }); let mut m = Meters { value = 1 }; m = Meters { value = x }; x = m.value; }",
    );
    compile(
        "export { main }; struct Meters { value: i32, } struct R { value: Meters, } fn main() -> ()  { let mut x = R { value = Meters { value = 1 } }; }",
    );
    compile(
        "export { main }; struct Meters { value: i32, } type Distance = Meters; fn main() -> ()  { let mut x = Distance(Meters { value = 1 }); let mut y = Meters(x); }",
    );
}

#[test]
fn record_layout_does_not_reorder_side_effects() {
    let module = compile(
        "struct R { a: i32, b: i32, } fn f (mut x: i32) -> i32 { let r = R { b = { x = 1; x }, a = { x = 2; x } }; x }",
    );
    let function = module
        .functions
        .iter()
        .find(|f| f.name.as_deref() == Some("f"))
        .unwrap();
    let instrs = &function.blocks[0].instrs;
    let literals: Vec<_> = instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Push {
                value: Value::Int32 { value },
            } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(literals, [1, 2]);
    let fields = instrs
        .iter()
        .find_map(|i| match i {
            Instr::MakeRecord { fields } => {
                Some(fields.iter().map(|f| f.as_ref()).collect::<Vec<_>>())
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(fields, ["a", "b"]);
}

#[test]
fn recursion_uses_immutable_function_references() {
    let module = compile(
        "fn f (n: i32) -> i32  { if (n == 0) { 7 } else { next(n - 1) } } fn next (n: i32) -> i32  { f(n) }",
    );
    for function in &module.functions {
        assert!(
            function
                .blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .any(|i| matches!(i, Instr::Function { .. }))
        );
    }
    let error = pipeline::generate(&parse(
        "export { main }; fn f () -> i32  { 1 } fn main() -> ()  { f = f; }",
    ))
    .unwrap_err();
    assert_eq!(error.kind, GenerateErrorKind::NotAPlace);
}

#[test]
fn lambdas_and_nested_definitions_are_parse_errors() {
    for source in [
        "fn main() -> ()  { let mut f = (n: i32) => n; }",
        "fn outer () -> i32 = { fn inner() -> i32  { 1 } inner() };",
        "fn missing (n)  { n }",
    ] {
        assert!(
            !support::frontend::ast(&support::frontend::cst(source, None))
                .errors
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn record_type_members_are_rejected_instead_of_panicking() {
    for source in [
        "fn main() { let x = { T = i32 }; }",
        "struct Item { a: i32 } fn main() { let x = Item { a = 1, T = i32 }; }",
    ] {
        assert!(
            support::pipeline::source_module(source).is_err(),
            "{source}"
        );
    }
    compile("export { main }; fn main() -> ()  { let mut x = { type T = i32; T(1) }; }");
}

#[test]
fn omitted_function_results_are_unit_not_inferred() {
    for source in [
        "export { main }; fn main()  {}",
        "fn discard()  { 42; } fn apply(f: () -> ())  { f() } fn main()  { apply(discard) }",
        "fn even(n: i32)  { if (n == 0) { () } else { odd(n - 1) } } fn odd(n: i32)  { even(n - 1) }",
    ] {
        let explicit = source.replace(") =", ") -> () =");
        let mut implicit = compile(source);
        let mut explicit = compile(&explicit);
        // Source positions differ when the annotation is spelled out.
        implicit.origins = Default::default();
        explicit.origins = Default::default();
        assert_eq!(implicit, explicit);
    }
    for source in [
        "fn answer()  { 42 }",
        "fn identity(n: i32)  { n }",
        "fn answer()  { if (1 == 1) { 42 } else { 0 } }",
        "struct Unit {} fn nominal()  { Unit {} }",
    ] {
        let error = pipeline::generate(&parse(source)).unwrap_err();
        assert!(
            matches!(
                error.kind,
                GenerateErrorKind::Type {
                    kind: TypeErrorKind::TypeMismatch { .. }
                }
            ),
            "{source}\n{error}"
        );
    }
    compile("fn answer() -> i32  { 42 }");
}

#[test]
fn signed_literals_respect_context_and_the_minimum_integer() {
    let module = compile(
        "export { main }; fn main() -> ()  { let mut x = -2147483648; let mut y: i64 = -1; let mut z = -0x80000000; let mut w = i64(1 + 2); let mut hex = -0xdead; }",
    );
    assert_eq!(
        module.functions[0]
            .locals
            .iter()
            .map(|g| g.ty.clone())
            .collect::<Vec<_>>(),
        [Ty::Int64, Ty::Int64, Ty::Int64, Ty::Int64, Ty::Int64]
    );
    assert!(
        module.functions[0].blocks[0]
            .instrs
            .iter()
            .any(|i| matches!(
                i,
                Instr::Push {
                    value: Value::Int64 { value: -2147483648 }
                }
            ))
    );
}

#[test]
fn returned_pointers_support_field_assignment() {
    compile(
        "struct FieldsX<T0> { x: T0, }\nfn id (p: Ptr<FieldsX<i32>>) -> Ptr<FieldsX<i32>>  { p } fn f (p: Ptr<FieldsX<i32>>) { id(p).x = 1 }",
    );
    compile(
        "struct FieldsX<T0> { x: T0, }\nstruct R { inner: FieldsX<i32>, } fn id (p: Ptr<R>) -> Ptr<R>  { p } fn f (p: Ptr<R>) { id(p).inner.x = 1 }",
    );
    let src = "struct FieldsX<T0> { x: T0, }\nfn id (r: FieldsX<i32>) -> FieldsX<i32>  { r } fn f (r: FieldsX<i32>) { id(r).x = 1 }";
    assert!(matches!(
        pipeline::generate(&parse(src)).unwrap_err().kind,
        GenerateErrorKind::NotAPlace
    ));
}

#[test]
fn nested_field_access_evaluates_its_base_once() {
    for src in [
        "struct FieldsInner<T0> { inner: T0, }\nstruct FieldsX<T0> { x: T0, }\nfn id (r: FieldsInner<FieldsX<i32>>) -> FieldsInner<FieldsX<i32>>  { r } fn f (r: FieldsInner<FieldsX<i32>>) -> i32  { id(r).inner.x }",
        "struct FieldsP<T0> { p: T0, }\nstruct FieldsX<T0> { x: T0, }\nfn id (r: FieldsP<Ptr<FieldsX<i32>>>) -> FieldsP<Ptr<FieldsX<i32>>>  { r } fn f (r: FieldsP<Ptr<FieldsX<i32>>>) { id(r).p.x = 1 }",
    ] {
        let module = compile(src);
        let f = module
            .functions
            .iter()
            .find(|f| f.name.as_deref() == Some("f"))
            .unwrap();
        assert_eq!(
            f.blocks
                .iter()
                .flat_map(|b| &b.instrs)
                .filter(|i| matches!(i, Instr::Call { .. }))
                .count(),
            1,
            "{src}"
        );
    }
}

#[test]
fn uninitialized_reads_are_rejected_on_all_paths() {
    for src in [
        "fn f () -> i32  { let mut x: i32; x }",
        "fn consume(a: i32, b: i32)  {} fn main()  { let mut value: i32; consume(value, { value = 1; value }); }",
        "fn f (c: i32) -> i32  { let mut x: i32; if (c == 0) { x = 1 } else {}; x }",
        "fn f (c: i32) -> i32  { let mut x: i32; (c == 0) && ({ x = 1; x } == 1); x }",
        "export { main }; fn main() -> ()  { let mut x: i32; let mut y = x; }",
        "struct FieldsX<T0> { x: T0, }\nfn f () -> i32  { let mut r: FieldsX<i32>; r.x }",
        "struct FieldsX<T0> { x: T0, }\nfn f () -> i32  { let mut r: FieldsX<i32>; r.x = 1; r.x }",
    ] {
        assert!(
            pipeline::generate(&parse(src))
                .unwrap_err()
                .to_string()
                .to_lowercase()
                .contains("uninitialized"),
            "{src}"
        );
    }
    compile("fn f () -> i32  { let mut x: i32; x = 1; x }");
    compile(
        "fn consume(a: i32, b: i32)  {} fn main()  { let mut value: i32; consume({ value = 1; value }, value); }",
    );
    compile("fn f (c: i32) -> i32  { let mut x: i32; if (c == 0) { x = 1 } else { x = 2 }; x }");
    compile(
        "fn even (n: i32) -> i32  { if (n == 0) { 1 } else { odd(n - 1) } } fn odd (n: i32) -> i32  { if (n == 0) { 0 } else { even(n - 1) } }",
    );
}
