use resin_types::prelude::*;
use support::pipeline::{self, nominal};

mod support;
use support::{module, parse};

fn rejects(source: &str, message: &str) {
    let error = pipeline::generate(&parse(source)).unwrap_err().to_string();
    assert!(error.contains(message), "{source}\n{error}");
}

#[test]
fn structs_mint_identities_and_aliases_do_not() {
    let m = module(
        "struct Point { x: i32, } type Position = Point; type Number = i32; fn copy(p: Position) -> Point  { p } fn number(n: Number) -> i32  { n }",
    );
    assert_eq!(m.types.iter().filter(|d| d.name().is_some()).count(), 1);
    assert_eq!(
        m.functions[0].result,
        Ty::Defined {
            definition: nominal(&m, "Point")
        }
    );
    assert_eq!(m.functions[1].result, Ty::Int32);
    rejects("struct A {} struct B {} fn f(a: A) -> B  { a }", "expected");
}

#[test]
fn unions_are_canonical_and_tags_belong_to_structs() {
    let m = module(
        "struct A {} struct B { n: i32, } type First = A | B | A; type Second = B | A; fn f(v: First) -> Second  { v } fn a() -> First  { A {} }",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Union {
            variants: vec![
                Ty::Defined {
                    definition: nominal(&m, "A")
                },
                Ty::Defined {
                    definition: nominal(&m, "B")
                }
            ]
        }
    );
    module("type Choice = i32 | bool;");
}

#[test]
fn errors_accumulate_across_propagation() {
    let m = module(
        "struct A {} struct B {} fn a() -> (i32 | Err<A>)  { Err(A {}) } fn b() -> (i32 | Err<B>)  { (2) } fn both() -> (i32 | Err<_>)  { let mut x = a()?; let mut y = b()?; (x + y) }",
    );
    assert_eq!(
        m.functions[2].result,
        Ty::union_of([
            Ty::Int32,
            Ty::Error {
                payload: Box::new(Ty::union([nominal(&m, "A"), nominal(&m, "B")]))
            }
        ])
    );
}

#[test]
fn nested_error_unions_flatten_and_holes_remain_monomorphic() {
    let m = module(
        "struct A {} struct B {} fn nested() -> ((i32 | Err<_>) | Err<_>)  { ((42)) } fn outer() -> ((Ptr<i32> | Err<A>) | Err<B>)  { Err(B {}) }",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::union_of([
            Ty::union_of([
                Ty::Int32,
                Ty::Error {
                    payload: Box::new(Ty::union([]))
                }
            ]),
            Ty::Error {
                payload: Box::new(Ty::union([]))
            }
        ])
    );
}

#[test]
fn result_and_union_matches_are_exhaustive() {
    module(
        "struct A {} struct B { n: i32, } fn f(v: A | B) -> i32  { match (v) { A(a) => { 0 }, B(b) => { b.n } } } fn g(v: (i32 | Err<A>)) -> i32  { match (v) { i32(n) => { n }, Err(e) => { 0 } } }",
    );
    rejects(
        "struct A {} struct B {} fn f(v: A | B) -> i32  { match (v) { A(a) => { 0 } } }",
        "every variant",
    );
}

#[test]
fn local_result_annotations_collect_errors_from_later_assignments() {
    module(
        "struct E {} fn f() -> (i32 | Err<_>)  { let mut r: (i32 | Err<_>); r = (1); r = Err(E {}); r }",
    );
}

#[test]
fn recursive_error_sets_reach_a_fixed_point() {
    let m = module(
        "struct A {} struct B {} fn a(n: i32) -> (i32 | Err<_>)  { if (n == 0) { Err(A {}) } else { (b(n - 1)?) } } fn b(n: i32) -> (i32 | Err<_>)  { if (n == 0) { Err(B {}) } else { (a(n - 1)?) } }",
    );
    let expected = Ty::union_of([
        Ty::Int32,
        Ty::Error {
            payload: Box::new(Ty::union([nominal(&m, "A"), nominal(&m, "B")])),
        },
    ]);
    for function in m.functions {
        assert_eq!(function.result, expected);
    }
}

#[test]
fn mutable_pointers_do_not_widen_and_bad_matches_are_rejected() {
    rejects(
        "struct A {} struct B {} fn f(p: Ptr<(i32 | Err<A>)>) -> Ptr<(i32 | Err<A | B>)>  { p }",
        "TypeMismatch",
    );
    rejects(
        "struct A {} struct B {} fn f(x: A | B) -> i32  { match (x) { A(a) => { 0 }, A(b) => { 1 } } }",
        "duplicate",
    );
    rejects(
        "struct A {} fn f(x: (i32 | Err<A>)) -> i32  { match (x) { A(a) => { 0 } } }",
        "unknown or duplicate",
    );
    rejects(
        "struct A {} struct B {} fn f(x: (i32 | Err<A>)) -> (i32 | Err<B>)  { x }",
        "TypeMismatch",
    );
    rejects("struct A {} fn f(x: (i32 | Err<_>))  {}", "only allowed");
    rejects("type Recursive = Ptr<Recursive>;", "recursive type alias");
}

#[test]
fn union_matches_intersect_initialization_and_scope_payloads() {
    module(
        "struct A {} struct B {} fn f(x: A | B) -> i32  { let mut n: i32; match (x) { A(a) => { n = 1; }, B(b) => { n = 2; } }; n }",
    );
    rejects(
        "struct A {} struct B {} fn f(x: A | B) -> i32  { let mut n: i32; match (x) { A(a) => { n = 1; }, B(b) => { () } }; n }",
        "UninitializedValue",
    );
    rejects(
        "struct A {} fn f(x: A)  { match (x) { A(a) => { () } }; a; }",
        "UnboundValue",
    );
}

#[test]
fn verifier_rejects_invalid_sum_instructions_and_types() {
    let source = "struct E {} fn f() -> (i32 | Err<E>)  { Err(E {}) }";
    let mut m = module(source);
    m.functions[0].blocks[0].instrs.insert(
        0,
        resin_lir::Instr::Push {
            value: Value::Bool { value: true },
        },
    );
    m.functions[0].blocks[0].instrs.insert(
        1,
        resin_lir::Instr::MakeVariant {
            ty: Ty::union_of([Ty::Int32, Ty::Str]),
            tag: Case::Type(Ty::Bool),
        },
    );
    assert!(matches!(
        resin_lir::verify(&m).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidVariant
    ));

    let mut m = module(source);
    m.functions[0].result = Ty::union_of([
        Ty::Int32,
        Ty::Error {
            payload: Box::new(Ty::Bool),
        },
    ]);
    assert!(matches!(
        resin_lir::verify(&m).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidReturnStack { .. }
    ));

    let mut m = module(source);
    m.functions[0].result = Ty::Union {
        variants: vec![
            Ty::Defined {
                definition: nominal(&m, "E"),
            },
            Ty::Defined {
                definition: nominal(&m, "E"),
            },
        ],
    };
    assert!(matches!(
        resin_lir::verify(&m).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidVariant
    ));
}

#[test]
fn contextual_tuple_arguments_can_widen_result_errors() {
    for result in ["i32", "_"] {
        module(&format!(
            "struct A {{}} struct B {{}} fn narrow() -> (i32 | Err<A>)  {{ (42) }} fn take(r: (i32 | Err<A | B>), n: i32) -> i32  {{ n }} fn main() -> {result}  {{ take(narrow(), 1) }}"
        ));
    }
}

#[test]
fn old_result_names_are_ordinary_identifiers() {
    module(
        "type Result<T, E> = T | Err<E>; fn ok(value: i32) -> i32  { value } fn err(value: i32) -> i32  { value } fn f() -> Result<i32, str>  { ok(err(42)) }",
    );
    module(
        "struct Fields { ok: i32, err: i32, } fn f() -> i32  { let mut fields = Fields { ok = 1, err = 2 }; fields.ok + fields.err }",
    );
}

#[test]
fn errors_discovered_through_recursive_payloads_join_before_sets_close() {
    let m = module(
        "struct B {} struct E { nested: (i32 | Err<B>), } fn f(n: i32) -> (i32 | Err<_>)  { match (g(n)) { i32(value) => { (value) }, Err(error) => { (error.nested?) } } } fn g(n: i32) -> (i32 | Err<_>)  { if (n > 0) { f(n - 1); () } else { () }; Err(E { nested = Err(B {}) }) }",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::union_of([
            Ty::Int32,
            Ty::Error {
                payload: Box::new(Ty::Defined {
                    definition: nominal(&m, "B")
                })
            }
        ])
    );
    assert_eq!(
        m.functions[1].result,
        Ty::union_of([
            Ty::Int32,
            Ty::Error {
                payload: Box::new(Ty::Defined {
                    definition: nominal(&m, "E")
                })
            }
        ])
    );
}

#[test]
fn propagation_rejects_wrong_types_and_narrower_errors() {
    rejects("fn f() -> i32  { 1? }", "Err");
    rejects(
        "struct E {} fn f(r: (i32 | Err<E>)) -> i32  { r? }",
        "containing Err",
    );
    rejects(
        "struct A {} struct B {} fn f(r: (i32 | Err<B>)) -> (i32 | Err<A>)  { (r?) }",
        "every propagated error",
    );
    module("fn f() -> (i32 | Err<bool>)  { (1) }");
}

#[test]
fn ir_never_elimination_cannot_consume_an_inhabited_value() {
    let mut m = support::module("fn impossible(n: Never) -> i32  { absurd(n) }");
    assert!(
        m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, resin_lir::Instr::Eliminate { .. }))
    );
    m.functions[0].locals[0].ty = Ty::Int32;
    assert!(resin_lir::verify(&m).is_err());
}

#[test]
fn wildcard_matches_ignore_payloads_and_cover_remaining_variants() {
    let source = r#"export { main };
        struct Failure {}
        fn choose(value: i32 | bool | None) -> i32  {
            match (value) { i32(number) => { number }, _ => { 7 } }
        }
        fn ignore<T>(value: T) -> i32  { match (value) { _ => { 3 } } }
        fn main()  {
            assert(choose(i32(42)) == 42);
            assert(choose(false) == 7);
            assert(choose(None) == 7);
            let mut result: (i32 | Err<Failure>) = (42);
            match (result) { i32(_) => {}, Err(_) => { assert(false); } };
            assert(ignore::<i32 | None>(None) == 3);
        }"#;
    let output = support::project::Project::new(&module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    rejects(
        "fn f(v: i32 | None)  { match (v) { _ => {}, None => {} } }",
        "final arm",
    );
    rejects(
        "fn f(v: i32 | None)  { match (v) { i32(_) => {}, None => {}, _ => {} } }",
        "remaining variant",
    );
}

#[test]
fn error_sets_collect_builtin_and_owned_values() {
    let source = r#"export { main }; import { "$/string.resin" };
        fn text() -> (i32 | Err<str>)  { Err("failed") }
        fn number() -> (i32 | Err<i32>)  { Err(i32(42)) }
        fn owned() -> (i32 | Err<String>)  { Err(string_from_str("owned")) }
        fn choose(which: i32) -> (i32 | Err<_>)  {
            if (which == 0) { (text()?) } else if (which == 1) { (number()?) } else { (owned()?) }
        }
        fn main()  {
            match (choose(0)) { i32(_) => { assert(false); }, Err(error) => {
                match (error) { str(text) => { assert(text.length == u64(6)); }, _ => { assert(false); } }
            } };
            match (choose(1)) { i32(_) => { assert(false); }, Err(error) => {
                match (error) { i32(value) => { assert(value == 42); }, _ => { assert(false); } }
            } };
            match (choose(2)) { i32(_) => { assert(false); }, Err(error) => {
                match (error) { String(text) => { assert(text:get().length == u64(5)); }, _ => { assert(false); } }
            } };
        }"#;
    let output = support::project::Project::new(&module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
