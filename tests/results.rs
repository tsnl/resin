use resin_lir::{self as lir, Ty, TypeId};
use support::pipeline;

mod support;
use support::{module, parse};

fn rejects(source: &str, message: &str) {
    let error = pipeline::generate(&parse(source)).unwrap_err().to_string();
    assert!(error.contains(message), "{source}\n{error}");
}

#[test]
fn structs_mint_identities_and_aliases_do_not() {
    let m = module(
        "struct Point { x: int }; type Position = Point; type Number = int; def copy(p: Position) -> Point = { p }; def number(n: Number) -> int = { n };",
    );
    assert_eq!(m.types.iter().filter(|d| d.name().is_some()).count(), 2);
    assert_eq!(
        m.functions[0].result,
        Ty::Defined {
            definition: TypeId::from_index(1)
        }
    );
    assert_eq!(m.functions[1].result, Ty::Int32);
    rejects(
        "struct A {}; struct B {}; def f(a: A) -> B = { a };",
        "expected",
    );
}

#[test]
fn unions_are_canonical_and_tags_belong_to_structs() {
    let m = module(
        "struct A {}; struct B { n: int }; type First = A | B | A; type Second = B | A; def f(v: First) -> Second = { v }; def a() -> First = { A {} };",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Union {
            variants: vec![
                Ty::Defined {
                    definition: TypeId::from_index(1)
                },
                Ty::Defined {
                    definition: TypeId::from_index(2)
                }
            ]
        }
    );
    assert_eq!(TypeId::from_index(0).tag(), 0);
    module("type Choice = int | bool;");
}

#[test]
fn errors_accumulate_across_propagation() {
    let m = module(
        "struct A {}; struct B {}; def a() -> Result<int, A> = { err(A {}) }; def b() -> Result<int, B> = { ok(2) }; def both() -> Result<int, _> = { var x = a()?; var y = b()?; ok(x + y) };",
    );
    assert_eq!(
        m.functions[2].result,
        Ty::Result {
            value: Box::new(Ty::Int32),
            error: Box::new(Ty::union([TypeId::from_index(1), TypeId::from_index(2)]))
        }
    );
}

#[test]
fn result_holes_nest_and_remain_monomorphic() {
    let m = module(
        "struct A {}; struct B {}; def nested() -> Result<Result<int, _>, _> = { ok(ok(42)) }; def outer() -> Result<Result<Ptr<int>, A>, B> = { err(B {}) };",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Result {
            value: Box::new(Ty::Result {
                value: Box::new(Ty::Int32),
                error: Box::new(Ty::union([]))
            }),
            error: Box::new(Ty::union([]))
        }
    );
}

#[test]
fn result_and_union_matches_are_exhaustive() {
    module(
        "struct A {}; struct B { n: int }; def f(v: A | B) -> int = { match (v) { A(a) => { 0 }, B(b) => { b.n } } }; def g(v: Result<int, A>) -> int = { match (v) { ok(n) => { n }, err(e) => { 0 } } };",
    );
    rejects(
        "struct A {}; struct B {}; def f(v: A | B) -> int = { match (v) { A(a) => { 0 } } };",
        "every variant",
    );
}

#[test]
fn local_result_annotations_collect_errors_from_later_assignments() {
    module(
        "struct E {}; def f() -> Result<int, _> = { var r: Result<int, _>; r := ok(1); r := err(E {}); r };",
    );
}

#[test]
fn recursive_error_sets_reach_a_fixed_point() {
    let m = module(
        "struct A {}; struct B {}; def a(n: int) -> Result<int, _> = { if (n == 0) { err(A {}) } else { ok(b(n - 1)?) } }; def b(n: int) -> Result<int, _> = { if (n == 0) { err(B {}) } else { ok(a(n - 1)?) } };",
    );
    let expected = Ty::Result {
        value: Box::new(Ty::Int32),
        error: Box::new(Ty::union([TypeId::from_index(1), TypeId::from_index(2)])),
    };
    for function in m.functions {
        assert_eq!(function.result, expected);
    }
}

#[test]
fn mutable_pointers_do_not_widen_and_bad_matches_are_rejected() {
    rejects(
        "struct A {}; struct B {}; def f(p: Ptr<Result<int, A>>) -> Ptr<Result<int, A | B>> = { p };",
        "TypeMismatch",
    );
    rejects(
        "struct A {}; struct B {}; def f(x: A | B) -> int = { match (x) { A(a) => { 0 }, A(b) => { 1 } } };",
        "duplicate",
    );
    rejects(
        "struct A {}; def f(x: Result<int, A>) -> int = { match (x) { A(a) => { 0 } } };",
        "pattern",
    );
    rejects(
        "struct A {}; struct B {}; def f(x: Result<int, A>) -> Result<int, B> = { x };",
        "every propagated error",
    );
    rejects(
        "struct A {}; def f(x: Result<int, _>) = {};",
        "only allowed",
    );
    rejects("type Recursive = Ptr<Recursive>;", "UnboundType");
}

#[test]
fn union_matches_intersect_initialization_and_scope_payloads() {
    module(
        "struct A {}; struct B {}; def f(x: A | B) -> int = { var n: int; match (x) { A(a) => { n := 1; }, B(b) => { n := 2; } }; n };",
    );
    rejects(
        "struct A {}; struct B {}; def f(x: A | B) -> int = { var n: int; match (x) { A(a) => { n := 1; }, B(b) => { () } }; n };",
        "UninitializedValue",
    );
    rejects(
        "struct A {}; def f(x: A) = { match (x) { A(a) => { () } }; a; };",
        "UnboundValue",
    );
}

#[test]
fn verifier_rejects_invalid_sum_instructions_and_types() {
    let source = "struct E {}; def f() -> Result<int, E> = { err(E {}) };";
    let mut m = module(source);
    let instruction = m.functions[0].blocks[0]
        .instrs
        .iter_mut()
        .find(|i| matches!(i, lir::Instr::MakeVariant { .. }))
        .unwrap();
    let lir::Instr::MakeVariant { tag, .. } = instruction else {
        unreachable!()
    };
    *tag = lir::Case::Type(Ty::Bool);
    assert!(matches!(
        resin_lir_verifier::verify(&m).unwrap_err().kind,
        resin_lir_verifier::VerifyErrorKind::InvalidVariant
    ));

    let mut m = module(source);
    m.functions[0].result = Ty::Result {
        value: Box::new(Ty::Int32),
        error: Box::new(Ty::Bool),
    };
    assert!(matches!(
        resin_lir_verifier::verify(&m).unwrap_err().kind,
        resin_lir_verifier::VerifyErrorKind::InvalidVariant
    ));

    let mut m = module(source);
    m.functions[0].result = Ty::Union {
        variants: vec![
            Ty::Defined {
                definition: TypeId::from_index(1),
            },
            Ty::Defined {
                definition: TypeId::from_index(1),
            },
        ],
    };
    assert!(matches!(
        resin_lir_verifier::verify(&m).unwrap_err().kind,
        resin_lir_verifier::VerifyErrorKind::InvalidVariant
    ));
}

#[test]
fn contextual_tuple_arguments_can_widen_result_errors() {
    for result in ["int", "_"] {
        module(&format!(
            "struct A {{}}; struct B {{}}; def narrow() -> Result<int, A> = {{ ok(42) }}; def take(r: Result<int, A | B>, n: int) -> int = {{ n }}; def main() -> {result} = {{ take(narrow(), 1) }};"
        ));
    }
}

#[test]
fn constructors_are_unshadowable_but_field_names_are_ordinary() {
    for name in ["ok", "err"] {
        rejects(&format!("def {name}() = {{}};"), "ReservedBuiltin");
        rejects(&format!("def f({name}: int) = {{}};"), "ReservedBuiltin");
        rejects(
            &format!("def f() = {{ var {name} = 1; }};"),
            "ReservedBuiltin",
        );
    }
    module(
        "struct Fields { ok: int, err: int }; def f() -> int = { var fields = Fields { ok = 1, err = 2 }; fields.ok + fields.err };",
    );
}

#[test]
fn errors_discovered_through_recursive_payloads_join_before_sets_close() {
    let m = module(
        "struct B {}; struct E { nested: Result<int, B> }; def f(n: int) -> Result<int, _> = { match (g(n)) { ok(value) => { ok(value) }, err(error) => { ok(error.nested?) } } }; def g(n: int) -> Result<int, _> = { if (n > 0) { f(n - 1); () } else { () }; err(E { nested = err(B {}) }) };",
    );
    assert_eq!(
        m.functions[0].result,
        Ty::Result {
            value: Box::new(Ty::Int32),
            error: Box::new(Ty::Defined {
                definition: TypeId::from_index(1)
            })
        }
    );
    assert_eq!(
        m.functions[1].result,
        Ty::Result {
            value: Box::new(Ty::Int32),
            error: Box::new(Ty::Defined {
                definition: TypeId::from_index(2)
            })
        }
    );
}

#[test]
fn propagation_rejects_wrong_types_and_narrower_errors() {
    rejects("def f() -> int = { 1? };", "Result");
    rejects(
        "struct E {}; def f(r: Result<int, E>) -> int = { r? };",
        "Result return",
    );
    rejects(
        "struct A {}; struct B {}; def f(r: Result<int, B>) -> Result<int, A> = { ok(r?) };",
        "every propagated error",
    );
    rejects("def f() -> Result<int, bool> = { ok(1) };", "structs");
}

#[test]
fn ir_never_elimination_cannot_consume_an_inhabited_value() {
    let mut m = support::module("def impossible(n: Never) -> int = { absurd(n) };");
    assert!(
        m.functions[0]
            .blocks
            .iter()
            .flat_map(|b| &b.instrs)
            .any(|i| matches!(i, resin_lir::Instr::Eliminate { .. }))
    );
    m.functions[0].locals[0].ty = resin_lir::Ty::Int32;
    assert!(resin_lir_verifier::verify(&m).is_err());
}
