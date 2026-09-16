use resin_hir::{MethodName, TermKind, Type};

mod common;
use common::hir_module;

#[test]
fn unary_and_binary_slots_resolve_to_ordinary_calls() {
    let module = hir_module(
        r#"
        struct Number { value: int,
            def __neg__(self: Number) -> Number = { Number { value = -self.value } };
            def __sub__(self: Number, other: int) -> int = { self.value - other };
        };
        def use(value: Number) -> int = { -value - 2 };
    "#,
    )
    .unwrap();
    let owner = module
        .types
        .iter()
        .find(|ty| ty.name.as_ref() == "Number")
        .unwrap();
    for arity in [1, 2] {
        let operator = MethodName::Operator {
            symbol: "-".into(),
            arity,
        };
        let name = if arity == 1 { "__neg__" } else { "__sub__" };
        assert_eq!(owner.methods[&operator], owner.methods[&name.into()]);
    }
    let function = module
        .functions
        .iter()
        .find(|f| f.name.as_ref() == "use")
        .unwrap();
    let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!()
    };
    assert!(matches!(tail.kind, TermKind::Call { .. }));
    assert_eq!(tail.ty, Type::Int32);
}

#[test]
fn generic_operators_retain_signature_queries_until_specialization() {
    let module = hir_module("def add<T, U>(left: T, right: U) -> _ = { left + right };").unwrap();
    let function = &module.functions[0];
    assert!(
        matches!(
            function.signature.result.ty,
            Type::FunctionResult { .. } | Type::Value { .. }
        ),
        "{:?}",
        function.signature.result.ty
    );
    let TermKind::Block { tail, .. } = &function.body.as_ref().unwrap().kind else {
        panic!()
    };
    let mut tail = tail.as_ref();
    while let TermKind::Use { arg } = &tail.kind {
        tail = arg;
    }
    let TermKind::DependentMethodCall {
        lookup,
        receiver,
        args,
    } = &tail.kind
    else {
        panic!("{tail:?}")
    };
    assert_eq!(
        lookup.name,
        MethodName::Operator {
            symbol: "+".into(),
            arity: 2
        }
    );
    assert!(lookup.associated && receiver.is_none());
    assert_eq!(args.len(), 2);
}

#[test]
fn unused_operator_declarations_are_validated() {
    for (source, expected) in [
        (
            "struct Bad { def __add__(self: Bad, x: Bad, y: Bad) -> Bad = { self }; };",
            "invalid number of operands",
        ),
        (
            "struct Bad { def __invert__(self: Bad, x: Bad) -> Bad = { self }; };",
            "invalid number of operands",
        ),
        (
            "struct Bad { def __add__<T>(self: Bad, x: T) -> Bad = { self }; };",
            "cannot declare additional",
        ),
        (
            "struct Bad { def __add__(self: Ptr<Bad>, x: Bad) -> Bad = { x }; };",
            "owning struct by value",
        ),
        (
            "struct Bad { def __neg__(self: int) -> int = { self }; };",
            "owning struct by value",
        ),
        (
            "struct Bad { def __eq__(self: Bad, other: Bad) -> int = { 1 }; };",
            "must return bool",
        ),
        (
            "struct Bad { def __not__(self: Bad) -> int = { 1 }; };",
            "must return bool",
        ),
    ] {
        let error = hir_module(source).unwrap_err().to_string();
        assert!(error.contains(expected), "{source}\n{error}");
    }
}

#[test]
fn duplicate_operators_and_missing_left_operand_overloads_are_rejected() {
    for source in [
        "struct A { def __add__(a: A, b: A) -> A = { a }; def __add__(a: A, b: int) -> A = { a }; };",
        "struct A {}; def add(a: A, b: A) -> A = { a + b };",
        "struct A { def __add__(a: A, b: int) -> A = { a }; }; def add(a: A) -> A = { 2 + a };",
        "struct A { def __add__(a: A, b: int) -> A = { a }; }; def add(a: A) -> A = { &a + 1 };",
        "struct A { def __eq__(a: A, b: A) -> bool = { 1 == 1 }; }; def compare(a: A) -> bool = { a != a };",
    ] {
        assert!(hir_module(source).is_err(), "{source}");
    }
}
