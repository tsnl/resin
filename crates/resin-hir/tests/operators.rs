use resin_hir::{Term, TermKind, Type};

mod common;
use common::hir_module;

fn tail(function: &resin_hir::Function) -> &Term {
    let mut term = function.body.as_ref().unwrap();
    loop {
        term = match &term.kind {
            TermKind::Block { tail, .. } => tail,
            TermKind::Use { arg } | TermKind::Convert { arg } => arg,
            TermKind::Read { place } | TermKind::Move { place } => place,
            _ => return term,
        };
    }
}

#[test]
fn unary_and_binary_operators_resolve_to_ordinary_free_calls() {
    let module = hir_module(
        "struct Number { value: i32 }\n         fn __neg__(value: Number) -> Number { Number { value = -value.value } }\n         fn __sub__(left: Number, right: i32) -> i32 { left.value - right }\n         fn use(value: Number) -> i32 { -value - 2 }",
    )
    .unwrap();
    let function = module
        .functions
        .iter()
        .find(|f| f.name.as_ref() == "use")
        .unwrap();
    let TermKind::Call { func, args } = &tail(function).kind else {
        panic!("ordinary call")
    };
    let TermKind::Function { function, .. } = func.kind else {
        panic!("resolved operation")
    };
    assert_eq!(module.functions[function.index()].name.as_ref(), "__sub__");
    assert!(matches!(args[0].kind, TermKind::Call { .. }));
    assert_eq!(
        module.functions[function.index()].signature.result.ty,
        Type::Int32
    );
}

#[test]
fn generic_operators_retain_all_operand_types_until_specialization() {
    let module = hir_module("fn add<T, U>(left: T, right: U) -> _ { left + right }").unwrap();
    let function = &module.functions[0];
    assert!(matches!(
        function.signature.result.ty,
        Type::FunctionResult { .. } | Type::Value { .. }
    ));
    let TermKind::OperationCall { lookup, args } = &tail(function).kind else {
        panic!("dependent operation")
    };
    assert_eq!(lookup.primitive.as_deref(), Some("+"));
    assert_eq!(
        lookup.arguments,
        function
            .signature
            .params
            .iter()
            .map(|p| p.annotation.ty.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(args.len(), 2);
}

#[test]
fn operators_support_borrowed_receivers_additional_binders_and_right_operand_dispatch() {
    hir_module(
        "struct Number { value: i32 }\n         fn __add__<T>(left: Ref<Number>, right: T) -> T { right }\n         fn __add__(left: i32, right: Ref<Number>) -> i32 { left + right.value }\n         fn sum(value: Number) -> i32 { value + 7 + (35 + value) }",
    )
    .unwrap();
}

#[test]
fn invalid_operator_calls_and_unused_body_errors_are_rejected() {
    for source in [
        "struct A {} fn __add__(a: A, b: A, c: A) -> A { a } fn use(a: A, b: A) -> A { a + b }",
        "struct A {} fn add(a: A, b: A) -> A { a + b }",
        "struct A {} fn __add__(a: A, b: i32) -> A { a } fn add(a: A) -> A { 2 + a }",
        "struct A {} fn __add__(a: A, b: i32) -> A { a } fn add(a: A) -> A { &a + 1 }",
        "struct A {} fn __eq__(a: A, b: A) -> bool { true } fn compare(a: A) -> bool { a != a }",
        "struct A {} fn __add__<T>(a: A, b: T) -> A { missing }",
        "struct A {} fn __add__(a: A, b: i32) -> A { a } fn __add__(a: A, b: i32) -> A { a } fn use(a: A) -> A { a + 1 }",
    ] {
        assert!(hir_module(source).is_err(), "{source}");
    }
}
