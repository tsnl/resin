use resin_hir::{Term, TermKind, Type};

mod common;
use common::hir_module;

fn tail(function: &resin_hir::Function) -> &Term {
    value(function.body.as_ref().unwrap())
}

fn value(mut term: &Term) -> &Term {
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
fn dependent_operations_retain_visible_candidates_and_determining_result() {
    let module = hir_module(
        "struct A {} struct B {} fn read(value: Ref<A>) -> i32 { 1 } fn read(value: Ref<B>) -> bool { true } fn relay<T>(value: T) -> _ { value:read() }",
    )
    .unwrap();
    let relay = &module.functions[2];
    let Type::Value { of } = &relay.signature.result.ty else {
        panic!("read dependent result as a value")
    };
    let Type::FunctionResult { function } = of.as_ref() else {
        panic!("operation result relation")
    };
    let Type::Operation { lookup } = function.as_ref() else {
        panic!("operation type relation")
    };
    assert_eq!(lookup.name.as_ref(), "read");
    assert_eq!(
        lookup.arguments,
        [Type::Parameter {
            parameter: relay.signature.type_params[0].id
        }]
    );
    assert_eq!(lookup.candidates.len(), 2);
    assert_eq!(
        module.functions[lookup.candidates[0].index()].name.as_ref(),
        "read"
    );
    assert!(lookup.type_args.is_none());
    assert!(matches!(tail(relay).kind, TermKind::OperationCall { .. }));
}

#[test]
fn dependent_fields_and_operation_results_compose() {
    let module = hir_module(
        "struct A { item: i32 } struct B { item: bool }\n         fn read(value: Ref<A>) -> B { B { item = true } }\n         fn read(value: Ref<B>) -> A { A { item = 1 } }\n         fn next(value: Ref<A>) -> i32 { value.item }\n         fn next(value: Ref<B>) -> bool { value.item }\n         fn field_operation<T>(value: T) -> _ { value.item:read() }\n         fn operation_field<T>(value: T) -> _ { value:read().item }\n         fn operation_operation<T>(value: T) -> _ { value:read():next() }",
    )
    .unwrap();
    assert!(matches!(
        module.functions[4].signature.result.ty,
        Type::Value { .. }
    ));
    assert!(matches!(
        module.functions[5].signature.result.ty,
        Type::Member { .. }
    ));
    assert!(matches!(
        module.functions[6].signature.result.ty,
        Type::Value { .. }
    ));
}

#[test]
fn generic_function_references_retain_explicit_arguments() {
    let module =
        hir_module("fn make<U>(value: U) -> U { value } fn select<T, U>() -> _ { make::<U> }")
            .unwrap();
    let select = &module.functions[1];
    let TermKind::Function {
        function,
        type_args,
    } = &tail(select).kind
    else {
        panic!("ordinary function reference")
    };
    assert_eq!(module.functions[function.index()].name.as_ref(), "make");
    assert_eq!(
        type_args,
        &[Type::Parameter {
            parameter: select.signature.type_params[1].id
        }]
    );
    assert!(matches!(select.signature.result.ty, Type::Function { .. }));
}

#[test]
fn dependent_callable_fields_remain_ordinary_calls() {
    let module = hir_module("fn call<T>(value: T) -> _ { (value.callback)(41) }").unwrap();
    let TermKind::Call { func, .. } = &tail(&module.functions[0]).kind else {
        panic!("function-valued field call")
    };
    assert!(matches!(value(func).kind, TermKind::Field { .. }));
    assert!(matches!(
        module.functions[0].signature.result.ty,
        Type::Value { .. }
    ));
}

#[test]
fn unknown_names_and_undetermined_arguments_fail_before_specialization() {
    let error = hir_module("fn unused<T>(value: T) { value:missing(); }").unwrap_err();
    assert!(error.to_string().contains("UnboundValue"), "{error}");
    let error = hir_module(
        "fn read<T, U>(value: T) -> T { value } fn relay<T>(value: T) -> T { value:read::<T, _>() }",
    ).unwrap_err();
    assert!(error.to_string().contains("annotat"), "{error}");
}
