use resin_hir::{TermKind, Type};

mod common;
use common::hir_module;

fn compile(source: &str) -> Result<resin_hir::Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn named_functions_keep_one_polymorphic_body_and_complete_weak_results() {
    let module = compile("fn identity<T>(value: T) -> _  { value } fn main() -> int  { identity(1) } fn other() -> bool  { identity::<bool>(1 == 1) }").unwrap();
    let identity = &module.functions[0];
    assert_eq!(identity.signature.type_params.len(), 1);
    assert_eq!(
        identity.signature.result.ty,
        Type::Parameter {
            parameter: identity.signature.type_params[0].id
        }
    );
    assert_eq!(module.functions.len(), 3);
    let printed = resin_hir::format_module(&module);
    assert!(printed.contains("(apply fn0 \"int\")"), "{printed}");
    assert!(printed.contains("(apply fn0 \"bool\")"), "{printed}");
}

#[test]
fn arithmetic_and_literals_remain_polymorphic_and_enclosing_binders_are_preserved() {
    let module = compile("fn increment<T>(value: T) -> T  { value + 1 } fn twice<U>(value: U) -> U  { increment(increment(value)) } fn main() -> uint  { twice(2) }").unwrap();
    assert_eq!(module.functions.len(), 3);
    let TermKind::Block { tail, .. } = &module.functions[0].body.as_ref().unwrap().kind else {
        panic!("block");
    };
    let mut tail = tail.as_ref();
    while let TermKind::Use { arg } | TermKind::Convert { arg } = &tail.kind {
        tail = arg;
    }
    let TermKind::DependentMethodCall { args, lookup, .. } = &tail.kind else {
        panic!("addition: {tail:?}");
    };
    assert_eq!(lookup.receiver, module.functions[0].signature.result.ty);
    assert!(matches!(
        args[1].ty,
        Type::FunctionParameter { index: 1, .. }
    ));
    assert!(matches!(args[1].kind, TermKind::Numeric { .. }));
}

#[test]
fn members_and_layout_queries_retain_determining_types() {
    let module = compile(
        "fn read<T>(value: Ptr<T>) -> _  { value.member } fn measure<T>() -> ulong  { size_of(T) }",
    )
    .unwrap();
    assert!(matches!(
        module.functions[0].signature.result.ty,
        Type::Member { .. }
    ));
    let TermKind::Block { tail, .. } = &module.functions[1].body.as_ref().unwrap().kind else {
        panic!("block");
    };
    assert!(matches!(
        tail.kind,
        TermKind::Layout {
            of: Type::Parameter { .. },
            ..
        }
    ));
}

#[test]
fn recursive_groups_complete_results_without_equating_distinct_binders() {
    let module = compile("fn left<T>(value: T, stop: bool) -> _  { if (stop) { value } else { right(value, 1 == 1) } } fn right<U>(value: U, stop: bool) -> _  { if (stop) { value } else { left(value, 1 == 1) } } fn main() -> int  { left(1, 1 == 1) }").unwrap();
    for function in &module.functions[..2] {
        assert_eq!(
            function.signature.result.ty,
            Type::Parameter {
                parameter: function.signature.type_params[0].id
            }
        );
    }
    assert_ne!(
        module.functions[0].signature.type_params[0].id,
        module.functions[1].signature.type_params[0].id
    );
}

#[test]
fn unresolved_results_and_undetermined_arguments_require_annotations() {
    for source in [
        "fn looped<T>() -> _  { looped::<T>() } fn main() -> int  { looped::<int>() }",
        "fn create<T>() -> T  { 0 } fn main()  { create(); }",
        "fn marker<T>()  {} fn main()  { marker(); }",
    ] {
        let error = compile(source).unwrap_err();
        assert!(error.to_string().contains("annotat"), "{source}: {error}");
    }
}

#[test]
fn applications_do_not_generalize_local_storage_or_discard_explicit_arity() {
    for source in [
        "fn identity<T>(value: T) -> T  { value } fn main()  { identity::<int, int>(1); }",
        "fn identity(value: int) -> int  { value } fn main()  { identity::<int>(1); }",
        "fn identity<T>(value: T) -> T  { value } fn main()  { let mut f = identity; f(1); f(1 == 1); }",
        "fn pair<T>(a: T, b: T) -> T  { a } fn main()  { pair(1_i, 2_l); }",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
}

#[test]
fn result_payloads_and_error_sets_retain_bound_parameters() {
    let module = compile("fn propagate<T, E>(input: (T | Err<E>)) -> (_ | Err<_>)  { (input?) } fn recover<T, E>(input: (T | Err<E>), fallback: T) -> T  { match (input) { T(value) => { value }, Err(error) => { fallback } } } fn combine<T, E, F>(a: (T | Err<E>), b: (T | Err<F>)) -> (T | Err<_>)  { a?; (b?) }").unwrap();
    let signature = &module.functions[0].signature;
    assert_eq!(
        signature.result.ty,
        Type::Union {
            variants: vec![
                Type::Parameter {
                    parameter: signature.type_params[0].id
                },
                Type::Error {
                    payload: Box::new(Type::Parameter {
                        parameter: signature.type_params[1].id
                    })
                }
            ]
        }
    );
    let Type::Union { variants } = &module.functions[2].signature.result.ty else {
        panic!("union")
    };
    let error = variants
        .iter()
        .find_map(|ty| match ty {
            Type::Error { payload } => Some(payload),
            _ => None,
        })
        .unwrap();
    assert!(matches!(error.as_ref(), Type::Union { variants } if variants.len() == 2));
}
