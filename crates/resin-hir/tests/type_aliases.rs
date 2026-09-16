use resin_hir::Type;

mod common;
use common::hir_module;

fn compile(source: &str) -> Result<resin_hir::Module, resin_source::SourceError> {
    hir_module(source)
}

#[test]
fn transparent_aliases_expose_structure_for_deduction() {
    let module = compile("type Pair<T, U> = (T, U); type Repeated<T> = Pair<T, T>; fn first<T>(value: Repeated<T>) -> T  { value.0 } fn main() -> int  { first((42, 0)) }").unwrap();
    assert!(matches!(
        module.functions[0].signature.params[0].annotation.ty,
        Type::Record { .. }
    ));
    assert_eq!(module.functions[1].signature.result.ty, Type::Int32);
    assert!(module.types.is_empty()); // Transparent aliases create no nominal definitions.
}

#[test]
fn local_aliases_capture_outer_binders_without_capturing_their_own_arguments() {
    let module = compile("type Pair<T, U> = (T, U); fn combine<T>(value: T) -> _  { type Captured<U> = Pair<T, U>; type Shadow<T> = Pair<T, int>; let mut first: Captured<int>; first = (value, 42); let mut second: Shadow<bool>; second = (true, 0); first } fn main() -> int  { combine(0).1 }").unwrap();
    let signature = &module.functions[0].signature;
    let Type::Record { fields } = &signature.result.ty else {
        panic!("record")
    };
    assert_eq!(
        fields[0].ty,
        Type::Parameter {
            parameter: signature.type_params[0].id
        }
    );
    assert_eq!(fields[1].ty, Type::Int32);
}

#[test]
fn alias_arity_holes_and_cycles_are_definition_errors() {
    for source in [
        "type Item<T> = Ptr<T>; fn f(value: Item)  {}",
        "type Item<T> = Ptr<T>; fn f(value: Item<int, int>)  {}",
        "type Item<T, T> = T;",
        "type Item<T> = Ptr<_>;",
        "type Item<T> = Ptr<Item<T>>;",
        "type Item<T> = Next<T>; type Next<T> = Item<T>;",
        "type Unused<T> = int; fn main()  { let mut item: Unused<_>; item = 0; }",
    ] {
        assert!(compile(source).is_err(), "{source}");
    }
    let error = compile("type Item<T> = Ptr<Item<T>>;").unwrap_err();
    assert!(
        error.to_string().contains("recursive type alias"),
        "{error}"
    );
}

#[test]
fn alias_expansion_has_its_own_size_limit() {
    let mut source = "type Double<T> = (T, T); type A0 = int;".to_owned();
    for index in 1..18 {
        source.push_str(&format!("type A{index} = Double<A{}>;", index - 1));
    }
    let error = compile(&source).unwrap_err();
    assert!(error.to_string().contains("size limit"), "{error}");
}

#[test]
fn alias_expansion_has_its_own_depth_limit() {
    let mut source = "type A0 = int;".to_owned();
    for index in 1..260 {
        source.push_str(&format!("type A{index} = Ptr<A{}>;", index - 1));
    }
    let error = compile(&source).unwrap_err();
    assert!(error.to_string().contains("depth limit"), "{error}");
}
