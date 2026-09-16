mod common;
use common::hir_module;

#[test]
fn reference_parameters_results_and_initialized_annotations_complete() {
    for source in [
        "fn identity<T>(value: Ref<T>) -> Ref<T>  { value }",
        "fn refer<T>(value: Ptr<T>) -> Ref<T>  { value.* }",
        "fn main() -> int  { let mut x: int = 1; let mut r: Ref<int> = x; r = 2; let mut copy = r; copy }",
        "fn id(value: Ref<int>) -> Ref<int>  { value } fn main() -> int  { let mut x = 1_i; id(x) = 2; x }",
        "fn choose(a: Ref<int>, b: Ref<int>, flag: bool) -> Ref<int>  { if (flag) { a } else { b } }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn reference_binding_requires_an_initialized_place() {
    for (source, diagnostic) in [
        (
            "fn main()  { let mut reference: Ref<int>; }",
            "require an initializer",
        ),
        (
            "fn main()  { let mut reference: Ref<int> = 1_i; }",
            "reference binding requires",
        ),
        (
            "fn main()  { let mut value: int; let mut reference: Ref<int> = value; }",
            "UninitializedValue",
        ),
        (
            "fn get() -> int  { 1 } fn main()  { let mut reference: Ref<int> = get(); }",
            "reference binding requires",
        ),
    ] {
        let error = hir_module(source).unwrap_err();
        assert!(error.diagnostic.contains(diagnostic), "{source}\n{error}");
    }
}

#[test]
fn references_cannot_be_hidden_in_value_storage_or_generic_arguments() {
    for source in [
        "struct Invalid { value: Ref<int>, }",
        "type R = Ref<int>; struct Invalid { value: R, }",
        "fn invalid(value: Ptr<Ref<int>>)  {}",
        "fn invalid(value: Ref<Ref<int>>)  {}",
        "fn invalid(value: (Ref<int> | Err<None>))  {}",
        "fn invalid(value: Ref<int> | None)  {}",
        "struct Cell<T> { value: T, } fn invalid(value: Cell<Ref<int>>)  {}",
        "fn identity<T>(value: T) -> T  { value } fn invalid()  { let mut x = 1_i; identity::<Ref<int>>(x); }",
        "struct Cell {  }\nfn cell_accept<T>(value: T)  {}\n fn invalid()  { let mut x = 1_i; cell_accept::<Ref<int>>(x); }",
        "fn invalid()  { let mut value: Ref<int> = Ref<int>(1_i); }",
        "extern { \"test.h\": { fn invalid(value: Ref<int>); } };",
        "extern { \"test.h\": { fn invalid() -> Ref<int>; } };",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted unsupported reference storage: {source}"
        );
    }
}

#[test]
fn reference_binding_does_not_widen_or_implicitly_dereference_pointers() {
    for source in [
        "fn invalid()  { let mut x = 1_i; let mut r: Ref<int | None> = x; }",
        "fn invalid(p: Ptr<int>)  { let mut r: Ref<int> = p; }",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted invalid binding: {source}"
        );
    }
}

#[test]
fn reference_parameters_accept_temporary_arguments() {
    for source in [
        "fn accept(value: Ref<int>) {} fn main() { accept(1_i); }",
        "struct Cell { value: int } fn get(self: Ref<Cell>) -> Ref<int> { self.value } fn main() { Cell { value = 1 }:get(); }",
        "fn get<T>(value: Ref<T>) -> Ref<T> { value } fn main() { get(1_i); }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}
