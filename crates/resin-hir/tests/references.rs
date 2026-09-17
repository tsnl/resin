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

#[test]
fn reference_contracts_do_not_grant_addresses() {
    for source in [
        "fn invalid(value: Ref<int>) -> Ptr<int> { &value }",
        "fn invalid<T>(value: Ref<T>) -> Ptr<T> { &value }",
        "struct Cell { value: int } fn invalid(cell: Ref<Cell>) -> Ptr<int> { &cell.value }",
        "struct Cell { value: int } struct Outer { cell: Cell } fn invalid(value: Ref<Outer>) -> Ptr<int> { &value.cell.value }",
        "fn identity(value: Ref<int>) -> Ref<int> { value } fn invalid(pointer: Ptr<int>) -> Ptr<int> { &identity(pointer.*) }",
        "fn invalid(pointer: Ptr<int>) -> Ptr<int> { let reference: Ref<int> = pointer.*; &reference }",
    ] {
        let error = hir_module(source).unwrap_err();
        assert!(
            error
                .diagnostic
                .contains("cannot take the address of a Ref"),
            "{source}\n{error}"
        );
    }
}

#[test]
fn reading_a_pointer_through_a_reference_preserves_pointer_access() {
    for source in [
        "fn address(pointer: Ref<Ptr<int>>) -> Ptr<int> { &pointer.* }",
        "struct Cell { value: int } fn address(pointer: Ref<Ptr<Cell>>) -> Ptr<int> { &pointer.value }",
        "struct Cell { value: int } struct Handle { pointer: Ptr<Cell> } fn address(handle: Ref<Handle>) -> Ptr<int> { &handle.pointer.value }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn local_storage_has_no_pointer_capability() {
    for source in [
        "fn invalid() { let value = 1_i; &value; }",
        "struct Cell { value: int } fn invalid() { let cell = Cell { value = 1 }; &cell.value; }",
        "fn invalid() { let values = [1_i, 2_i]; &values:at(0_ul); }",
        "fn invalid() { let values = [1_i, 2_i]; values:lea(0_ul); }",
        "fn invalid() { let values = [1_i]; let alias: Ref<_> = values; alias:lea(0_ul); }",
        "struct Cell { value: int } fn drop(cell: Ptr<Cell>) {}",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted local address: {source}"
        );
    }
}
