mod common;
use common::hir_module;

#[test]
fn reference_parameters_results_and_initialized_annotations_complete() {
    for source in [
        "fn identity<T>(value: Ref<T>) -> Ref<T>  { value }",
        "fn refer<T>(value: Ptr<T>) -> Ref<T>  { value.* }",
        "fn main() -> i32  { let mut x: i32 = 1; let mut r: Ref<i32> = x; r = 2; let mut copy = r; copy }",
        "fn id(value: Ref<i32>) -> Ref<i32>  { value } fn main() -> i32  { let mut x: i32 = 1; id(x) = 2; x }",
        "fn choose(a: Ref<i32>, b: Ref<i32>, flag: bool) -> Ref<i32>  { if (flag) { a } else { b } }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn reference_binding_requires_an_initialized_place() {
    for (source, diagnostic) in [
        (
            "fn main()  { let mut reference: Ref<i32>; }",
            "require an initializer",
        ),
        (
            "fn main()  { let mut reference: Ref<i32> = i32(1); }",
            "reference binding requires",
        ),
        (
            "fn main()  { let mut value: i32; let mut reference: Ref<i32> = value; }",
            "UninitializedValue",
        ),
        (
            "fn get() -> i32  { 1 } fn main()  { let mut reference: Ref<i32> = get(); }",
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
        "struct Invalid { value: Ref<i32>, }",
        "type R = Ref<i32>; struct Invalid { value: R, }",
        "fn invalid(value: Ptr<Ref<i32>>)  {}",
        "fn invalid(value: Ref<Ref<i32>>)  {}",
        "fn invalid(value: (Ref<i32> | Err<None>))  {}",
        "fn invalid(value: Ref<i32> | None)  {}",
        "struct Cell<T> { value: T, } fn invalid(value: Cell<Ref<i32>>)  {}",
        "fn identity<T>(value: T) -> T  { value } fn invalid()  { let mut x: i32 = 1; identity::<Ref<i32>>(x); }",
        "struct Cell {  }\nfn cell_accept<T>(value: T)  {}\n fn invalid()  { let mut x: i32 = 1; cell_accept::<Ref<i32>>(x); }",
        "fn invalid()  { let mut value: Ref<i32> = Ref<i32>(i32(1)); }",
        "extern { \"test.h\": { fn invalid(value: Ref<i32>); } };",
        "extern { \"test.h\": { fn invalid() -> Ref<i32>; } };",
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
        "fn invalid()  { let mut x: i32 = 1; let mut r: Ref<i32 | None> = x; }",
        "fn invalid(p: Ptr<i32>)  { let mut r: Ref<i32> = p; }",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted invalid binding: {source}"
        );
    }
}

#[test]
fn reference_parameters_reject_temporary_arguments() {
    for source in [
        "fn accept(value: Ref<i32>) {} fn main() { accept(i32(1)); }",
        "struct Cell { value: i32 } fn get(self: Ref<Cell>) -> Ref<i32> { self.value } fn main() { Cell { value = 1 }:get(); }",
        "fn get<T>(value: Ref<T>) -> Ref<T> { value } fn main() { get(i32(1)); }",
        "fn accept(value: Ref<i32>) {} fn main() { let call = accept; call(1 + 2); }",
        "struct Item { n: i32 } fn accept(value: Ref<i32>) {} fn main() { accept(Item { n = 1 }.n); }",
        "fn main() { [1, 2]:at(0); }",
        "fn main() { ([1, 2])(0); }",
        "fn accept(value: Ref<u32>) {} @compute_shader fn main(i: u64, p: Ptr<u32>) { accept(u32(1)); }",
    ] {
        let error = hir_module(source).unwrap_err();
        assert!(
            error.diagnostic.contains("bind the temporary to a local"),
            "{source}\n{error}"
        );
    }
}

#[test]
fn reference_contracts_do_not_grant_addresses() {
    for source in [
        "fn invalid(value: Ref<i32>) -> Ptr<i32> { &value }",
        "fn invalid<T>(value: Ref<T>) -> Ptr<T> { &value }",
        "struct Cell { value: i32 } fn invalid(cell: Ref<Cell>) -> Ptr<i32> { &cell.value }",
        "struct Cell { value: i32 } struct Outer { cell: Cell } fn invalid(value: Ref<Outer>) -> Ptr<i32> { &value.cell.value }",
        "fn identity(value: Ref<i32>) -> Ref<i32> { value } fn invalid(pointer: Ptr<i32>) -> Ptr<i32> { &identity(pointer.*) }",
        "fn invalid(pointer: Ptr<i32>) -> Ptr<i32> { let reference: Ref<i32> = pointer.*; &reference }",
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
        "fn address(pointer: Ref<Ptr<i32>>) -> Ptr<i32> { &pointer.* }",
        "struct Cell { value: i32 } fn address(pointer: Ref<Ptr<Cell>>) -> Ptr<i32> { &pointer.value }",
        "struct Cell { value: i32 } struct Handle { pointer: Ptr<Cell> } fn address(handle: Ref<Handle>) -> Ptr<i32> { &handle.pointer.value }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn local_storage_has_no_pointer_capability() {
    for source in [
        "fn invalid() { let value: i32 = 1; &value; }",
        "struct Cell { value: i32 } fn invalid() { let cell = Cell { value = 1 }; &cell.value; }",
        "fn invalid() { let values = [i32(1), i32(2)]; &values:at(u64(0)); }",
        "fn invalid() { let values = [i32(1), i32(2)]; values:lea(u64(0)); }",
        "fn invalid() { let values = [i32(1)]; let alias: Ref<_> = values; alias:lea(u64(0)); }",
        "struct Cell { value: i32 } fn drop(cell: Ptr<Cell>) {}",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted local address: {source}"
        );
    }
}
