mod common;
use common::hir_module;

#[test]
fn reference_parameters_results_and_initialized_annotations_complete() {
    for source in [
        "fn identity<T>(value: Ref<T>) -> Ref<T>  { value }",
        "fn refer<T>(value: PtrMut<T>) -> Ref<T>  { value.* }",
        "fn main() -> i32  { let mut x: i32 = 1; let r: RefMut<i32> = x; r = 2; let mut copy = r; copy }",
        "fn id(value: RefMut<i32>) -> RefMut<i32>  { value } fn main() -> i32  { let mut x: i32 = 1; id(x) = 2; x }",
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
        "fn invalid(value: PtrMut<Ref<i32>>)  {}",
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
        "fn invalid(p: PtrMut<i32>)  { let mut r: Ref<i32> = p; }",
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
        "fn accept(value: Ref<u32>) {} @compute_shader fn main(i: u64, p: PtrMut<u32>) { accept(u32(1)); }",
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
        "fn invalid(value: Ref<i32>) -> PtrMut<i32> { &value }",
        "fn invalid<T>(value: Ref<T>) -> PtrMut<T> { &value }",
        "struct Cell { value: i32 } fn invalid(cell: Ref<Cell>) -> PtrMut<i32> { &cell.value }",
        "struct Cell { value: i32 } struct Outer { cell: Cell } fn invalid(value: Ref<Outer>) -> PtrMut<i32> { &value.cell.value }",
        "fn identity(value: Ref<i32>) -> Ref<i32> { value } fn invalid(pointer: PtrMut<i32>) -> PtrMut<i32> { &identity(pointer.*) }",
        "fn invalid(pointer: PtrMut<i32>) -> PtrMut<i32> { let reference: Ref<i32> = pointer.*; &reference }",
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
        "struct Cell { value: i32 } fn address(pointer: PtrMut<Ptr<Cell>>) -> Ptr<i32> { &pointer.value }",
        "struct Cell { value: i32 } fn address(pointer: Ptr<PtrMut<Cell>>) -> PtrMut<i32> { &pointer.value }",
        "fn address(pointer: Ref<PtrMut<i32>>) -> PtrMut<i32> { &pointer.* }",
        "struct Cell { value: i32 } fn address(pointer: Ref<PtrMut<Cell>>) -> PtrMut<i32> { &pointer.value }",
        "struct Cell { value: i32 } struct Handle { pointer: PtrMut<Cell> } fn address(handle: Ref<Handle>) -> PtrMut<i32> { &handle.pointer.value }",
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
        "struct Cell { value: i32 } fn drop(cell: PtrMut<Cell>) {}",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted local address: {source}"
        );
    }
}

#[test]
fn mutable_references_require_writable_places() {
    for source in [
        "fn bad(value: Ref<i32>) { value = 2; }",
        "struct Cell { value: i32 } fn bad(cell: Ref<Cell>) { cell.value = 2; }",
        "fn bad(value: Ref<i32>) -> RefMut<i32> { value }",
        "fn bad(value: Ref<i32>) { let alias: RefMut<i32> = value; }",
        "fn change(value: RefMut<i32>) {} fn bad(value: Ref<i32>) { change(value); }",
        "fn bad() { let value: i32 = 1; let alias: RefMut<i32> = value; }",
        "struct Cell { value: i32 } fn bad() { let cell = Cell { value = 1 }; let alias: RefMut<i32> = cell.value; }",
        "fn change(value: RefMut<i32>) {} fn bad(value: i32) { change(value); }",
        "fn bad() { let mut items = [i32(1), i32(2)]; let values: Ref<_> = items; values:at_mut(0) = 2; }",
        "fn bad() { let values = [i32(1), i32(2)]; values:at_mut(0) = 2; }",
        "fn bad() { let mut values = [i32(1), i32(2)]; values:at(0) = 2; }",
        "fn bad() { let mut value: RefMut<i32> = i32(1); }",
        "fn bad() { let mut values = [i32(1)]; &values:at_mut(0); }",
        "fn bad(value: RefMut<i32>) -> PtrMut<i32> { &value }",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted invalid writable access: {source}"
        );
    }
}

#[test]
fn mutable_references_weaken_and_preserve_pointer_field_capabilities() {
    for source in [
        "fn read(value: Ref<i32>) -> i32 { value } fn use(value: RefMut<i32>) -> i32 { read(value) }",
        "fn read(value: RefMut<i32>) -> Ref<i32> { value }",
        "fn change(value: RefMut<i32>) { value = 42; } fn main() { let mut value: i32 = 0; change(value); }",
        "fn change(mut value: i32) { let alias: RefMut<i32> = value; alias = 42; }",
        "fn change(pointer: Ref<PtrMut<i32>>) { pointer.* = 42; }",
        "struct Cell { value: i32 } struct Handle { pointer: PtrMut<Cell> } fn change(handle: Ref<Handle>) { handle.pointer.value = 42; }",
        "fn change() { let mut values = [i32(1), i32(2)]; let item: RefMut<i32> = values:at_mut(0); item = 42; }",
        "fn read() -> i32 { let values = [i32(1), i32(2)]; values:at(0) }",
    ] {
        hir_module(source).unwrap_or_else(|error| panic!("{source}\n{error}"));
    }
}

#[test]
fn mutable_references_share_reference_storage_restrictions() {
    for source in [
        "struct Invalid { value: RefMut<i32> }",
        "fn invalid(value: PtrMut<RefMut<i32>>) {}",
        "fn invalid(value: Ref<RefMut<i32>>) {}",
        "fn invalid(value: RefMut<Ref<i32>>) {}",
        "fn invalid(value: RefMut<i32> | None) {}",
        "fn invalid() { let reference: RefMut<i32>; }",
        "extern { \"test.h\": { fn invalid(value: RefMut<i32>); } };",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted unsupported reference storage: {source}"
        );
    }
}

#[test]
fn pointer_permissions_survive_borrowing_fields_and_addressing() {
    for source in [
        "fn read(p: Ptr<i32>) -> i32 { p.* }",
        "fn weaken<T>(p: PtrMut<T>) -> Ptr<T> { p }",
        "fn write(p: PtrMut<i32>) { p.* = 3; }",
        "fn borrow(p: Ptr<i32>) -> Ref<i32> { p.* }",
        "struct Cell { value: i32 } fn address(p: Ptr<Cell>) -> Ptr<i32> { &p.value }",
        "fn nested(p: Ptr<PtrMut<i32>>) { p.*.* = 3; }",
        "struct Handle { p: PtrMut<i32> } fn write(h: Ref<Handle>) { h.p.* = 3; }",
    ] {
        hir_module(source).unwrap_or_else(|e| panic!("{source}\n{e}"));
    }
    for source in [
        "fn bad(p: Ptr<i32>) { p.* = 3; }",
        "fn bad(p: Ptr<i32>) { let mut q = p; q.* = 3; }",
        "fn bad(p: Ptr<i32>) -> RefMut<i32> { p.* }",
        "fn bad(p: Ptr<i32>) -> PtrMut<i32> { p }",
        "fn bad(p: Ptr<i32>) -> PtrMut<i32> { PtrMut<i32>(p) }",
        "struct Cell { value: i32 } fn bad(p: Ptr<Cell>) -> PtrMut<i32> { &p.value }",
        "fn bad(p: Ptr<PtrMut<i32>>, q: PtrMut<i32>) { p.* = q; }",
        "fn bad(p: PtrMut<PtrMut<i32>>) -> Ptr<Ptr<i32>> { p }",
        "fn bad(p: Ptr<[i32; 2]>) { p:at_mut(0) = 3; }",
        "fn bad(p: Ptr<[i32; 2]>) -> PtrMut<i32> { p:lea(0) }",
    ] {
        assert!(
            hir_module(source).is_err(),
            "accepted permission strengthening: {source}"
        );
    }
}
