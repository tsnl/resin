mod support;

#[test]
fn readonly_views_keep_aliases_without_granting_write_access() {
    let source = r#"
        export { main };
        import { "$/span.resin", "$/shared.resin" };
        fn read<T>(p: Ptr<T>) -> T { p.* }
        fn through_generic<P>(p: P) -> i32 { read(p) }
        fn address<T>(p: Ptr<T>) -> Ptr<T> { &p.* }
        fn borrowed_address<T>(p: Ref<Ptr<T>>) -> Ptr<T> { &p.* }
        struct Cell { n: i32 }
        fn field_address(p: PtrMut<Ptr<Cell>>) -> Ptr<i32> { &p.n }
        fn main() -> i32 | Err<_> {
            let owner = arc_span_alloc(3, i32(4))?;
            let writable = owner:get();
            let readonly = writable:read_only();
            let bytes = readonly:as_bytes();
            let part = readonly:slice(1, 2);
            writable:store(1, 17);
            assert(readonly:load(1) == 17 && part:load(0) == 17);
            assert(bytes.length == 12);
            let pointer = writable:lea(1);
            let observed: Ptr<i32> = pointer;
            assert(through_generic(pointer) == 17 && address(observed).* == 17);
            assert(borrowed_address(observed).* == 17);
            let cell = arc_ptr_alloc(Cell { n = 23 })?;
            let cell_pointer: Ptr<Cell> = cell:get();
            let slot = arc_ptr_alloc(cell_pointer)?;
            assert(field_address(slot:get()).* == 23);
            let reference: RefMut<i32> = pointer.*;
            reference = 19;
            assert(readonly:load(1) == 19);
            0
        }
    "#;
    let output = support::project::Project::new(&support::module(source), Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn readonly_views_cannot_mutate_or_forge_writable_intrinsics() {
    for body in [
        "fn bad(s: Ref<Span<i32>>) { s:store(0, 1); }",
        "fn bad(s: Ref<Span<i32>>) { s:at_mut(0) = 1; }",
        "fn bad(s: Ref<Span<i32>>) -> PtrMut<i32> { s:lea(0) }",
        "fn bad(s: Ref<Span<i32>>) { let x = s:as_bytes(); x:at_mut(0) = 1; }",
        "intrinsic \"pointer_index\" fn bad<T>(p: Ptr<T>, length: u64, index: u64) -> PtrMut<T>;",
        "intrinsic \"pointer_bytes\" fn bad<T>(p: Ptr<T>, length: u64) -> (PtrMut<u8>, u64);",
        "fn bad<T>(p: T) { p.* = 1; } fn main(p: Ptr<i32>) { bad(p); }",
        "fn bad() { let p = \"literal\".data; p.* = 0; }",
    ] {
        let source = format!("import {{ \"$/span.resin\" }}; {body}");
        assert!(
            support::pipeline::source_module(&source).is_err(),
            "accepted {source}"
        );
    }
}
