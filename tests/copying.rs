mod support;

fn succeeds(source: &str) {
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn value_arguments_and_borrowed_reads_copy_nested_structs() {
    succeeds(
        r#"export { main };
        struct Cell<T> { value: T }
        fn identity<T>(value: T) -> T { value }
        fn copy<T>(value: Ref<T>) -> T { value }
        fn copy_cell<T>(value: Ref<Cell<T>>) -> Cell<T> { value }
        fn sum(pair: Cell<Cell<i32>>) -> i32 { pair.value.value + pair.value.value }
        fn main() -> i32 {
            let value = Cell<Cell<i32>> { value = Cell<i32> { value = 21 } };
            let mut other = copy_cell(value);
            let third = copy(value);
            assert(third.value.value == 21);
            other.value.value = 7;
            assert(sum(value) == 42);
            assert(identity(value).value.value == 21);
            assert(value.value.value == 21 && other.value.value == 7);
            let tuple = (value, [value, value], Err(value));
            let duplicate = tuple;
            assert(tuple.0.value.value == duplicate.0.value.value);
            0
        }
    "#,
    );
}

#[test]
fn implicit_arc_copies_retain_move_only_payloads_through_calls_and_aggregates() {
    succeeds(
        r#"export { main }; import { "$/shared.resin", "$/span.resin" };
        struct Resource { drops: Ptr<i32> }
        fn drop(value: RefMut<Resource>) { value.drops.* = value.drops.* + 1; }
        struct Holder<T> { value: T }
        fn pass<T>(value: T) -> T { value }
        fn inspect(value: ArcPtr<Resource>) { assert(value:get().drops.* == 0); }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            let weak: WeakPtr<Resource>;
            {
                let owner = arc_ptr_alloc(Resource { drops = drops:get() })?;
                weak = owner:downgrade();
                inspect(owner);
                let kept = {
                    let optional: ArcPtr<Resource> | None = owner;
                    let optional_copy = optional;
                    match (optional_copy) { ArcPtr<Resource>(handle) => { inspect(handle); }, None => { assert(false); } };
                    match (optional) { ArcPtr<Resource>(handle) => { inspect(handle); }, None => { assert(false); } };
                    let error = Err(owner);
                    let error_copy = error;
                    match (error) { Err(handle) => { inspect(handle); } };
                    match (error_copy) { Err(handle) => { inspect(handle); } };
                    let holder = Holder<ArcPtr<Resource>> { value = owner };
                    let duplicate = holder;
                    let array = [holder, duplicate];
                    let array_copy = array;
                    pass(array_copy:at(0).value)
                };
                assert(drops:get().* == 0);
                inspect(kept);
                inspect(owner);
                let weak_copy = weak;
                { let upgraded = weak_copy:upgrade()!; inspect(upgraded); };
                assert(drops:get().* == 0);
            };
            assert(drops:get().* == 1);
            assert(match (weak:upgrade()) { None => { true }, ArcPtr<Resource>(_) => { false } });
            0
        }
    "#,
    );
}

#[test]
fn repeated_allocation_copies_structs_and_retains_shared_fields() {
    succeeds(
        r#"export { main }; import { "$/shared.resin", "$/span.resin" };
        struct Resource { drops: Ptr<i32> }
        fn drop(value: RefMut<Resource>) { value.drops.* = value.drops.* + 1; }
        struct Holder { owner: ArcPtr<Resource>, number: i32 }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            {
                let initial = Holder { owner = arc_ptr_alloc(Resource { drops = drops:get() })?, number = 42 };
                let values = arc_span_alloc(u64(3), initial)?;
                assert(initial.number == 42);
                let view = values:get();
                let element = view:at(1);
                assert(element.number == 42 && element.owner:get().drops.* == 0);
                let empty = arc_span_alloc(u64(0), initial)?;
                assert(empty:get().length == 0);
            };
            assert(drops:get().* == 1);
            0
        }
    "#,
    );
}

#[test]
fn phantom_box_propagates_move_only_ownership_without_a_struct_annotation() {
    for body in [
        "let item = Item { marker = PhantomBox {} }; let moved = item; item;",
        "let item = Item { marker = PhantomBox {} }; let items = [item]; let moved = items; items;",
        "let item = Item { marker = PhantomBox {} }; let error = Err(item); let moved = error; error;",
        "let item = Item { marker = PhantomBox {} }; let pointer = arc_ptr_alloc(item)?; let copied = pointer:get().*;",
        "let item = Item { marker = PhantomBox {} }; let values = arc_span_alloc(u64(2), item)?;",
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/ownership.resin", "$/shared.resin", "$/status.resin" }};
            struct Item {{ marker: PhantomBox }}
            fn main() -> () | Err<OutOfMemory> {{ {body} }}
        "#
        );
        let error = support::pipeline::source_module(&source)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("moved")
                || error.contains("cannot move")
                || error.contains("implicitly copyable"),
            "{error}"
        );
    }
    succeeds(
        r#"export { main }; import { "$/ownership.resin" };
        struct Item { marker: PhantomBox, value: i32 }
        fn consume(item: Item) -> i32 { item.value }
        fn main() -> i32 {
            let item = Item { marker = PhantomBox {}, value = 42 };
            assert(consume(item) == 42);
            0
        }
    "#,
    );
}
