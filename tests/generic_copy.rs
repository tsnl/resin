mod support;

fn succeeds(source: &str) {
    let module = support::module(source);
    let output = support::project::Project::new(&module, Some("main"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn rejects(source: &str, message: &str) {
    let error = support::pipeline::source_module(source)
        .unwrap_err()
        .to_string();
    assert!(error.contains(message), "{error}");
    assert!(
        !error.contains("invalid HIR") && !error.contains("invalid IR"),
        "{error}"
    );
}

#[test]
fn repeated_generic_values_copy_structs_and_retain_shared_owners() {
    succeeds(
        r#"export { main }; import { "$/shared.resin" };
        struct Cell<T> { value: T }
        struct Resource { drops: Ptr<i32> }
        fn drop(value: RefMut<Resource>) { value.drops.* = value.drops.* + 1; }
        fn twice<T>(value: T) -> (T, T) { (value, value) }
        fn branch<T>(value: T, flag: bool) -> T {
            if (flag) { let other = value; };
            value
        }
        fn repeat<T>(value: T) -> T {
            let mut index = 0;
            while (index < 3) { let other = value; index = index + 1; continue; };
            value
        }
        fn fields<T>(value: Cell<T>) -> (T, Cell<T>) { (value.value, value) }
        fn main() -> i32 | Err<_> {
            let pair = twice(Cell<i32> { value = 21 });
            assert(pair.0.value + pair.1.value == 42);
            let drops = arc_ptr_alloc(i32(0))?;
            {
                let owner = arc_ptr_alloc(Resource { drops = drops:get() })?;
                let pair = twice(owner);
                let kept = repeat(branch(pair.0, true));
                let cell = fields(Cell<_> { value = kept });
                assert(cell.0:get().drops.* == 0);
                assert(cell.1.value:get().drops.* == 0);
                assert(owner:get().drops.* == 0);
            };
            assert(drops:get().* == 1);
            0
        }
    "#,
    );
}

#[test]
fn repeated_uses_require_copy_across_branches_loops_and_partial_moves() {
    for (definition, call) in [
        (
            "fn exercise<T>(value: T) { let other = value; value; }",
            "exercise(item)",
        ),
        (
            "fn exercise<T>(value: T) { if (true) { let other = value; }; value; }",
            "exercise(item)",
        ),
        (
            "fn exercise<T>(value: T) { while (true) { let other = value; continue; }; }",
            "exercise(item)",
        ),
        (
            "fn exercise<T>(value: T) { while (true) { let other = value; break; }; value; }",
            "exercise(item)",
        ),
        (
            "struct Cell<T> { value: T } fn exercise<T>(value: Cell<T>) { let field = value.value; value; }",
            "exercise(Cell<_> { value = item })",
        ),
        (
            "struct Cell<T> { value: T } fn exercise<T>(mut value: Cell<T>, replacement: T) { let other = value; value.value = replacement; }",
            "exercise(Cell<_> { value = item }, Item {})",
        ),
    ] {
        rejects(
            &format!(
                r#"
            struct Item {{}} fn drop(value: RefMut<Item>) {{}}
            {definition}
            fn main() {{ let item = Item {{}}; {call}; }}
        "#
            ),
            "implicitly copyable",
        );
    }
}

#[test]
fn single_use_generics_move_and_reinitialization_restores_availability() {
    succeeds(
        r#"export { main }; import { "$/shared.resin" };
        struct Item { drops: Ptr<i32> }
        fn drop(value: RefMut<Item>) { value.drops.* = value.drops.* + 1; }
        struct Pair<T> { first: T, second: T }
        fn identity<T>(value: T) -> T { value }
        fn choose<T>(value: T, flag: bool) -> T {
            if (flag) { return value; };
            value
        }
        fn replace<T>(mut value: T, replacement: T) -> T {
            let old = value;
            value = replacement;
            value
        }
        fn replace_field<T>(mut value: Pair<T>, replacement: T) -> Pair<T> {
            let old = value.first;
            value.first = replacement;
            value
        }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            {
                let kept = identity(choose(Item { drops = drops:get() }, true));
                assert(drops:get().* == 0);
                let replaced = replace(kept, Item { drops = drops:get() });
                assert(drops:get().* == 1);
                let pair = replace_field(Pair<_> {
                    first = replaced, second = Item { drops = drops:get() }
                }, Item { drops = drops:get() });
                assert(drops:get().* == 2);
            };
            assert(drops:get().* == 4);
            0
        }
    "#,
    );
}

#[test]
fn dependent_callbacks_borrow_without_requiring_copy() {
    succeeds(
        r#"export { main }; import { "$/shared.resin" };
        struct Item { drops: Ptr<i32>, number: i32 }
        fn drop(value: RefMut<Item>) { value.drops.* = value.drops.* + 1; }
        fn read(value: Ref<Item>) -> i32 { value.number }
        fn increment(value: RefMut<Item>) -> i32 { value.number = value.number + 1; value.number }
        fn twice<T, F>(mut value: T, action: F) -> i32 { action(value) + action(value) }
        fn twice_known<F>(value: Item, action: F) -> i32 { action(value) + action(value) }
        fn followed<T, F>(value: T, action: F) -> T { action(value); value }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            assert(twice(Item { drops = drops:get(), number = 21 }, read) == 42);
            assert(drops:get().* == 1);
            assert(twice(Item { drops = drops:get(), number = 20 }, increment) == 43);
            assert(drops:get().* == 2);
            { let kept = followed(Item { drops = drops:get(), number = 42 }, read); assert(kept.number == 42); };
            assert(drops:get().* == 3);
            assert(twice_known(Item { drops = drops:get(), number = 21 }, read) == 42);
            assert(drops:get().* == 4);
            0
        }
    "#,
    );
    rejects(
        r#"
        struct Item {} fn drop(value: RefMut<Item>) {}
        fn consume(value: Item) {}
        fn twice<T, F>(value: T, action: F) { action(value); action(value); }
        fn main() { twice(Item {}, consume); }
    "#,
        "implicitly copyable",
    );
}

#[test]
fn dependent_field_reads_respect_pointer_and_drop_boundaries() {
    for expression in ["value.value", "identity(value).value"] {
        rejects(
            &format!(
                r#"import {{ "$/shared.resin" }};
        struct Item {{}} fn drop(value: RefMut<Item>) {{}}
        struct Cell {{ value: Item }}
        fn identity<T>(value: T) -> T {{ value }}
        fn field<T>(value: T) -> _ {{ {expression} }}
        fn main() -> () | Err<_> {{
            let owner = arc_ptr_alloc(Cell {{ value = Item {{}} }})?;
            field(owner:get());
        }}
    "#
            ),
            "cannot move a value through a reference or pointer",
        );
    }
    succeeds(
        r#"export { main }; import { "$/shared.resin" };
        struct Item { drops: Ptr<i32> }
        fn drop(value: RefMut<Item>) { value.drops.* = value.drops.* + 1; }
        struct Cell<T> { value: T }
        fn identity<T>(value: T) -> T { value }
        fn field<T>(value: T) -> _ { identity(value).value }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            { let kept = field(Cell<Item> { value = Item { drops = drops:get() } }); assert(drops:get().* == 0); };
            assert(drops:get().* == 1);
            0
        }
    "#,
    );
    for (generic, message) in [
        ("fn field<T>(value: T) -> _ { value.value }", "drop hook"),
        ("fn field<T>(value: Cell<T>) -> T { value.value }", "copy"),
    ] {
        rejects(
            &format!(
                r#"
            struct Item {{}} fn drop(value: RefMut<Item>) {{}}
            struct Cell<T> {{ value: T }} fn drop<T>(value: RefMut<Cell<T>>) {{}}
            {generic}
            fn main() {{ field(Cell<Item> {{ value = Item {{}} }}); }}
        "#
            ),
            message,
        );
    }
}
