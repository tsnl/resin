mod support;

#[test]
fn dependent_reads_distinguish_references_from_fresh_owned_results() {
    let error = support::pipeline::source_module(
        r#"
        struct Item { value: int }
        fn copy<T>(value: Ref<T>) -> T { value }
        fn main() { let original = Item { value = 42 }; let duplicate = copy(original); }
    "#,
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("cannot move a value through a reference"),
        "{error}"
    );
    let source = r#"
        export { main };
        struct Item { value: int }
        fn duplicate(value: Ref<Item>) -> Item { Item { value = value.value } }
        fn forwarded<T>(value: Ref<T>) -> _ { value:duplicate() }
        fn copy<T>(value: Ref<T>) -> T { value }
        fn main() -> int {
            let original = Item { value = 42 };
            let fresh = forwarded(original);
            assert(fresh.value == original.value && copy(42_i) == 42);
            0
        }
    "#;
    let output = support::project::Project::new(&support::module(source), Some("main"))
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
fn reference_arguments_keep_temporaries_until_the_full_expression_finishes() {
    let source = r#"
        export { main };
import { "$/shared.resin" };

        struct Item { trace: Ptr<int>, digit: int }
        fn drop(value: Ref<Item>) { value.trace.* = value.trace.* * 10 + value.digit; }
        fn borrow<T>(value: Ref<T>) -> Ref<T> { value }
        fn digit(value: Ref<Item>) -> int { assert(value.trace.* == 0); value.digit }
        fn observe(a: Ref<Item>, b: Ref<Item>) -> int {
            assert(a.trace.* == 0 && b.trace.* == 0);
            a.digit + b.digit
        }
        fn forwarded<T>(value: T) -> int { value:digit() }
        fn condition(value: Ref<Item>, iteration: Ref<int>) -> bool {
            iteration = iteration + 1;
            iteration <= 2
        }
        fn fail() -> int | Err<None> { Err(None) }
        fn early(trace: Ptr<int>) -> int | Err<None> {
            observe(Item { trace = trace, digit = 3 }, Item { trace = trace, digit = fail()? })
        }
        fn main() -> int | Err<_> {
            let trace_owner = arc_ptr_alloc(0_i)?; let trace: Ref<_> = trace_owner:get().*;
            let result = observe(Item { trace = trace_owner:get(), digit = 1 }:borrow(), Item { trace = trace_owner:get(), digit = 2 });
            assert(result == 3 && trace == 21);
            trace = 0;
            assert(Item { trace = trace_owner:get(), digit = 4 }:digit() == 4 && trace == 0);
            assert(trace == 4);
            trace = 0;
            match (early(trace_owner:get())) { int(_) => {}, Err(_) => {} };
            assert(trace == 3);
            trace = 0;
            assert(forwarded(Item { trace = trace_owner:get(), digit = 5 }) == 5);
            assert(trace == 5);
            trace = 0;
            let mut iteration = 0_i;
            while (condition(Item { trace = trace_owner:get(), digit = 6 }, iteration)) {
                assert(trace == 6 || trace == 66);
            };
            assert(trace == 666 && iteration == 3);
            0
        }
    "#;
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
fn arc_allocation_moves_payloads_and_explicit_clones_keep_them_alive() {
    let source = r#"
        export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Item { trace: Ptr<int>, value: int }
        fn drop(value: Ref<Item>) { value.trace.* = value.trace.* + 1; }
        fn main() -> int | Err<OutOfMemory> {
            let trace_owner = arc_ptr_alloc(0)?; let trace: Ref<int> = trace_owner:get().*;
            {
                let original = arc_ptr_alloc(Item { trace = trace_owner:get(), value = 42 })?;
                assert(trace == 0);
                let retained = original:clone();
                let weak = original:downgrade();
                { let moved = original; };
                assert(trace == 0);
                assert(retained:get().value == 42);
                { let upgraded = weak:upgrade()!; assert(upgraded:get().value == 42); };
                assert(trace == 0);
            };
            assert(trace == 1);
            0
        }
    "#;
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
fn repeated_allocation_rejects_move_only_elements() {
    let source = r#"
        export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Item { value: int }
        fn main() -> () | Err<OutOfMemory> {
            let values = arc_span_alloc(2_ul, Item { value = 42 })?;
        }
    "#;
    let error = support::pipeline::source_module(source).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("repeated allocation requires an implicitly copyable element"),
        "{error}"
    );
}

#[test]
fn moves_and_partial_replacements_destroy_each_value_once() {
    let source = r#"export { main };
import { "$/shared.resin" };

        struct Item { trace: Ptr<int>, digit: int,
            
        }
fn drop(value: Ref<Item>) { value.trace.* = value.trace.* * 10 + value.digit; }

        struct Pair { a: Item, b: Item, }
        fn main() -> int | Err<_> {
            let trace_owner = arc_ptr_alloc(0)?; let trace: Ref<int> = trace_owner:get().*;
            {
                let mut pair = Pair { a= Item { trace= trace_owner:get(), digit= 1 }, b= Item { trace= trace_owner:get(), digit= 2 } };
                let saved = pair.a;
                pair.a = Item { trace= trace_owner:get(), digit= 3 };
                let moved = pair;
            };
            if (trace == 231) { 0 } else { 1 }
        }
    "#;
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
fn returned_error_payloads_keep_exactly_one_owner() {
    let source = r#"export { main };
import { "$/shared.resin" };

        struct Item { trace: Ptr<int>, }
        fn drop(value: Ref<Item>) { value.trace.* = value.trace.* + 1; }
        fn route(value: Item, failed: bool) -> Item | Err<Item> {
            if (failed) { return Err(value); };
            value
        }
        fn main() -> int | Err<_> {
            let trace_owner = arc_ptr_alloc(0)?; let trace: Ref<int> = trace_owner:get().*;
            {
                let result = route(Item { trace = trace_owner:get() }, true);
                match (result) {
                    Err(error) => { let moved = error; },
                    Item(value) => { let moved = value; },
                };
            };
            if (trace == 1) { 0 } else { 1 }
        }
    "#;
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
