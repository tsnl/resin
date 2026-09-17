mod support;

#[test]
fn primitive_operations_and_source_overloads_share_free_call_resolution() {
    let source = r#"
        export { main };
import { "$/shared.resin" };

        struct Item { value: i32 }
        fn at_mut(value: RefMut<Item>, index: u64) -> RefMut<i32> { value.value }
        fn first<T>(value: Ref<T>) -> _ { value:at(u64(0)) }
        fn main() -> i32 | Err<_> {
            let mut values = [i32(19), i32(23)];
            let item_owner = arc_ptr_alloc(Item { value = 42 })?; let item: RefMut<_> = item_owner:get().*;
            at_mut(values, u64(0)) = i32(20);
            values:at_mut(u64(1)) = i32(22);
            at_mut(item, u64(0)) = first(values) + values:at(u64(1));
            let old = replace(&item_owner:get().value, i32(0));
            assert(old == 42 && item.value == 0);
            assert(first(values) == 20);
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
fn generic_relations_and_operators_select_visible_signatures() {
    let source = r#"
        export { main };
        struct Left { value: i32, }
        struct Right { value: i32, }
        fn relate(a: Left, b: Right) -> i32 { a.value + b.value }
        fn relate(a: Right, b: Left) -> i32 { a.value - b.value }
        fn relate(a: i32, b: i32) -> i32 { a + b }
        fn relation<A, B>(a: A, b: B) -> _ { a:relate(b) }
        fn __add__(a: Left, b: Right) -> i32 { a.value + b.value }
        fn sum<A, B>(a: A, b: B) -> _ { a + b }
        fn main() -> i32 {
            let a = relation(Left { value= 19 }, Right { value= 23 });
            let b = relation(Right { value= 23 }, Left { value= 19 });
            let c = sum(Left { value= 19 }, Right { value= 23 });
            let d = sum(i32(19), i32(23));
            if (a == 42 && b == 4 && c == 42 && d == 42) { 0 } else { 1 }
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
fn gpu_bridges_borrow_receivers_and_pipelines_through_native_generation() {
    let source = r#"
        export { main };
        import { "$/gpu.resin" };
        @compute_shader fn kernel(index: u64, root: Ptr<u32>) { root.* = u32(index); }
        fn main() -> i32 | Err<_> {
            if (false) {
                let gpu = gpu_new()?;
                let value = gpu:create(u32(0))?;
                let pipeline = gpu:create_compute_pipeline(kernel)?;
                let commands = gpu:start_command_recording()?;
                commands:dispatch(pipeline, u32(0), u32(1), u32(1), u32(1))?;
                commands:cancel();
                let retained = pipeline:clone();
            };
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
