mod support;

#[test]
fn primitive_operations_and_source_overloads_share_free_call_resolution() {
    let source = r#"
        export { main };
        struct Item { value: int }
        fn at(value: Ref<Item>, index: ulong) -> Ref<int> { value.value }
        fn first<T>(value: Ref<T>) -> _ { value:at(0_ul) }
        fn main() -> int {
            let mut values = [19_i, 23_i];
            let mut item = Item { value = 42 };
            at(values, 0_ul) = 20_i;
            values:at(1_ul) = 22_i;
            at(item, 0_ul) = first(values) + values:at(1_ul);
            let old = replace(&item.value, 0_i);
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
        struct Left { value: int, }
        struct Right { value: int, }
        fn relate(a: Left, b: Right) -> int { a.value + b.value }
        fn relate(a: Right, b: Left) -> int { a.value - b.value }
        fn relate(a: int, b: int) -> int { a + b }
        fn relation<A, B>(a: A, b: B) -> _ { a:relate(b) }
        fn __add__(a: Left, b: Right) -> int { a.value + b.value }
        fn sum<A, B>(a: A, b: B) -> _ { a + b }
        fn main() -> int {
            let a = relation(Left { value= 19 }, Right { value= 23 });
            let b = relation(Right { value= 23 }, Left { value= 19 });
            let c = sum(Left { value= 19 }, Right { value= 23 });
            let d = sum(19_i, 23_i);
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
        @compute_shader fn kernel(index: ulong, root: Ptr<uint>) { root.* = uint(index); }
        fn main() -> int | Err<_> {
            if (false) {
                let gpu = gpu_new()?;
                let value = gpu:create(0_ui)?;
                let pipeline = gpu:create_compute_pipeline(kernel)?;
                let commands = gpu:start_command_recording()?;
                commands:dispatch(pipeline, 0_ui, 1_ui, 1_ui, 1_ui)?;
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
