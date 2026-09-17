mod support;

fn succeeds(source: &str) {
    let module = support::module(source);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{source}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn manual_example_runs_as_an_ordinary_host_function() {
    succeeds(include_str!("../examples/parallel_blocks.resin"));
}

#[test]
fn serial_schedule_handles_empty_nested_and_different_element_types() {
    succeeds(
        r#"export { main };
        fn main() -> i32 {
            let ys = parallel_map([1, 2, 3, 4, 5]) |x| { f32(x) * f32(0.5) };
            assert(ys(0) == f32(0.5) && ys(4) == f32(2.5));
            let sum = parallel_reduce(ys, f32(0)) |a, b| { a + b };
            assert(sum == f32(7.5));
            let empty = parallel_map([]) |x| { let value: i64 = x; value + 1 };
            assert(parallel_reduce(empty, 0) |a, b| { a + b } == 0);
            let nested = parallel_map([1, 2, 3]) |x| {
                let inner = parallel_map([10, 20]) |y| { x * y };
                parallel_reduce(inner, 0) |a, b| { a + b }
            };
            assert(nested(0) == 30 && nested(1) == 60 && nested(2) == 90);
            let one = parallel_reduce([42], 0) |mut a, b| { a = a + b; a };
            assert(one == 42);
            0
        }
    "#,
    );
}

#[test]
fn map_inputs_and_reduce_identity_are_evaluated_once_in_source_order() {
    succeeds(
        r#"export { main };
        fn step(value: RefMut<i64>) -> i64 { value = value + 1; value }
        fn main() -> i32 {
            let mut counter = 0;
            let mapped = parallel_map([step(counter), step(counter)]) |mut x| {
                x = x * 2;
                x
            };
            assert(counter == 2 && mapped(0) == 2 && mapped(1) == 4);
            let sum = parallel_reduce([step(counter), step(counter)], step(counter)) |a, b| { a + b };
            assert(counter == 5 && sum == 12);
            0
        }
    "#,
    );
}

#[test]
fn generic_helpers_preserve_copy_and_capture_contracts() {
    succeeds(
        r#"export { main };
        fn identity<T>(value: T) -> T { value }
        fn repeat<T>(value: T) -> T {
            let results = parallel_map([0, 1]) |x| { identity(value) };
            results(1)
        }
        fn main() -> i32 { assert(repeat(42) == 42); 0 }
    "#,
    );
    let source = r#"
        struct Cell { value: i64 }
        fn update(value: RefMut<Cell>) -> i64 { value.value = 1; value.value }
        fn apply<T>(mut value: T) -> i64 {
            let results = parallel_map([0]) |x| { value:update() };
            results(0)
        }
        fn main() -> i64 { apply(Cell { value = 0 }) }
    "#;
    let error = support::pipeline::source_module(source)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("RefMut") || error.contains("read-only"),
        "{error}"
    );
}

#[test]
fn shader_helpers_use_the_serial_schedule() {
    let module = support::pipeline::source_module(
        r#"
        fn sum() -> i64 { parallel_reduce([1, 2, 3], 0) |a, b| { a + b } }
        @compute_shader fn kernel(index: u64, output: Ptr<i64>) { output.* = sum(); }
    "#,
    )
    .unwrap();
    let project = support::project::Project::new(&module, None).unwrap();
    support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
}

#[test]
fn iteration_results_and_shared_captures_keep_their_owners_until_scope_exit() {
    succeeds(
        r#"export { main }; import { "$/shared.resin" };
        struct Resource { drops: Ptr<i32> }
        fn drop(value: RefMut<Resource>) { value.drops.* = value.drops.* + 1; }
        fn main() -> i32 | Err<_> {
            let drops = arc_ptr_alloc(i32(0))?;
            {
                let results = parallel_map([1, 2, 3]) |x| {
                    Resource { drops = drops:get() }
                };
                assert(drops:get().* == 0);
            };
            assert(drops:get().* == 3);
            {
                let owner = arc_ptr_alloc(Resource { drops = drops:get() })?;
                let copies = parallel_map([owner, owner]) |item| { item };
                let kept = parallel_reduce(copies, owner) |a, b| { a };
                assert(drops:get().* == 3);
                assert(kept:get().drops.* == 3);
            };
            assert(drops:get().* == 4);
            0
        }
    "#,
    );
}

#[test]
fn cooperative_lir_checks_region_parameters_and_boundaries() {
    let module = support::module(
        r#"
        @compute_shader fn kernel(group: u64, output: Ptr<u32>) {
            let values = parallel_map([u32(1), u32(2)]) |x| { x + u32(1) };
            output.* = values(0);
        }
    "#,
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let bytes = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert!(
        support::shaders::instructions(&bytes, 224).next().is_some(),
        "map completion must synchronize its workgroup"
    );
    assert!(
        support::shaders::instructions(&bytes, 59).any(|args| args[2] == 4),
        "map results require Workgroup storage"
    );
    let function = module
        .functions
        .iter()
        .position(|function| function.profile == resin_lir::Profile::Compute)
        .unwrap();
    let block = module.functions[function]
        .blocks
        .iter()
        .position(|block| matches!(block.terminator, resin_lir::Terminator::Parallel { .. }))
        .unwrap();
    let mut bad = module.clone();
    if let resin_lir::Terminator::Parallel { private_locals, .. } =
        &mut bad.functions[function].blocks[block].terminator
    {
        private_locals.start = 0;
    }
    assert!(matches!(
        resin_lir::verify(&bad).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidParallelRegion
    ));
    let mut bad = module.clone();
    let resin_lir::Terminator::Parallel { body, .. } =
        bad.functions[function].blocks[block].terminator
    else {
        unreachable!()
    };
    bad.functions[function].blocks[body.index()].terminator = resin_lir::Terminator::Return;
    assert!(matches!(
        resin_lir::verify(&bad).unwrap_err().kind,
        resin_lir::VerifyErrorKind::InvalidParallelRegion
    ));
}
