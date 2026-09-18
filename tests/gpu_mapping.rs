mod support;

const GRAPH: &str = r#"
export { main, host_main, kernel };
import { "$/gpu.resin", "$/span.resin", "$/shared.resin", "$/workgroup.resin" };
struct Node { value: u32, next: Ptr<Node> }
struct Root { node: Ptr<Node>, output: SpanMut<u32> }
fn sum(node: Ptr<Node>) -> u32 { node.value + node.next.value }
@compute_shader
fn kernel(index: u64, root: Ptr<Root>, group: Workgroup<u32>) {
    if (group:lane_index() == 0) { group = sum(root.node); };
    group:sync();
    root.output:at_mut(index) = group;
}
fn host_main() -> i32 | Err<_> {
    assert(size_of(Ptr<Node>) == 8 && size_of(Span<u32>) == 16);
    let tail = arc_ptr_alloc(Node { value = 19, next = Ptr<Node>(u64(0)) })?;
    let head = arc_ptr_alloc(Node { value = 23, next = tail:get() })?;
    let output = arc_span_alloc(1, u32(0))?;
    let root = arc_ptr_alloc(Root { node = head:get(), output = output:get() })?;
    let mut state: u32 = 0;
    kernel(0, root:get(), state);
    let values = output:get();
    assert(values:at(0) == 42);
    0
}
fn main() -> i32 | Err<_> {
    let gpu = gpu_new()?;
    let tail = gpu:create(Node { value = 19, next = Ptr<Node>(u64(0)) })?;
    let head = gpu:create(Node { value = 23, next = tail:device() })?;
    let mapped = head:map()?;
    // Mapping exposes the stored device pointer, without rewriting it.
    assert(u64(mapped.next) == u64(tail:device()));
    let output = gpu:alloc::<u32>(gpu:compute_workgroup_size() * 3)?;
    let pipeline = gpu:create_compute_pipeline(kernel)?;
    let commands = gpu:start_command_recording()?;
    commands:dispatch(pipeline, Root { node = head:device(), output = output:device() }, 3, 1, 1)?;
    commands:submit()?;
    let values = output:map()?;
    let mut i: u64 = 0;
    while (i < values.length) { assert(values:at(i) == 42); i = i + 1; };
    // A mapped slice and an element view refer to the same interior storage.
    let part = output:slice(1, 2);
    let part_host = part:map()?;
    part_host:at_mut(0) = 73;
    assert(values:at(1) == 73);
    let part_device = part:device();
    let output_device = output:device();
    assert(u64(part_device.data) == u64(output_device:lea(1)));
    let readonly = part:read_only();
    let read_host: Span<u32> = readonly:map()?;
    let read_device: Span<u32> = readonly:device();
    assert(read_host:at(0) == 73 && read_device.length == 2);
    let empty = gpu:alloc::<u32>(0)?;
    let empty_host = empty:map()?;
    assert(empty_host.length == 0);
    let unmapped = gpu:alloc_in::<u32>(1, memory_gpu)?;
    match (unmapped:map()) {
        Err(_) => {},
        SpanMut<u32>(_) => { assert(false); },
    };
    0
}
"#;

#[test]
fn pointer_graph_algorithm_remains_host_callable() {
    let project =
        support::project::Project::new(&support::module(GRAPH), Some("host_main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(feature = "gpu")]
#[test]
fn explicit_addresses_support_nested_graphs_mapping_and_workgroups() {
    let _lock = resin_runtime::testing::lock_gpu();
    match resin_runtime::ResinGpu::create() {
        Ok(_) => {}
        Err(
            resin_runtime::ResinStatus::Unsupported | resin_runtime::ResinStatus::VulkanUnavailable,
        ) => {
            assert_ne!(std::env::var("RESIN_REQUIRE_GPU").as_deref(), Ok("1"));
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
    let project = support::project::Project::new(&support::module(GRAPH), Some("main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mapping_preserves_permissions_and_rejects_managed_payloads() {
    for source in [
        "fn bad(p: Ref<GpuPtr<u32>>) { p:slice(0, 1); }",
        "fn bad(p: Ref<GpuPtrMut<u32>>) { p:slice(0, 1); }",
        "fn bad(p: Ref<GpuPtr<u32>>) -> PtrMut<u32> | Err<_> { p:map() }",
        "fn bad(p: Ref<GpuSpan<u32>>) -> SpanMut<u32> | Err<_> { p:map() }",
        "fn bad(p: Ref<GpuPtr<u32>>) -> PtrMut<u32> { p:device() }",
        "intrinsic \"gpu_view_map\" fn bad<T>(view: GpuView, count: u64) -> (PtrMut<T> | None, i32);",
        "fn bad(p: Ref<GpuPtr<StrongOwner>>) { p:device(); }",
    ] {
        let source = format!("import {{ \"$/gpu.resin\", \"$/span.resin\" }}; {source}");
        assert!(
            support::pipeline::source_module(&source).is_err(),
            "accepted {source}"
        );
    }
}
