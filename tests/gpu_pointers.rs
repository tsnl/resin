#![cfg(feature = "gpu")]

#[allow(dead_code)]
mod support;

use support::pipeline;
use support::project;

use resin_runtime::{ResinGpu, ResinStatus, testing::lock_gpu};
use std::{
    fs,
    process::{Command, Output},
    rc::Rc,
};
use tempfile::TempDir;

fn run(source: &str) -> Option<Output> {
    run_with_gpu_library(source, None)
}

fn run_with_gpu_library(source: &str, library: Option<&str>) -> Option<Output> {
    let _gpu = gpu_singleton()?;
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, source).unwrap();
    if let Some(library) = library {
        fs::write(directory.path().join("gpu.resin"), library).unwrap();
    }
    let module =
        pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{source}\n{error}"));
    let project = project::Project::new(&module, Some("main")).unwrap();
    // This fixture owns its native cache; compilation can overlap other GPU tests.
    let executable = project.build_executable();
    let _lock = lock_gpu();
    Some(Command::new(executable.path()).output().unwrap())
}

fn gpu_singleton() -> Option<Rc<ResinGpu>> {
    // ResinGpu is not Send or Sync, so each test thread owns its lazy singleton.
    thread_local! {
        static GPU: Option<Rc<ResinGpu>> = {
            let _lock = lock_gpu();
            match ResinGpu::create() {
                Ok(gpu) => Some(Rc::new(gpu)),
                Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
                    assert!(
                        std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                        "a suitable Vulkan device is required"
                    );
                    None
                }
                Err(error) => panic!("GPU initialization failed: {error:?}"),
            }
        };
    }
    GPU.with(Clone::clone)
}

fn success(output: &Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn inferred_values_and_slices_keep_their_allocation_and_gpu_alive() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/span.resin", "$/shared.resin" };
        struct Pair { left: i32, right: i32, }
        fn value() -> (GpuPtr<i32> | Err<_>)  {
            let mut gpu = gpu_new()?;
            gpu:create(i32(42))
        }
        fn field() -> (GpuPtr<i32> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pair = gpu:alloc::<i32>(u64(2))?;
            { let borrowed = pair:at(u64(0)); borrowed:store(i32(3)) };
            { let borrowed = pair:at(u64(1)); borrowed:store(i32(5)) };
            (pair:at(u64(1)))
        }
        fn slice() -> (GpuSpan<i32> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<i32>(u64(5))?;
            let mut i: u64 = 0;
            while (i < values.length) { { let borrowed = values:at(i); borrowed:store(i32(i) + i32(10)) }; i = i + u64(1); };
            (values:slice(u64(1), u64(3)))
        }
        fn main() -> (i32 | Err<_>)  {
            let mut number = value()?;
            let mut member = field()?;
            let mut values = slice()?;
            let alias = values.data:clone();
            let mut tail = alias:slice(u64(1), u64(2));
            number:store(number:load() + i32(1));
            member:store(member:load() + i32(2));
            { let borrowed = tail:at(u64(0)); borrowed:store(i32(24)) };
            let copied_owner = arc_ptr_alloc([i32(0), i32(0), i32(0)])?; let copied: Ref<_> = copied_owner:get().*;
            { let borrowed = values:read_only(); borrowed:copy_to(Span<i32> { data = copied_owner:get():lea(u64(0)), length = u64(3) }) };
            (if (number:load() == i32(43) && member:load() == i32(7) && copied:at(u64(0)) == i32(11)
                && copied:at(u64(1)) == i32(24) && copied:at(u64(2)) == i32(13) && tail.length == u64(2)) { 0 } else { 1 })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn nested_host_owners_and_explicit_gpu_loads_preserve_allocation_lifetimes() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/shared.resin" };
        struct Item { value: i32,
            
            
        }
fn increment(self: RefMut<Item>)  { self.value = self.value + i32(1); }

fn read(self: Ref<Item>) -> i32  { self.value }

        struct Outer { item: Item, }
        fn field() -> (GpuPtr<Outer> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pointer = gpu:create(Outer { item = Item { value = i32(40) } })?;
            let mut owner = arc_ptr_alloc::<GpuPtr<Outer>>(pointer:clone())?;
            let indirect: Ref<_> = owner;
            let mut item = indirect:get().*:load();
            item.item:increment();
            let previous = indirect:get().*:replace(item);
            item = pointer:load();
            item.item.value = item.item:read() + previous.item.value - i32(39);
            pointer:store(item);
            (indirect:get().*:clone())
        }
        fn main() -> (i32 | Err<_>)  {
            let mut result = field()?;
            (if (result:load().item.value == i32(42)) { 0 } else { 1 })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn custom_allocators_must_return_the_requested_size_and_alignment() {
    for (bytes, value) in [
        ("u64(1)", "allocation.0!"),
        (
            "bytes + u64(1)",
            "invalid_offset(allocation.0!, u64(1), u64(0), u64(1))",
        ),
    ] {
        let library = allocator_library(bytes, value);
        for action in [
            "let mut values = gpu:alloc::<i64>(u64(2))?;",
            "let mut pipeline = gpu:create_compute_pipeline(kernel)?; let mut commands = gpu:start_command_recording()?; commands:dispatch(pipeline, Root { left = i64(1), right = i64(2) }, u32(1), u32(1), u32(1))?;",
        ] {
            let source = format!(
                r#"export {{ main }};
                import {{ "gpu.resin", "$/status.resin" }};
                struct Root {{ left: i64, right: i64, }}
                @compute_shader fn kernel(index: u64, root: Ptr<Root>)  {{}}
                fn main() -> (i32 | Err<_>)  {{
                    let mut gpu = gpu_new()?;
                    {action}(i32(0))
                }}
            "#
            );
            let Some(output) = run_with_gpu_library(&source, Some(&library)) else {
                return;
            };
            assert!(
                !output.status.success(),
                "invalid allocation was accepted: {bytes}; {value}; {action}"
            );
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("GPU"), "{error}");
        }
    }
}

fn allocator_library(bytes: &str, value: &str) -> String {
    let mut library = include_str!("../resin/gpu.resin")
        .replace("\"status.resin\"", "\"$/status.resin\"")
        .replace(
            "\"internal/status_codes.resin\"",
            "\"$/internal/status_codes.resin\"",
        )
        .replace("\"window.resin\"", "\"$/window.resin\"")
        .replace("\"shared.resin\"", "\"$/shared.resin\"")
        .replace("\"span.resin\"", "\"$/span.resin\"");
    let start = library.find("@gpu_allocator").unwrap();
    let end = start + library[start..].find("@gpu_compute_pipeline").unwrap();
    library.replace_range(start..end, &format!(r#"@gpu_allocator
        fn malloc(self: Ref<Gpu>, bytes: u64, alignment: u64, memory: i32) -> (GpuView | Err<RuntimeError>)  {{
            let mut allocation = gpu_view_allocate(self:native(), self.owner.owner, {bytes}, alignment, memory);
            runtime_status_from_code(allocation.1)?;
            ({value})
        }}
    "#));
    library.push_str("intrinsic \"gpu_view_offset\" fn invalid_offset(view: GpuView, bytes: u64, size: u64, alignment: u64) -> GpuView;\n");
    library
}

const COMPUTE: &str = r#"export { main };
    import { "$/gpu.resin", "$/status.resin", "$/span.resin", "$/shared.resin" };
    struct FieldsIncrementValues<T0, T1> { increment: T0, values: T1, }
struct Parameters { increment: u32, values: Span<u32>, }
fn clone<A, B>(value: Ref<FieldsIncrementValues<A, B>>) -> FieldsIncrementValues<A, B> {
    FieldsIncrementValues<A, B> { increment = value.increment, values = value.values:clone() }
}
    @compute_shader fn kernel(index: u64, root: Ptr<Parameters>)  {
        if (index < root.values.length) {
            let mut item: RefMut<u32> = root.values:at_mut(index);
            item = item + root.increment;
        };
    }
    fn main() -> (i32 | Err<_>)  {
        let mut gpu = gpu_new()?;
        let mut values = gpu:alloc::<u32>(u64(4))?;
        let mut i: u64 = 0;
        while (i < values.length) { { let element = values:at(i); element:store(u32(i)) }; i = i + u64(1); };
        let mut arguments = FieldsIncrementValues<_, _> { increment = u32(5), values = values:slice(u64(1), u64(2)) };
        let mut pipeline = gpu:create_compute_pipeline(kernel)?;
        ACTION
    }
"#;

#[test]
fn projected_scalar_and_span_arguments_dispatch_and_allow_readback_after_submit() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))?;
        commands:submit()?;
        let mut again = gpu:start_command_recording()?;
        again:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))?;
        again:submit()?;
        let result_owner = arc_ptr_alloc([u32(0), u32(0), u32(0), u32(0)])?; let result: Ref<_> = result_owner:get().*;
        values:copy_to(Span<u32> { data = result_owner:get():lea(u64(0)), length = u64(4) });
        (if (result:at(u64(0)) == u32(0) && result:at(u64(1)) == u32(11)
            && result:at(u64(2)) == u32(12) && result:at(u64(3)) == u32(3)) { 0 } else { 1 })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn generic_operator_overloads_execute_on_the_gpu() {
    let source = r#"export { main };
        import { "$/gpu.resin", "$/span.resin" };
        struct Cell<T> { value: T,
            
        }
fn __add__<T>(a: Cell<T>, b: Cell<T>) -> Cell<T>  { Cell<T> { value = a.value + b.value } }

        struct Parameters { values: Span<Cell<u32>>, }
        struct HostParameters { values: GpuSpan<Cell<u32>>, }
        fn add<T>(a: T, b: T) -> _  { a + b }
        @compute_shader fn kernel(index: u64, root: Ptr<Parameters>)  {
            if (index < root.values.length) {
                let mut cell: RefMut<Cell<u32>> = root.values:at_mut(index);
                cell = add(Cell<u32> { value = cell.value }, Cell<u32> { value = 40 });
            };
        }
        fn main() -> i32 | Err<_>  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<Cell<u32>>(3)?;
            let mut index: u64 = 0;
            while (index < values.length) {
                { let borrowed = values:at(index); borrowed:store(Cell<u32> { value = u32(index) }) };
                index = index + 1;
            };
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(pipeline, HostParameters { values = values:clone() }, u32(1), u32(1), u32(1))?;
            commands:submit()?;
            if ({ let borrowed = values:at(0); borrowed:load() }.value == 40 && { let borrowed = values:at(1); borrowed:load() }.value == 41
                && { let borrowed = values:at(2); borrowed:load() }.value == 42) { 0 } else { 1 }
        }
    "#;
    let Some(output) = run(source) else { return };
    success(&output);
}

#[test]
fn source_sequences_project_offsets_and_retain_resources_through_submit() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/span.resin" };
        struct FieldsValuesScalar<T0, T1> { values: T0, scalar: T1, }
struct Root { values: Span<u32>, scalar: Ptr<u32>, }
        @compute_shader fn kernel(index: u64, root: Ptr<Root>)  {
            if (index < root.values.length) { root.values:at_mut(index) = root.values:at(index) + u32(10); };
            if (index == u64(0)) { root.scalar.* = u32(42); };
        }
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<u32>(u64(5))?;
            let mut i: u64 = 0;
            while (i < u64(5)) { { let borrowed = values:at(i); borrowed:store(u32(i)) }; i = i + u64(1); };
            let mut scalar = gpu:create(u32(0))?;
            let mut commands = gpu:start_command_recording()?;
            {
                let mut pipeline = gpu:create_compute_pipeline(kernel)?;
                commands:dispatch(pipeline, FieldsValuesScalar<_, _> { values = values:slice(u64(2), u64(2)), scalar = scalar:clone() }, u32(2), u32(1), u32(1))?;
            };
            commands:submit()?;
            (if ({ let borrowed = values:at(u64(1)); borrowed:load() } == u32(1) && { let borrowed = values:at(u64(2)); borrowed:load() } == u32(12) && { let borrowed = values:at(u64(3)); borrowed:load() } == u32(13) && { let borrowed = values:at(u64(4)); borrowed:load() } == u32(4) && scalar:load() == u32(42)) { i32(0) } else { i32(1) })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn source_pipeline_contract_retagging_cannot_change_the_shader_root() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin" };
        struct FieldsValue<T0> { value: T0, }
struct Root { value: u32, }
        struct Other { value: u32, }
        @compute_shader fn kernel(index: u64, root: Ptr<Root>)  {}
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut forged = GpuComputePipeline<Other, GpuPipelineOwner> { contract = pipeline.contract };
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(forged, FieldsValue<_> { value = u32(0) }, u32(1), u32(1), u32(1))?;
            (i32(0))
        }
    "#) else {
        return;
    };
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("pipeline contract"));
}

#[test]
fn failed_dispatch_and_cancel_restore_cpu_access_and_last_command_alias_cancels() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"let mut cancelled = gpu:start_command_recording()?;
        cancelled:cancel();
        let mut failed = match (cancelled:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))) {
            ()(value) => { i32(0) }, Err(error) => { runtime_status_code(error) },
        };
        { let element = values:at(u64(1)); element:store(u32(20)) };
        let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))?;
        let mut alias = commands;
        alias:cancel();
        { let element = values:at(u64(1)); element:store(u32(21)) };
        {
            let mut abandoned = gpu:start_command_recording()?;
            let mut last = abandoned;
            last:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))?;
        };
        (if (failed == i32(1) && { let element = values:at(u64(1)); element:load() } == u32(21)) { 0 } else { 1 })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn recorded_gpu_work_denies_cpu_access_through_all_aliases() {
    for access in [
        "let mut value = { let element = values:at(u64(0)); element:load() };",
        "{ let element = values:at(u64(0)); element:store(u32(7)) };",
        "let source = arc_ptr_alloc([u32(7)])?; let view = Span<u32> { data = source:get():lea(u64(0)), length = u64(1) }; values:copy_from(view);",
    ] {
        let source = COMPUTE.replace(
            "ACTION",
            &format!(
                r#"let mut commands = gpu:start_command_recording()?;
            commands:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))?;
            {access}
            (i32(0))
        "#
            ),
        );
        let Some(output) = run(&source) else { return };
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("retained by a recording"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn restricted_gpu_pointers_and_spans_trap_on_disallowed_access() {
    for access in [
        "let restricted = number:read_only(); restricted:store(i32(8));",
        "let restricted = number:write_only(); let mut value = restricted:load();",
        "let restricted = number:read_only(); restricted:replace(i32(8));",
        "let restricted = number:write_only(); restricted:replace(i32(8));",
        "let restricted = values:read_only(); let element = restricted:at(0); element:store(8);",
        "let restricted = values:write_only(); let element = restricted:at(0); let mut value = element:load();",
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin" }};
            fn main() -> (i32 | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut number = gpu:create(i32(7))?;
                let mut values = gpu:alloc::<i32>(u64(1))?;
                {{ let element = values:at(u64(0)); element:store(i32(7)) }};
                {access}(i32(0))
            }}
        "#
        );
        let Some(output) = run(&source) else { return };
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("access permission denied"),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn inferred_signed_long_pointers_project_and_precomputed_inputs_evaluate_once() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/span.resin", "$/shared.resin" };
        struct FieldsValueValuesIncrement<T0, T1, T2> { value: T0, values: T1, increment: T2, }
struct Parameters { value: Ptr<i64>, values: Span<i64>, increment: i64, }
        @compute_shader fn kernel(index: u64, root: Ptr<Parameters>)  {
            if (index == u64(0) && root.value.* < i64(0)) {
                root.value.* = -root.value.* + root.increment;
                root.values:at_mut(u64(0)) = root.value.* * i64(2);
            };
        }
        fn allocation(gpu: Gpu, calls: Ptr<i32>) -> (Gpu, u64)  {
            calls.* = calls.* + i32(1);
            (gpu, u64(1))
        }
        fn launch(pipeline: GpuComputePipeline<Parameters, GpuPipelineOwner>, value: GpuPtr<i64>, values: GpuSpan<i64>, calls: Ptr<i32>) -> _  {
            calls.* = calls.* + i32(1);
            (pipeline, FieldsValueValuesIncrement<_, _, _> { value = value, values = values, increment = i64(7) }, u32(1), u32(1), u32(1))
        }
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let calls_owner = arc_ptr_alloc(i32(0))?; let calls: Ref<_> = calls_owner:get().*;
            let mut value = gpu:create(-42)?;
            let mut allocation_request = allocation(gpu:clone(), calls_owner:get());
            let mut values = allocation_request.0:alloc::<i64>(allocation_request.1)?;
            { let borrowed = values:at(u64(0)); borrowed:store(i64(0)) };
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut commands = gpu:start_command_recording()?;
            let mut launch_request = launch(pipeline, value:clone(), values:clone(), calls_owner:get());
            commands:dispatch(launch_request.0, launch_request.1, launch_request.2, launch_request.3, launch_request.4)?;
            commands:submit()?;
            (if (calls == i32(2) && value:load() == i64(49) && { let borrowed = values:at(u64(0)); borrowed:load() } == i64(98)) { 0 } else { 1 })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn returned_typed_pipelines_and_recordings_keep_scoped_resources_alive() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/span.resin", "$/shared.resin" };
        struct FieldsCommandsValues<T0, T1> { commands: T0, values: T1, }
struct FieldsValues<T0> { values: T0, }
struct Parameters { values: Span<u32>, }
        @compute_shader fn kernel(index: u64, root: Ptr<Parameters>)  {
            if (index < root.values.length) { root.values:at_mut(index) = u32(42); };
        }
        fn make_pipeline(gpu: Ref<Gpu>) -> (GpuComputePipeline<Parameters, GpuPipelineOwner> | Err<_>)  {
            gpu:create_compute_pipeline(kernel)
        }
        fn record() -> (FieldsCommandsValues<GpuCommands, GpuSpan<u32>> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<u32>(u64(1))?;
            { let borrowed = values:at(u64(0)); borrowed:store(u32(0)) };
            let mut pipeline = make_pipeline(gpu)?;
            let alias = pipeline:clone();
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(alias, FieldsValues<_> { values = values:clone() }, u32(1), u32(1), u32(1))?;
            (FieldsCommandsValues<_, _> { commands = commands, values = values })
        }
        fn main() -> (i32 | Err<_>)  {
            let mut recorded = record()?;
            recorded.commands:submit()?;
            (if ({ let borrowed = recorded.values:at(u64(0)); borrowed:load() } == u32(42)) { i32(0) } else { i32(1) })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn pipelines_from_another_device_fail_recording_without_locking_arguments() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"let mut other = gpu_new()?;
        let mut commands = other:start_command_recording()?;
        let mut failed = match (commands:dispatch(pipeline, arguments:clone(), u32(1), u32(1), u32(1))) {
            ()(value) => { i32(0) }, Err(error) => { runtime_status_code(error) },
        };
        { let element = values:at(u64(1)); element:store(u32(42)) };
        commands:cancel();
        (if (failed == i32(1) && { let element = values:at(u64(1)); element:load() } == u32(42)) { i32(0) } else { i32(1) })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn dispatch_rejects_argument_views_from_another_device() {
    let source = COMPUTE.replace("ACTION", r#"let mut other = gpu_new()?;
        let mut foreign_values = other:alloc::<u32>(u64(1))?;
        let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, FieldsIncrementValues<_, _> { increment = u32(5), values = foreign_values }, u32(1), u32(1), u32(1))?;
        (i32(0))
    "#);
    let Some(output) = run(&source) else { return };
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("different device"));
}

#[test]
fn rooted_graphics_stages_receive_automatically_projected_arguments() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/graphics.resin" };
        struct FieldsColorOffset<T0, T1> { color: T0, offset: T1, }
struct Parameters { color: Ptr<Color>, offset: f32, }
        @vertex_shader fn vertex(index: i32, root: Ptr<Parameters>) -> Vertex  {
            Vertex {
                position = Position {
                    x = (if (index == i32(1)) { f32(3.0) } else { f32(-1.0) }) + root.offset,
                    y = if (index == i32(2)) { f32(3.0) } else { f32(-1.0) },
                    z = f32(0.0), w = f32(1.0),
                },
                color = Color { r = f32(1.0), g = f32(1.0), b = f32(1.0), a = f32(1.0) },
            }
        }
        @fragment_shader fn fragment(color: Color, root: Ptr<Parameters>) -> Color  {
            Color { r = root.color.r, g = root.color.g, b = root.color.b, a = root.color.a }
        }
        fn make_pipeline(gpu: Ref<Gpu>) -> (GpuGraphicsPipeline<Parameters, GpuPipelineOwner> | Err<_>)  {
            gpu:create_graphics_pipeline(vertex, fragment)
        }
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pipeline = make_pipeline(gpu)?;
            let mut color = gpu:create(Color { r = f32(1.0), g = f32(0.0), b = f32(0.0), a = f32(1.0) })?;
            let mut image = gpu:create_image(u32(8), u32(8))?;
            let mut pixels = gpu:alloc::<u8>(u64(256))?;
            let mut commands = gpu:start_command_recording()?;
            commands:begin_rendering(image, f32(0.0), f32(0.0), f32(0.0), f32(1.0))?;
            commands:draw(pipeline, FieldsColorOffset<_, _> { color = color, offset = f32(0.0) }, u32(3))?;
            commands:end_rendering()?;
            commands:copy_image_to_buffer(image, pixels)?;
            commands:submit()?;
            (if ({ let borrowed = pixels:at(u64(0)); borrowed:load() } == u8(255) && { let borrowed = pixels:at(u64(1)); borrowed:load() } == u8(0)
                && { let borrowed = pixels:at(u64(2)); borrowed:load() } == u8(0) && { let borrowed = pixels:at(u64(3)); borrowed:load() } == u8(255)) { i32(0) } else { i32(1) })
        }
    "#) else {
        return;
    };
    success(&output);
}

const VIEW_PRIMITIVES: &str = r#"intrinsic "gpu_view_allocate" fn allocate<N>(gpu: Ptr<N>, owner: StrongOwner, bytes: u64, alignment: u64, memory: i32) -> (GpuView | None, i32);
    intrinsic "gpu_view_offset" fn offset(view: GpuView, bytes: u64, size: u64, alignment: u64) -> GpuView;
    intrinsic "gpu_view_restrict" fn restrict(view: GpuView, access: u32) -> GpuView;
    intrinsic "gpu_view_load" fn load<T>(view: GpuView) -> T;
    intrinsic "gpu_view_store" fn store<T>(view: GpuView, value: T) -> ();
    intrinsic "gpu_view_replace" fn replace<T>(view: GpuView, value: T) -> T;
    intrinsic "gpu_view_copy_to" fn copy_to<T>(view: GpuView, count: u64, destination: Ptr<T>, length: u64) -> ();
    intrinsic "gpu_view_copy_from" fn copy_from<T>(view: GpuView, capacity: u64, source: Ptr<T>, count: u64) -> ();
    struct DeviceScalar<T> {
        view: GpuView,
        
        
    }
fn read<T>(self: Ref<DeviceScalar<T>>) -> T  { load::<T>(self.view) }

fn write<T>(self: Ref<DeviceScalar<T>>, value: T)  { store(self.view, value); }

    fn allocate_ints(count: u64) -> (GpuView | Err<RuntimeError>)  {
        let mut gpu = gpu_new()?;
        let mut allocated = allocate(gpu:native(), gpu.owner.owner, count * size_of(i32), align_of(i32), 0);
        runtime_status_from_code(allocated.1)?;
        (allocated.0!)
    }
"#;

#[test]
fn gpu_view_primitives_keep_owners_offsets_and_typed_source_methods() {
    let source = format!(
        r#"export {{ main }};
        import {{ "$/shared.resin", "$/gpu.resin", "$/status.resin" }};
        {VIEW_PRIMITIVES}
        fn main() -> (i32 | Err<_>)  {{
            let mut original = allocate_ints(3)?;
            let mut first = DeviceScalar<i32> {{ view = original }};
            let mut second = DeviceScalar<i32> {{ view = offset(original, 4, 4, 4) }};
            first:write(7);
            second:write(11);
            let mut previous = replace(offset(original, 4, 4, 4), i32(42));
            store(offset(original, 8, 4, 4), i32(19));
            let copied_owner = arc_ptr_alloc([i32(0), i32(0), i32(0)])?; let copied: Ref<_> = copied_owner:get().*;
            copy_to(restrict(original, 1), 3, copied_owner:get():lea(0), 3);
            (if (first:read() == 7 && second:read() == 42 && previous == 11 && copied:at(2) == 19) {{ 0 }} else {{ 1 }})
        }}
    "#
    );
    let Some(output) = run(&source) else {
        return;
    };
    success(&output);
}

#[test]
fn gpu_view_primitives_preserve_access_bounds_and_alignment_checks() {
    for (operation, diagnostic) in [
        ("store(restrict(view, 1), i32(8));", "permission"),
        (
            "let mut value = load::<i32>(restrict(view, 2));",
            "permission",
        ),
        (
            "let mut value = replace(restrict(view, 1), i32(8));",
            "permission",
        ),
        ("let mut value = offset(view, 1, 4, 4);", "misaligned"),
        ("let mut value = offset(view, 8, 4, 4);", "out of bounds"),
        ("copy_to(view, 2, result:get():lea(0), 1);", "too short"),
        ("copy_from(view, 1, result:get():lea(0), 2);", "too short"),
        (
            "copy_from(restrict(view, 1), 2, result:get():lea(0), 2);",
            "permission",
        ),
        (
            "copy_from(view, 3, result:get():lea(0), 3);",
            "out of bounds",
        ),
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin", "$/status.resin", "$/shared.resin" }};
            {VIEW_PRIMITIVES}
            fn main() -> (i32 | Err<_>)  {{
                let mut view = allocate_ints(2)?;
                // Keep the host source valid when testing a three-element copy
                // against the shorter GPU allocation.
                let result = arc_ptr_alloc([i32(0), i32(0), i32(0)])?;
                {operation}(0)
            }}
        "#
        );
        let Some(output) = run(&source) else {
            return;
        };
        assert!(!output.status.success(), "{operation}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(diagnostic),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn gpu_allocation_rejects_managed_and_opaque_elements_before_access() {
    for (element, diagnostic) in [
        ("ArcPtr<i32>", "plain shared storage"),
        ("WeakPtr<i32>", "plain shared storage"),
        ("Ptr<Native>", "plain shared storage"),
        ("Native", "no shared host/device layout"),
    ] {
        let source = format!(
            r#"import {{ "$/gpu.resin", "$/shared.resin" }};
            extern type Native;
            fn invalid(gpu: Gpu) -> (() | Err<_>)  {{
                gpu:alloc::<{element}>(u64(0))?;
                (())
            }}
        "#
        );
        let error = pipeline::source_module(&source).unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{element}: {error}");
    }
}

#[test]
fn gpu_sequences_allow_empty_tail_views_and_report_allocation_overflow() {
    let Some(output) = run(r#"export { main };
        import { "$/gpu.resin", "$/span.resin", "$/status.resin" };
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<u32>(u64(3))?;
            let mut empty = values:slice(u64(3), u64(0));
            let mut alias = empty.data:slice(u64(0), u64(0));
            alias:copy_to(Span<u32> { data = Ptr<u32>(u64(0)), length = u64(0) });
            let mut zero = gpu:alloc::<u32>(u64(0))?;
            zero:copy_to(Span<u32> { data = Ptr<u32>(u64(0)), length = u64(0) });
            { let borrowed = Span<u32> { data = Ptr<u32>(u64(0)), length = u64(0) }; alias:copy_from(borrowed) };
            { let borrowed = Span<u32> { data = Ptr<u32>(u64(0)), length = u64(0) }; zero:copy_from(borrowed) };
            let mut overflow = match (gpu:alloc::<u32>(u64(18446744073709551615))) {
                GpuSpan<u32>(allocated) => { i32(0) },
                Err(error) => { runtime_status_code(error) },
            };
            (if (empty.length == u64(0) && alias.length == u64(0) && zero.length == u64(0) && overflow == i32(4)) { i32(0) } else { i32(1) })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn gpu_sequences_reject_out_of_bounds_indices_and_overflowing_ranges() {
    for operation in [
        "values:at(u64(3));",
        "values:at(u64(18446744073709551615));",
        "let tail = values:slice(3, 0); tail:at(0);",
        "values:slice(u64(4), u64(0));",
        "values:slice(u64(2), u64(2));",
        "values:slice(u64(18446744073709551615), u64(2));",
        "values.data:slice(u64(18446744073709551615), u64(1));",
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin" }};
            fn main() -> (() | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut values = gpu:alloc::<u32>(u64(3))?;
                {operation}(())
            }}
        "#
        );
        let Some(output) = run(&source) else {
            return;
        };
        assert_eq!(output.status.code(), Some(1), "{operation}");
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .lines()
                .any(|line| line == "resin: GPU view out of bounds"),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn host_upload_copies_only_the_source_length_into_a_gpu_slice() {
    let Some(output) = run(r#"export { main };
        import { "$/shared.resin", "$/gpu.resin", "$/span.resin" };
        fn main() -> () | Err<_> {
            let gpu = gpu_new()?;
            let values = gpu:alloc::<u32>(u64(5))?;
            let initial_owner = arc_ptr_alloc([u32(10), u32(20), u32(30), u32(40), u32(50)])?; let initial: Ref<_> = initial_owner:get().*;
            { let borrowed = Span<u32> { data = initial_owner:get():lea(u64(0)), length = u64(5) }; values:copy_from(borrowed) };
            let patch_owner = arc_ptr_alloc([u32(7), u32(8)])?; let patch: Ref<_> = patch_owner:get().*;
            { let borrowed_1 = { let borrowed = values:slice(u64(1), u64(3)); borrowed:write_only() }; { let borrowed = Span<u32> { data = patch_owner:get():lea(u64(0)), length = u64(2) }; borrowed_1:copy_from(borrowed) } };
            let result_owner = arc_ptr_alloc([u32(0), u32(0), u32(0), u32(0), u32(0)])?; let result: Ref<_> = result_owner:get().*;
            values:copy_to(Span<u32> { data = result_owner:get():lea(u64(0)), length = u64(5) });
            assert(result:at(u64(0)) == u32(10) && result:at(u64(1)) == u32(7) && result:at(u64(2)) == u32(8));
            assert(result:at(u64(3)) == u32(40) && result:at(u64(4)) == u32(50));
        }
    "#) else {
        return;
    };
    success(&output);
}
