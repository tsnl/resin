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
        struct Pair { left: int, right: int, }
        fn value() -> (GpuPtr<int> | Err<_>)  {
            let mut gpu = gpu_new()?;
            gpu:create(42_i)
        }
        fn field() -> (GpuPtr<int> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pair = gpu:alloc::<int>(2_ul)?;
            pair:at(0_ul):store(3_i);
            pair:at(1_ul):store(5_i);
            (pair:at(1_ul))
        }
        fn slice() -> (GpuSpan<int> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<int>(5_ul)?;
            let mut i = 0_ul;
            while (i < values.length) { values:at(i):store(int(i) + 10_i); i = i + 1_ul; };
            (values:slice(1_ul, 3_ul))
        }
        fn main() -> (int | Err<_>)  {
            let mut number = value()?;
            let mut member = field()?;
            let mut values = slice()?;
            let alias = values.data:clone();
            let mut tail = alias:slice(1_ul, 2_ul);
            number:store(number:load() + 1_i);
            member:store(member:load() + 2_i);
            tail:at(0_ul):store(24_i);
            let mut copied = [0_i, 0_i, 0_i];
            values:read_only():copy_to(Span<int> { data = &copied:at(0_ul), length = 3_ul });
            (if (number:load() == 43_i && member:load() == 7_i && copied:at(0_ul) == 11_i
                && copied:at(1_ul) == 24_i && copied:at(2_ul) == 13_i && tail.length == 2_ul) { 0 } else { 1 })
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
        struct Item { value: int,
            
            
        }
fn increment(self: Ref<Item>)  { self.value = self.value + 1_i; }

fn read(self: Ref<Item>) -> int  { self.value }

        struct Outer { item: Item, }
        fn field() -> (GpuPtr<Outer> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pointer = gpu:create(Outer { item = Item { value = 40_i } })?;
            let mut owner = arc_ptr_alloc::<GpuPtr<Outer>>(pointer:clone())?;
            let mut indirect = &owner;
            let mut item = indirect.*:get().*:load();
            item.item:increment();
            let previous = indirect.*:get().*:replace(item);
            item = pointer:load();
            item.item.value = item.item:read() + previous.item.value - 39_i;
            pointer:store(item);
            (indirect.*:get().*:clone())
        }
        fn main() -> (int | Err<_>)  {
            let mut result = field()?;
            (if (result:load().item.value == 42_i) { 0 } else { 1 })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn custom_allocators_must_return_the_requested_size_and_alignment() {
    for (bytes, value) in [
        ("1_ul", "allocation.0!"),
        (
            "bytes + 1_ul",
            "invalid_offset(allocation.0!, 1_ul, 0_ul, 1_ul)",
        ),
    ] {
        let library = allocator_library(bytes, value);
        for action in [
            "let mut values = gpu:alloc::<long>(2_ul)?;",
            "let mut pipeline = gpu:create_compute_pipeline(kernel)?; let mut commands = gpu:start_command_recording()?; commands:dispatch(pipeline, Root { left = 1_l, right = 2_l }, 1_ui, 1_ui, 1_ui)?;",
        ] {
            let source = format!(
                r#"export {{ main }};
                import {{ "gpu.resin", "$/status.resin" }};
                struct Root {{ left: long, right: long, }}
                @compute_shader fn kernel(index: ulong, root: Ptr<Root>)  {{}}
                fn main() -> (int | Err<_>)  {{
                    let mut gpu = gpu_new()?;
                    {action}(0_i)
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
        fn malloc(self: Ref<Gpu>, bytes: ulong, alignment: ulong, memory: int) -> (GpuView | Err<RuntimeError>)  {{
            let mut allocation = gpu_view_allocate(self:native(), self.owner.owner, {bytes}, alignment, memory);
            runtime_status_from_code(allocation.1)?;
            ({value})
        }}
    "#));
    library.push_str("intrinsic \"gpu_view_offset\" fn invalid_offset(view: GpuView, bytes: ulong, size: ulong, alignment: ulong) -> GpuView;\n");
    library
}

const COMPUTE: &str = r#"export { main };
    import { "$/gpu.resin", "$/status.resin", "$/span.resin" };
    struct FieldsIncrementValues<T0, T1> { increment: T0, values: T1, }
struct Parameters { increment: uint, values: Span<uint>, }
fn clone<A, B>(value: Ref<FieldsIncrementValues<A, B>>) -> FieldsIncrementValues<A, B> {
    FieldsIncrementValues<A, B> { increment = value.increment, values = value.values:clone() }
}
    @compute_shader fn kernel(index: ulong, root: Ptr<Parameters>)  {
        if (index < root.values.length) {
            let mut item: Ref<uint> = root.values:at(index);
            item = item + root.increment;
        };
    }
    fn main() -> (int | Err<_>)  {
        let mut gpu = gpu_new()?;
        let mut values = gpu:alloc::<uint>(4_ul)?;
        let mut i = 0_ul;
        while (i < values.length) { values:at(i):store(uint(i)); i = i + 1_ul; };
        let mut arguments = FieldsIncrementValues<_, _> { increment = 5_ui, values = values:slice(1_ul, 2_ul) };
        let mut pipeline = gpu:create_compute_pipeline(kernel)?;
        ACTION
    }
"#;

#[test]
fn projected_scalar_and_span_arguments_dispatch_and_allow_readback_after_submit() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)?;
        commands:submit()?;
        let mut again = gpu:start_command_recording()?;
        again:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)?;
        again:submit()?;
        let mut result = [0_ui, 0_ui, 0_ui, 0_ui];
        values:copy_to(Span<uint> { data = &result:at(0_ul), length = 4_ul });
        (if (result:at(0_ul) == 0_ui && result:at(1_ul) == 11_ui
            && result:at(2_ul) == 12_ui && result:at(3_ul) == 3_ui) { 0 } else { 1 })
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

        struct Parameters { values: Span<Cell<uint>>, }
        struct HostParameters { values: GpuSpan<Cell<uint>>, }
        fn add<T>(a: T, b: T) -> _  { a + b }
        @compute_shader fn kernel(index: ulong, root: Ptr<Parameters>)  {
            if (index < root.values.length) {
                let mut cell: Ref<Cell<uint>> = root.values:at(index);
                cell = add(Cell<uint> { value = cell.value }, Cell<uint> { value = 40 });
            };
        }
        fn main() -> int | Err<_>  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<Cell<uint>>(3)?;
            let mut index = 0_ul;
            while (index < values.length) {
                values:at(index):store(Cell<uint> { value = uint(index) });
                index = index + 1;
            };
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(pipeline, HostParameters { values = values:clone() }, 1_ui, 1_ui, 1_ui)?;
            commands:submit()?;
            if (values:at(0):load().value == 40 && values:at(1):load().value == 41
                && values:at(2):load().value == 42) { 0 } else { 1 }
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
struct Root { values: Span<uint>, scalar: Ptr<uint>, }
        @compute_shader fn kernel(index: ulong, root: Ptr<Root>)  {
            if (index < root.values.length) { root.values:at(index) = root.values:at(index) + 10_ui; };
            if (index == 0_ul) { root.scalar.* = 42_ui; };
        }
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<uint>(5_ul)?;
            let mut i = 0_ul;
            while (i < 5_ul) { values:at(i):store(uint(i)); i = i + 1_ul; };
            let mut scalar = gpu:create(0_ui)?;
            let mut commands = gpu:start_command_recording()?;
            {
                let mut pipeline = gpu:create_compute_pipeline(kernel)?;
                commands:dispatch(pipeline, FieldsValuesScalar<_, _> { values = values:slice(2_ul, 2_ul), scalar = scalar:clone() }, 2_ui, 1_ui, 1_ui)?;
            };
            commands:submit()?;
            (if (values:at(1_ul):load() == 1_ui && values:at(2_ul):load() == 12_ui && values:at(3_ul):load() == 13_ui && values:at(4_ul):load() == 4_ui && scalar:load() == 42_ui) { 0_i } else { 1_i })
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
struct Root { value: uint, }
        struct Other { value: uint, }
        @compute_shader fn kernel(index: ulong, root: Ptr<Root>)  {}
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut forged = GpuComputePipeline<Other, GpuPipelineOwner> { contract = pipeline.contract };
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(forged, FieldsValue<_> { value = 0_ui }, 1_ui, 1_ui, 1_ui)?;
            (0_i)
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
        let mut failed = match (cancelled:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)) {
            ()(value) => { 0_i }, Err(error) => { runtime_status_code(error) },
        };
        values:at(1_ul):store(20_ui);
        let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)?;
        let mut alias = commands;
        alias:cancel();
        values:at(1_ul):store(21_ui);
        {
            let mut abandoned = gpu:start_command_recording()?;
            let mut last = abandoned;
            last:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)?;
        };
        (if (failed == 1_i && values:at(1_ul):load() == 21_ui) { 0 } else { 1 })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn recorded_gpu_work_denies_cpu_access_through_all_aliases() {
    for access in [
        "let mut value = values:at(0_ul):load();",
        "values:at(0_ul):store(7_ui);",
        "let source = [7_ui]; values:copy_from(Span<uint> { data = &source:at(0_ul), length = 1_ul });",
    ] {
        let source = COMPUTE.replace(
            "ACTION",
            &format!(
                r#"let mut commands = gpu:start_command_recording()?;
            commands:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)?;
            {access}
            (0_i)
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
        "number:read_only():store(8_i);",
        "let mut value = number:write_only():load();",
        "number:read_only():replace(8_i);",
        "number:write_only():replace(8_i);",
        "values:read_only():at(0_ul):store(8_i);",
        "let mut value = values:write_only():at(0_ul):load();",
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin" }};
            fn main() -> (int | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut number = gpu:create(7_i)?;
                let mut values = gpu:alloc::<int>(1_ul)?;
                values:at(0_ul):store(7_i);
                {access}(0_i)
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
struct Parameters { value: Ptr<long>, values: Span<long>, increment: long, }
        @compute_shader fn kernel(index: ulong, root: Ptr<Parameters>)  {
            if (index == 0_ul && root.value.* < 0_l) {
                root.value.* = -root.value.* + root.increment;
                root.values:at(0_ul) = root.value.* * 2_l;
            };
        }
        fn allocation(gpu: Gpu, calls: Ptr<int>) -> (Gpu, ulong)  {
            calls.* = calls.* + 1_i;
            (gpu, 1_ul)
        }
        fn launch(pipeline: GpuComputePipeline<Parameters, GpuPipelineOwner>, value: GpuPtr<long>, values: GpuSpan<long>, calls: Ptr<int>) -> _  {
            calls.* = calls.* + 1_i;
            (pipeline, FieldsValueValuesIncrement<_, _, _> { value = value, values = values, increment = 7_l }, 1_ui, 1_ui, 1_ui)
        }
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut calls = 0_i;
            let mut value = gpu:create(-42)?;
            let mut allocation_request = allocation(gpu:clone(), &calls);
            let mut values = allocation_request.0:alloc::<long>(allocation_request.1)?;
            values:at(0_ul):store(0_l);
            let mut pipeline = gpu:create_compute_pipeline(kernel)?;
            let mut commands = gpu:start_command_recording()?;
            let mut launch_request = launch(pipeline, value:clone(), values:clone(), &calls);
            commands:dispatch(launch_request.0, launch_request.1, launch_request.2, launch_request.3, launch_request.4)?;
            commands:submit()?;
            (if (calls == 2_i && value:load() == 49_l && values:at(0_ul):load() == 98_l) { 0 } else { 1 })
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
struct Parameters { values: Span<uint>, }
        @compute_shader fn kernel(index: ulong, root: Ptr<Parameters>)  {
            if (index < root.values.length) { root.values:at(index) = 42_ui; };
        }
        fn make_pipeline(gpu: Ref<Gpu>) -> (GpuComputePipeline<Parameters, GpuPipelineOwner> | Err<_>)  {
            gpu:create_compute_pipeline(kernel)
        }
        fn record() -> (FieldsCommandsValues<GpuCommands, GpuSpan<uint>> | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<uint>(1_ul)?;
            values:at(0_ul):store(0_ui);
            let mut pipeline = make_pipeline(gpu)?;
            let alias = pipeline:clone();
            let mut commands = gpu:start_command_recording()?;
            commands:dispatch(alias, FieldsValues<_> { values = values:clone() }, 1_ui, 1_ui, 1_ui)?;
            (FieldsCommandsValues<_, _> { commands = commands, values = values })
        }
        fn main() -> (int | Err<_>)  {
            let mut recorded = record()?;
            recorded.commands:submit()?;
            (if (recorded.values:at(0_ul):load() == 42_ui) { 0_i } else { 1_i })
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
        let mut failed = match (commands:dispatch(pipeline, arguments:clone(), 1_ui, 1_ui, 1_ui)) {
            ()(value) => { 0_i }, Err(error) => { runtime_status_code(error) },
        };
        values:at(1_ul):store(42_ui);
        commands:cancel();
        (if (failed == 1_i && values:at(1_ul):load() == 42_ui) { 0_i } else { 1_i })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn dispatch_rejects_argument_views_from_another_device() {
    let source = COMPUTE.replace("ACTION", r#"let mut other = gpu_new()?;
        let mut foreign_values = other:alloc::<uint>(1_ul)?;
        let mut commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, FieldsIncrementValues<_, _> { increment = 5_ui, values = foreign_values }, 1_ui, 1_ui, 1_ui)?;
        (0_i)
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
struct Parameters { color: Ptr<Color>, offset: float32, }
        @vertex_shader fn vertex(index: int, root: Ptr<Parameters>) -> Vertex  {
            Vertex {
                position = Position {
                    x = (if (index == 1_i) { 3.0_f } else { -1.0_f }) + root.offset,
                    y = if (index == 2_i) { 3.0_f } else { -1.0_f },
                    z = 0.0_f, w = 1.0_f,
                },
                color = Color { r = 1.0_f, g = 1.0_f, b = 1.0_f, a = 1.0_f },
            }
        }
        @fragment_shader fn fragment(color: Color, root: Ptr<Parameters>) -> Color  {
            Color { r = root.color.r, g = root.color.g, b = root.color.b, a = root.color.a }
        }
        fn make_pipeline(gpu: Ref<Gpu>) -> (GpuGraphicsPipeline<Parameters, GpuPipelineOwner> | Err<_>)  {
            gpu:create_graphics_pipeline(vertex, fragment)
        }
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut pipeline = make_pipeline(gpu)?;
            let mut color = gpu:create(Color { r = 1.0_f, g = 0.0_f, b = 0.0_f, a = 1.0_f })?;
            let mut image = gpu:create_image(8_ui, 8_ui)?;
            let mut pixels = gpu:alloc::<ubyte>(256_ul)?;
            let mut commands = gpu:start_command_recording()?;
            commands:begin_rendering(image, 0.0_f, 0.0_f, 0.0_f, 1.0_f)?;
            commands:draw(pipeline, FieldsColorOffset<_, _> { color = color, offset = 0.0_f }, 3_ui)?;
            commands:end_rendering()?;
            commands:copy_image_to_buffer(image, pixels)?;
            commands:submit()?;
            (if (pixels:at(0_ul):load() == 255_ub && pixels:at(1_ul):load() == 0_ub
                && pixels:at(2_ul):load() == 0_ub && pixels:at(3_ul):load() == 255_ub) { 0_i } else { 1_i })
        }
    "#) else {
        return;
    };
    success(&output);
}

const VIEW_PRIMITIVES: &str = r#"intrinsic "gpu_view_allocate" fn allocate<N>(gpu: Ptr<N>, owner: StrongOwner, bytes: ulong, alignment: ulong, memory: int) -> (GpuView | None, int);
    intrinsic "gpu_view_offset" fn offset(view: GpuView, bytes: ulong, size: ulong, alignment: ulong) -> GpuView;
    intrinsic "gpu_view_restrict" fn restrict(view: GpuView, access: uint) -> GpuView;
    intrinsic "gpu_view_load" fn load<T>(view: GpuView) -> T;
    intrinsic "gpu_view_store" fn store<T>(view: GpuView, value: T) -> ();
    intrinsic "gpu_view_replace" fn replace<T>(view: GpuView, value: T) -> T;
    intrinsic "gpu_view_copy_to" fn copy_to<T>(view: GpuView, count: ulong, destination: Ptr<T>, length: ulong) -> ();
    intrinsic "gpu_view_copy_from" fn copy_from<T>(view: GpuView, capacity: ulong, source: Ptr<T>, count: ulong) -> ();
    struct DeviceScalar<T> {
        view: GpuView,
        
        
    }
fn read<T>(self: Ref<DeviceScalar<T>>) -> T  { load::<T>(self.view) }

fn write<T>(self: Ref<DeviceScalar<T>>, value: T)  { store(self.view, value); }

    fn allocate_ints(count: ulong) -> (GpuView | Err<RuntimeError>)  {
        let mut gpu = gpu_new()?;
        let mut allocated = allocate(gpu:native(), gpu.owner.owner, count * size_of(int), align_of(int), 0);
        runtime_status_from_code(allocated.1)?;
        (allocated.0!)
    }
"#;

#[test]
fn gpu_view_primitives_keep_owners_offsets_and_typed_source_methods() {
    let source = format!(
        r#"export {{ main }};
        import {{ "$/gpu.resin", "$/status.resin" }};
        {VIEW_PRIMITIVES}
        fn main() -> (int | Err<_>)  {{
            let mut original = allocate_ints(3)?;
            let mut first = DeviceScalar<int> {{ view = original }};
            let mut second = DeviceScalar<int> {{ view = offset(original, 4, 4, 4) }};
            first:write(7);
            second:write(11);
            let mut previous = replace(offset(original, 4, 4, 4), 42_i);
            store(offset(original, 8, 4, 4), 19_i);
            let mut copied = [0_i, 0_i, 0_i];
            copy_to(restrict(original, 1), 3, &copied:at(0), 3);
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
        ("store(restrict(view, 1), 8_i);", "permission"),
        (
            "let mut value = load::<int>(restrict(view, 2));",
            "permission",
        ),
        (
            "let mut value = replace(restrict(view, 1), 8_i);",
            "permission",
        ),
        ("let mut value = offset(view, 1, 4, 4);", "misaligned"),
        ("let mut value = offset(view, 8, 4, 4);", "out of bounds"),
        ("copy_to(view, 2, &result:at(0), 1);", "too short"),
        ("copy_from(view, 1, &result:at(0), 2);", "too short"),
        (
            "copy_from(restrict(view, 1), 2, &result:at(0), 2);",
            "permission",
        ),
        ("copy_from(view, 3, &result:at(0), 3);", "out of bounds"),
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin", "$/status.resin" }};
            {VIEW_PRIMITIVES}
            fn main() -> (int | Err<_>)  {{
                let mut view = allocate_ints(2)?;
                // Keep the host source valid when testing a three-element copy
                // against the shorter GPU allocation.
                let mut result = [0_i, 0_i, 0_i];
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
        ("ArcPtr<int>", "plain shared storage"),
        ("WeakPtr<int>", "plain shared storage"),
        ("Ptr<Native>", "plain shared storage"),
        ("Native", "no shared host/device layout"),
    ] {
        let source = format!(
            r#"import {{ "$/gpu.resin", "$/shared.resin" }};
            extern type Native;
            fn invalid(gpu: Gpu) -> (() | Err<_>)  {{
                gpu:alloc::<{element}>(0_ul)?;
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
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut values = gpu:alloc::<uint>(3_ul)?;
            let mut empty = values:slice(3_ul, 0_ul);
            let mut alias = empty.data:slice(0_ul, 0_ul);
            alias:copy_to(Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul });
            let mut zero = gpu:alloc::<uint>(0_ul)?;
            zero:copy_to(Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul });
            alias:copy_from(Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul });
            zero:copy_from(Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul });
            let mut overflow = match (gpu:alloc::<uint>(18446744073709551615_ul)) {
                GpuSpan<uint>(allocated) => { 0_i },
                Err(error) => { runtime_status_code(error) },
            };
            (if (empty.length == 0_ul && alias.length == 0_ul && zero.length == 0_ul && overflow == 4_i) { 0_i } else { 1_i })
        }
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn gpu_sequences_reject_out_of_bounds_indices_and_overflowing_ranges() {
    for operation in [
        "values:at(3_ul);",
        "values:at(18446744073709551615_ul);",
        "values:slice(3_ul, 0_ul):at(0_ul);",
        "values:slice(4_ul, 0_ul);",
        "values:slice(2_ul, 2_ul);",
        "values:slice(18446744073709551615_ul, 2_ul);",
        "values.data:slice(18446744073709551615_ul, 1_ul);",
    ] {
        let source = format!(
            r#"export {{ main }};
            import {{ "$/gpu.resin" }};
            fn main() -> (() | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut values = gpu:alloc::<uint>(3_ul)?;
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
        import { "$/gpu.resin", "$/span.resin" };
        fn main() -> () | Err<_> {
            let gpu = gpu_new()?;
            let values = gpu:alloc::<uint>(5_ul)?;
            let initial = [10_ui, 20_ui, 30_ui, 40_ui, 50_ui];
            values:copy_from(Span<uint> { data = &initial:at(0_ul), length = 5_ul });
            let patch = [7_ui, 8_ui];
            values:slice(1_ul, 3_ul):write_only():copy_from(Span<uint> { data = &patch:at(0_ul), length = 2_ul });
            let result = [0_ui, 0_ui, 0_ui, 0_ui, 0_ui];
            values:copy_to(Span<uint> { data = &result:at(0_ul), length = 5_ul });
            assert(result:at(0_ul) == 10_ui && result:at(1_ul) == 7_ui && result:at(2_ul) == 8_ui);
            assert(result:at(3_ul) == 40_ui && result:at(4_ul) == 50_ui);
        }
    "#) else {
        return;
    };
    success(&output);
}
