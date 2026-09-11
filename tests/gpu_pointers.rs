#![cfg(feature = "gpu")]

#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "support/project.rs"]
mod project;
#[path = "support/shaders.rs"]
mod shaders;

use resin_runtime::{ResinGpu, ResinStatus, testing::lock_gpu};
use std::{fs, process::Output};
use tempfile::TempDir;

fn run(source: &str) -> Option<Output> {
    run_with_gpu_library(source, None)
}

fn run_with_gpu_library(source: &str, library: Option<&str>) -> Option<Output> {
    let _lock = lock_gpu();
    match ResinGpu::create() {
        Ok(gpu) => drop(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            return None;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, source).unwrap();
    if let Some(library) = library {
        fs::write(directory.path().join("gpu.resin"), library).unwrap();
    }
    let module = pipeline::generate_program(&pipeline::load(&path).unwrap())
        .unwrap_or_else(|error| panic!("{source}\n{error}"));
    Some(project::Project::new(&module, Some("main")).unwrap().run())
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
fn inferred_values_fields_and_slices_keep_their_allocation_and_gpu_alive() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin" };
        struct Pair { left: int, right: int };
        def value() -> Result<GpuPtr<int>, _> = {
            var gpu = Gpu.new()?;
            gpu.new(42_i)
        };
        def field() -> Result<GpuPtr<int>, _> = {
            var gpu = Gpu.new()?;
            var pair = gpu.new(Pair { left = 3_i, right = 5_i })?;
            ok(&pair.right)
        };
        def slice() -> Result<GpuSpan<int>, _> = {
            var gpu = Gpu.new()?;
            var values = GpuSpan<int>.allocate(gpu, 5_ul)?;
            var i = 0_ul;
            while (i < values.length) { values.at(i).* := int(i) + 10_i; i := i + 1_ul; };
            ok(values.slice(1_ul, 3_ul))
        };
        def main() -> Result<int, _> = {
            var number = value()?;
            var member = field()?;
            var values = slice()?;
            var alias = values.data;
            var tail = alias.slice(1_ul, 2_ul);
            number.* := number.* + 1_i;
            member.* := member.* + 2_i;
            tail.at(0_ul).* := 24_i;
            var copied = [0_i, 0_i, 0_i];
            values.read_only().copy_to(Span<int> { data = copied.at(0_ul), length = 3_ul });
            ok(if (number.* == 43_i && member.* == 7_i && copied.at(0_ul).* == 11_i
                && copied.at(1_ul).* == 24_i && copied.at(2_ul).* == 13_i && tail.length == 2_ul) { 0 } else { 1 })
        };
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn nested_host_owners_and_gpu_method_receivers_preserve_allocation_lifetimes() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin" };
        struct Item { value: int };
        struct Outer { item: Item };
        impl Item {
            def increment(self: GpuPtr<Item>) = { self.value := self.value + 1_i; };
            def read(self: Item) -> int = { self.value };
        }
        def field() -> Result<GpuPtr<int>, _> = {
            var gpu = Gpu.new()?;
            var pointer = gpu.new(Outer { item = Item { value = 40_i } })?;
            var owner = Arc<GpuPtr<Outer>>(pointer);
            var indirect = &owner;
            indirect.item.increment();
            var field = &indirect.item.value;
            var previous = field.replace(42_i);
            field.* := pointer.item.read() + previous - 41_i;
            ok(field)
        };
        def main() -> Result<int, _> = {
            var result = field()?;
            ok(if (result.* == 42_i) { 0 } else { 1 })
        };
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn custom_allocators_must_return_the_requested_size_and_alignment() {
    for (bytes, value) in [
        ("1_ul", "allocation.value!"),
        ("bytes + 1_ul", "allocation.value!.at(1_ul)"),
    ] {
        let library = allocator_library(bytes, value);
        for action in [
            "var values = GpuSpan<long>.allocate(gpu, 2_ul)?;",
            "var pipeline = gpu.create_compute_pipeline(kernel)?; var commands = gpu.start_command_recording()?; commands.dispatch(pipeline, { left = 1_l, right = 2_l }, 1_ui, 1_ui, 1_ui)?;",
        ] {
            let source = format!(
                r#"
                export {{ main }};
                import {{ "gpu.resin", "$/status.resin" }};
                struct Root {{ left: long, right: long }};
                @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {{}};
                def main() -> Result<int, _> = {{
                    var gpu = Gpu.new()?;
                    {action}
                    ok(0_i)
                }};
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
        .replace("\"window.resin\"", "\"$/window.resin\"");
    let start = library.find("\t@gpu_allocator").unwrap();
    let end = start + library[start..].find("\t@gpu_compute_pipeline").unwrap();
    library.replace_range(start..end, &format!(r#"
        @gpu_allocator
        def malloc(self: Gpu, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuPtr<ubyte>, RuntimeError> = {{
            var allocation = GpuPtr<ubyte>.allocate_native(self.handle, self, {bytes}, alignment, memory);
            RuntimeStatus.from_code(allocation.status)?;
            ok({value})
        }};
    "#));
    library
}

const COMPUTE: &str = r#"
    export { main };
    import { "$/gpu.resin", "$/status.resin" };
    struct Parameters { increment: uint, values: Span<uint> };
    @compute_shader def kernel(index: ulong, root: Ptr<Parameters>) = {
        if (index < root.values.length) {
            var item = root.values.at(index);
            item.* := item.* + root.increment;
        };
    };
    def main() -> Result<int, _> = {
        var gpu = Gpu.new()?;
        var values = GpuSpan<uint>.allocate(gpu, 4_ul)?;
        var i = 0_ul;
        while (i < values.length) { values.at(i).* := uint(i); i := i + 1_ul; };
        var arguments = { increment = 5_ui, values = values.slice(1_ul, 2_ul) };
        var pipeline = gpu.create_compute_pipeline(kernel)?;
        ACTION
    };
"#;

#[test]
fn projected_scalar_and_span_arguments_dispatch_and_allow_readback_after_submit() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"
        var commands = gpu.start_command_recording()?;
        commands.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)?;
        commands.submit()?;
        var again = gpu.start_command_recording()?;
        again.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)?;
        again.submit()?;
        var result = [0_ui, 0_ui, 0_ui, 0_ui];
        values.copy_to(Span<uint> { data = result.at(0_ul), length = 4_ul });
        ok(if (result.at(0_ul).* == 0_ui && result.at(1_ul).* == 11_ui
            && result.at(2_ul).* == 12_ui && result.at(3_ul).* == 3_ui) { 0 } else { 1 })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn failed_dispatch_and_cancel_restore_cpu_access_and_last_command_alias_cancels() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"
        var cancelled = gpu.start_command_recording()?;
        cancelled.cancel();
        var failed = match (cancelled.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)) {
            ok(value) => { 0_i }, err(error) => { RuntimeStatus.code(error) },
        };
        values.at(1_ul).* := 20_ui;
        var commands = gpu.start_command_recording()?;
        commands.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)?;
        var alias = commands;
        alias.cancel();
        values.at(1_ul).* := 21_ui;
        {
            var abandoned = gpu.start_command_recording()?;
            var last = abandoned;
            last.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)?;
        };
        ok(if (failed == 1_i && values.at(1_ul).* == 21_ui) { 0 } else { 1 })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn recorded_gpu_work_denies_cpu_access_through_all_aliases() {
    for access in [
        "var value = values.at(0_ul).*;",
        "values.at(0_ul).* := 7_ui;",
    ] {
        let source = COMPUTE.replace(
            "ACTION",
            &format!(
                r#"
            var commands = gpu.start_command_recording()?;
            commands.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)?;
            {access}
            ok(0_i)
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
        "number.read_only().* := 8_i;",
        "var value = number.write_only().*;",
        "number.read_only().replace(8_i);",
        "number.write_only().replace(8_i);",
        "values.read_only().at(0_ul).* := 8_i;",
        "var value = values.write_only().at(0_ul).*;",
    ] {
        let source = format!(
            r#"
            export {{ main }};
            import {{ "$/gpu.resin" }};
            def main() -> Result<int, _> = {{
                var gpu = Gpu.new()?;
                var number = gpu.new(7_i)?;
                var values = GpuSpan<int>.allocate(gpu, 1_ul)?;
                values.at(0_ul).* := 7_i;
                {access}
                ok(0_i)
            }};
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
fn inferred_signed_long_pointers_project_and_tuple_arguments_evaluate_once() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin" };
        struct Parameters { value: Ptr<long>, values: Span<long>, increment: long };
        @compute_shader def kernel(index: ulong, root: Ptr<Parameters>) = {
            if (index == 0_ul && root.value.* < 0_l) {
                root.value.* := -root.value.* + root.increment;
                root.values.at(0_ul).* := root.value.* * 2_l;
            };
        };
        def allocation(gpu: Gpu, calls: Ptr<int>) -> (Gpu, ulong) = {
            calls.* := calls.* + 1_i;
            (gpu, 1_ul)
        };
        def launch(pipeline: GpuComputePipeline<Parameters, GpuPipelineOwner>, value: GpuPtr<long>, values: GpuSpan<long>, calls: Ptr<int>) -> _ = {
            calls.* := calls.* + 1_i;
            (pipeline, { value = value, values = values, increment = 7_l }, 1_ui, 1_ui, 1_ui)
        };
        def main() -> Result<int, _> = {
            var gpu = Gpu.new()?;
            var calls = 0_i;
            var value = gpu.new(-42)?;
            var values = GpuSpan<long>.allocate(allocation(gpu, &calls))?;
            values.at(0_ul).* := 0_l;
            var pipeline = gpu.create_compute_pipeline(kernel)?;
            var commands = gpu.start_command_recording()?;
            commands.dispatch(launch(pipeline, value, values, &calls))?;
            commands.submit()?;
            ok(if (calls == 2_i && value.* == 49_l && values.at(0_ul).* == 98_l) { 0 } else { 1 })
        };
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn returned_typed_pipelines_and_recordings_keep_scoped_resources_alive() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin" };
        struct Parameters { values: Span<uint> };
        @compute_shader def kernel(index: ulong, root: Ptr<Parameters>) = {
            if (index < root.values.length) { root.values.at(index).* := 42_ui; };
        };
        def make_pipeline(gpu: Gpu) -> Result<GpuComputePipeline<Parameters, GpuPipelineOwner>, _> = {
            gpu.create_compute_pipeline(kernel)
        };
        def record() -> Result<{ commands: GpuCommands, values: GpuSpan<uint> }, _> = {
            var gpu = Gpu.new()?;
            var values = GpuSpan<uint>.allocate(gpu, 1_ul)?;
            values.at(0_ul).* := 0_ui;
            var pipeline = make_pipeline(gpu)?;
            var alias = pipeline;
            var commands = gpu.start_command_recording()?;
            commands.dispatch(alias, { values = values }, 1_ui, 1_ui, 1_ui)?;
            ok({ commands = commands, values = values })
        };
        def main() -> Result<int, _> = {
            var recorded = record()?;
            recorded.commands.submit()?;
            ok(if (recorded.values.at(0_ul).* == 42_ui) { 0_i } else { 1_i })
        };
    "#) else {
        return;
    };
    success(&output);
}

#[test]
fn pipelines_from_another_device_fail_recording_without_locking_arguments() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"
        var other = Gpu.new()?;
        var commands = other.start_command_recording()?;
        var failed = match (commands.dispatch(pipeline, arguments, 1_ui, 1_ui, 1_ui)) {
            ok(value) => { 0_i }, err(error) => { RuntimeStatus.code(error) },
        };
        values.at(1_ul).* := 42_ui;
        commands.cancel();
        ok(if (failed == 1_i && values.at(1_ul).* == 42_ui) { 0_i } else { 1_i })
    "#,
    );
    let Some(output) = run(&source) else { return };
    success(&output);
}

#[test]
fn dispatch_rejects_argument_views_from_another_device() {
    let source = COMPUTE.replace("ACTION", r#"
        var other = Gpu.new()?;
        var foreign_values = GpuSpan<uint>.allocate(other, 1_ul)?;
        var commands = gpu.start_command_recording()?;
        commands.dispatch(pipeline, { increment = 5_ui, values = foreign_values }, 1_ui, 1_ui, 1_ui)?;
        ok(0_i)
    "#);
    let Some(output) = run(&source) else { return };
    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("different device"));
}

#[test]
fn rooted_graphics_stages_receive_automatically_projected_arguments() {
    let Some(output) = run(r#"
        export { main };
        import { "$/gpu.resin", "$/graphics.resin" };
        struct Parameters { color: Ptr<Color>, offset: float32 };
        @vertex_shader def vertex(index: int, root: Ptr<Parameters>) -> Vertex = {
            Vertex {
                position = Position {
                    x = (if (index == 1_i) { 3.0_f } else { -1.0_f }) + root.offset,
                    y = if (index == 2_i) { 3.0_f } else { -1.0_f },
                    z = 0.0_f, w = 1.0_f,
                },
                color = Color { r = 1.0_f, g = 1.0_f, b = 1.0_f, a = 1.0_f },
            }
        };
        @fragment_shader def fragment(color: Color, root: Ptr<Parameters>) -> Color = {
            root.color.*
        };
        def make_pipeline(gpu: Gpu) -> Result<GpuGraphicsPipeline<Parameters, GpuPipelineOwner>, _> = {
            gpu.create_graphics_pipeline(vertex, fragment)
        };
        def main() -> Result<int, _> = {
            var gpu = Gpu.new()?;
            var pipeline = make_pipeline(gpu)?;
            var color = gpu.new(Color { r = 1.0_f, g = 0.0_f, b = 0.0_f, a = 1.0_f })?;
            var image = gpu.create_image(8_ui, 8_ui)?;
            var pixels = GpuSpan<ubyte>.allocate(gpu, 256_ul)?;
            var commands = gpu.start_command_recording()?;
            commands.begin_rendering(image, 0.0_f, 0.0_f, 0.0_f, 1.0_f)?;
            commands.draw(pipeline, { color = color, offset = 0.0_f }, 3_ui)?;
            commands.end_rendering()?;
            commands.copy_image_to_buffer(image, pixels)?;
            commands.submit()?;
            ok(if (pixels.at(0_ul).* == 255_ub && pixels.at(1_ul).* == 0_ub
                && pixels.at(2_ul).* == 0_ub && pixels.at(3_ul).* == 255_ub) { 0_i } else { 1_i })
        };
    "#) else {
        return;
    };
    success(&output);
}
