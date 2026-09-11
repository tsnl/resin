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
    for allocate in [
        "self.gpu.malloc(1_ul, alignment, memory)",
        "ok(self.gpu.malloc(bytes + 1_ul, alignment, memory)?.at(1_ul))",
    ] {
        for action in [
            "var values = GpuSpan<long>.allocate(gpu, 2_ul)?;",
            "var arguments = kernel.project(gpu, { left = 1_l, right = 2_l })?;",
        ] {
            let source = format!(
                r#"
                export {{ main }};
                import {{ "$/gpu.resin", "$/status.resin" }};
                struct Custom {{ gpu: Gpu }};
                impl Custom {{
                    @gpu_allocator
                    def malloc(self: Custom, bytes: ulong, alignment: ulong, memory: int) -> Result<GpuPtr<ubyte>, RuntimeError> = {{ {allocate} }};
                }}
                struct Root {{ left: long, right: long }};
                @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {{}};
                def main() -> Result<int, _> = {{
                    var gpu = Custom {{ gpu = Gpu.new()? }};
                    {action}
                    ok(0_i)
                }};
            "#
            );
            let Some(output) = run(&source) else {
                return;
            };
            assert!(
                !output.status.success(),
                "invalid allocation was accepted: {allocate}; {action}"
            );
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(error.contains("GPU"), "{error}");
        }
    }
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
        var arguments = kernel.project(gpu, { increment = 5_ui, values = values.slice(1_ul, 2_ul) })?;
        var pipeline = gpu.create_compute_pipeline(kernel.spirv)?;
        ACTION
    };
"#;

#[test]
fn projected_scalar_and_span_arguments_dispatch_and_allow_readback_after_submit() {
    let source = COMPUTE.replace(
        "ACTION",
        r#"
        var commands = gpu.start_command_recording()?;
        commands.set_pipeline(pipeline)?;
        commands.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
        commands.submit()?;
        var again = gpu.start_command_recording()?;
        again.set_pipeline(pipeline)?;
        again.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
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
        var commands = gpu.start_command_recording()?;
        var failed = match (commands.dispatch(arguments, 1_ui, 1_ui, 1_ui)) {
            ok(value) => { 0_i }, err(error) => { RuntimeStatus.code(error) },
        };
        values.at(1_ul).* := 20_ui;
        commands.set_pipeline(pipeline)?;
        commands.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
        var alias = commands;
        alias.cancel();
        values.at(1_ul).* := 21_ui;
        {
            var abandoned = gpu.start_command_recording()?;
            var last = abandoned;
            last.set_pipeline(pipeline)?;
            last.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
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
            commands.set_pipeline(pipeline)?;
            commands.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
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
        def launch(gpu: Gpu, value: GpuPtr<long>, values: GpuSpan<long>, calls: Ptr<int>) -> _ = {
            calls.* := calls.* + 1_i;
            (gpu, { value = value, values = values, increment = 7_l })
        };
        def main() -> Result<int, _> = {
            var gpu = Gpu.new()?;
            var calls = 0_i;
            var value = gpu.new(-42)?;
            var values = GpuSpan<long>.allocate(allocation(gpu, &calls))?;
            values.at(0_ul).* := 0_l;
            var arguments = kernel.project(launch(gpu, value, values, &calls))?;
            var pipeline = gpu.create_compute_pipeline(kernel.spirv)?;
            var commands = gpu.start_command_recording()?;
            commands.set_pipeline(pipeline)?;
            commands.dispatch(arguments, 1_ui, 1_ui, 1_ui)?;
            commands.submit()?;
            ok(if (calls == 2_i && value.* == 49_l && values.at(0_ul).* == 98_l) { 0 } else { 1 })
        };
    "#) else {
        return;
    };
    success(&output);
}
