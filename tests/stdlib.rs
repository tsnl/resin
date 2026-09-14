use resin_types::prelude::*;
use tempfile::TempDir;
#[path = "support/pipeline.rs"]
mod pipeline;
#[path = "support/project.rs"]
mod project;
#[path = "support/shaders.rs"]
mod shaders;
#[path = "support/toolchain.rs"]
mod toolchain;
use std::{fs, path::Path, process::Command};

fn run(source: &str, native: &str) -> std::process::Output {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("main.resin");
    fs::write(&path, source).unwrap();
    let program = pipeline::load(&path).unwrap();
    let module = pipeline::generate_program(&program).unwrap();
    let project = project::Project::new(&module, Some("main")).unwrap();
    let path = project.generated.c_source().unwrap();
    let c = format!("{native}\n{}", fs::read_to_string(path).unwrap());
    fs::write(path, c).unwrap();
    let cc = std::env::var_os("CC").unwrap_or_else(|| resin_toolchain::DEFAULT_C_COMPILER.into());
    let built = project.build(&toolchain::c(&cc)).unwrap();
    let executable = built
        .executable(project.generated.program().unwrap().file_name().unwrap())
        .unwrap();
    Command::new(executable.path())
        .current_dir(temp.path())
        .output()
        .unwrap()
}

fn success(output: &std::process::Output) {
    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn source_strings_format_explicit_byte_views_and_keep_the_terminator_outside_length() {
    let output = run(
        r#"
        export { main };
        import { "$/span.resin", "$/shared.resin", "$/string.resin" };
        def main() -> int = {
            var bytes = [65_ub, 0_ub, 66_ub];
            var text = String.from_bytes(Span<ubyte> { data = bytes.at(0), length = 3_ul });
            var weak = text.storage.downgrade();
            var formatted = fmt("{0}:{1}:{2}", (42, text.bytes(), "end"));
            var raw = formatted.get();
            var terminated = Span<ubyte> { data = raw.data, length = raw.length + 1_ul };
            print(formatted);
            if (raw.length == 10_ul && terminated.at(raw.length).* == 0_ub && weak.upgrade()!.get().length == 3_ul) { 0 } else { 1 }
        };
    "#,
        "",
    );
    success(&output);
    assert_eq!(output.stdout, b"42:A\0B:end");
}

#[test]
fn source_shared_elements_drop_in_reverse_and_unwind_on_allocation_failure() {
    success(&run(
        r#"
        export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Item { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Item>) = {
                if (self.digit != 0) { self.trace.* := self.trace.* * 10 + self.digit; };
            };
        };
        def fail(trace: Ptr<int>) -> Result<(), OutOfMemory> = {
            var owner = ArcPtr<Item>.alloc(Item { trace = trace, digit = 0 })?;
            owner.get().digit := 4;
            ArcSpan<ulong>.alloc(0xffffffffffffffff_ul, 0_ul)?;
            ok(())
        };
        def main() -> Result<int, _> = {
            var trace = 0;
            {
                var items = ArcSpan<Item>.alloc(3, Item { trace = &trace, digit = 0 })?;
                items.get().at(0).digit := 1;
                items.get().at(1).digit := 2;
                items.get().at(2).digit := 3;
            };
            var failed = match (fail(&trace)) { ok(value) => { 1 == 0 }, err(error) => { 1 == 1 } };
            ok(if (failed && trace == 3214) { 0 } else { 1 })
        };
    "#,
        "",
    ));
}

#[test]
fn source_owned_wrappers_retain_payloads_and_borrow_temporary_receivers() {
    success(&run(
        r#"
        export { main };
        import { "$/shared.resin", "$/status.resin" };
        struct Item { trace: Ptr<int>, digit: int,
            def drop(self: Ptr<Item>) = {
                if (self.digit != 0) { self.trace.* := self.trace.* * 10 + self.digit; };
            };
        };
        def main() -> Result<int, _> = {
            var trace = 0;
            var weak = WeakPtr<Item>.empty();
            var valid = 1 == 1;
            {
                var owner = ArcPtr<Item>.alloc(Item { trace = &trace, digit = 0 })?;
                owner.get().digit := 7;
                weak := owner.downgrade();
                var copy = owner;
                valid := valid && copy.get().digit == 7 && weak.upgrade()!.get().digit == 7;
                var values = ArcSpan<uint>.alloc(3, 42_ui)?;
                values.get().at(2).* := 9_ui;
                valid := valid && values.get().at(0).* == 42_ui && values.get().at(2).* == 9_ui;
                var borrowed = ArcSpan<uint>.alloc(1, 13_ui)?.get();
                valid := valid && borrowed.at(0).* == 13_ui;
                var empty = ArcSpan<uint>.alloc(0, 0_ui)?;
                valid := valid && empty.get().length == 0_ul;
            };
            valid := valid && trace == 7;
            valid := valid && match (weak.upgrade()) { ArcPtr<Item>(live) => { 1 == 0 }, None => { 1 == 1 } };
            valid := valid && match (ArcSpan<uint>.alloc(0xffffffffffffffff_ul, 0_ui)) {
                ok(owner) => { 1 == 0 }, err(error) => { 1 == 1 },
            };
            ok(if (valid) { 0 } else { 1 })
        };
    "#,
        "",
    ));
}

#[test]
fn host_allocates_initialized_typed_storage_and_reports_overflow() {
    let output = run(
        r#"
        export { main };
        import { "$/host.resin", "$/status.resin" };
        struct Empty {};
        def main() -> Result<int, _> = {
            var weak = WeakSpan<uint>();
            var valid = 1 == 1;
            {
                var memory: ArcSpan<uint>;
                memory := Host.alloc(4, 7_ui)?;
                weak := memory.downgrade();
                var alias = memory;
                var values = memory.get();
                valid := valid && values.length == 4_ul && values.at(3).* == 7_ui;
                values.at(3).* := 42_ui;
                valid := valid && alias.get().at(3).* == 42_ui;
                valid := valid && values.as_bytes().length == 4_ul * size_of(uint);
                var upgraded = weak.upgrade()!;
                valid := valid && upgraded.get().at(3).* == 42_ui;
                var descriptor = ArcPtr<Span<uint>>(values);
                var previous = descriptor.get().replace(Span<uint> { data = values.data, length = 2_ul });
                valid := valid && previous.length == 4_ul && descriptor.length == 2_ul;
                valid := valid && memory.get().length == 4_ul;
            };
            var expired = match (weak.upgrade()) {
                ArcSpan<uint>(owner) => { 1 == 0 },
                None => { 1 == 1 },
            };
            var empty = Host.alloc(0, 0_ui)?;
            valid := valid && empty.get().length == 0_ul;
            var empty_elements = Host.alloc(19, Empty {})?;
            valid := valid && empty_elements.get().length == 19_ul;
            var failure: Result<ArcSpan<uint>, OutOfMemory>;
            failure := Host.alloc(0xffffffffffffffff_ul, 0_ui);
            var failed = match (failure) {
                ok(memory) => { 1 == 0 },
                err(error) => { 1 == 1 },
            };
            ok(if (valid && expired && failed) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);

    // Error propagation does not require importing the module's private dependencies.
    let output = run(
        r#"
        export { main };
        import { "$/host.resin" };
        def main() -> Result<(), _> = {
            Host.alloc(0xffffffffffffffff_ul, 0_ui)?;
            ok(())
        };
        "#,
        "",
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(
        String::from_utf8_lossy(&output.stderr).replace("\r\n", "\n"),
        "unhandled error: OutOfMemory\n"
    );
}

#[test]
fn owned_spans_release_managed_elements_on_success_and_error() {
    let output = run(
        r#"
        export { main };
        import { "$/host.resin" };
        struct Failed {};
        struct Item {
            trace: Ptr<int>,
            digit: int,
            def drop(self: Ptr<Item>) = { self.trace.* := self.trace.* * 10 + self.digit; };
        };
        def work(trace: Ptr<int>, fail: bool) -> Result<(), _> = {
            var values = Host.alloc(2, ArcPtr<Item> { trace = trace, digit = 1 })?;
            values.get().at(1).* := ArcPtr<Item> { trace = trace, digit = 2 };
            var alias = values;
            if (fail) { err(Failed {}) } else { ok(()) }
        };
        def main() -> Result<int, _> = {
            var trace = 0;
            work(&trace, 1 == 0)?;
            var valid = trace == 21;
            trace := 0;
            var failed = match (work(&trace, 1 == 1)) {
                ok(value) => { 1 == 0 },
                err(error) => { 1 == 1 },
            };
            valid := valid && failed && trace == 21;
            trace := 0;
            var rejected = match (Host.alloc(0xffffffffffffffff_ul, ArcPtr<Item> { trace = &trace, digit = 3 })) {
                ok(values) => { 1 == 0 },
                err(error) => { 1 == 1 },
            };
            ok(if (valid && rejected && trace == 3) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
}

#[test]
fn direct_owned_span_construction_returns_optional_initialized_storage() {
    let output = run(
        r#"
        export { main };
        def main() -> int = {
            var values = ArcSpan<uint>.try_new(3, 7_ui)!;
            var view = values.get();
            var valid = view.length == 3_ul && view.at(0).* == 7_ui && view.at(2).* == 7_ui;
            var rejected = match (ArcSpan<uint>.try_new(0xffffffffffffffff_ul, 0_ui)) {
                ArcSpan<uint>(owner) => { 1 == 0 },
                None => { 1 == 1 },
            };
            if (valid && rejected) { 0 } else { 1 }
        };
        "#,
        "",
    );
    success(&output);
}

#[test]
fn generic_owned_span_methods_preserve_lifetimes_and_widened_results() {
    let output = run(
        r#"
        export { main };
        import { "$/host.resin", "$/status.resin" };
        struct Other {};
        def allocate<T>(count: ulong, initial: T) -> Result<ArcSpan<T>, OutOfMemory | Other> = {
            Host.alloc(count, initial)
        };
        def optional<T>(count: ulong, initial: T) -> ArcSpan<T> | None = {
            ArcSpan<T>.try_new(count, initial)
        };
        def borrowed<T>(owner: Ptr<ArcSpan<T>>) -> Span<T> = { owner.get() };
        def weaken<T>(owner: ArcSpan<T>) -> WeakSpan<T> = { owner.downgrade() };
        def upgrade<T>(weak: WeakSpan<T>) -> ArcSpan<T> | None | Other = { weak.upgrade() };
        def first<T>(initial: T) -> T = {
            var owner = ArcSpan<T>.try_new(1, initial)!;
            owner.get().at(0).*
        };
        def main() -> Result<int, _> = {
            var weak = WeakSpan<uint>();
            var valid = 1 == 1;
            {
                var owner = allocate(2, 7_ui)?;
                weak := weaken(owner);
                var view = borrowed(&owner);
                view.at(1).* := 42_ui;
                valid := valid && view.length == 2_ul && owner.get().at(1).* == 42_ui;
                valid := valid && match (upgrade(weak)) {
                    ArcSpan<uint>(live) => { live.get().at(1).* == 42_ui },
                    None => { 1 == 0 },
                    Other(other) => { 1 == 0 },
                };
                var another = optional(3, 9_ui)!;
                valid := valid && another.get().length == 3_ul && another.get().at(2).* == 9_ui;
                valid := valid && first(17_ui) == 17_ui;
            };
            valid := valid && match (upgrade(weak)) {
                ArcSpan<uint>(live) => { 1 == 0 },
                None => { 1 == 1 },
                Other(other) => { 1 == 0 },
            };
            valid := valid && match (allocate(0xffffffffffffffff_ul, 0_ui)) {
                ok(owner) => { 1 == 0 },
                err(error) => {
                    match (error) {
                        OutOfMemory(error) => { 1 == 1 },
                        Other(other) => { 1 == 0 },
                    }
                },
            };
            ok(if (valid) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
}

#[test]
fn borrowed_span_slices_preserve_aliases_and_accept_empty_null_views() {
    let output = run(
        r#"
        export { main };
        def main() -> int = {
            var values = ArcSpan<uint>.try_new(4, 0_ui)!;
            var view = values.get();
            var middle = view.slice(1, 2);
            var alias = middle;
            alias.at(1).* := 42_ui;
            var valid = middle.length == 2_ul && view.at(2).* == 42_ui;
            valid := valid && view.at(0).* == 0_ui && view.at(3).* == 0_ui;
            var end = view.slice(view.length, 0);
            valid := valid && end.length == 0_ul;
            var null_view = Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul };
            var empty = null_view.slice(0, 0);
            valid := valid && empty.length == 0_ul && ulong(empty.data) == 0_ul;
            if (valid) { 0 } else { 1 }
        };
        "#,
        "",
    );
    success(&output);
}

#[test]
fn borrowed_span_slice_and_byte_length_overflow_trap_before_memory_access() {
    for (operation, diagnostic) in [
        ("view.slice(3, 0)", "span slice out of bounds"),
        ("view.slice(1, 2)", "span slice out of bounds"),
        (
            "view.slice(0xffffffffffffffff_ul, 1)",
            "span slice out of bounds",
        ),
        (
            "Span<uint> { data = Ptr<uint>(0_ul), length = 0xffffffffffffffff_ul }.as_bytes()",
            "span byte length overflow",
        ),
    ] {
        let output = run(
            &format!(
                r#"
                export {{ main }};
                def main() = {{
                    var view = Span<uint> {{ data = Ptr<uint>(0_ul), length = 2_ul }};
                    {operation};
                    print("unreachable");
                }};
                "#
            ),
            "",
        );
        assert!(!output.status.success(), "{operation}");
        assert!(output.stdout.is_empty(), "{operation}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(diagnostic),
            "{operation}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn every_native_status_operation_has_a_public_result_wrapper() {
    let root = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), ""));
    let source = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = source.path().join("main.resin");
    fs::write(
        &path,
        "import { \"$/gpu.resin\", \"$/window.resin\", \"$/image.resin\", \"$/console.resin\" };",
    )
    .unwrap();
    let module = pipeline::generate_program(&pipeline::load(&path).unwrap()).unwrap();
    for name in ["gpu", "window", "image", "console"] {
        let public = pipeline::generate_program(
            &pipeline::load(&root.join(format!("resin/{name}.resin"))).unwrap(),
        )
        .unwrap();
        assert!(
            public.entries.is_empty(),
            "operations are methods, not free function exports"
        );
        let header = fs::read_to_string(
            Path::new(resin_runtime::INCLUDE_DIR).join(format!("resin_runtime/{name}.h")),
        )
        .unwrap();
        let mut checked = 0;
        for line in header.lines() {
            let Some(declaration) = line.strip_prefix("ResinStatus resin_") else {
                continue;
            };
            let name = declaration.split('(').next().unwrap();
            let (receiver, method) = match name {
                "gpu_create" => ("GpuOwner", "new"),
                "gpu_create_at" => ("GpuOwner", "new_at"),
                "gpu_device_count" => ("GpuOwner", "device_count"),
                "gpu_enumerate_devices" => ("GpuOwner", "enumerate_devices"),
                "gpu_create_for_window" => ("GpuOwner", "new_for_window"),
                "window_create" => ("WindowOwner", "new"),
                "image_read_png" => ("ImageDataOwner", "read_png"),
                "image_write_png" => ("ImageDataOwner", "write_pixels"),
                // These raw native operations have been replaced by owning
                // views and compiler-generated projection in Resin source.
                "gpu_malloc" | "gpu_dispatch" | "gpu_copy_image_to_buffer" | "gpu_set_pipeline" => {
                    continue;
                }
                "gpu_ptr_allocate" => ("GpuOwner", "malloc"),
                "gpu_create_compute_pipeline" => ("GpuOwner", "create_compute_pipeline"),
                "gpu_create_graphics_pipeline" => ("GpuOwner", "create_graphics_pipeline"),
                "gpu_create_image" => ("GpuOwner", "create_image"),
                "gpu_start_command_recording" => ("GpuOwner", "start_command_recording"),
                "gpu_projected_dispatch" => ("CommandsOwner", "dispatch"),
                "gpu_begin_rendering" => ("CommandsOwner", "begin_rendering"),
                "gpu_end_rendering" => ("CommandsOwner", "end_rendering"),
                "gpu_draw" | "gpu_projected_draw" => ("CommandsOwner", "draw"),
                "gpu_copy_image_to_span" => ("CommandsOwner", "copy_image_to_buffer"),
                "gpu_submit" => ("CommandsOwner", "submit"),
                "gpu_cancel_command_buffer" => ("CommandsOwner", "cancel"),
                "window_poll_events" => ("WindowOwner", "poll_events"),
                "window_should_close" => ("WindowOwner", "should_close"),
                "window_set_should_close" => ("WindowOwner", "set_should_close"),
                "window_framebuffer_size" => ("WindowOwner", "framebuffer_size"),
                "window_set_size" => ("WindowOwner", "set_size"),
                "window_key_pressed" => ("WindowOwner", "key_pressed"),
                "window_key_state" => ("WindowOwner", "key_state"),
                "window_mouse_button_state" => ("WindowOwner", "mouse_button_state"),
                "window_cursor_position" => ("WindowOwner", "cursor_position"),
                "window_scroll_delta" => ("WindowOwner", "scroll_delta"),
                "window_focused" => ("WindowOwner", "focused"),
                "window_capture_cursor" => ("WindowOwner", "capture_cursor"),
                "gpu_present" => ("GpuOwner", "present"),
                _ => panic!("missing method mapping for native operation: {name}"),
            };
            let qualified = format!("{receiver}.{method}");
            let function = module
                .functions
                .iter()
                .find(|function| function.name.as_deref() == Some(qualified.as_str()))
                .unwrap_or_else(|| panic!("missing lowered function {qualified}"));
            assert!(function.foreign.is_none(), "{name}");
            assert!(matches!(function.result, Ty::Result { .. }), "{name}");
            checked += 1;
        }
        assert!(checked > 0 || name == "console");
    }
}

#[test]
fn native_statuses_become_named_errors_and_keep_unknown_codes() {
    let output = run(
        r#"
        export { main };
        import { "$/status.resin" };
        extern "string.h" def strcmp(a: Ptr<ubyte>, b: Ptr<ubyte>) -> int;
        def main() -> Result<int, _> = {
            RuntimeStatus.from_code(0)?;
            var code = -1;
            var valid = 1 == 1;
            while (code <= 9) {
                var actual = match (RuntimeStatus.from_code(code)) {
                    ok(value) => { 0 },
                    err(error) => { RuntimeStatus.code(error) },
                };
                valid := valid && actual == code;
                code := code + 1;
            };
            var message = "io error";
            valid := valid && strcmp(RuntimeStatus.message(IoError {}), message.data) == 0;
            ok(if (valid) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);

    for (code, name) in [
        (1, "InvalidArgument"),
        (2, "VulkanUnavailable"),
        (3, "Unsupported"),
        (4, "OutOfMemory"),
        (5, "VulkanError"),
        (6, "IoError"),
        (7, "Incomplete"),
        (8, "WindowUnavailable"),
        (99, "UnknownRuntimeError"),
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"$/status.resin\" }}; def main() -> Result<(), _> = {{ RuntimeStatus.from_code({code}) }};"
            ),
            "",
        );
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(&format!("unhandled error: {name}"))
        );
    }
}

#[test]
fn png_wrappers_return_image_data_and_propagate_io_errors() {
    let output = run(
        r#"
        export { main };
        import { "$/image.resin", "$/status.resin" };
        def main() -> Result<int, _> = {
            var path = "pixel.png";
            var pixels = [ubyte(1), ubyte(2), ubyte(3), ubyte(255)];
            ImageData.write_pixels(path.data, 1, 1, 4, Span<ubyte> { data = pixels.at(0), length = 4_ul }, 0)?;
            var image = ImageData.read_png(path.data, 0)?;
            var alias = image;
            var copy_path = "copy.png";
            alias.write_png(copy_path.data)?;
            var copied = ImageData.read_png(copy_path.data, 0)?;
            ok(if (copied.width() == image.width() && copied.height() == image.height() && copied.pixels().data.* == image.pixels().data.*
                && image.width() == uint(1) && image.height() == uint(1) && image.channels() == uint(4)
                && image.pixels().data.* == ubyte(1) && Ptr<ubyte>(ulong(image.pixels().data) + ulong(3)).* == ubyte(255)) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
    for call in [
        "ImageData.read_png(path.data, 4)?",
        "ImageData.write_pixels(path.data, 1, 1, 4, Span<ubyte> { data = pixels.at(0), length = 4_ul }, 0)?",
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"$/image.resin\", \"$/string.resin\" }}; struct Cleanup {{ def drop(self: Ptr<Cleanup>) = {{ print(fmt(\"cleanup\\n\", ())); }}; }};  def main() -> Result<(), _> = {{ var path = \"missing/pixel.png\"; var pixels = [0_ub, 0_ub, 0_ub, 0_ub]; var cleanup = Cleanup {{}}; {call}; ok(()) }};"
            ),
            "",
        );
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, b"cleanup\n");
        assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: IoError"));
    }
}

#[test]
fn png_pixel_views_check_dimensions_padding_and_storage_before_native_access() {
    for (width, height, channels, length, stride) in [
        (1, 1, 4, 3, "0_ul"),
        (1, 2, 4, 8, "5_ul"),
        (1, 1, 4, 4, "3_ul"),
        (1, 2, 4, 4, "0xffffffffffffffff_ul"),
        (0, 1, 4, 4, "0_ul"),
        (1, 0, 4, 4, "0_ul"),
        (1, 1, 0, 4, "0_ul"),
        (1, 1, 5, 4, "0_ul"),
    ] {
        let output = run(
            &format!(
                r#"
                export {{ main }};
                import {{ "$/image.resin", "$/status.resin" }};
                def main() -> int = {{
                    var pixels = [0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub];
                    var bytes = Span<ubyte> {{ data = pixels.at(0), length = {length}_ul }};
                    match (ImageData.write_pixels("missing/pixel.png".data, {width}, {height}, {channels}, bytes, {stride})) {{
                        ok(value) => {{ 1 }},
                        err(error) => {{ if (RuntimeStatus.code(error) == 1) {{ 0 }} else {{ 1 }} }},
                    }}
                }};
                "#
            ),
            "",
        );
        success(&output);
    }

    let output = run(
        r#"
        export { main };
        import { "$/image.resin" };
        def main() -> Result<int, _> = {
            var pixels = [1_ub, 2_ub, 3_ub, 255_ub, 99_ub, 4_ub, 5_ub, 6_ub, 255_ub, 99_ub];
            var bytes = Span<ubyte> { data = pixels.at(0), length = 10_ul };
            ImageData.write_pixels("padded.png".data, 1, 2, 4, bytes, 5)?;
            var image = ImageData.read_png("padded.png".data, 0)?;
            var loaded = Span<ubyte> { data = image.pixels, length = 8_ul };
            ok(if (loaded.at(0).* == 1_ub && loaded.at(4).* == 4_ub) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
}

#[test]
fn gpu_cleanup_covers_acquisition_recording_and_submission_failures() {
    if shaders::optimizer().is_none() {
        return;
    }
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        extern "resin_runtime.h" def test_mode(mode: int);
        extern "resin_runtime.h" def test_verify(code: int);
        @vertex_shader
        def vertex(index: int) -> Vertex = {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        };
        @fragment_shader
        def fragment(color: Color) -> Color = { color };
        def work() -> Result<(), _> = {
            var gpu = Gpu.new()?;

            var allocation = gpu.malloc(16, 8, Memory.default())?;

            var commands = gpu.start_command_recording()?;

            var pipeline = gpu.create_graphics_pipeline(vertex, fragment)?;
            commands.draw(pipeline, None, 3)?;
            commands.submit()?;
            ok(())
        };
        def main() = {
            var mode = 0;
            while (mode < 6) {
                test_mode(mode);
                var result = match (work()) {
                    ok(value) => { 0 },
                    err(error) => { RuntimeStatus.code(error) },
                };
                test_verify(result);
                mode := mode + 1;
            };
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static int mode, cleanup;
        static void test_mode(int value) { mode = value; cleanup = 0; }
        static void test_verify(int code) {
            const int codes[] = {RESIN_STATUS_UNSUPPORTED, RESIN_STATUS_OUT_OF_MEMORY,
                RESIN_STATUS_VULKAN_ERROR, RESIN_STATUS_INVALID_ARGUMENT, RESIN_STATUS_VULKAN_ERROR, 0};
            const int cleanups[] = {0, 1, 21, 321, 421, 421};
            assert(code == codes[mode]);
            assert(cleanup == cleanups[mode]);
        }
        static ResinStatus mock_create(ResinGpu **out) {
            *out = (ResinGpu *)(uintptr_t)1;
            return mode == 0 ? RESIN_STATUS_UNSUPPORTED : RESIN_STATUS_SUCCESS;
        }
        static void mock_destroy(ResinGpu *gpu) {
            assert(gpu == (ResinGpu *)(uintptr_t)1);
            cleanup = cleanup * 10 + 1;
        }
        static void mock_free(void *payload) {
            cleanup = cleanup * 10 + 2;
            resin_arc_release(*(ResinArc **)payload);
        }
        static ResinStatus mock_malloc(ResinGpu *gpu, ResinArc *gpu_owner, size_t bytes, size_t align, int32_t memory, ResinGpuPtr *out) {
            assert(gpu == (ResinGpu *)(uintptr_t)1 && bytes == 16 && align == 8 && memory == RESIN_MEMORY_DEFAULT);
            *out = (ResinGpuPtr){0};
            if (mode == 1) return RESIN_STATUS_OUT_OF_MEMORY;
            out->owner = resin_arc_new(sizeof(ResinArc *), _Alignof(ResinArc *), mock_free);
            *(ResinArc **)resin_arc_data(out->owner) = gpu_owner;
            resin_arc_retain(gpu_owner);
            out->access = RESIN_GPU_ACCESS_READ | RESIN_GPU_ACCESS_WRITE;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_record(ResinGpu *gpu, ResinCommandBuffer **out) {
            assert(gpu == (ResinGpu *)(uintptr_t)1);
            *out = (ResinCommandBuffer *)(uintptr_t)3;
            return mode == 2 ? RESIN_STATUS_VULKAN_ERROR : RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_graphics(ResinGpu *gpu, const uint8_t *vertex, size_t vertex_length, const uint8_t *fragment, size_t fragment_length, ResinPipeline **out) {
            assert(gpu == (ResinGpu *)(uintptr_t)1 && vertex && fragment && vertex_length > 20 && fragment_length > 20);
            *out = NULL;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_draw(ResinCommandBuffer *commands, ResinDeviceAddress root, uint32_t count) {
            assert(commands == (ResinCommandBuffer *)(uintptr_t)3 && root == 0 && count == 3);
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_pipeline(ResinCommandBuffer *commands, const ResinPipeline *pipeline) {
            assert(commands == (ResinCommandBuffer *)(uintptr_t)3 && pipeline == NULL);
            return mode == 3 ? RESIN_STATUS_INVALID_ARGUMENT : RESIN_STATUS_SUCCESS;
        }
        static void mock_cancel(ResinGpu *gpu, ResinCommandBuffer *commands) {
            assert(gpu == (ResinGpu *)(uintptr_t)1);
            if (commands) {
                assert(cleanup == 0);
                cleanup = 3;
            }
        }
        static ResinStatus mock_submit(ResinGpu *gpu, ResinCommandBuffer *commands) {
            assert(gpu == (ResinGpu *)(uintptr_t)1);
            assert(commands == (ResinCommandBuffer *)(uintptr_t)3);
            cleanup = 4;
            return mode == 4 ? RESIN_STATUS_VULKAN_ERROR : RESIN_STATUS_SUCCESS;
        }
        #define resin_gpu_create mock_create
        #define resin_gpu_destroy mock_destroy
        #define resin_gpu_ptr_allocate mock_malloc
        #define resin_gpu_start_command_recording mock_record
        #define resin_gpu_create_graphics_pipeline mock_graphics
        #define resin_gpu_draw mock_draw
        #define resin_gpu_set_pipeline mock_pipeline
        #define resin_gpu_cancel_command_buffer mock_cancel
        #define resin_gpu_submit mock_submit
        "#,
    );
    success(&output);
}

#[test]
fn presentation_distinguishes_skipped_frames_from_errors_without_opening_windows() {
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/window.resin", "$/status.resin" };
        def main() -> int = {
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            var image = GpuImage { handle = Ptr<ResinImage>(0_ul), gpu = gpu };
            var first = match (gpu.present(image)) { ok(shown) => { shown }, err(e) => { 1 == 0 } };
            var second = match (gpu.present(image)) { ok(shown) => { !shown }, err(e) => { 1 == 0 } };
            var third = match (gpu.present(image)) { ok(shown) => { 0 }, err(e) => { RuntimeStatus.code(e) } };
            var fourth = match (gpu.present(image)) { ok(shown) => { 0 }, err(e) => { RuntimeStatus.code(e) } };
            if (first && second && third == 5 && fourth == 99) { 0 } else { 1 }
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_present(ResinGpu *gpu, ResinImage *image) {
            assert(gpu == NULL && image == NULL);
            static int call;
            const ResinStatus codes[] = {RESIN_STATUS_SUCCESS, RESIN_STATUS_INCOMPLETE, RESIN_STATUS_VULKAN_ERROR, (ResinStatus)99};
            return codes[call++];
        }
        #define resin_gpu_present mock_present
        "#,
    );
    success(&output);
}

#[test]
fn queries_return_values_and_enumeration_preserves_incomplete_errors() {
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/window.resin", "$/status.resin" };
        def valid_size(width: uint, height: uint) -> bool = {
            width == uint(640) && height == uint(480)
        };
        def main() -> Result<int, _> = {
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            var window = Window { handle = Ptr<ResinWindow>(0_ul) };
            var count = Gpu.device_count()?;
            var size = window.framebuffer_size()?;
            var incomplete = match (Gpu.enumerate_devices(Ptr<ResinGpuDeviceInfo>(ulong(0)), 0)) {
                ok(value) => { 0 },
                err(error) => { RuntimeStatus.code(error) },
            };
            var valid = !window.should_close() && window.key_pressed(Window.key_escape());
            window.set_should_close(1 == 1)?;
            valid := valid && window.should_close();
            window.set_should_close(1 == 0)?;
            valid := valid && !window.should_close();
            ok(if (valid && count == uint(2)
                && valid_size(size.0, size.1) && incomplete == 7) { 0 } else { 1 })
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static int closed;
        static ResinStatus mock_count(uint32_t *out) { *out = 2; return RESIN_STATUS_SUCCESS; }
        static ResinStatus mock_size(const ResinWindow *window, uint32_t *width, uint32_t *height) {
            assert(window == NULL);
            *width = 640;
            *height = 480;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_enumerate(ResinGpuDeviceInfo *infos, uint32_t count) {
            assert(infos == NULL && count == 0);
            return RESIN_STATUS_INCOMPLETE;
        }
        static int mock_close(const ResinWindow *window) { assert(window == NULL); return closed; }
        static ResinStatus mock_set_close(const ResinWindow *window, int close) {
            assert(window == NULL && (close == 0 || close == 1));
            closed = close;
            return RESIN_STATUS_SUCCESS;
        }
        static int mock_key(const ResinWindow *window, int key) {
            assert(window == NULL && key == RESIN_KEY_ESCAPE);
            return 1;
        }
        #define resin_gpu_device_count mock_count
        #define resin_window_framebuffer_size mock_size
        #define resin_gpu_enumerate_devices mock_enumerate
        #define resin_window_should_close mock_close
        #define resin_window_set_should_close mock_set_close
        #define resin_window_key_pressed mock_key
        "#,
    );
    success(&output);
}

#[test]
fn byte_input_reports_stream_errors_instead_of_eof() {
    let output = run(
        r#"
        export { main };
        import { "$/console.resin" };
        struct Cleanup {
            def drop(self: Ptr<Cleanup>) = { print("cleanup\n"); };
        };

        def main() -> Result<(), _> = {
            var cleanup = Cleanup {};
            Console.read_byte()?;
            ok(())
        };
        "#,
        r#"
        #include <resin_runtime.h>
        static int mock_getchar(void) { return EOF; }
        static int mock_error(void) { return 1; }
        #define getchar mock_getchar
        #define resin_stdin_error mock_error
        "#,
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout, b"cleanup\n");
    assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: InputReadError"));
}

#[test]
fn typed_pipeline_factories_embed_shaders_and_keep_shared_ownership() {
    if shaders::optimizer().is_none() {
        return;
    }
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        extern "resin_runtime.h" def test_finished();
        extern "resin_runtime.h" def test_fail();
        @compute_shader
        def kernel(index: ulong, root: Ptr<int>) = { root.* := int(index); };
        @vertex_shader
        def vertex(index: int) -> Vertex = {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        };
        @fragment_shader
        def fragment(color: Color) -> Color = { color };
        def copy_pipeline(value: GpuComputePipeline<int, GpuPipelineOwner>) -> GpuComputePipeline<int, GpuPipelineOwner> = { value };
        def main() -> Result<int, _> = {
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            {
                var compute = gpu.create_compute_pipeline(kernel)?;
                var alias = copy_pipeline(compute);
                var graphics = gpu.create_graphics_pipeline(vertex, fragment)?;
                var graphics_alias = graphics;
            };
            test_fail();
            var compute_code = match (gpu.create_compute_pipeline(kernel)) {
                ok(pipeline) => { 0 }, err(error) => { RuntimeStatus.code(error) },
            };
            var graphics_code = match (gpu.create_graphics_pipeline(vertex, fragment)) {
                ok(pipeline) => { 0 }, err(error) => { RuntimeStatus.code(error) },
            };
            test_finished();
            ok(if (compute_code == 5 && graphics_code == 5) { 0 } else { 1 })
        };
    "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        #include <string.h>
        static unsigned created, freed;
        static int failing;
        static void test_fail(void) { failing = 1; }
        static void check_shader(const uint8_t *data, size_t length) {
            uint32_t magic;
            assert(data && length > 20 && length % 4 == 0);
            memcpy(&magic, data, sizeof(magic));
            assert(magic == 0x07230203);
        }
        static void test_finished(void) { assert(created == 3 && freed == 3); }
        static ResinStatus mock_compute(ResinGpu *gpu, const uint8_t *data, size_t length, ResinPipeline **out) {
            assert(!gpu);
            check_shader(data, length);
            if (failing) return RESIN_STATUS_VULKAN_ERROR;
            assert(created == 0);
            created |= 1;
            *out = (ResinPipeline *)(uintptr_t)1;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_graphics(ResinGpu *gpu, const uint8_t *vertex, size_t vertex_length, const uint8_t *fragment, size_t fragment_length, ResinPipeline **out) {
            assert(!gpu);
            check_shader(vertex, vertex_length);
            check_shader(fragment, fragment_length);
            if (failing) return RESIN_STATUS_VULKAN_ERROR;
            assert(created == 1);
            created |= 2;
            *out = (ResinPipeline *)(uintptr_t)2;
            return RESIN_STATUS_SUCCESS;
        }
        static void mock_free_pipeline(ResinGpu *gpu, ResinPipeline *pipeline) {
            assert(!gpu && ((uintptr_t)pipeline == 1 || (uintptr_t)pipeline == 2));
            assert(!(freed & (uintptr_t)pipeline));
            freed |= (uintptr_t)pipeline;
        }
        #define resin_gpu_free_pipeline mock_free_pipeline
        #define resin_gpu_create_compute_pipeline mock_compute
        #define resin_gpu_create_graphics_pipeline mock_graphics
    "#,
    );
    success(&output);
}

#[test]
fn window_input_snapshots_expose_edges_coordinates_and_named_controls() {
    let output = run(
        r#"
        export { main };
        import { "$/window.resin" };
        def coordinates(point: (float64, float64)) -> bool = { point.0 == 12.5_d && point.1 == -3.25_d };
        def main() -> Result<int, _> = {
            var window = Window { handle = Ptr<ResinWindow>(0_ul) };
            var key = window.key_state(Window.keys().w);
            var mouse = window.mouse_button_state(Window.mouse_buttons().left);
            var valid = !key.down && key.pressed && key.released && mouse.down && mouse.pressed && !mouse.released;
            valid := valid && coordinates(window.cursor_position()?);
            valid := valid && coordinates(window.scroll_delta()?);
            valid := valid && window.focused() && Window.keys().escape == Window.key_escape() && Window.keys().f25 == 314;
            window.capture_cursor(1 == 1)?;
            window.capture_cursor(1 == 0)?;
            ok(if (valid) { 0 } else { 1 })
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static uint32_t mock_key_state(const ResinWindow *window, int key) {
            assert(window == NULL && key == 87);
            return RESIN_INPUT_PRESSED | RESIN_INPUT_RELEASED;
        }
        static uint32_t mock_mouse_state(const ResinWindow *window, int button) {
            assert(window == NULL && button == 0);
            return RESIN_INPUT_DOWN | RESIN_INPUT_PRESSED;
        }
        static ResinStatus mock_coordinates(const ResinWindow *window, double *x, double *y) {
            assert(window == NULL); *x = 12.5; *y = -3.25; return RESIN_STATUS_SUCCESS;
        }
        static int mock_focus(const ResinWindow *window) { assert(window == NULL); return 1; }
        static ResinStatus mock_capture(const ResinWindow *window, int capture) {
            static int call; assert(window == NULL && capture == (call++ == 0)); return RESIN_STATUS_SUCCESS;
        }
        #define resin_window_key_state mock_key_state
        #define resin_window_mouse_button_state mock_mouse_state
        #define resin_window_cursor_position mock_coordinates
        #define resin_window_scroll_delta mock_coordinates
        #define resin_window_focused mock_focus
        #define resin_window_capture_cursor mock_capture
        "#,
    );
    success(&output);
}

#[test]
fn gpu_views_do_not_expose_unowned_address_conversions() {
    for expression in [
        "gpu.host_to_device_pointer(Ptr<ubyte>(0_ul))",
        "value.host_pointer()",
        "value.device_pointer()",
        "value.host",
        "value.owner",
        "Ptr<int>(value)",
        "ulong(value)",
    ] {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("main.resin");
        fs::write(
            &path,
            format!(
                r#"export {{ main }}; import {{ "$/gpu.resin" }};
            def main() -> Result<(), _> = {{
                var gpu = Gpu.new()?;
                var value = gpu.new(42_i)?;
                {expression};
                ok(())
            }};"#
            ),
        )
        .unwrap();
        assert!(
            pipeline::load(&path)
                .and_then(|program| pipeline::generate_program(&program))
                .is_err(),
            "accepted unowned GPU address escape: {expression}"
        );
    }
}

#[test]
fn commands_retain_resources_until_submit_cancel_or_last_alias_drop() {
    if shaders::optimizer().is_none() {
        return;
    }
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        extern "resin_runtime.h" def test_mode(mode: int);
        extern "resin_runtime.h" def test_recorded();
        extern "resin_runtime.h" def test_code(code: int);
        extern "resin_runtime.h" def test_completed();
        extern "resin_runtime.h" def test_finished();
        @vertex_shader
        def vertex(index: int) -> Vertex = {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        };
        @fragment_shader
        def fragment(color: Color) -> Color = { color };
        def main() -> Result<(), _> = {
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            var mode = 0;
            while (mode < 4) {
                test_mode(mode);
                {
                    var commands = {
                        var original = gpu.start_command_recording()?;
                        var pipeline = gpu.create_graphics_pipeline(vertex, fragment)?;
                        original.begin_rendering(GpuImage { handle = Ptr<ResinImage>(2_ul), gpu = gpu }, 0_f, 0_f, 0_f, 1_f)?;
                        var drawing = original;
                        drawing.draw(pipeline, None, 7)?;
                        original.end_rendering()?;
                        original
                    };
                    var alias = commands;
                    test_recorded();
                    if (mode < 2) {
                        var code = match (alias.submit()) {
                            ok(value) => { 0 },
                            err(error) => { RuntimeStatus.code(error) },
                        };
                        test_code(code);
                    } else if (mode == 2) {
                        alias.cancel();
                    };
                    test_completed();
                };
                test_finished();
                mode := mode + 1;
            };
            ok(())
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static int mode, freed, completed, drawn;
        static void test_mode(int value) { mode = value; freed = completed = drawn = 0; }
        static void test_recorded(void) { assert(freed == 0 && completed == 0 && drawn == 1); }
        static void test_code(int code) { assert(code == (mode == 0 ? RESIN_STATUS_SUCCESS : RESIN_STATUS_VULKAN_ERROR)); }
        static void test_completed(void) {
            assert(freed == (mode == 3 ? 0 : 3));
            assert(completed == (mode == 3 ? 0 : 1));
        }
        static void test_finished(void) { assert(freed == 3 && completed == 1); }
        static ResinStatus mock_record(ResinGpu *gpu, ResinCommandBuffer **out) {
            assert(gpu == NULL); *out = (ResinCommandBuffer *)(uintptr_t)3; return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_graphics(ResinGpu *gpu, const uint8_t *vertex, size_t vertex_length, const uint8_t *fragment, size_t fragment_length, ResinPipeline **out) {
            assert(!gpu && vertex && fragment && vertex_length > 20 && fragment_length > 20);
            *out = (ResinPipeline *)(uintptr_t)1;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_pipeline(ResinCommandBuffer *commands, const ResinPipeline *pipeline) {
            assert((uintptr_t)commands == 3 && (uintptr_t)pipeline == 1); return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_begin(ResinCommandBuffer *commands, ResinImage *image, float r, float g, float b, float a) {
            assert((uintptr_t)commands == 3 && (uintptr_t)image == 2);
            assert(r == 0 && g == 0 && b == 0 && a == 1); return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_end(ResinCommandBuffer *commands) { assert((uintptr_t)commands == 3); return RESIN_STATUS_SUCCESS; }
        static ResinStatus mock_draw(ResinCommandBuffer *commands, ResinDeviceAddress root, uint32_t count) {
            assert((uintptr_t)commands == 3 && root == 0 && count == 7 && freed == 0 && drawn++ == 0);
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_submit(ResinGpu *gpu, ResinCommandBuffer *commands) {
            assert(gpu == NULL && (uintptr_t)commands == 3 && mode < 2 && freed == 0 && completed++ == 0);
            return mode == 0 ? RESIN_STATUS_SUCCESS : RESIN_STATUS_VULKAN_ERROR;
        }
        static void mock_cancel(ResinGpu *gpu, ResinCommandBuffer *commands) {
            assert(gpu == NULL); if (!commands) return;
            assert((uintptr_t)commands == 3 && mode >= 2 && freed == 0 && completed++ == 0);
        }
        static void mock_free_pipeline(ResinGpu *gpu, ResinPipeline *pipeline) {
            assert(gpu == NULL && completed == 1 && (uintptr_t)pipeline == 1 && !(freed & 1)); freed |= 1;
        }
        static void mock_free_image(ResinGpu *gpu, ResinImage *image) {
            assert(gpu == NULL && completed == 1 && (uintptr_t)image == 2 && !(freed & 2)); freed |= 2;
        }
        #define resin_gpu_start_command_recording mock_record
        #define resin_gpu_create_graphics_pipeline mock_graphics
        #define resin_gpu_set_pipeline mock_pipeline
        #define resin_gpu_begin_rendering mock_begin
        #define resin_gpu_end_rendering mock_end
        #define resin_gpu_draw mock_draw
        #define resin_gpu_submit mock_submit
        #define resin_gpu_cancel_command_buffer mock_cancel
        #define resin_gpu_free_pipeline mock_free_pipeline
        #define resin_gpu_free_image mock_free_image
        "#,
    );
    success(&output);
}

#[test]
fn window_constructor_accepts_owned_titles_until_the_native_call_returns() {
    let output = run(
        r#"
        export { main };
        import { "$/window.resin", "$/shared.resin", "$/string.resin" };
        extern "resin_runtime.h" def window_counts() -> int;
        def main() -> Result<int, _> = {
            var weak = WeakSpan<ubyte>.empty();
            {
                var title = String.from_str("named {0}");
                weak := title.storage.downgrade();
                var a = Window.new(32_ui, 24_ui, title)?;
                var b = Window.new(32_ui, 24_ui, String.from_str("temporary"))?;
                print(title);
            };
            var released = match (weak.upgrade()) {
                None => { 1 == 1 },
                ArcSpan<ubyte>(live) => { 1 == 0 },
            };
            ok(if (released && window_counts() == 22) { 0 } else { 1 })
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        #include <string.h>
        static int created, destroyed;
        static ResinStatus mock_window_create(uint32_t width, uint32_t height, const char *title, ResinWindow **out) {
            assert(width == 32 && height == 24);
            assert(strcmp(title, created == 0 ? "named {0}" : "temporary") == 0);
            *out = (ResinWindow *)(uintptr_t)++created;
            return RESIN_STATUS_SUCCESS;
        }
        static void mock_window_destroy(ResinWindow *window) {
            assert(window != NULL);
            ++destroyed;
        }
        static int window_counts(void) { return created * 10 + destroyed; }
        #define resin_window_create mock_window_create
        #define resin_window_destroy mock_window_destroy
        "#,
    );
    success(&output);
    assert_eq!(output.stdout, b"named {0}");
}
