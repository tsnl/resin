#[allow(dead_code)]
mod support;

use std::{fs, path::Path, process::Command};
use support::pipeline;
use support::project;
use support::shaders;
use support::toolchain;
use tempfile::TempDir;

fn run(source: &str, native: &str) -> std::process::Output {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("main.resin");
    fs::write(&path, source).unwrap();
    let module = pipeline::file_module(&path).unwrap_or_else(|error| panic!("{error}"));
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
        r#"export { main };
        import { "$/span.resin", "$/shared.resin", "$/string.resin" };
        fn main() -> int  {
            let buffer = [65_ub, 0_ub, 66_ub];
            let mut text = string_from_bytes(Span<ubyte> { data = &buffer:at(0), length = 3_ul });
            let mut weak = text.storage:downgrade();
            let mut formatted = fmt("{0}:{1}:{2}", (42, text:bytes(), "end"));
            let mut raw = formatted:get();
            let mut terminated = Span<ubyte> { data = raw.data, length = raw.length + 1_ul };
            print(formatted);
            if (raw.length == 10_ul && terminated:at(raw.length) == 0_ub && weak:upgrade()!:get().length == 3_ul) { 0 } else { 1 }
        }
    "#,
        "",
    );
    success(&output);
    assert_eq!(output.stdout, b"42:A\0B:end");
}

#[test]
fn source_shared_elements_drop_in_reverse_and_unwind_on_allocation_failure() {
    success(&run(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin", "$/status.resin" };
        struct Item { trace: Ptr<int>, digit: int,
            
        }
fn drop(self: Ptr<Item>)  {
                if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
            }

        fn fail(trace: Ptr<int>) -> (() | Err<OutOfMemory>)  {
            let mut owner = arc_ptr_alloc::<Item>(Item { trace = trace, digit = 0 })?;
            owner:get().digit = 4;
            arc_span_alloc::<ulong>(0xffffffffffffffff_ul, 0_ul)?;
            (())
        }
        fn main() -> (int | Err<_>)  {
            let mut trace = 0;
            {
                let items = arc_ptr_alloc([Item { trace = &trace, digit = 0 }, Item { trace = &trace, digit = 0 }, Item { trace = &trace, digit = 0 }])?;
                items:get().*:at(0).digit = 1;
                items:get().*:at(1).digit = 2;
                items:get().*:at(2).digit = 3;
            };
            let mut failed = match (fail(&trace)) { ()(value) => { 1 == 0 }, Err(error) => { 1 == 1 } };
            (if (failed && trace == 3214) { 0 } else { 1 })
        }
    "#,
        "",
    ));
}

#[test]
fn source_owned_wrappers_retain_payloads_and_borrow_temporary_receivers() {
    success(&run(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin", "$/status.resin" };
        struct Item { trace: Ptr<int>, digit: int,
            
        }
fn drop(self: Ptr<Item>)  {
                if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
            }

        fn main() -> (int | Err<_>)  {
            let mut trace = 0;
            let mut weak = weak_ptr_empty::<Item>();
            let mut valid = 1 == 1;
            {
                let mut owner = arc_ptr_alloc::<Item>(Item { trace = &trace, digit = 0 })?;
                owner:get().digit = 7;
                weak = owner:downgrade();
                let copy = owner:clone();
                valid = valid && copy:get().digit == 7 && weak:upgrade()!:get().digit == 7;
                let mut values = arc_span_alloc::<uint>(3, 42_ui)?;
                values:get():at(2) = 9_ui;
                valid = valid && values:get():at(0) == 42_ui && values:get():at(2) == 9_ui;
                valid = valid && arc_span_alloc::<uint>(1, 13_ui)?:get():at(0) == 13_ui;
                let mut empty = arc_span_alloc::<uint>(0, 0_ui)?;
                valid = valid && empty:get().length == 0_ul;
            };
            valid = valid && trace == 7;
            valid = valid && match (weak:upgrade()) { ArcPtr<Item>(live) => { 1 == 0 }, None => { 1 == 1 } };
            valid = valid && match (arc_span_alloc::<uint>(0xffffffffffffffff_ul, 0_ui)) {
                ArcSpan<uint>(owner) => { 1 == 0 }, Err(error) => { 1 == 1 },
            };
            (if (valid) { 0 } else { 1 })
        }
    "#,
        "",
    ));
}

#[test]
fn source_owners_allocate_initialized_typed_storage_and_reports_overflow() {
    let output = run(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin", "$/status.resin" };
        type Empty = ();
        fn main() -> (int | Err<_>)  {
            let mut weak = weak_span_empty::<uint>();
            let mut valid = 1 == 1;
            {
                let mut memory: ArcSpan<uint>;
                memory = arc_span_alloc::<uint>(4, 7_ui)?;
                weak = memory:downgrade();
                let alias = memory:clone();
                let mut values = memory:get();
                valid = valid && values.length == 4_ul && values:at(3) == 7_ui;
                values:at(3) = 42_ui;
                valid = valid && alias:get():at(3) == 42_ui;
                valid = valid && values:as_bytes().length == 4_ul * size_of(uint);
                let mut upgraded = weak:upgrade()!;
                valid = valid && upgraded:get():at(3) == 42_ui;
                let mut descriptor = arc_ptr_alloc::<Span<uint>>(values:clone())?;
                let mut previous = descriptor:get():replace(Span<uint> { data = values.data, length = 2_ul });
                valid = valid && previous.length == 4_ul && descriptor:get().length == 2_ul;
                valid = valid && memory:get().length == 4_ul;
            };
            let mut expired = match (weak:upgrade()) {
                ArcSpan<uint>(owner) => { 1 == 0 },
                None => { 1 == 1 },
            };
            let mut empty = arc_span_alloc::<uint>(0, 0_ui)?;
            valid = valid && empty:get().length == 0_ul;
            let mut empty_elements = arc_span_alloc::<Empty>(19, ())?;
            valid = valid && empty_elements:get().length == 19_ul;
            let mut failure: (ArcSpan<uint> | Err<OutOfMemory>);
            failure = arc_span_alloc::<uint>(0xffffffffffffffff_ul, 0_ui);
            let mut failed = match (failure) {
                ArcSpan<uint>(memory) => { 1 == 0 },
                Err(error) => { 1 == 1 },
            };
            (if (valid && expired && failed) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);

    // Error propagation does not require importing the module's private dependencies.
    let output = run(
        r#"export { main };
        import { "$/shared.resin" };
        fn main() -> (() | Err<_>)  {
            arc_span_alloc::<uint>(0xffffffffffffffff_ul, 0_ui)?;
            (())
        }
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
fn shared_arrays_release_managed_elements_on_success_and_error() {
    let output = run(
        r#"export { main };
        import { "$/shared.resin" };
        struct Failed {}
        struct Item {
            trace: Ptr<int>,
            digit: int,
            
        }
fn drop(self: Ptr<Item>)  { if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; }; }

        fn item(trace: Ptr<int>, digit: int) -> (ArcPtr<Item> | Err<_>)  {
            let mut owner = arc_ptr_alloc::<Item>(Item { trace = trace, digit = 0 })?;
            owner:get().digit = digit;
            (owner)
        }
        fn work(trace: Ptr<int>, fail: bool) -> (() | Err<_>)  {
            let values = arc_ptr_alloc([item(trace, 1)?, item(trace, 2)?])?;
            let alias = values:clone();
            if (fail) { Err(Failed {}) } else { (()) }
        }
        fn main() -> (int | Err<_>)  {
            let mut trace = 0;
            work(&trace, 1 == 0)?;
            let mut valid = trace == 21;
            trace = 0;
            let mut failed = match (work(&trace, 1 == 1)) {
                ()(value) => { 1 == 0 },
                Err(error) => { 1 == 1 },
            };
            valid = valid && failed && trace == 21;
            trace = 0;
            let mut rejected = match ({
                let owner = item(&trace, 3)?;
                arc_span_alloc::<ulong>(0xffffffffffffffff_ul, 0_ul)
            }) {
                ArcSpan<ulong>(values) => { 1 == 0 },
                Err(error) => { 1 == 1 },
            };
            (if (valid && rejected && trace == 3) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);
}

#[test]
fn generic_owned_span_methods_preserve_lifetimes_and_widened_results() {
    let output = run(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin", "$/status.resin" };
        struct Other {}
        fn allocate<T>(count: ulong, initial: T) -> (ArcSpan<T> | Err<OutOfMemory | Other>)  {
            arc_span_alloc::<T>(count, initial)
        }
        fn optional<T>(count: ulong, initial: T) -> ArcSpan<T> | None  {
            match (arc_span_alloc::<T>(count, initial)) { ArcSpan<T>(owner) => { owner }, Err(error) => { None } }
        }
        fn borrowed<T>(owner: Ref<ArcSpan<T>>) -> Span<T> { owner:get() }
        fn weaken<T>(owner: Ref<ArcSpan<T>>) -> WeakSpan<T>  { owner:downgrade() }
        fn widen_upgrade<T>(weak: Ref<WeakSpan<T>>) -> ArcSpan<T> | None | Other  { weak:upgrade() }
        fn first<T>(initial: T) -> T  {
            let mut owner = optional(1, initial)!;
            owner:get():at(0)
        }
        fn main() -> (int | Err<_>)  {
            let mut weak = weak_span_empty::<uint>();
            let mut valid = 1 == 1;
            {
                let mut owner = allocate(2, 7_ui)?;
                weak = weaken(owner);
                let mut view = borrowed(owner);
                view:at(1) = 42_ui;
                valid = valid && view.length == 2_ul && owner:get():at(1) == 42_ui;
                valid = valid && match (widen_upgrade(weak)) {
                    ArcSpan<uint>(live) => { live:get():at(1) == 42_ui },
                    None => { 1 == 0 },
                    Other(other) => { 1 == 0 },
                };
                let mut another = optional(3, 9_ui)!;
                valid = valid && another:get().length == 3_ul && another:get():at(2) == 9_ui;
                valid = valid && first(17_ui) == 17_ui;
            };
            valid = valid && match (widen_upgrade(weak)) {
                ArcSpan<uint>(live) => { 1 == 0 },
                None => { 1 == 1 },
                Other(other) => { 1 == 0 },
            };
            valid = valid && match (allocate(0xffffffffffffffff_ul, 0_ui)) {
                ArcSpan<uint>(owner) => { 1 == 0 },
                Err(error) => {
                    match (error) {
                        OutOfMemory(error) => { 1 == 1 },
                        Other(other) => { 1 == 0 },
                    }
                },
            };
            (if (valid) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);
}

#[test]
fn borrowed_span_slices_preserve_aliases_and_accept_empty_null_views() {
    let output = run(
        r#"export { main }; import { "$/shared.resin", "$/span.resin" };
        fn main() -> (int | Err<_>)  {
            let mut values = arc_span_alloc::<uint>(4, 0_ui)?;
            let mut view = values:get();
            let mut middle = view:slice(1, 2);
            let alias = middle:clone();
            alias:at(1) = 42_ui;
            let mut valid = middle.length == 2_ul && view:at(2) == 42_ui;
            valid = valid && view:at(0) == 0_ui && view:at(3) == 0_ui;
            let mut end = view:slice(view.length, 0);
            valid = valid && end.length == 0_ul;
            let mut null_view = Span<uint> { data = Ptr<uint>(0_ul), length = 0_ul };
            let mut empty = null_view:slice(0, 0);
            valid = valid && empty.length == 0_ul && ulong(empty.data) == 0_ul;
            (if (valid) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);
}

#[test]
fn every_native_status_operation_has_a_public_result_wrapper() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let libraries = ["gpu", "window", "image", "console"];
    let modules = libraries.map(|name| {
        support::frontend::check_hir(
            &pipeline::load(&root.join(format!("resin/{name}.resin"))).unwrap(),
        )
        .into_module()
        .unwrap()
    });
    for name in libraries {
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
            let operation = match name {
                "gpu_create" => "gpu_new",
                "gpu_create_at" => "gpu_new_at",
                "gpu_device_count" => "gpu_device_count",
                "gpu_enumerate_devices" => "gpu_enumerate_devices",
                "gpu_create_for_window" => "gpu_new_for_window",
                "window_create" => "window_new",
                "image_read_png" => "image_data_read_png",
                "image_write_png" => "image_data_write_pixels",
                // These raw native operations have been replaced by owning
                // views and compiler-generated projection in Resin source.
                "gpu_malloc" | "gpu_dispatch" | "gpu_copy_image_to_buffer" | "gpu_set_pipeline" => {
                    continue;
                }
                "gpu_ptr_allocate" => "malloc",
                "gpu_create_compute_pipeline" => "create_compute_pipeline",
                "gpu_create_graphics_pipeline" => "create_graphics_pipeline",
                "gpu_create_image" => "create_image",
                "gpu_start_command_recording" => "start_command_recording",
                "gpu_projected_dispatch" => "dispatch",
                "gpu_begin_rendering" => "begin_rendering",
                "gpu_end_rendering" => "end_rendering",
                "gpu_draw" | "gpu_projected_draw" => "draw",
                "gpu_copy_image_to_span" => "copy_image_to_buffer",
                "gpu_submit" => "submit",
                "gpu_cancel_command_buffer" => "cancel",
                "window_poll_events" => "poll_events",
                "window_wait_events" => "wait_events",
                "window_should_close" => "should_close",
                "window_set_should_close" => "set_should_close",
                "window_framebuffer_size" => "framebuffer_size",
                "window_set_size" => "set_size",
                "window_key_pressed" => "key_pressed",
                "window_key_state" => "key_state",
                "window_mouse_button_state" => "mouse_button_state",
                "window_cursor_position" => "cursor_position",
                "window_scroll_delta" => "scroll_delta",
                "window_focused" => "focused",
                "window_capture_cursor" => "capture_cursor",
                "gpu_present" => "present",
                _ => panic!("missing operation mapping for native operation: {name}"),
            };
            let function = modules
                .iter()
                .find_map(|module| {
                    module
                        .entries
                        .get(operation)
                        .map(|id| &module.functions[id.index()])
                })
                .unwrap_or_else(|| panic!("missing exported operation {operation}"));
            assert!(function.foreign_header.is_none(), "{name}");
            assert!(
                matches!(function.signature.result.ty, resin_hir::Type::Union { .. }),
                "{name}"
            );
            checked += 1;
        }
        assert!(checked > 0 || name == "console");
    }
}

#[test]
fn native_statuses_become_named_errors_and_keep_unknown_codes() {
    let output = run(
        r#"export { main };

        extern {
            "string.h": {
                fn strcmp(a: Ptr<ubyte>, b: Ptr<ubyte>) -> int;
            },
        };
       import { "$/status.resin" };
        fn main() -> (int | Err<_>)  {
            runtime_status_from_code(0)?;
            let mut code = -1;
            let mut valid = 1 == 1;
            while (code <= 9) {
                let mut actual = match (runtime_status_from_code(code)) {
                    ()(value) => { 0 },
                    Err(error) => { runtime_status_code(error) },
                };
                valid = valid && actual == code;
                code = code + 1;
            };
            let mut message = "io error";
            valid = valid && strcmp(runtime_status_message(IoError {}), message.data) == 0;
            (if (valid) { 0 } else { 1 })
        }"#,
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
                "export {{ main }}; import {{ \"$/status.resin\" }}; fn main() -> (() | Err<_>)  {{ runtime_status_from_code({code}) }}"
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
        r#"export { main };
        import { "$/image.resin", "$/span.resin", "$/status.resin" };
        fn main() -> (int | Err<_>)  {
            let mut path = "pixel.png";
            let mut buffer = [ubyte(1), ubyte(2), ubyte(3), ubyte(255)];
            image_data_write_pixels(path.data, 1, 1, 4, Span<ubyte> { data = &buffer:at(0), length = 4_ul }, 0)?;
            let mut image = image_data_read_png(path.data, 0)?;
            let alias = image:clone();
            let mut copy_path = "copy.png";
            alias:write_png(copy_path.data)?;
            let mut copied = image_data_read_png(copy_path.data, 0)?;
            (if (copied:width() == image:width() && copied:height() == image:height() && copied:pixels().data.* == image:pixels().data.*
                && image:width() == uint(1) && image:height() == uint(1) && image:channels() == uint(4)
                && image:pixels().data.* == ubyte(1) && Ptr<ubyte>(ulong(image:pixels().data) + ulong(3)).* == ubyte(255)) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);
    for call in [
        "image_data_read_png(path.data, 4)?",
        "image_data_write_pixels(path.data, 1, 1, 4, Span<ubyte> { data = &buffer:at(0), length = 4_ul }, 0)?",
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"$/image.resin\", \"$/span.resin\", \"$/string.resin\" }}; struct Cleanup {{  }}\nfn drop(self: Ptr<Cleanup>)  {{ print(fmt(\"cleanup\\n\", ())); }}\n  fn main() -> (() | Err<_>)  {{ let mut path = \"missing/pixel.png\"; let mut buffer = [0_ub, 0_ub, 0_ub, 0_ub]; let mut cleanup = Cleanup {{}}; {call}; (()) }}"
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
                r#"export {{ main }};
                import {{ "$/image.resin", "$/span.resin", "$/status.resin" }};
                fn main() -> int  {{
                    let mut buffer = [0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub, 0_ub];
                    let mut bytes = Span<ubyte> {{ data = &buffer:at(0), length = {length}_ul }};
                    match (image_data_write_pixels("missing/pixel.png".data, {width}, {height}, {channels}, bytes, {stride})) {{
                        ()(value) => {{ 1 }},
                        Err(error) => {{ if (runtime_status_code(error) == 1) {{ 0 }} else {{ 1 }} }},
                    }}
                }}
                "#
            ),
            "",
        );
        success(&output);
    }

    let output = run(
        r#"export { main };
        import { "$/image.resin", "$/span.resin" };
        fn main() -> (int | Err<_>)  {
            let mut buffer = [1_ub, 2_ub, 3_ub, 255_ub, 99_ub, 4_ub, 5_ub, 6_ub, 255_ub];
            let mut bytes = Span<ubyte> { data = &buffer:at(0), length = 9_ul };
            image_data_write_pixels("padded.png".data, 1, 2, 4, bytes, 5)?;
            let mut image = image_data_read_png("padded.png".data, 0)?;
            let mut loaded = image:pixels();
            image_data_write_pixels("single.png".data, 1, 1, 4, bytes:slice(0, 4), 0xffffffffffffffff_ul)?;
            let mut single = image_data_read_png("single.png".data, 0)?;
            (if (loaded:at(0) == 1_ub && loaded:at(4) == 4_ub
                && single:height() == 1_ui && single:pixels():at(3) == 255_ub) { 0 } else { 1 })
        }
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
        r#"export { main };

        extern {
            "resin_runtime.h": {
                fn test_mode(mode: int);
                fn test_verify(code: int);
            },
        };
       import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        @vertex_shader
        fn vertex(index: int) -> Vertex  {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        }
        @fragment_shader
        fn fragment(color: Color) -> Color  { color }
        fn work() -> (() | Err<_>)  {
            let mut gpu = gpu_new()?;

            let mut allocation = gpu:malloc(16, 8, memory_default)?;

            let mut commands = gpu:start_command_recording()?;

            let mut pipeline = gpu:create_graphics_pipeline(vertex, fragment)?;
            commands:draw(pipeline, None, 3)?;
            commands:submit()?;
            (())
        }
        fn main()  {
            let mut mode = 0;
            while (mode < 6) {
                test_mode(mode);
                let mut result = match (work()) {
                    ()(value) => { 0 },
                    Err(error) => { runtime_status_code(error) },
                };
                test_verify(result);
                mode = mode + 1;
            };
        }"#,
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
            *out = mode == 0 ? NULL : (ResinGpu *)(uintptr_t)1;
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
            *out = mode == 2 ? NULL : (ResinCommandBuffer *)(uintptr_t)3;
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
        r#"export { main };
        import { "$/gpu.resin", "$/window.resin", "$/status.resin" };
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut image = gpu:create_image(1_ui, 1_ui)?;
            let mut first = match (gpu:present(image)) { bool(shown) => { shown }, Err(e) => { 1 == 0 } };
            let mut second = match (gpu:present(image)) { bool(shown) => { !shown }, Err(e) => { 1 == 0 } };
            let mut third = match (gpu:present(image)) { bool(shown) => { 0 }, Err(e) => { runtime_status_code(e) } };
            let mut fourth = match (gpu:present(image)) { bool(shown) => { 0 }, Err(e) => { runtime_status_code(e) } };
            (if (first && second && third == 5 && fourth == 99) { 0 } else { 1 })
        }
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_create(ResinGpu **out) { *out = NULL; return RESIN_STATUS_SUCCESS; }
        #define resin_gpu_create mock_create
        static ResinStatus mock_image(ResinGpu *gpu, uint32_t width, uint32_t height, ResinImage **out) {
            assert(gpu == NULL && width == 1 && height == 1); *out = NULL; return RESIN_STATUS_SUCCESS;
        }
        #define resin_gpu_create_image mock_image
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
        r#"export { main };
        import { "$/gpu.resin", "$/window.resin", "$/status.resin", "$/string.resin" };
        fn valid_size(width: uint, height: uint) -> bool  {
            width == uint(640) && height == uint(480)
        }
        fn main() -> (int | Err<_>)  {
            let mut window = window_new(1_ui, 1_ui, string_from_str("queries"))?;
            let mut count = gpu_device_count()?;
            let mut size = window:framebuffer_size()?;
            let mut incomplete = match (gpu_enumerate_devices(Ptr<ResinGpuDeviceInfo>(ulong(0)), 0)) {
                ()(value) => { 0 },
                Err(error) => { runtime_status_code(error) },
            };
            let mut valid = !window:should_close() && window:key_pressed(key_escape);
            window:set_should_close(1 == 1)?;
            valid = valid && window:should_close();
            window:set_should_close(1 == 0)?;
            valid = valid && !window:should_close();
            (if (valid && count == uint(2)
                && valid_size(size.0, size.1) && incomplete == 7) { 0 } else { 1 })
        }
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_window(uint32_t width, uint32_t height, const char *title, ResinWindow **out) {
            assert(width == 1 && height == 1 && title); *out = NULL; return RESIN_STATUS_SUCCESS;
        }
        #define resin_window_create mock_window
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
        r#"export { main };
        import { "$/console.resin", "$/string.resin" };
        struct Cleanup {
            
        }
fn drop(self: Ptr<Cleanup>)  { print("cleanup\n"); }


        fn main() -> (() | Err<_>)  {
            let mut cleanup = Cleanup {};
            console_read_byte()?;
            (())
        }
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
        r#"export { main };

        extern {
            "resin_runtime.h": {
                fn test_finished();
                fn test_fail();
            },
        };
       import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        @compute_shader
        fn kernel(index: ulong, root: Ptr<int>)  { root.* = int(index); }
        @vertex_shader
        fn vertex(index: int) -> Vertex  {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        }
        @fragment_shader
        fn fragment(color: Color) -> Color  { color }
        fn copy_pipeline(value: GpuComputePipeline<int, GpuPipelineOwner>) -> GpuComputePipeline<int, GpuPipelineOwner>  { value }
        fn main() -> (int | Err<_>)  {
            let mut gpu = gpu_new()?;
            {
                let mut compute = gpu:create_compute_pipeline(kernel)?;
                let mut alias = copy_pipeline(compute);
                let mut graphics = gpu:create_graphics_pipeline(vertex, fragment)?;
                let mut graphics_alias = graphics;
            };
            test_fail();
            let mut compute_code = match (gpu:create_compute_pipeline(kernel)) {
                GpuComputePipeline<int, GpuPipelineOwner>(pipeline) => { 0 }, Err(error) => { runtime_status_code(error) },
            };
            let mut graphics_code = match (gpu:create_graphics_pipeline(vertex, fragment)) {
                GpuGraphicsPipeline<None, GpuPipelineOwner>(pipeline) => { 0 }, Err(error) => { runtime_status_code(error) },
            };
            test_finished();
            (if (compute_code == 5 && graphics_code == 5) { 0 } else { 1 })
        }"#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_create(ResinGpu **out) { *out = NULL; return RESIN_STATUS_SUCCESS; }
        #define resin_gpu_create mock_create
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
        r#"export { main };
        import { "$/window.resin", "$/string.resin" };
        fn coordinates(point: (float64, float64)) -> bool  { point.0 == 12.5_d && point.1 == -3.25_d }
        fn main() -> (int | Err<_>)  {
            let mut window = window_new(16_ui, 16_ui, string_from_str("input snapshot"))?;
            let mut key = window:key_state(key_w);
            let mut mouse = window:mouse_button_state(mouse_button_left);
            let mut valid = !key.down && key.pressed && key.released && mouse.down && mouse.pressed && !mouse.released;
            valid = valid && coordinates(window:cursor_position()?);
            valid = valid && coordinates(window:scroll_delta()?);
            valid = valid && window:focused() && key_escape == 256 && key_f25 == 314;
            window:capture_cursor(1 == 1)?;
            window:capture_cursor(1 == 0)?;
            (if (valid) { 0 } else { 1 })
        }
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_window_create(uint32_t width, uint32_t height, const char *title, ResinWindow **out) {
            assert(width == 16 && height == 16 && title);
            *out = NULL;
            return RESIN_STATUS_SUCCESS;
        }
        #define resin_window_create mock_window_create
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
            fn main() -> (() | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut value = gpu:create(42_i)?;
                {expression};
                (())
            }}"#
            ),
        )
        .unwrap();
        assert!(
            pipeline::file_module(&path).is_err(),
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
        r#"export { main };

        extern {
            "resin_runtime.h": {
                fn test_mode(mode: int);
                fn test_recorded();
                fn test_code(code: int);
                fn test_completed();
                fn test_finished();
            },
        };
       import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        @vertex_shader
        fn vertex(index: int) -> Vertex  {
            Vertex { position = Position { x = 0_f, y = 0_f, z = 0_f, w = 1_f },
                color = Color { r = 1_f, g = 0_f, b = 0_f, a = 1_f } }
        }
        @fragment_shader
        fn fragment(color: Color) -> Color  { color }
        fn main() -> (() | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut mode = 0;
            while (mode < 4) {
                test_mode(mode);
                {
                    let mut commands = {
                        let mut original = gpu:start_command_recording()?;
                        let mut pipeline = gpu:create_graphics_pipeline(vertex, fragment)?;
                        original:begin_rendering(gpu:create_image(1_ui, 1_ui)?, 0_f, 0_f, 0_f, 1_f)?;
                        let drawing = original:clone();
                        drawing:draw(pipeline, None, 7)?;
                        original:end_rendering()?;
                        original
                    };
                    let alias = commands:clone();
                    test_recorded();
                    if (mode < 2) {
                        let mut code = match (alias:submit()) {
                            ()(value) => { 0 },
                            Err(error) => { runtime_status_code(error) },
                        };
                        test_code(code);
                    } else if (mode == 2) {
                        alias:cancel();
                    };
                    test_completed();
                };
                test_finished();
                mode = mode + 1;
            };
            (())
        }"#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_create(ResinGpu **out) { *out = NULL; return RESIN_STATUS_SUCCESS; }
        #define resin_gpu_create mock_create
        static ResinStatus mock_image(ResinGpu *gpu, uint32_t width, uint32_t height, ResinImage **out) {
            assert(gpu == NULL && width == 1 && height == 1); *out = (ResinImage *)(uintptr_t)2; return RESIN_STATUS_SUCCESS;
        }
        #define resin_gpu_create_image mock_image
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
        r#"export { main };

        extern {
            "resin_runtime.h": {
                fn window_counts() -> int;
            },
        };
       import { "$/window.resin", "$/shared.resin", "$/string.resin" };
        fn main() -> (int | Err<_>)  {
            let mut weak = weak_span_empty::<ubyte>();
            {
                let mut title = string_from_str("named {0}");
                weak = title.storage:downgrade();
                let mut a = window_new(32_ui, 24_ui, title)?;
                let mut b = window_new(32_ui, 24_ui, string_from_str("temporary"))?;
                print(title);
            };
            let mut released = match (weak:upgrade()) {
                None => { 1 == 1 },
                ArcSpan<ubyte>(live) => { 1 == 0 },
            };
            (if (released && window_counts() == 22) { 0 } else { 1 })
        }"#,
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
