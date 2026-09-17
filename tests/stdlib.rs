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
fn slice_lea_returns_pointers_and_evaluates_the_index_once() {
    success(&run(
        r#"export { main };
        import { "$/span.resin", "$/shared.resin" };
        fn index(calls: RefMut<i32>) -> u64 { calls = calls + 1; u64(1) }
        fn first<T>(items: Ref<Span<T>>) -> Ptr<T> { items:lea(u64(0)) }
        fn main() -> () | Err<_> {
            let owner = arc_span_alloc::<i32>(u64(3), i32(0))?;
            let mut calls: i32 = 0;
            let pointer: Ptr<i32> = { let borrowed = owner:get(); borrowed:lea(index(calls)) };
            pointer.* = 41;
            { let borrowed = owner:get(); first(borrowed) }.* = 7;
            assert(calls == 1 && { let borrowed = owner:get(); borrowed:at(u64(1)) } == 41);
            assert({ let borrowed = owner:get(); borrowed:at(u64(0)) } == 7 && { let borrowed = owner:get(); borrowed:at(u64(2)) } == 0);
            let array = arc_ptr_alloc([[i32(1), i32(2)], [i32(3), i32(4)]])?;
            let row = array:get():lea(u64(1));
            row:lea(u64(0)).* = 42;
            let reference: RefMut<i32> = row:at_mut(u64(1));
            reference = 19;
            assert(array:get():at(u64(1)):at(u64(0)) == 42 && row:at(u64(1)) == 19);

            assert("abc":lea(u64(1)).* == u8(98) && "abc"(u64(2)) == u8(99));
        }
        "#,
        "",
    ));
}

#[test]
fn source_strings_format_explicit_byte_views_and_keep_the_terminator_outside_length() {
    let output = run(
        r#"export { main };
        import { "$/span.resin", "$/shared.resin", "$/string.resin", "$/stdio.resin" };
        fn main() -> i32 | Err<_> {
            let buffer_owner = arc_ptr_alloc([u8(65), u8(0), u8(66)])?; let buffer: Ref<_> = buffer_owner:get().*;
            let mut text = { let borrowed = Span<u8> { data = buffer_owner:get():lea(0), length = u64(3) }; string_from_bytes(borrowed) };
            let mut weak = text.storage:downgrade();
            let mut formatted = fmt("{0}:{1}:{2}", (42, text:bytes(), "end"));
            let mut raw = formatted:get();
            let mut terminated = Span<u8> { data = raw.data, length = raw.length + u64(1) };
            print(formatted);
            if (raw.length == u64(10) && terminated:at(raw.length) == u8(0) && { let borrowed = weak:upgrade()!; borrowed:get() }.length == u64(3)) { 0 } else { 1 }
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
        struct Item { trace: Ptr<i32>, digit: i32,
            
        }
fn drop(self: RefMut<Item>)  {
                if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
            }

        fn fail(trace: Ptr<i32>) -> (() | Err<OutOfMemory>)  {
            let mut owner = arc_ptr_alloc::<Item>(Item { trace = trace, digit = 0 })?;
            owner:get().digit = 4;
            arc_span_alloc::<u64>(u64(0xffffffffffffffff), u64(0))?;
            (())
        }
        fn main() -> (i32 | Err<_>)  {
            let trace_owner = arc_ptr_alloc(0)?; let trace: RefMut<_> = trace_owner:get().*;
            {
                let items = arc_ptr_alloc([Item { trace = trace_owner:get(), digit = 0 }, Item { trace = trace_owner:get(), digit = 0 }, Item { trace = trace_owner:get(), digit = 0 }])?;
                items:get().*:at_mut(0).digit = 1;
                items:get().*:at_mut(1).digit = 2;
                items:get().*:at_mut(2).digit = 3;
            };
            let mut failed = match (fail(trace_owner:get())) { ()(value) => { 1 == 0 }, Err(error) => { 1 == 1 } };
            (if (failed && trace == 3214) { 0 } else { 1 })
        }
    "#,
        "",
    ));
}

#[test]
fn source_owned_wrappers_retain_payloads_and_borrow_named_receivers() {
    success(&run(
        r#"export { main };
        import { "$/shared.resin", "$/span.resin", "$/status.resin" };
        struct Item { trace: Ptr<i32>, digit: i32,
            
        }
fn drop(self: RefMut<Item>)  {
                if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; };
            }

        fn main() -> (i32 | Err<_>)  {
            let trace_owner = arc_ptr_alloc(0)?; let trace: RefMut<_> = trace_owner:get().*;
            let mut weak = weak_ptr_empty::<Item>();
            let mut valid = 1 == 1;
            {
                let mut owner = arc_ptr_alloc::<Item>(Item { trace = trace_owner:get(), digit = 0 })?;
                owner:get().digit = 7;
                weak = owner:downgrade();
                let copy = owner:clone();
                valid = valid && copy:get().digit == 7 && { let borrowed = weak:upgrade()!; borrowed:get() }.digit == 7;
                let mut values = arc_span_alloc::<u32>(3, u32(42))?;
                let view = values:get();
                view:at_mut(2) = u32(9);
                valid = valid && { let borrowed = values:get(); borrowed:at(0) } == u32(42) && { let borrowed = values:get(); borrowed:at(2) } == u32(9);
                {
                    let extra = arc_span_alloc::<u32>(1, u32(13))?;
                    let view = extra:get();
                    valid = valid && view:at(0) == u32(13);
                };
                let mut empty = arc_span_alloc::<u32>(0, u32(0))?;
                valid = valid && empty:get().length == u64(0);
            };
            valid = valid && trace == 7;
            valid = valid && match (weak:upgrade()) { ArcPtr<Item>(live) => { 1 == 0 }, None => { 1 == 1 } };
            valid = valid && match (arc_span_alloc::<u32>(u64(0xffffffffffffffff), u32(0))) {
                ArcSpan<u32>(owner) => { 1 == 0 }, Err(error) => { 1 == 1 },
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
        fn main() -> (i32 | Err<_>)  {
            let mut weak = weak_span_empty::<u32>();
            let mut valid = 1 == 1;
            {
                let mut memory: ArcSpan<u32>;
                memory = arc_span_alloc::<u32>(4, u32(7))?;
                weak = memory:downgrade();
                let alias = memory:clone();
                let mut values = memory:get();
                valid = valid && values.length == u64(4) && values:at(3) == u32(7);
                values:at_mut(3) = u32(42);
                valid = valid && { let borrowed = alias:get(); borrowed:at(3) } == u32(42);
                valid = valid && values:as_bytes().length == u64(4) * size_of(u32);
                let mut upgraded = weak:upgrade()!;
                valid = valid && { let borrowed = upgraded:get(); borrowed:at(3) } == u32(42);
                let mut descriptor = arc_ptr_alloc::<Span<u32>>(values:clone())?;
                let mut previous = descriptor:get():replace(Span<u32> { data = values.data, length = u64(2) });
                valid = valid && previous.length == u64(4) && descriptor:get().length == u64(2);
                valid = valid && memory:get().length == u64(4);
            };
            let mut expired = match (weak:upgrade()) {
                ArcSpan<u32>(owner) => { 1 == 0 },
                None => { 1 == 1 },
            };
            let mut empty = arc_span_alloc::<u32>(0, u32(0))?;
            valid = valid && empty:get().length == u64(0);
            let mut empty_elements = arc_span_alloc::<Empty>(19, ())?;
            valid = valid && empty_elements:get().length == u64(19);
            let mut failure: (ArcSpan<u32> | Err<OutOfMemory>);
            failure = arc_span_alloc::<u32>(u64(0xffffffffffffffff), u32(0));
            let mut failed = match (failure) {
                ArcSpan<u32>(memory) => { 1 == 0 },
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
            arc_span_alloc::<u32>(u64(0xffffffffffffffff), u32(0))?;
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
            trace: Ptr<i32>,
            digit: i32,
            
        }
fn drop(self: RefMut<Item>)  { if (self.digit != 0) { self.trace.* = self.trace.* * 10 + self.digit; }; }

        fn item(trace: Ptr<i32>, digit: i32) -> (ArcPtr<Item> | Err<_>)  {
            let mut owner = arc_ptr_alloc::<Item>(Item { trace = trace, digit = 0 })?;
            owner:get().digit = digit;
            (owner)
        }
        fn work(trace: Ptr<i32>, fail: bool) -> (() | Err<_>)  {
            let values = arc_ptr_alloc([item(trace, 1)?, item(trace, 2)?])?;
            let alias = values:clone();
            if (fail) { Err(Failed {}) } else { (()) }
        }
        fn main() -> (i32 | Err<_>)  {
            let trace_owner = arc_ptr_alloc(0)?; let trace: RefMut<_> = trace_owner:get().*;
            work(trace_owner:get(), 1 == 0)?;
            let mut valid = trace == 21;
            trace = 0;
            let mut failed = match (work(trace_owner:get(), 1 == 1)) {
                ()(value) => { 1 == 0 },
                Err(error) => { 1 == 1 },
            };
            valid = valid && failed && trace == 21;
            trace = 0;
            let mut rejected = match ({
                let owner = item(trace_owner:get(), 3)?;
                arc_span_alloc::<u64>(u64(0xffffffffffffffff), u64(0))
            }) {
                ArcSpan<u64>(values) => { 1 == 0 },
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
        fn allocate<T>(count: u64, initial: T) -> (ArcSpan<T> | Err<OutOfMemory | Other>)  {
            arc_span_alloc::<T>(count, initial)
        }
        fn optional<T>(count: u64, initial: T) -> ArcSpan<T> | None  {
            match (arc_span_alloc::<T>(count, initial)) { ArcSpan<T>(owner) => { owner }, Err(error) => { None } }
        }
        fn borrowed<T>(owner: Ref<ArcSpan<T>>) -> Span<T> { owner:get() }
        fn weaken<T>(owner: Ref<ArcSpan<T>>) -> WeakSpan<T>  { owner:downgrade() }
        fn widen_upgrade<T>(weak: Ref<WeakSpan<T>>) -> ArcSpan<T> | None | Other  { weak:upgrade() }
        fn first<T>(initial: T) -> T  {
            let mut owner = optional(1, initial)!;
            let view = owner:get();
            view:at(0)
        }
        fn main() -> (i32 | Err<_>)  {
            let mut weak = weak_span_empty::<u32>();
            let mut valid = 1 == 1;
            {
                let mut owner = allocate(2, u32(7))?;
                weak = weaken(owner);
                let mut view = borrowed(owner);
                view:at_mut(1) = u32(42);
                valid = valid && view.length == u64(2) && { let borrowed = owner:get(); borrowed:at(1) } == u32(42);
                valid = valid && match (widen_upgrade(weak)) {
                    ArcSpan<u32>(live) => { { let borrowed = live:get(); borrowed:at(1) } == u32(42) },
                    None => { 1 == 0 },
                    Other(other) => { 1 == 0 },
                };
                let mut another = optional(3, u32(9))!;
                valid = valid && another:get().length == u64(3) && { let borrowed = another:get(); borrowed:at(2) } == u32(9);
                valid = valid && first(u32(17)) == u32(17);
            };
            valid = valid && match (widen_upgrade(weak)) {
                ArcSpan<u32>(live) => { 1 == 0 },
                None => { 1 == 1 },
                Other(other) => { 1 == 0 },
            };
            valid = valid && match (allocate(u64(0xffffffffffffffff), u32(0))) {
                ArcSpan<u32>(owner) => { 1 == 0 },
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
        fn main() -> (i32 | Err<_>)  {
            let mut values = arc_span_alloc::<u32>(4, u32(0))?;
            let mut view = values:get();
            let mut middle = view:slice(1, 2);
            let alias = middle:clone();
            alias:at_mut(1) = u32(42);
            let mut valid = middle.length == u64(2) && view:at(2) == u32(42);
            valid = valid && view:at(0) == u32(0) && view:at(3) == u32(0);
            let mut end = view:slice(view.length, 0);
            valid = valid && end.length == u64(0);
            let mut null_view = Span<u32> { data = Ptr<u32>(u64(0)), length = u64(0) };
            let mut empty = null_view:slice(0, 0);
            valid = valid && empty.length == u64(0) && u64(empty.data) == u64(0);
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
    let libraries = [
        ("gpu", "gpu"),
        ("window", "window"),
        ("image", "image"),
        ("stdio", "console"),
    ];
    let modules = libraries.map(|(name, _)| {
        support::frontend::check_hir(
            &pipeline::load(&root.join(format!("resin/{name}.resin"))).unwrap(),
        )
        .into_module()
        .unwrap()
    });
    for (_, name) in libraries {
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
                fn strcmp(a: Ptr<u8>, b: Ptr<u8>) -> i32;
            },
        };
       import { "$/status.resin" };
        fn main() -> (i32 | Err<_>)  {
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
        import { "$/shared.resin", "$/image.resin", "$/span.resin", "$/status.resin" };
        fn main() -> (i32 | Err<_>)  {
            let mut path = "pixel.png";
            let buffer_owner = arc_ptr_alloc([u8(1), u8(2), u8(3), u8(255)])?; let buffer: Ref<_> = buffer_owner:get().*;
            { let borrowed = Span<u8> { data = buffer_owner:get():lea(0), length = u64(4) }; image_data_write_pixels(path.data, 1, 1, 4, borrowed, 0) }?;
            let mut image = image_data_read_png(path.data, 0)?;
            let alias = image:clone();
            let mut copy_path = "copy.png";
            alias:write_png(copy_path.data)?;
            let mut copied = image_data_read_png(copy_path.data, 0)?;
            (if (copied:width() == image:width() && copied:height() == image:height() && copied:pixels().data.* == image:pixels().data.*
                && image:width() == u32(1) && image:height() == u32(1) && image:channels() == u32(4)
                && image:pixels().data.* == u8(1) && Ptr<u8>(u64(image:pixels().data) + u64(3)).* == u8(255)) { 0 } else { 1 })
        }
        "#,
        "",
    );
    success(&output);
    for call in [
        "image_data_read_png(path.data, 4)?",
        "let view = Span<u8> { data = buffer:get():lea(0), length = u64(4) }; image_data_write_pixels(path.data, 1, 1, 4, view, 0)?",
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"$/shared.resin\", \"$/image.resin\", \"$/span.resin\", \"$/string.resin\", \"$/stdio.resin\" }}; struct Cleanup {{  }}\nfn drop(self: RefMut<Cleanup>)  {{ print(\"cleanup\\n\"); }}\n  fn main() -> (() | Err<_>)  {{ let mut path = \"missing/pixel.png\"; let buffer = arc_ptr_alloc([u8(0), u8(0), u8(0), u8(0)])?; let mut cleanup = Cleanup {{}}; {call}; (()) }}"
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
        (1, 1, 4, 3, "u64(0)"),
        (1, 2, 4, 8, "u64(5)"),
        (1, 1, 4, 4, "u64(3)"),
        (1, 2, 4, 4, "u64(0xffffffffffffffff)"),
        (0, 1, 4, 4, "u64(0)"),
        (1, 0, 4, 4, "u64(0)"),
        (1, 1, 0, 4, "u64(0)"),
        (1, 1, 5, 4, "u64(0)"),
    ] {
        let output = run(
            &format!(
                r#"export {{ main }};
                import {{ "$/shared.resin", "$/image.resin", "$/span.resin", "$/status.resin" }};
                fn main() -> i32 | Err<_> {{
                    let buffer_owner = arc_ptr_alloc([u8(0), u8(0), u8(0), u8(0), u8(0), u8(0), u8(0), u8(0)])?; let buffer: Ref<_> = buffer_owner:get().*;
                    let mut bytes = Span<u8> {{ data = buffer_owner:get():lea(0), length = {length} }};
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
        import { "$/shared.resin", "$/image.resin", "$/span.resin" };
        fn main() -> (i32 | Err<_>)  {
            let buffer_owner = arc_ptr_alloc([u8(1), u8(2), u8(3), u8(255), u8(99), u8(4), u8(5), u8(6), u8(255)])?; let buffer: Ref<_> = buffer_owner:get().*;
            let mut bytes = Span<u8> { data = buffer_owner:get():lea(0), length = u64(9) };
            image_data_write_pixels("padded.png".data, 1, 2, 4, bytes, 5)?;
            let mut image = image_data_read_png("padded.png".data, 0)?;
            let mut loaded = image:pixels();
            { let borrowed = bytes:slice(0, 4); image_data_write_pixels("single.png".data, 1, 1, 4, borrowed, u64(0xffffffffffffffff)) }?;
            let mut single = image_data_read_png("single.png".data, 0)?;
            (if (loaded:at(0) == u8(1) && loaded:at(4) == u8(4)
                && single:height() == u32(1) && { let borrowed = single:pixels(); borrowed:at(3) } == u8(255)) { 0 } else { 1 })
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
                fn test_mode(mode: i32);
                fn test_verify(code: i32);
            },
        };
       import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        @vertex_shader
        fn vertex(index: i32) -> Vertex  {
            Vertex { position = Position { x = f32(0), y = f32(0), z = f32(0), w = f32(1) },
                color = Color { r = f32(1), g = f32(0), b = f32(0), a = f32(1) } }
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
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            let mut image = gpu:create_image(u32(1), u32(1))?;
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
        fn valid_size(width: u32, height: u32) -> bool  {
            width == u32(640) && height == u32(480)
        }
        fn main() -> (i32 | Err<_>)  {
            let mut window = { let borrowed = string_from_str("queries"); window_new(u32(1), u32(1), borrowed) }?;
            let mut count = gpu_device_count()?;
            let mut size = window:framebuffer_size()?;
            let mut incomplete = match (gpu_enumerate_devices(Ptr<ResinGpuDeviceInfo>(u64(0)), 0)) {
                ()(value) => { 0 },
                Err(error) => { runtime_status_code(error) },
            };
            let mut valid = !window:should_close() && window:key_pressed(key_escape);
            window:set_should_close(1 == 1)?;
            valid = valid && window:should_close();
            window:set_should_close(1 == 0)?;
            valid = valid && !window:should_close();
            (if (valid && count == u32(2)
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
        import { "$/stdio.resin", "$/string.resin" };
        struct Cleanup {
            
        }
fn drop(self: RefMut<Cleanup>)  { print("cleanup\n"); }


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
        fn kernel(index: u64, root: Ptr<i32>)  { root.* = i32(index); }
        @vertex_shader
        fn vertex(index: i32) -> Vertex  {
            Vertex { position = Position { x = f32(0), y = f32(0), z = f32(0), w = f32(1) },
                color = Color { r = f32(1), g = f32(0), b = f32(0), a = f32(1) } }
        }
        @fragment_shader
        fn fragment(color: Color) -> Color  { color }
        fn copy_pipeline(value: GpuComputePipeline<i32, GpuPipelineOwner>) -> GpuComputePipeline<i32, GpuPipelineOwner>  { value }
        fn main() -> (i32 | Err<_>)  {
            let mut gpu = gpu_new()?;
            {
                let mut compute = gpu:create_compute_pipeline(kernel)?;
                let mut alias = copy_pipeline(compute);
                let mut graphics = gpu:create_graphics_pipeline(vertex, fragment)?;
                let mut graphics_alias = graphics;
            };
            test_fail();
            let mut compute_code = match (gpu:create_compute_pipeline(kernel)) {
                GpuComputePipeline<i32, GpuPipelineOwner>(pipeline) => { 0 }, Err(error) => { runtime_status_code(error) },
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
        fn coordinates(point: (f64, f64)) -> bool  { point.0 == f64(12.5) && point.1 == f64(-3.25) }
        fn main() -> (i32 | Err<_>)  {
            let mut window = { let borrowed = string_from_str("input snapshot"); window_new(u32(16), u32(16), borrowed) }?;
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
        "gpu.host_to_device_pointer(Ptr<u8>(u64(0)))",
        "value.host_pointer()",
        "value.device_pointer()",
        "value.host",
        "value.owner",
        "Ptr<i32>(value)",
        "u64(value)",
    ] {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("main.resin");
        fs::write(
            &path,
            format!(
                r#"export {{ main }}; import {{ "$/gpu.resin" }};
            fn main() -> (() | Err<_>)  {{
                let mut gpu = gpu_new()?;
                let mut value = gpu:create(i32(42))?;
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
                fn test_mode(mode: i32);
                fn test_recorded();
                fn test_code(code: i32);
                fn test_completed();
                fn test_finished();
            },
        };
       import { "$/gpu.resin", "$/graphics.resin", "$/status.resin" };
        @vertex_shader
        fn vertex(index: i32) -> Vertex  {
            Vertex { position = Position { x = f32(0), y = f32(0), z = f32(0), w = f32(1) },
                color = Color { r = f32(1), g = f32(0), b = f32(0), a = f32(1) } }
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
                        { let borrowed = gpu:create_image(u32(1), u32(1))?; original:begin_rendering(borrowed, f32(0), f32(0), f32(0), f32(1)) }?;
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
fn window_constructor_borrows_owned_titles() {
    let output = run(
        r#"export { main };

        extern {
            "resin_runtime.h": {
                fn window_counts() -> i32;
            },
        };
       import { "$/window.resin", "$/shared.resin", "$/string.resin", "$/stdio.resin" };
        fn main() -> (i32 | Err<_>)  {
            let mut weak = weak_span_empty::<u8>();
            {
                let mut title = string_from_str("named {0}");
                weak = title.storage:downgrade();
                let mut a = window_new(u32(32), u32(24), title)?;
                let mut b = { let borrowed = string_from_str("temporary"); window_new(u32(32), u32(24), borrowed) }?;
                print(title);
            };
            let mut released = match (weak:upgrade()) {
                None => { 1 == 1 },
                ArcSpan<u8>(live) => { 1 == 0 },
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

#[test]
fn argparse_yields_aliases_values_and_duplicates_in_order() {
    success(&run(
        r#"export { main };
        import { "$/shared.resin", "$/argparse.resin", "$/span.resin", "$/string.resin" };
        fn main() -> () | Err<_> {
            let argv_owner = arc_ptr_alloc(["app".data, "-v".data, "--count=12".data, "-o".data, "a path.png".data,
                "--count".data, "4294967295".data, "--real".data, "-0.125".data, "--output=".data])?; let argv: Ref<_> = argv_owner:get().*;
            let mut parser = { let borrowed = Span<Ptr<u8>> { data = argv_owner:get():lea(u64(0)), length = u64(10) }; argparse(borrowed,
                "  --verbose|-v --count= --output|-o= --real= ") };
            let flag = parser:next()?!;
            assert(flag:named("--verbose") && flag.value.length == u64(0));
            let count = parser:next()?!;
            assert(count:named("--count") && argument_integer(count.value)? == u32(12));
            let path = parser:next()?!;
            assert(path:named("--output") && path.value.length == u64(10));
            let repeated = parser:next()?!;
            assert(argument_integer(repeated.value)? == u32(4294967295));
            let real = parser:next()?!;
            assert(argument_number(real.value)? == f32(-0.125));
            let empty = parser:next()?!;
            assert(empty:named("--output") && empty.value.length == u64(0));
            assert(match (parser:next()?) { None => { true }, Argument(item) => { false } });
        }
    "#,
        "",
    ));
}

#[test]
fn argparse_reports_errors_and_numeric_parsing_respects_span_bounds() {
    success(&run(
        r#"export { main };
        import { "$/shared.resin", "$/argparse.resin", "$/span.resin", "$/string.resin" };
        fn rejected(option: str) -> () | Err<_> {
            let argv = arc_ptr_alloc(["app".data, option.data])?;
            let mut parser = { let borrowed = Span<Ptr<u8>> { data = argv:get():lea(u64(0)), length = u64(2) }; argparse(borrowed, "--flag --count=") };
            assert(match (parser:next()) {
                Err(message) => { message:get().length > u64(0) },
                Argument(item) => { false }, None => { false },
            });
        }
        fn bad_integer(text: str) {
            assert(match ({ let borrowed = bytes(text); argument_integer(borrowed) }) { Err(message) => { true }, u32(value) => { false } });
        }
        fn bad_number(text: str) {
            assert(match ({ let borrowed = bytes(text); argument_number(borrowed) }) { Err(message) => { true }, f32(value) => { false } });
        }
        fn main() -> () | Err<_> {
            rejected("--unknown")?; rejected("--count")?; rejected("--flag=yes")?; rejected("positional")?;
            bad_integer(""); bad_integer("4294967296"); bad_integer("-1"); bad_integer("1x");
            bad_number(""); bad_number("nan"); bad_number("inf"); bad_number("1e100"); bad_number("1.0junk");
            let bounded_owner = arc_ptr_alloc([u8(49), u8(46), u8(50), u8(53), u8(57)])?; let bounded: Ref<_> = bounded_owner:get().*;
            assert({ let borrowed = Span<u8> { data = bounded_owner:get():lea(u64(0)), length = u64(4) }; argument_number(borrowed) }? == f32(1.25));
            let embedded_owner = arc_ptr_alloc([u8(49), u8(0), u8(50)])?; let embedded: Ref<_> = embedded_owner:get().*;
            assert(match ({ let borrowed = Span<u8> { data = embedded_owner:get():lea(u64(0)), length = u64(3) }; argument_number(borrowed) }) {
                Err(message) => { true }, f32(value) => { false },
            });
        }
    "#,
        "",
    ));
}
