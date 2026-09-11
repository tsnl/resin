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
                "gpu_malloc" | "gpu_dispatch" | "gpu_copy_image_to_buffer" => continue,
                "gpu_ptr_allocate" => ("GpuOwner", "malloc"),
                "gpu_create_compute_pipeline" => ("GpuOwner", "create_compute_pipeline"),
                "gpu_create_graphics_pipeline" => ("GpuOwner", "create_graphics_pipeline"),
                "gpu_create_image" => ("GpuOwner", "create_image"),
                "gpu_start_command_recording" => ("GpuOwner", "start_command_recording"),
                "gpu_set_pipeline" => ("CommandsOwner", "set_pipeline"),
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
            ImageData.write_pixels(path.data, 1, 1, 4, Ptr<ubyte>(&pixels), 0)?;
            var image = ImageData.read_png(path.data, 0)?;
            var alias = image;
            var copy_path = "copy.png";
            alias.write_png(copy_path.data)?;
            var copied = ImageData.read_png(copy_path.data, 0)?;
            ok(if (copied.width == image.width && copied.height == image.height && copied.pixels.* == image.pixels.*
                && image.width == uint(1) && image.height == uint(1) && image.channels == uint(4)
                && image.pixels.* == ubyte(1) && Ptr<ubyte>(ulong(image.pixels) + ulong(3)).* == ubyte(255)) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
    for call in [
        "ImageData.read_png(path.data, 4)?",
        "ImageData.write_pixels(path.data, 1, 1, 4, Ptr<ubyte>(&pixels), 0)?",
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"$/image.resin\" }}; struct Cleanup {{}}; impl Cleanup {{ def drop(self: Ptr<Cleanup>) = {{ print(fmt(\"cleanup\\n\", ())); }}; }} def main() -> Result<(), _> = {{ var path = \"missing/pixel.png\"; var pixels = [uint(0)]; var cleanup = Cleanup {{}}; {call}; ok(()) }};"
            ),
            "",
        );
        assert_eq!(output.status.code(), Some(1));
        assert_eq!(output.stdout, b"cleanup\n");
        assert!(String::from_utf8_lossy(&output.stderr).contains("unhandled error: IoError"));
    }
}

#[test]
fn gpu_cleanup_covers_acquisition_recording_and_submission_failures() {
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/status.resin" };
        extern "resin_runtime.h" def test_mode(mode: int);
        extern "resin_runtime.h" def test_verify(code: int);
        def work() -> Result<(), _> = {
            var gpu = Gpu.new()?;

            var allocation = gpu.malloc(16, 8, Memory.default())?;

            var commands = gpu.start_command_recording()?;

            commands.set_pipeline(GpuPipeline { handle = Ptr<ResinPipeline>(0_ul), gpu = gpu })?;
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
                && valid_size(size) && incomplete == 7) { 0 } else { 1 })
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
        struct Cleanup {};
        impl Cleanup { def drop(self: Ptr<Cleanup>) = { print("cleanup\n"); }; }
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
fn pipeline_wrappers_unpack_shader_spans_at_the_c_boundary() {
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin" };
        def main() -> Result<int, _> = {
            var a = [ubyte(1), ubyte(2)];
            var b = [ubyte(3), ubyte(4), ubyte(5)];
            var vertex = Span<ubyte> { data = Ptr<ubyte>(&a), length = ulong(2) };
            var fragment = Span<ubyte> { data = Ptr<ubyte>(&b), length = ulong(3) };
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            var compute = gpu.create_compute_pipeline(vertex)?;
            var graphics = gpu.create_graphics_pipeline(vertex, fragment)?;
            ok(if (ulong(compute.handle) == ulong(1) && ulong(graphics.handle) == ulong(2)) { 0 } else { 1 })
        };
    "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static ResinStatus mock_compute(ResinGpu *gpu, const uint8_t *data, size_t length, ResinPipeline **out) {
            assert(!gpu && length == 2 && data[0] == 1 && data[1] == 2);
            *out = (ResinPipeline *)(uintptr_t)1;
            return RESIN_STATUS_SUCCESS;
        }
        static ResinStatus mock_graphics(ResinGpu *gpu, const uint8_t *vertex, size_t vertex_length, const uint8_t *fragment, size_t fragment_length, ResinPipeline **out) {
            assert(!gpu && vertex_length == 2 && fragment_length == 3 && vertex[1] == 2 && fragment[2] == 5);
            *out = (ResinPipeline *)(uintptr_t)2;
            return RESIN_STATUS_SUCCESS;
        }
        static void mock_free_pipeline(ResinGpu *gpu, ResinPipeline *pipeline) {
            assert(!gpu && ((uintptr_t)pipeline == 1 || (uintptr_t)pipeline == 2));
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
        def coordinates(x: float64, y: float64) -> bool = { x == 12.5_d && y == -3.25_d };
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
    let output = run(
        r#"
        export { main };
        import { "$/gpu.resin", "$/status.resin" };
        extern "resin_runtime.h" def test_mode(mode: int);
        extern "resin_runtime.h" def test_recorded();
        extern "resin_runtime.h" def test_code(code: int);
        extern "resin_runtime.h" def test_completed();
        extern "resin_runtime.h" def test_finished();
        def main() -> Result<(), _> = {
            var gpu = Gpu { handle = Ptr<ResinGpu>(0_ul), window = None };
            var mode = 0;
            while (mode < 4) {
                test_mode(mode);
                {
                    var commands = {
                        var original = gpu.start_command_recording()?;
                        original.set_pipeline(GpuPipeline { handle = Ptr<ResinPipeline>(1_ul), gpu = gpu })?;
                        original.begin_rendering(GpuImage { handle = Ptr<ResinImage>(2_ul), gpu = gpu }, 0_f, 0_f, 0_f, 1_f)?;
                        original.end_rendering()?;
                        original
                    };
                    var alias = commands;
                    alias.draw(None, 7)?;
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
        import { "$/window.resin" };
        extern "resin_runtime.h" def window_counts() -> int;
        def main() -> Result<int, _> = {
            var weak = Weak<Span<ubyte>>();
            {
                var title = String.from_str("named {0}");
                weak := title.bytes.downgrade();
                var a = Window.new(32_ui, 24_ui, title)?;
                var b = Window.new(32_ui, 24_ui, String.from_str("temporary"))?;
                print(title);
            };
            var released = match (weak.upgrade()) {
                None => { 1 == 1 },
                Arc<Span<ubyte>>(live) => { 1 == 0 },
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
