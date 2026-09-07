use resin::{ast, backend::c, ir, toolchain};
use std::{fs, path::Path, process::Command};

fn run(source: &str, native: &str) -> std::process::Output {
    let temp = toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    let path = temp.path().join("main.resin");
    fs::write(&path, source).unwrap();
    let program = ast::load(&path).unwrap();
    let module = ir::generate_program(&program).unwrap();
    let c = format!("{native}\n{}", c::emit(&module, "main").unwrap());
    let executable = temp
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC").unwrap_or_else(|| toolchain::DEFAULT_C_COMPILER.into());
    toolchain::compile_c(&c, &executable, &cc).unwrap();
    Command::new(executable)
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
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for name in ["gpu", "window", "image", "console"] {
        let program = ast::load(&root.join(format!("stdlib/{name}.resin"))).unwrap();
        let module = ir::generate_program(&program).unwrap();
        for (name, id) in &module.entries {
            assert!(!name.starts_with("resin_"), "raw native export: {name}");
            assert!(module.functions[id.index()].foreign.is_none(), "{name}");
        }
        let header =
            fs::read_to_string(root.join(format!("resin-runtime/include/resin_runtime/{name}.h")))
                .unwrap();
        let mut checked = 0;
        for line in header.lines() {
            let Some(declaration) = line.strip_prefix("ResinStatus resin_") else {
                continue;
            };
            let name = declaration.split('(').next().unwrap();
            let id = module
                .entries
                .get(name)
                .unwrap_or_else(|| panic!("missing wrapper: {name}"));
            assert!(
                matches!(module.functions[id.index()].result, ir::Ty::Result { .. }),
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
        r#"
        export { main };
        import { "std/status.resin" };
        extern "string.h" def strcmp(a: Ptr<ubyte>, b: Ptr<ubyte>) -> int;
        def main() -> Result<int, _> = {
            status(0)?;
            var code = -1;
            var valid = 1 == 1;
            while (code <= 9) {
                var actual = match (status(code)) {
                    ok(value) => { 0 },
                    err(error) => { runtime_error_code(error) },
                };
                valid := valid && actual == code;
                code := code + 1;
            };
            var message = "io error";
            valid := valid && strcmp(runtime_error_message(IoError {}), Ptr<ubyte>(&message)) == 0;
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
                "export {{ main }}; import {{ \"std/status.resin\" }}; def main() -> Result<(), _> = {{ status({code}) }};"
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
        import { "std/image.resin" };
        def main() -> Result<int, _> = {
            var path = "pixel.png";
            var pixels = [ubyte(1), ubyte(2), ubyte(3), ubyte(255)];
            image_write_png(Ptr<ubyte>(&path), 1, 1, 4, Ptr<ubyte>(&pixels), 0)?;
            var image = image_read_png(Ptr<ubyte>(&path), 0)?;
            defer image_free(image.pixels);
            ok(if (image.width == uint(1) && image.height == uint(1) && image.channels == uint(4)
                && image.pixels.* == ubyte(1) && Ptr<ubyte>(ulong(image.pixels) + ulong(3)).* == ubyte(255)) { 0 } else { 1 })
        };
        "#,
        "",
    );
    success(&output);
    for call in [
        "image_read_png(Ptr<ubyte>(&path), 4)?",
        "image_write_png(Ptr<ubyte>(&path), 1, 1, 4, Ptr<ubyte>(&pixels), 0)?",
    ] {
        let output = run(
            &format!(
                "export {{ main }}; import {{ \"std/image.resin\" }}; def main() -> Result<(), _> = {{ var path = \"missing/pixel.png\"; var pixels = [uint(0)]; defer print(\"cleanup\\n\", ()); {call}; ok(()) }};"
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
        import { "std/gpu.resin", "std/status.resin" };
        extern "resin_runtime.h" def test_mode(mode: int);
        extern "resin_runtime.h" def test_verify(code: int);
        def work() -> Result<(), _> = {
            var gpu = gpu_create()?;
            defer gpu_destroy(gpu);
            var allocation = gpu_malloc(gpu, 16, 8, memory_default())?;
            defer gpu_free(gpu, allocation);
            var commands = gpu_start_command_recording(gpu)?;
            defer gpu_cancel_command_buffer(gpu, &commands);
            gpu_set_pipeline(commands, Ptr<ResinPipeline>(ulong(0)))?;
            gpu_submit(gpu, &commands)?;
            ok(())
        };
        def main() = {
            var mode = 0;
            while (mode < 6) {
                test_mode(mode);
                var result = match (work()) {
                    ok(value) => { 0 },
                    err(error) => { runtime_error_code(error) },
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
        static ResinStatus mock_malloc(ResinGpu *gpu, size_t bytes, size_t align, ResinMemory memory, ResinAllocation **out) {
            assert(gpu == (ResinGpu *)(uintptr_t)1 && bytes == 16 && align == 8 && memory == RESIN_MEMORY_DEFAULT);
            *out = (ResinAllocation *)(uintptr_t)2;
            return mode == 1 ? RESIN_STATUS_OUT_OF_MEMORY : RESIN_STATUS_SUCCESS;
        }
        static void mock_free(ResinGpu *gpu, ResinAllocation *allocation) {
            assert(gpu == (ResinGpu *)(uintptr_t)1);
            assert(allocation == (ResinAllocation *)(uintptr_t)2);
            cleanup = cleanup * 10 + 2;
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
        #define resin_gpu_malloc mock_malloc
        #define resin_gpu_free mock_free
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
        import { "std/gpu.resin", "std/window.resin", "std/status.resin" };
        def main() -> int = {
            var gpu = Ptr<ResinGpu>(ulong(0));
            var image = Ptr<ResinImage>(ulong(0));
            var first = match (gpu_present(gpu, image)) { ok(shown) => { shown }, err(e) => { 1 == 0 } };
            var second = match (gpu_present(gpu, image)) { ok(shown) => { !shown }, err(e) => { 1 == 0 } };
            var third = match (gpu_present(gpu, image)) { ok(shown) => { 0 }, err(e) => { runtime_error_code(e) } };
            var fourth = match (gpu_present(gpu, image)) { ok(shown) => { 0 }, err(e) => { runtime_error_code(e) } };
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
        import { "std/gpu.resin", "std/window.resin", "std/status.resin" };
        def valid_size(width: uint, height: uint) -> bool = {
            width == uint(640) && height == uint(480)
        };
        def main() -> Result<int, _> = {
            var gpu = Ptr<ResinGpu>(ulong(0));
            var window = Ptr<ResinWindow>(ulong(0));
            var count = gpu_device_count()?;
            var address = gpu_host_to_device_pointer(gpu, Ptr<ubyte>(ulong(0)))?;
            var size = window_framebuffer_size(window)?;
            var incomplete = match (gpu_enumerate_devices(Ptr<ResinGpuDeviceInfo>(ulong(0)), 0)) {
                ok(value) => { 0 },
                err(error) => { runtime_error_code(error) },
            };
            var valid = !window_should_close(window) && window_key_pressed(window, key_escape());
            window_set_should_close(window, 1 == 1)?;
            valid := valid && window_should_close(window);
            window_set_should_close(window, 1 == 0)?;
            valid := valid && !window_should_close(window);
            ok(if (valid && count == uint(2) && address == ulong(4294967303)
                && valid_size(size) && incomplete == 7) { 0 } else { 1 })
        };
        "#,
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        static int closed;
        static ResinStatus mock_count(uint32_t *out) { *out = 2; return RESIN_STATUS_SUCCESS; }
        static ResinStatus mock_address(const ResinGpu *gpu, const void *host, ResinDeviceAddress *out) {
            assert(gpu == NULL && host == NULL);
            *out = UINT64_C(4294967303);
            return RESIN_STATUS_SUCCESS;
        }
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
        #define resin_gpu_host_to_device_pointer mock_address
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
        import { "std/console.resin" };
        def main() -> Result<(), _> = {
            defer print("cleanup\n", ());
            read_byte()?;
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
        import { "std/gpu.resin" };
        def main() -> Result<int, _> = {
            var a = [ubyte(1), ubyte(2)];
            var b = [ubyte(3), ubyte(4), ubyte(5)];
            var vertex = Span<ubyte> { data = Ptr<ubyte>(&a), length = ulong(2) };
            var fragment = Span<ubyte> { data = Ptr<ubyte>(&b), length = ulong(3) };
            var gpu = Ptr<ResinGpu>(ulong(0));
            var compute = gpu_create_compute_pipeline(gpu, vertex)?;
            var graphics = gpu_create_graphics_pipeline(gpu, vertex, fragment)?;
            ok(if (ulong(compute) == ulong(1) && ulong(graphics) == ulong(2)) { 0 } else { 1 })
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
        #define resin_gpu_create_compute_pipeline mock_compute
        #define resin_gpu_create_graphics_pipeline mock_graphics
    "#,
    );
    success(&output);
}
