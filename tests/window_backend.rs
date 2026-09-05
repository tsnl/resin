use std::{
    ffi::OsStr,
    path::Path,
    process::{Command, Output},
};

use resin::toolchain::{TempDir, compile_c};
use resin_runtime::testing::lock_gpu;

fn compile(source: &str, path: &Path) {
    let cc = std::env::var_os("CC").unwrap_or_else(|| "cc".into());
    compile_c(source, path, &cc).unwrap();
}

fn window_required() -> bool {
    std::env::var("RESIN_REQUIRE_WINDOW").as_deref() == Ok("1")
}

fn display_available() -> bool {
    if !cfg!(feature = "gpu") {
        assert!(
            !window_required(),
            "RESIN_REQUIRE_WINDOW needs --features gpu"
        );
        return false;
    }
    let available = cfg!(target_os = "macos")
        || ["DISPLAY", "WAYLAND_DISPLAY"]
            .iter()
            .any(|name| std::env::var_os(name).is_some_and(|value| !value.is_empty()));
    assert!(
        available || !window_required(),
        "RESIN_REQUIRE_WINDOW needs a display (or Xvfb)"
    );
    available
}

fn succeeded(output: &Output) -> bool {
    if matches!(output.status.code(), Some(2 | 3 | 8 | 77)) && !window_required() {
        eprintln!("skipping: window or presentation unavailable");
        return false;
    }
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let errors = String::from_utf8_lossy(&output.stderr);
    assert!(
        !errors.contains("Validation Error") && !errors.contains("VUID-"),
        "{errors}"
    );
    true
}

#[test]
#[cfg(target_os = "linux")]
fn missing_display_returns_status_and_clears_output() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("unavailable");
    compile(
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        #include <string.h>
        int main(void) {
            for (int i = 0; i < 2; ++i) {
                ResinWindow *window = (ResinWindow *)(uintptr_t)1;
                assert(resin_window_create(64, 64, "test", &window) == RESIN_STATUS_WINDOW_UNAVAILABLE);
                assert(window == NULL);
            }
            assert(strcmp(resin_status_string(RESIN_STATUS_WINDOW_UNAVAILABLE), "window unavailable") == 0);
            return 0;
        }
    "#,
        &executable,
    );
    let output = Command::new(executable)
        .env("DISPLAY", ":999999")
        .env("WAYLAND_DISPLAY", "/missing/resin-wayland-socket")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let errors = String::from_utf8_lossy(&output.stderr);
    if errors.contains("resin: could not load libglfw.so.3:") && !window_required() {
        return;
    }
    let prefix = "resin: glfwInit failed: GLFW error 0x";
    assert_eq!(errors.matches(prefix).count(), 2, "{errors}");
    assert!(!errors.contains("no error description"), "{errors}");
}

#[test]
#[cfg(target_os = "linux")]
fn glfw_loader_errors_are_reported() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("load-error");
    compile(
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        int main(void) {
            ResinWindow *window = (ResinWindow *)(uintptr_t)1;
            assert(resin_window_create(64, 64, "test", &window) == RESIN_STATUS_WINDOW_UNAVAILABLE);
            assert(window == NULL);
            return 0;
        }
        "#,
        &executable,
    );
    std::fs::write(temp.path().join("libglfw.so.3"), b"not a shared library").unwrap();
    let paths = std::env::join_paths(std::iter::once(temp.path().to_path_buf()).chain(
        std::env::split_paths(&std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default()),
    ))
    .unwrap();
    let output = Command::new(executable)
        .env("LD_LIBRARY_PATH", paths)
        .output()
        .unwrap();
    let errors = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{errors}");
    assert!(
        errors.contains("resin: could not load libglfw.so.3:"),
        "{errors}"
    );
    assert!(errors.contains("file too short"), "{errors}");
    assert!(!errors.contains("glfwInit failed"), "{errors}");
}

#[test]
fn windows_present_resize_and_release_resources() {
    if !display_available() {
        return;
    }
    let _lock = lock_gpu();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp.path().join("window");
    compile(
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        #include <stdio.h>
        static void check(ResinStatus status) {
            if (status != RESIN_STATUS_SUCCESS) fprintf(stderr, "runtime: %s\n", resin_status_string(status));
            assert(status == RESIN_STATUS_SUCCESS);
        }
        int main(void) {
            ResinWindow *window = NULL, *other = NULL;
            ResinGpu *gpu = NULL;
            ResinImage *image = NULL;
            ResinStatus status = resin_window_create(96, 64, "Resin test", &window);
            if (status == RESIN_STATUS_WINDOW_UNAVAILABLE) return 77;
            check(status);
            status = resin_gpu_create_for_window(window, &gpu);
            if (status == RESIN_STATUS_UNSUPPORTED || status == RESIN_STATUS_VULKAN_UNAVAILABLE) {
                resin_window_destroy(window);
                return 77;
            }
            check(status);
            ResinGpu *duplicate = (ResinGpu *)(uintptr_t)1;
            assert(resin_gpu_create_for_window(window, &duplicate) == RESIN_STATUS_INVALID_ARGUMENT);
            assert(duplicate == NULL);
            check(resin_window_create(64, 64, "Other window", &other));
            check(resin_gpu_create_image(gpu, 64, 64, &image));
            assert(resin_gpu_present(gpu, image) == RESIN_STATUS_INVALID_ARGUMENT);
            assert(resin_window_set_size(window, 0, 64) == RESIN_STATUS_INVALID_ARGUMENT);
            assert(!resin_window_key_pressed(window, -1));
            int presented = 0, resized = 0;
            uint32_t initial_width = 0, initial_height = 0;
            check(resin_window_framebuffer_size(window, &initial_width, &initial_height));
            for (int frame = 0; frame < 24; ++frame) {
                if (frame == 4) check(resin_window_set_size(window, 160, 120));
                if (frame == 16) check(resin_window_set_size(window, 96, 64));
                check(resin_window_poll_events(window));
                uint32_t width = 0, height = 0;
                check(resin_window_framebuffer_size(window, &width, &height));
                assert(width > 0 && height > 0);
                resized |= width != initial_width || height != initial_height;
                ResinCommandBuffer *commands = NULL;
                check(resin_gpu_start_command_recording(gpu, &commands));
                check(resin_gpu_begin_rendering(commands, image, 0.25f, 0.5f, (float)frame / 24.0f, 1.0f));
                check(resin_gpu_end_rendering(commands));
                check(resin_gpu_submit(gpu, commands));
                ResinCommandBuffer *stale = NULL;
                check(resin_gpu_start_command_recording(gpu, &stale));
                check(resin_gpu_begin_rendering(stale, image, 0, 0, 0, 1));
                check(resin_gpu_end_rendering(stale));
                status = resin_gpu_present(gpu, image);
                if (status == RESIN_STATUS_SUCCESS) {
                    ++presented;
                    assert(resin_gpu_submit(gpu, stale) == RESIN_STATUS_INVALID_ARGUMENT);
                } else {
                    assert(status == RESIN_STATUS_INCOMPLETE);
                    resin_gpu_cancel_command_buffer(gpu, stale);
                }
            }
            assert(presented >= 20 && resized);
            check(resin_window_set_should_close(window, 1));
            assert(resin_window_should_close(window));
            check(resin_window_set_should_close(window, 0));
            assert(!resin_window_should_close(window));
            resin_gpu_free_image(gpu, image);
            resin_window_destroy(window);
            resin_gpu_destroy(gpu);
            check(resin_window_poll_events(other));
            resin_window_destroy(other);
            check(resin_window_create(64, 64, "Reinitialized", &window));
            resin_window_destroy(window);
            return 0;
        }
    "#,
        &executable,
    );
    succeeded(&Command::new(executable).output().unwrap());
}

#[test]
fn resin_window_example_compiles_without_linking_glfw() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/window.resin");
    let executable = temp.path().join("window-example");
    let output = Command::new(env!("CARGO_BIN_EXE_resin"))
        .current_dir(temp.path())
        .arg(source)
        .args(["--output", "exe", "-o"])
        .arg(&executable)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    if !display_available() {
        return;
    }
    let _lock = lock_gpu();
    let output = Command::new(executable)
        .env("GLSLC", OsStr::new("/missing/glslc"))
        .output()
        .unwrap();
    if succeeded(&output) {
        assert_eq!(output.stdout, b"window demo complete\n");
    }
}
