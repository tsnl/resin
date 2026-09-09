use tempfile::TempDir;
#[path = "support/toolchain.rs"]
mod toolchain;
use std::{
    ffi::OsStr,
    path::Path,
    process::{Command, Output, Stdio},
    time::{Duration, Instant},
};
use support::pipeline;

use resin_ast::{StmtKind, TermKind};
use resin_runtime::testing::lock_gpu;

use support::shaders;
#[allow(dead_code)]
mod support;

fn compile(source: &str, path: &Path) {
    let cc = std::env::var_os("CC").unwrap_or_else(|| resin_toolchain::DEFAULT_C_COMPILER.into());
    toolchain::compile_c(source, path, &cc).unwrap();
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
    let available = !cfg!(target_os = "linux")
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
    let errors = String::from_utf8_lossy(&output.stderr);
    let unavailable = matches!(output.status.code(), Some(2 | 3 | 8 | 77))
        || (output.status.code() == Some(1)
            && errors.lines().any(|line| {
                matches!(
                    line,
                    "unhandled error: VulkanUnavailable"
                        | "unhandled error: Unsupported"
                        | "unhandled error: WindowUnavailable"
                )
            }));
    if unavailable && !window_required() {
        eprintln!("skipping: window or presentation unavailable");
        return false;
    }
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !errors.contains("Validation Error") && !errors.contains("VUID-"),
        "{errors}"
    );
    true
}

#[test]
#[cfg(target_os = "linux")]
fn missing_display_returns_status_and_clears_output() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("unavailable{}", std::env::consts::EXE_SUFFIX));
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
    let prefix = "resin: glfwInit failed: GLFW error 0x";
    assert_eq!(errors.matches(prefix).count(), 2, "{errors}");
    assert!(!errors.contains("no error description"), "{errors}");
}

#[test]
fn glfw_is_linked_into_c_executables() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("static-glfw{}", std::env::consts::EXE_SUFFIX));
    compile(
        r#"
        #include <resin_runtime.h>
        #include <assert.h>
        extern void glfwGetVersion(int *major, int *minor, int *revision);
        int main(void) {
            int major = 0, minor = 0, revision = 0;
            glfwGetVersion(&major, &minor, &revision);
            assert(major == 3 && minor >= 4);
            return 0;
        }
        "#,
        &executable,
    );
    let output = Command::new(executable)
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .unwrap();
    let errors = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{errors}");
    assert!(errors.is_empty(), "{errors}");
}

#[test]
fn windows_present_resize_and_release_resources() {
    if !display_available() {
        return;
    }
    let _lock = lock_gpu();
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("window{}", std::env::consts::EXE_SUFFIX));
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
            assert(resin_window_key_state(window, -1) == 0);
            assert(resin_window_mouse_button_state(window, 8) == 0);
            double cursor_x = 0, cursor_y = 0, scroll_x = 1, scroll_y = 1;
            check(resin_window_cursor_position(window, &cursor_x, &cursor_y));
            check(resin_window_scroll_delta(window, &scroll_x, &scroll_y));
            assert(scroll_x == 0 && scroll_y == 0);
            check(resin_window_capture_cursor(window, 1));
            check(resin_window_poll_events(window));
            check(resin_window_cursor_position(window, &cursor_x, &cursor_y));
            check(resin_window_capture_cursor(window, 0));
            assert(resin_window_focused(window) == 0 || resin_window_focused(window) == 1);

            int presented = 0, enlarged = 0, restored = 0;
            uint32_t initial_width = 0, initial_height = 0;
            check(resin_window_framebuffer_size(window, &initial_width, &initial_height));
            for (int frame = 0; frame < 24; ++frame) {
                /* Drain pending compositor configures before applying a new size.
                   On Wayland, polling afterward can restore the previous size
                   before it has been presented. */
                check(resin_window_poll_events(window));
                if (frame == 4) check(resin_window_set_size(window, 160, 120));
                if (frame == 16) check(resin_window_set_size(window, 96, 64));
                uint32_t width = 0, height = 0;
                check(resin_window_framebuffer_size(window, &width, &height));
                assert(width > 0 && height > 0);
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
                    if (frame >= 4 && frame < 16)
                        enlarged |= width > initial_width && height > initial_height;
                    if (frame >= 16)
                        restored |= width == initial_width && height == initial_height;
                    assert(resin_gpu_submit(gpu, stale) == RESIN_STATUS_INVALID_ARGUMENT);
                } else {
                    assert(status == RESIN_STATUS_INCOMPLETE);
                    resin_gpu_cancel_command_buffer(gpu, stale);
                }
            }
            if (presented < 20 || !enlarged || !restored)
                fprintf(stderr, "presentation/resize failed: presented=%d/24 enlarged=%d restored=%d initial=%ux%u\n",
                        presented, enlarged, restored, initial_width, initial_height);
            assert(presented >= 20);
            assert(enlarged);
            assert(restored);
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
fn resin_window_example_uses_bundled_glfw() {
    run_example("window");
}

#[test]
fn resin_particles_example_computes_and_presents() {
    run_example("particles");
}

fn run_example(name: &str) {
    let Some(compiler) = shaders::optimizer() else {
        return;
    };
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("examples/{name}.resin"));
    let executable = temp
        .path()
        .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
    let mut ast = pipeline::load(&source).unwrap();
    let body = ast
        .modules
        .last_mut()
        .unwrap()
        .file
        .stmts
        .iter_mut()
        .find_map(|stmt| match &mut stmt.val {
            StmtKind::Function { name, body, .. } if name.val.as_ref() == "main" => Some(body),
            _ => None,
        })
        .unwrap();
    let TermKind::Block { stmts, .. } = &mut body.val else {
        panic!("main body")
    };
    stmts.insert(0, support::statements("var test_frames = 0;").remove(0));
    let body = stmts
        .iter_mut()
        .find_map(|stmt| match &mut stmt.val {
            StmtKind::Expr { term } => match &mut term.val {
                TermKind::While { body, .. } => Some(body),
                _ => None,
            },
            _ => None,
        })
        .unwrap();
    let TermKind::Block { stmts, .. } = &mut body.val else {
        panic!("loop body")
    };
    // Close through the runtime after three frames; leave the interactive demo unbounded.
    stmts.extend(support::statements("test_frames := test_frames + 1; if (test_frames == 3) { window.set_should_close(1 == 1)?; } else { () };"));
    let module = pipeline::generate_program(&ast).unwrap();
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let built = project.build(&toolchain::spirv(&compiler)).unwrap();
    built
        .executable(project.generated.program().unwrap().file_name().unwrap())
        .unwrap()
        .copy_to(&executable)
        .unwrap();
    if !display_available() {
        return;
    }
    let _lock = lock_gpu();
    let mut child = Command::new(executable)
        .env("SPIRV_OPT", OsStr::new("/missing/spirv-opt"))
        .env("CC", OsStr::new("/missing/cc"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "window example did not close: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let output = child.wait_with_output().unwrap();
    if succeeded(&output) {
        assert_eq!(output.stdout, format!("{name} demo complete\n").as_bytes());
    }
}
