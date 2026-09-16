#[allow(dead_code)]
mod support;

use std::{
    path::PathBuf,
    process::Command,
    time::{Duration, Instant},
};

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/eg011_mandelbrot.resin")
}

#[test]
fn known_orbits_pixel_coordinates_and_zoom_limits() {
    let module =
        support::pipeline::host_entry(&example(), "test").unwrap_or_else(|error| panic!("{error}"));
    let output = support::project::Project::new(&module, Some("test"))
        .unwrap()
        .run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"Mandelbrot tests passed\n");
}

#[test]
fn explorer_builds_and_can_render_resize_and_close() {
    let original = std::fs::read_to_string(example()).unwrap();
    let source = original.replace(
        "while (true) {",
        r#"
        let mut frames = 0_i;
        while (true) {
            frames = frames + 1;
            if (frames == 2) { plot:zoom(0.5_d); dirty = true; };
            if (frames == 3) { window:set_size(800, 500)?; };
            if (frames == 4) { window:set_should_close(true)?; };
    "#,
    );
    assert_ne!(source, original, "instrument the interactive loop");
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("mandelbrot.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, "main").unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    let display = !cfg!(target_os = "linux")
        || ["DISPLAY", "WAYLAND_DISPLAY"]
            .iter()
            .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()));
    let required = std::env::var("RESIN_REQUIRE_WINDOW").as_deref() == Ok("1");
    assert!(
        cfg!(feature = "gpu") || !required,
        "RESIN_REQUIRE_WINDOW needs --features gpu"
    );
    assert!(display || !required, "RESIN_REQUIRE_WINDOW needs a display");
    if !cfg!(feature = "gpu") || !display {
        return;
    }
    let _gpu = resin_runtime::testing::lock_gpu();
    let mut child = Command::new(executable.path())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            let output = child.wait_with_output().unwrap();
            panic!(
                "explorer did not close: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let output = child.wait_with_output().unwrap();
    let errors = String::from_utf8_lossy(&output.stderr);
    if !required
        && ["VulkanUnavailable", "Unsupported", "WindowUnavailable"]
            .iter()
            .any(|name| errors.contains(&format!("unhandled error: {name}")))
    {
        return;
    }
    assert!(output.status.success(), "{}", errors);
    assert!(
        !errors.contains("Validation Error") && !errors.contains("VUID-"),
        "{errors}"
    );
}
