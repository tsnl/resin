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
fn explorer_builds_and_can_render_resize_and_close_in_both_modes() {
    let original = std::fs::read_to_string(example()).unwrap();
    let source = original.replace(
        "while (true) {",
        r#"
        let mut frames = 0_i;
        while (true) {
            frames = frames + 1;
            if (frames == 3) { plot:zoom(0.5_f); dirty = true; };
            if (frames == 4) { window:set_size(800, 500)?; };
            if (frames == 6) { window:set_should_close(true)?; };
    "#,
    );
    assert_ne!(source, original, "instrument the interactive loop");
    run_example(&source, "main", true);
    let cpu = source.replace("const use_gpu = true;", "const use_gpu = false;");
    assert_ne!(cpu, source, "switch to the CPU solver");
    run_example(&cpu, "main", true);
}

#[test]
fn compute_matches_cpu_for_single_and_four_samples() {
    let mut source = std::fs::read_to_string(example())
        .unwrap()
        .replace("export { main, test };", "export { main, test, compare };");
    source.push_str(
        r#"
        fn close(actual: float32, expected: float32) {
            assert(actual - expected < 0.0001_f && expected - actual < 0.0001_f);
        }
        fn compare() -> () | Err<_> {
            let gpu = gpu_new()?;
            let pipeline = gpu:create_compute_pipeline(compute)?;
            // A non-square image and a partially filled final workgroup.
            let plot = plot_new(37, 19, 37.0_f / 19.0_f);
            let solver = mandelbrot_new(32);
            let count = ulong(plot.width) * ulong(plot.height);
            let pixels = gpu:alloc::<Color>(count + 1_ul)?;
            pixels:at(count):store(Color { r = -1.0_f, g = -2.0_f, b = -3.0_f, a = -4.0_f });
            let group_size = gpu:compute_workgroup_size();
            let groups = uint((count + group_size - 1_ul) / group_size);
            let mut samples = 1_i;
            while (samples <= 4) {
                let root = HostParameters {
                    plot = plot:clone(), solver = mandelbrot_new(solver.max_iters),
                    samples = samples, pixels = pixels:clone(),
                };
                let commands = gpu:start_command_recording()?;
                commands:dispatch(pipeline, root, groups, 1, 1)?;
                commands:submit()?;
                let mut y = 0_ui;
                while (y < plot.height) {
                    let mut x = 0_ui;
                    while (x < plot.width) {
                        let actual = pixels:at(ulong(y) * ulong(plot.width) + ulong(x)):load();
                        let expected = solver:solve(plot, x, y, samples);
                        close(actual.r, expected.r);
                        close(actual.g, expected.g);
                        close(actual.b, expected.b);
                        close(actual.a, expected.a);
                        x = x + 1;
                    };
                    y = y + 1;
                };
                let sentinel = pixels:at(count):load();
                assert(sentinel.r == -1.0_f && sentinel.g == -2.0_f);
                assert(sentinel.b == -3.0_f && sentinel.a == -4.0_f);
                samples = samples + 3;
            };
            // These four subpixels escape after 2, 1, 2, and 1 iterations.
            let mut edge = plot_new(1, 1, 1.0_f);
            edge.center.real = 2.0_f;
            edge.span = 2.0_f;
            let averaged = solver:solve(edge, 0, 0, 4);
            let first = palette(1);
            let second = palette(2);
            close(averaged.r, (first.r + second.r) * 0.5_f);
            close(averaged.g, (first.g + second.g) * 0.5_f);
            close(averaged.b, (first.b + second.b) * 0.5_f);
        }
    "#,
    );
    run_example(&source, "compare", false);
}

fn run_example(source: &str, entry: &str, needs_window: bool) {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("mandelbrot.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, entry).unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some(entry)).unwrap();
    let executable = project.build_executable();
    let display = !cfg!(target_os = "linux")
        || ["DISPLAY", "WAYLAND_DISPLAY"]
            .iter()
            .any(|key| std::env::var_os(key).is_some_and(|value| !value.is_empty()));
    let window_required =
        needs_window && std::env::var("RESIN_REQUIRE_WINDOW").as_deref() == Ok("1");
    let required = window_required || std::env::var("RESIN_REQUIRE_GPU").as_deref() == Ok("1");
    assert!(
        cfg!(feature = "gpu") || !required,
        "required GPU execution needs --features gpu"
    );
    assert!(
        display || !window_required,
        "RESIN_REQUIRE_WINDOW needs a display"
    );
    if !cfg!(feature = "gpu") || (needs_window && !display) {
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
                "Mandelbrot {entry} did not finish: {}",
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
