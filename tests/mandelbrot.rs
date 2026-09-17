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
    let source = original.replacen(
        "while (true) {",
        r#"
        let mut frames: i32 = 0;
        while (true) {
            frames = frames + 1;
            if (frames == 3) { plot:zoom(f32(0.5)); dirty = true; };
            if (frames == 4) { window:set_size(800, 500)?; };
            if (frames == 6) { window:set_should_close(true)?; };
    "#,
        1,
    );
    assert_ne!(source, original, "instrument the interactive loop");
    let source = source.replace(
        "if (frames == 6)",
        "if (frames == 5) { renderer:screenshot(options.screenshot)?; }; if (frames == 6)",
    );
    let (_project, executable) = build_example(&source, "main");
    for mode in ["--gpu", "--cpu"] {
        let output = tempfile::TempDir::new().unwrap();
        let path = output.path().join("screenshot with spaces.png");
        if run_example(
            &executable,
            "main",
            true,
            &[
                mode,
                "--width",
                "97",
                "--height",
                "61",
                "--screenshot",
                path.to_str().unwrap(),
            ],
        ) {
            let image = resin_runtime::image_read_png(&path, 4).unwrap();
            assert_eq!((image.width, image.height), (800, 500));
            assert!(
                image
                    .pixels
                    .chunks_exact(4)
                    .any(|pixel| pixel[..3] != [0, 0, 0])
            );
        }
    }
}

#[test]
fn a_segment_writes_only_its_own_row_range() {
    let mut source = std::fs::read_to_string(example()).unwrap().replace(
        "export { main, test };",
        "export { main, test, segment_bounds };",
    );
    source.push_str(
        r#"
        fn segment_bounds() -> () | Err<_> {
            let plot = plot_new(35, 2);
            let count = u64(plot.width) * u64(plot.height) * u64(4);
            let pixels = arc_span_alloc::<u8>(count + u64(1), u8(123))?;
            let view = pixels:get();
            let root = Parameters<_> { plot = plot, solver = mandelbrot_new(32),
                start_segment = 0, segment_count = plot:segment_count(), pixels = view:slice(u64(0), count) };
            assert(plot:segment_count() == u64(4));
            evaluate_segment(u64(0), root);
            let mut i: u64 = 0;
            while (i < u64(32)) {
                assert(view:at(i * u64(4) + u64(3)) == u8(255));
                i = i + u64(1);
            };
            i = u64(32) * u64(4);
            while (i <= count) {
                assert(view:at(i) == u8(123));
                i = i + u64(1);
            };
            evaluate_segment(u64(1), root);
            i = u64(32);
            while (i < u64(35)) {
                assert(view:at(i * u64(4) + u64(3)) == u8(255));
                i = i + u64(1);
            };
            i = u64(35) * u64(4);
            while (i <= count) {
                assert(view:at(i) == u8(123));
                i = i + u64(1);
            };
            dispatch_host(root);
            evaluate_segment(u64(4), root);
            i = u64(0);
            while (i < u64(70)) {
                assert(view:at(i * u64(4) + u64(3)) == u8(255));
                i = i + u64(1);
            };
            assert(view:at(count) == u8(123));
        }
    "#,
    );
    let (_project, executable) = build_example(&source, "segment_bounds");
    let output = Command::new(executable.path()).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn compute_matches_cpu_for_partial_segments() {
    let mut source = std::fs::read_to_string(example())
        .unwrap()
        .replace("export { main, test };", "export { main, test, compare };");
    source.push_str(
        r#"
        fn compare() -> () | Err<_> {
            let gpu = gpu_new()?;
            let pipeline = gpu:create_compute_pipeline(compute)?;
            // Non-square, short row tails and an incomplete final workgroup.
            let plot = plot_new(37, 19);
            let solver = mandelbrot_new(32);
            let count = u64(plot.width) * u64(plot.height) * u64(4);
            let pixels = gpu:alloc::<u8>(count + u64(1))?;
            let segments = plot:segment_count();
            let group_size = gpu:compute_workgroup_size();
            {
                let sentinel = pixels:at(count);
                sentinel:store(u8(123));
                let commands = gpu:start_command_recording()?;
                // Two explicit ranges exercise batch-local invocation indices.
                let first = Parameters<_> {
                    plot = plot, solver = solver,
                    start_segment = 0, segment_count = 7, pixels = pixels,
                };
                let second = Parameters<_> {
                    plot = plot, solver = solver,
                    start_segment = 7, segment_count = segments - 7, pixels = pixels,
                };
                commands:dispatch(pipeline, first, u32((u64(7) + group_size - u64(1)) / group_size), 1, 1)?;
                commands:dispatch(pipeline, second, u32((segments - u64(7) + group_size - u64(1)) / group_size), 1, 1)?;
                commands:submit()?;
                let actual = arc_span_alloc::<u8>(count + u64(1), u8(0))?;
                pixels:copy_to(actual:get());
                let expected = host_image(plot, solver)?;
                let actual_view = actual:get();
                let expected_view = expected:get();
                let mut i: u64 = 0;
                while (i < count) {
                    let difference = i32(actual_view:at(i)) - i32(expected_view:at(i));
                    assert(difference >= -1 && difference <= 1);
                    i = i + u64(1);
                };
                assert(actual_view:at(count) == u8(123));
            };
        }
    "#,
    );
    let (_project, executable) = build_example(&source, "compare");
    run_example(&executable, "compare", false, &[]);
}

#[test]
fn cli_validates_options_and_writes_headless_pngs() {
    let source = std::fs::read_to_string(example()).unwrap();
    let (_project, executable) = build_example(&source, "main");
    let directory = tempfile::TempDir::new().unwrap();
    let cpu_command = || {
        let mut command = Command::new(executable.path());
        command
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .env("XDG_SESSION_TYPE", "x11")
            .env(
                "VK_DRIVER_FILES",
                directory.path().join("no-vulkan-driver.json"),
            )
            .env(
                "VK_ICD_FILENAMES",
                directory.path().join("no-vulkan-driver.json"),
            );
        command
    };
    let help = cpu_command().arg("--help").output().unwrap();
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("--output"));
    for args in [
        vec!["--unknown"],
        vec!["--width"],
        vec!["--width", "0"],
        vec!["--width", "8193"],
        vec!["--width", "4294967296"],
        vec!["--height", "-1"],
        vec!["--iterations", "31"],
        vec!["--iterations", "4097"],
        vec!["--samples", "17"],
        vec!["--samples", "0"],
        vec!["--real", "NaN"],
        vec!["--imag", "inf"],
        vec!["--real", "1.2junk"],
        vec!["--real", ""],
        vec!["--span", "0"],
        vec!["--span", "1e-20"],
        vec!["--output", ""],
        vec!["--screenshot", ""],
    ] {
        let output = cpu_command().args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{args:?}: {:?}", output);
        assert!(String::from_utf8_lossy(&output.stderr).contains("Use --help"));
    }
    let mut images = Vec::new();
    {
        let path = directory.path().join("cpu.png");
        let output = cpu_command()
            .args([
                "--cpu",
                "--width",
                "37",
                "--height",
                "19",
                "--iterations",
                "32",
                "--output",
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let image = resin_runtime::image_read_png(&path, 4).unwrap();
        assert_eq!((image.width, image.height), (37, 19));
        assert!(image.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
        images.push(image);
    }
    let zoomed_path = directory.path().join("zoomed.png");
    let output = cpu_command()
        .args([
            "--cpu",
            "--width",
            "37",
            "--height",
            "19",
            "--iterations",
            "32",
            "--real",
            "-0.7435",
            "--imag",
            "0.1314",
            "--span",
            "0.005",
            "--output",
            zoomed_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_ne!(
        images[0].pixels,
        resin_runtime::image_read_png(&zoomed_path, 4)
            .unwrap()
            .pixels
    );
    let failure = cpu_command()
        .args([
            "--cpu",
            "--width",
            "1",
            "--height",
            "1",
            "--output",
            directory
                .path()
                .join("missing/failed.png")
                .to_str()
                .unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!failure.status.success(), "report image write errors");

    if !cfg!(feature = "gpu") {
        return;
    }
    let _gpu = resin_runtime::testing::lock_gpu();
    {
        let path = directory.path().join("gpu.png");
        let output = Command::new(executable.path())
            .env_remove("DISPLAY")
            .env_remove("WAYLAND_DISPLAY")
            .env("XDG_SESSION_TYPE", "x11")
            .args([
                "--gpu",
                "--width",
                "37",
                "--height",
                "19",
                "--iterations",
                "32",
                "--output",
                path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        let errors = String::from_utf8_lossy(&output.stderr);
        if std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1")
            && ["VulkanUnavailable", "Unsupported"]
                .iter()
                .any(|name| errors.contains(&format!("unhandled error: {name}")))
        {
            return;
        }
        assert!(output.status.success(), "{errors}");
        assert!(
            !errors.contains("Validation Error") && !errors.contains("VUID-"),
            "{errors}"
        );
        let image = resin_runtime::image_read_png(&path, 4).unwrap();
        assert_eq!((image.width, image.height), (37, 19));
        for (actual, expected) in image.pixels.iter().zip(&images[0].pixels) {
            assert!(
                actual.abs_diff(*expected) <= 1,
                "GPU {actual} != CPU {expected}"
            );
        }
    }
}

fn build_example(
    source: &str,
    entry: &str,
) -> (support::project::Project, resin_toolchain::Executable) {
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("mandelbrot.resin");
    std::fs::write(&path, source).unwrap();
    let module =
        support::pipeline::host_entry(&path, entry).unwrap_or_else(|error| panic!("{error}"));
    let project = support::project::Project::new(&module, Some(entry)).unwrap();
    let executable = project.build_executable();
    (project, executable)
}

fn run_example(
    executable: &resin_toolchain::Executable,
    entry: &str,
    needs_window: bool,
    args: &[&str],
) -> bool {
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
        return false;
    }
    let _gpu = resin_runtime::testing::lock_gpu();
    let mut child = Command::new(executable.path())
        .args(args)
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
        return false;
    }
    assert!(output.status.success(), "{}", errors);
    assert!(
        !errors.contains("Validation Error") && !errors.contains("VUID-"),
        "{errors}"
    );
    true
}
