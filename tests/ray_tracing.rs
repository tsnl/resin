mod support;

const SOURCE: &str = r#"
export { main };
import { "$/gpu.resin", "$/span.resin", "$/shared.resin" };
struct Payload { value: f32, }
@ray_generation_shader
fn generation(index: u64, output: Ptr<f32>) {
    let hit = trace_ray(f32(0.0), f32(0.0), f32(-1.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0), f32(100.0), Payload { value = f32(0.0) });
    output.* = hit.value;
}
@miss_shader
fn miss(payload: Payload, output: Ptr<f32>) -> Payload { Payload { value = f32(-1.0) } }
@closest_hit_shader
fn closest(payload: Payload, output: Ptr<f32>) -> Payload {
    let info = ray_hit_info();
    Payload { value = info.0 }
}
fn main() -> i32 | Err<_> {
    if (false) {
        let gpu = gpu_new()?;
        let vertices = arc_ptr_alloc([f32(-1.0), f32(-1.0), f32(0.0), f32(1.0), f32(-1.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0)])?;
        let transform = arc_ptr_alloc([f32(1.0), f32(0.0), f32(0.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0), f32(0.0), f32(0.0), f32(0.0), f32(1.0), f32(0.0)])?;
        let scene = gpu:create_ray_scene(Span<f32> { data = vertices:get():lea(0), length = 9 }, Span<f32> { data = transform:get():lea(0), length = 12 })?;
        let pipeline = scene:create_ray_tracing_pipeline(generation, miss, closest)?;
        let commands = gpu:start_command_recording()?;
        let retained = pipeline:clone();
        commands:trace_rays(retained, f32(0.0), 1, 1, 1)?;
        commands:cancel();
    };
    0
}
"#;

#[test]
fn triangle_pipeline_generates_valid_spirv_and_native_bridges() {
    let module = support::module(SOURCE);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    assert_eq!(project.generated.shaders().len(), 3);
    for shader in project.generated.shaders() {
        support::shaders::validate(shader.unoptimized_spirv());
    }
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn nested_tracing_and_hit_data_in_other_stages_are_rejected() {
    for source in [
        SOURCE.replace("let info = ray_hit_info();", "let nested = trace_ray(f32(0), f32(0), f32(0), f32(0), f32(0), f32(1), f32(0), f32(10), payload); let info = ray_hit_info();"),
        SOURCE.replace("let hit = trace_ray", "let invalid = ray_hit_info(); let hit = trace_ray"),
    ] {
        let error = support::pipeline::source_module(&source).unwrap_err();
        assert!(error.to_string().contains("not available in this shader stage"), "{error}");
    }
}

#[test]
fn mismatched_payload_and_pipeline_kind_are_rejected() {
    let wrong_payload = SOURCE
        .replace("Payload { value = f32(0.0) });", "f32(0.0));")
        .replace("output.* = hit.value;", "output.* = hit;");
    assert!(support::pipeline::source_module(&wrong_payload).is_err());
    let wrong_kind = SOURCE.replace("commands:trace_rays(", "commands:dispatch(");
    assert!(support::pipeline::source_module(&wrong_kind).is_err());
}

#[cfg(feature = "gpu")]
#[test]
fn hardware_triangle_hits_misses_and_instance_transforms() {
    use resin_runtime::*;
    use resin_types::shader::Stage;
    let _lock = testing::lock_gpu();
    let Ok(mut gpu) = ResinGpu::create() else {
        assert_ne!(
            std::env::var("RESIN_REQUIRE_RAY_TRACING").as_deref(),
            Ok("1"),
            "a ray tracing GPU is required"
        );
        eprintln!("skipping: no suitable GPU");
        return;
    };
    if unsafe { resin_gpu_supports_ray_tracing(&gpu) } == 0 {
        assert_ne!(
            std::env::var("RESIN_REQUIRE_RAY_TRACING").as_deref(),
            Ok("1"),
            "ray tracing pipelines are required"
        );
        eprintln!("skipping: ray tracing pipelines unavailable");
        return;
    }
    let module = support::module(SOURCE);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let code = |stage| {
        std::fs::read(
            project
                .generated
                .shaders()
                .iter()
                .find(|shader| shader.stage() == stage)
                .unwrap()
                .unoptimized_spirv(),
        )
        .unwrap()
    };
    let shaders = [
        code(Stage::RayGeneration),
        code(Stage::Miss),
        code(Stage::ClosestHit),
    ];
    // One triangle; a transform can move it away from the ray or further along it.
    let vertices = [-1.0f32, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 0.0];
    unsafe {
        let output = gpu.malloc(4, 4, ResinMemory::Default).unwrap();
        for (x, z, expected) in [(0.0, 0.0, 1.0), (4.0, 0.0, -1.0), (0.0, 3.0, 4.0)] {
            let transform = [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z];
            let mut scene = std::ptr::null_mut();
            assert_eq!(
                resin_gpu_create_ray_scene(
                    &gpu,
                    vertices.as_ptr(),
                    3,
                    transform.as_ptr(),
                    1,
                    &mut scene
                ),
                ResinStatus::Success
            );
            let mut pipeline = std::ptr::null_mut();
            assert_eq!(
                resin_gpu_create_ray_pipeline(
                    &gpu,
                    scene,
                    shaders[0].as_ptr(),
                    shaders[0].len(),
                    shaders[1].as_ptr(),
                    shaders[1].len(),
                    shaders[2].as_ptr(),
                    shaders[2].len(),
                    &mut pipeline
                ),
                ResinStatus::Success
            );
            resin_gpu_free_ray_scene(scene); // The pipeline must retain the scene.
            let mut commands = gpu.start_command_recording().unwrap();
            commands.set_pipeline(&*pipeline).unwrap();
            assert_eq!(
                commands.dispatch(output.device_pointer(), 1, 1, 1),
                Err(ResinStatus::InvalidArgument)
            );
            assert_eq!(
                resin_gpu_trace_rays(
                    &mut commands,
                    output.device_pointer(),
                    u32::MAX,
                    u32::MAX,
                    u32::MAX
                ),
                ResinStatus::InvalidArgument
            );
            assert_eq!(
                resin_gpu_trace_rays(&mut commands, output.device_pointer(), 1, 1, 1),
                ResinStatus::Success
            );
            gpu.submit(commands).unwrap();
            assert_eq!(*output.host_pointer().cast::<f32>(), expected);
            resin_gpu_free_pipeline(&mut gpu, pipeline);
        }
        gpu.free(&output);
    }
}

#[cfg(feature = "gpu")]
#[test]
fn source_example_projects_and_retains_output_through_submission() {
    let _lock = resin_runtime::testing::lock_gpu();
    let module = support::pipeline::host_entry(
        std::path::Path::new("examples/eg013_ray_tracing.resin"),
        "main",
    )
    .unwrap();
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    let directory = tempfile::TempDir::new().unwrap();
    let output = std::process::Command::new(executable.path())
        .current_dir(directory.path())
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    if output.stdout == b"Ray tracing pipelines are unavailable on this GPU.\n" {
        assert_ne!(
            std::env::var("RESIN_REQUIRE_RAY_TRACING").as_deref(),
            Ok("1")
        );
        assert!(!directory.path().join("ray-tracing.png").exists());
        return;
    }
    assert!(
        !String::from_utf8_lossy(&output.stderr).contains("VUID-"),
        "{output:?}"
    );
    assert_eq!(output.stdout, b"wrote ray-tracing.png\n");
    let image = resin_runtime::image_read_png(directory.path().join("ray-tracing.png"), 4).unwrap();
    assert_eq!((image.width, image.height), (640, 480));
    assert!(image.pixels.chunks_exact(4).all(|pixel| pixel[3] == 255));
    let mut counts = [0; 4];
    for y in (0..480).step_by(8) {
        for x in (0..640).step_by(8) {
            let (expected, instance) = fisheye_reference_pixel(x, y);
            counts[instance] += 1;
            let offset = (y * 640 + x) * 4;
            let actual = &image.pixels[offset..offset + 3];
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(&a, b)| a.abs_diff(b) <= 2),
                "pixel ({x}, {y}): {actual:?}, expected {expected:?}"
            );
        }
    }
    assert!(counts[..3].iter().all(|&hits| hits > 50), "{counts:?}");
    assert!(counts[3] > 1000, "{counts:?}");
}

// Independent f64 reference: bisect the forward lens model, intersect the Z=2
// plane analytically, and locate the point within each translated triangle.
#[cfg(feature = "gpu")]
fn fisheye_reference_pixel(x: usize, y: usize) -> ([u8; 3], usize) {
    let px = (x as f64 + 0.5 - 320.0) / 230.0;
    let py = (240.0 - y as f64 - 0.5) / 230.0;
    let radius = px.hypot(py);
    let (mut low, mut high) = (0.0, std::f64::consts::FRAC_PI_2);
    for _ in 0..50 {
        let theta = (low + high) * 0.5;
        let projected = theta
            + 0.05 * theta.powi(3)
            + 0.005 * theta.powi(5)
            + 0.0005 * theta.powi(7)
            + 0.00005 * theta.powi(9);
        if projected < radius {
            low = theta;
        } else {
            high = theta;
        }
    }
    let scale = 2.0 * ((low + high) * 0.5).tan() / radius;
    for (instance, translation) in [-3.0, 0.0, 3.0].into_iter().enumerate() {
        let local_x = px * scale - translation;
        let blue = (py * scale + 1.0) * 0.5;
        let green = (local_x + 1.0 - blue) * 0.5;
        let red = 1.0 - green - blue;
        if red >= 0.0 && green >= 0.0 && blue >= 0.0 {
            return ([red, green, blue].map(|v| (255.0 * v) as u8), instance);
        }
    }
    ([10, 15, 25], 3)
}

#[test]
fn fisheye_camera_round_trips_projection_on_cpu() {
    let example = include_str!("../examples/eg013_ray_tracing.resin")
        .replace("export { main };", "export { check_camera };");
    let checks = r#"
        fn check_camera() {
            let camera = example_camera();
            let center = direction(camera, 0, 0);
            assert(center.0 == 0 && center.1 == 0 && center.2 == 1);
            let equidistant = Fisheye { focal = 230, k1 = 0, k2 = 0, k3 = 0, k4 = 0 };
            assert(incident_angle(equidistant, f32(1)) == 1);
            let mut i = 0;
            while (i <= 100) {
                let theta = f32(i) * f32(0.0157);
                let t2 = theta * theta;
                let radius = theta + camera.k1 * theta * t2
                    + camera.k2 * theta * t2 * t2
                    + camera.k3 * theta * t2 * t2 * t2
                    + camera.k4 * theta * t2 * t2 * t2 * t2;
                let error = incident_angle(camera, radius) - theta;
                assert(error > f32(-0.00001) && error < f32(0.00001));
                i = i + 1;
            };
            let corner = direction(camera, 320, 240);
            let opposite = direction(camera, -320, -240);
            let length = corner.0 * corner.0 + corner.1 * corner.1 + corner.2 * corner.2;
            assert(length > f32(0.99999) && length < f32(1.00001));
            assert(corner.0 == -opposite.0 && corner.1 == -opposite.1 && corner.2 == opposite.2);
            assert(corner.2 > 0);
        }
    "#;
    let directory = tempfile::TempDir::new().unwrap();
    let path = directory.path().join("camera.resin");
    std::fs::write(&path, format!("{example}\n{checks}")).unwrap();
    let module = support::pipeline::host_entry(&path, "check_camera").unwrap();
    let project = support::project::Project::new(&module, Some("check_camera")).unwrap();
    assert!(project.generated.shaders().is_empty());
    let output = project.run();
    assert!(output.status.success(), "{output:?}");
}
