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
        commands:trace_rays(pipeline, f32(0.0), 1, 1, 1)?;
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
fn nested_tracing_and_hit_queries_in_other_stages_are_rejected() {
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
