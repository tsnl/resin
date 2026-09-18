mod support;

const EXAMPLE: &str = include_str!("../examples/resource_bindings.resin");

#[test]
fn storage_algorithm_runs_on_host_spans() {
    let module = support::module(EXAMPLE);
    let project = support::project::Project::new(&module, Some("cpu_main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn resource_entry_emits_valid_spirv() {
    for source in [EXAMPLE, GRAPHICS] {
        let module = support::module(source);
        let project = support::project::Project::new(&module, None).unwrap();
        for shader in project.generated.shaders() {
            support::shaders::validate(shader.unoptimized_spirv());
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn typed_bindings_keep_offsets_permissions_and_recorded_values() {
    let _lock = resin_runtime::testing::lock_gpu();
    match resin_runtime::ResinGpu::create() {
        Ok(_) => (),
        Err(
            resin_runtime::ResinStatus::Unsupported | resin_runtime::ResinStatus::VulkanUnavailable,
        ) => {
            assert_ne!(std::env::var("RESIN_REQUIRE_GPU").as_deref(), Ok("1"));
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
    for source in [EXAMPLE, BOUNDS] {
        let module = support::module(source);
        let project = support::project::Project::new(&module, Some("main")).unwrap();
        let output = project.run();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn readonly_bindings_and_constants_cannot_be_written_or_retyped() {
    for body in [
        "fn bad(input: Ref<Buffer<f32>>) { input:store(0, f32(1)); }",
        "intrinsic \"gpu_buffer_store\" fn forge<T>(input: Ref<Buffer<T>>, index: u64, value: T) -> ();",
        "intrinsic \"gpu_read_buffer_type\" fn wrong<T>(input: Buffer<T>) -> Ptr<u32>;",
        "struct Resources { input: Buffer<f32>, dt: f32 } @compute_shader fn bad(index: u64, resources: Ref<Resources>) { resources.dt = 1; }",
        "struct Resources { input: Buffer<Ptr<f32>> } @compute_shader fn bad(index: u64, resources: Ref<Resources>) { let pointer = resources.input:load(index); }",
        "struct Resources { input: Buffer<Ptr<f32>> } @compute_shader fn bad(index: u64, resources: Ref<Resources>) {}",
        "struct Resources { raw: GpuView } @compute_shader fn bad(index: u64, resources: Ref<Resources>) {}",
        "struct Resources { input: Buffer<f32> } @compute_shader fn bad(index: u64, resources: Ref<Resources>) { let copy = resources.input; }",
    ] {
        let source = format!("import {{ \"$/buffer.resin\" }}; {body}");
        assert!(
            support::pipeline::source_module(&source).is_err(),
            "accepted {source}"
        );
    }
}

const GRAPHICS: &str = r#"
export { vertex, fragment };
import { "$/buffer.resin" };
struct Position { x: f32, y: f32, z: f32, w: f32 }
struct Color { r: f32, g: f32, b: f32, a: f32 }
struct Vertex { position: Position, color: Color }
struct Nested { colors: Buffer<Color> }
struct Resources { nested: Nested, scale: f32 }
@vertex_shader
fn vertex(index: i32, resources: Ref<Resources>) -> Vertex {
    Vertex {
        position = Position { x = resources.scale, y = 0, z = 0, w = 1 },
        color = resources.nested.colors:load(0),
    }
}
@fragment_shader
fn fragment(color: Color, resources: Ref<Resources>) -> Color {
    let tint = resources.nested.colors:load(0);
    Color { r = color.r * tint.r, g = color.g, b = color.b, a = color.a }
}
"#;

const BOUNDS: &str = r#"
export { main };
import { "$/buffer.resin", "$/gpu.resin" };
struct Resources { input: Buffer<f32>, output: BufferMut<f32> }
@compute_shader
fn bounds(index: u64, resources: Ref<Resources>) {
    if (index == 0) {
        resources.output:store(0, resources.input:load(resources.input.length));
        resources.output:store(1, resources.input:load(18446744073709551615));
        resources.output:store(resources.output.length, f32(91));
        resources.output:store(18446744073709551615, f32(92));
    };
}
fn main() -> i32 | Err<_> {
    let gpu = gpu_new()?;
    let input = gpu:alloc::<f32>(0)?;
    let output = gpu:alloc::<f32>(3)?;
    let first = output:at(0);
    let second = output:at(1);
    let guard = output:at(2);
    first:store(f32(77)); second:store(f32(77)); guard:store(f32(77));
    let range = output:slice(0, 2);
    let resources = Resources { input = input:read_buffer(), output = range:write_buffer() };
    bounds(0, resources);
    assert(first:load() == f32(0) && second:load() == f32(0) && guard:load() == f32(77));
    first:store(f32(88)); second:store(f32(88));
    let pipeline = gpu:create_compute_pipeline(bounds)?;
    let commands = gpu:start_command_recording()?;
    commands:dispatch(pipeline, resources, 1, 1, 1)?;
    commands:submit()?;
    assert(first:load() == f32(0) && second:load() == f32(0) && guard:load() == f32(77));
    0
}
"#;
