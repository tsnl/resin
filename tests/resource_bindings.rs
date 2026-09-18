mod support;

const EXAMPLE: &str = include_str!("../examples/resource_bindings.resin");

#[cfg(feature = "gpu")]
#[test]
fn descriptors_execute_without_buffer_device_addresses() {
    use resin_runtime::{ResinMemory, ResinStatus, testing};
    let Some(optimizer) = support::shaders::optimizer() else {
        return;
    };
    let _lock = testing::lock_gpu();
    let mut gpu = match testing::gpu_without_device_addresses() {
        Ok(gpu) => gpu,
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert_ne!(std::env::var("RESIN_REQUIRE_GPU").as_deref(), Ok("1"));
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    };
    let source = r#"
        export { kernel };
        import { "$/buffer.resin" };
        struct Resources { input: Buffer<u32>, output: BufferMut<u32>, bias: u32 }
        @compute_shader
        fn kernel(i: u64, resources: Ref<Resources>) {
            resources.output:store(i, resources.input:load(i) + resources.bias);
        }
    "#;
    let module = support::module(source);
    let project = support::project::Project::new(&module, None).unwrap();
    let built = project
        .build(&support::toolchain::spirv(&optimizer))
        .unwrap();
    let shader = &project.generated.shaders()[0];
    let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
    assert!(support::shaders::instructions(&bytes, 120).next().is_none());
    #[repr(C)]
    struct Binding {
        offset: u64,
        reserved: [u64; 2],
        length: u64,
    }
    #[repr(C)]
    struct Parameters {
        input: Binding,
        output: Binding,
        bias: u32,
    }
    // Mirror the compiler's wire layout. Allocations, pipeline, and descriptor
    // sets stay live until synchronous submission; mapped accesses do not overlap it.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&bytes).unwrap();
        let input = gpu.malloc(24, 4, ResinMemory::Default).unwrap();
        let output = gpu.malloc(24, 4, ResinMemory::Default).unwrap();
        let root = gpu
            .malloc(size_of::<Parameters>(), 8, ResinMemory::Default)
            .unwrap();
        for allocation in [&input, &output, &root] {
            assert_eq!(allocation.device_pointer(), 0);
        }
        input
            .host_pointer()
            .cast::<[u32; 6]>()
            .write([99, 1, 2, 3, 4, 99]);
        output.host_pointer().cast::<[u32; 6]>().write([99; 6]);
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        let offsets = testing::bind_resource_buffers(
            &gpu,
            &mut commands,
            &[
                (&root, 0, size_of::<Parameters>()),
                (&input, 4, 16),
                (&output, 4, 16),
            ],
        )
        .unwrap();
        root.host_pointer().cast::<Parameters>().write(Parameters {
            input: Binding {
                offset: offsets[1],
                reserved: [0; 2],
                length: 4,
            },
            output: Binding {
                offset: offsets[2],
                reserved: [0; 2],
                length: 4,
            },
            bias: 10,
        });
        commands.dispatch(offsets[0], 1, 1, 1).unwrap();
        gpu.submit(commands).unwrap();
        assert_eq!(
            output.host_pointer().cast::<[u32; 6]>().read(),
            [99, 11, 12, 13, 14, 99]
        );
        drop(pipeline);
        gpu.free(&root);
        gpu.free(&output);
        gpu.free(&input);
    }
    // A stage without a resource argument must not force the whole graphics
    // pipeline back onto the physical-address profile.
    let source = GRAPHICS
        .replace(
            "fn vertex(index: i32, resources: Ref<Resources>)",
            "fn vertex(index: i32)",
        )
        .replace("resources.scale", "f32(1)")
        .replace(
            "color = resources.nested.colors:load(0)",
            "color = Color { r = 1, g = 0, b = 0, a = 1 }",
        );
    let module = support::module(&source);
    let project = support::project::Project::new(&module, None).unwrap();
    let built = project
        .build(&support::toolchain::spirv(&optimizer))
        .unwrap();
    let shader_bytes = |stage| {
        let shader = project
            .generated
            .shaders()
            .iter()
            .find(|s| s.stage() == stage)
            .unwrap();
        let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
        assert!(support::shaders::instructions(&bytes, 14).all(|args| args[0] == 0));
        bytes
    };
    let vertex = shader_bytes(resin_types::Stage::Vertex);
    let fragment = shader_bytes(resin_types::Stage::Fragment);
    drop(unsafe { gpu.create_graphics_pipeline(&vertex, &fragment) }.unwrap());
    let module = support::module(
        "export { kernel }; @compute_shader fn kernel(i: u64, p: Ptr<u32>) { p.* = u32(i); }",
    );
    let project = support::project::Project::new(&module, None).unwrap();
    let bytes = std::fs::read(project.generated.shaders()[0].unoptimized_spirv()).unwrap();
    assert!(matches!(
        unsafe { gpu.create_compute_pipeline(&bytes) },
        Err(ResinStatus::Unsupported)
    ));
}

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
            let bytes = std::fs::read(shader.unoptimized_spirv()).unwrap();
            assert!(
                support::shaders::instructions(&bytes, 14).all(|args| args[0] == 0),
                "descriptor shader must use Logical addressing"
            );
            assert!(
                !support::shaders::instructions(&bytes, 17).any(|args| args[0] == 5347),
                "unexpected PhysicalStorageBufferAddresses capability"
            );
            assert!(
                support::shaders::instructions(&bytes, 120).next().is_none(),
                "unexpected OpConvertUToPtr"
            );
            assert!(
                support::shaders::instructions(&bytes, 71).any(|args| args[1] == 33),
                "missing Binding decoration"
            );
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
    for source in [EXAMPLE, BOUNDS, SNAPSHOTS, GRAPHICS_EXECUTION] {
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

const SNAPSHOTS: &str = r#"
export { main };
import { "$/buffer.resin", "$/gpu.resin" };
struct Element { byte: u8, value: f32, word: u32 }
struct Resources { output: BufferMut<Element>, value: f32 }
@compute_shader
fn kernel(i: u64, resources: Ref<Resources>) {
    resources.output:store(i, Element { byte = u8(17), value = resources.value, word = u32(i) });
}
fn main() -> i32 | Err<_> {
    let gpu = gpu_new()?;
    let output = gpu:alloc::<Element>(42)?;
    let pipeline = gpu:create_compute_pipeline(kernel)?;
    let commands = gpu:start_command_recording()?;
    let mut i: u64 = 0;
    // More recordings than fit in one descriptor pool, changing both the buffer
    // subrange and constants. Each iteration drops the binding's original owner.
    while (i < 40) {
        let range = output:slice(i + 1, 1);
        let resources = Resources { output = range:write_buffer(), value = f32(i) };
        commands:dispatch(pipeline, resources, 1, 1, 1)?;
        i = i + 1;
    };
    commands:submit()?;
    i = 0;
    while (i < 40) {
        let value = output:at(i + 1);
        let element = value:load();
        assert(element.byte == u8(17) && element.value == f32(i) && element.word == u32(0));
        i = i + 1;
    };
    0
}
"#;

const GRAPHICS_EXECUTION: &str = r#"
export { main };
import { "$/buffer.resin", "$/graphics.resin", "$/gpu.resin" };
struct Resources { colors: Buffer<Color>, observed: BufferMut<u32>, multiplier: f32 }
@vertex_shader
fn vertex(index: i32, resources: Ref<Resources>) -> Vertex {
    let x: f32 = if (index == 1) { 3 } else { -1 };
    let y: f32 = if (index == 2) { 3 } else { -1 };
    Vertex { position = Position { x = x, y = y, z = 0, w = 1 }, color = resources.colors:load(0) }
}
@fragment_shader
fn fragment(color: Color, resources: Ref<Resources>) -> Color {
    resources.observed:store(0, u32(23));
    Color { r = color.r * resources.multiplier, g = color.g, b = color.b, a = color.a }
}
fn main() -> i32 | Err<_> {
    let gpu = gpu_new()?;
    let colors = gpu:alloc::<Color>(1)?;
    let color = colors:at(0);
    color:store(Color { r = 1, g = 0, b = 0, a = 1 });
    let observed = gpu:alloc::<u32>(1)?;
    let resources = Resources { colors = colors:read_buffer(), observed = observed:write_buffer(), multiplier = 1 };
    let pipeline = gpu:create_graphics_pipeline(vertex, fragment)?;
    let image = gpu:create_image(1, 1)?;
    let pixels = gpu:alloc::<u8>(4)?;
    let commands = gpu:start_command_recording()?;
    commands:begin_rendering(image, 0, 0, 0, 1)?;
    commands:draw(pipeline, resources, 3)?;
    commands:end_rendering()?;
    commands:copy_image_to_buffer(image, pixels)?;
    commands:submit()?;
    let red = pixels:at(0); let green = pixels:at(1); let blue = pixels:at(2); let alpha = pixels:at(3);
    let written = observed:at(0);
    assert(written:load() == u32(23));
    assert(red:load() == u8(255) && green:load() == u8(0) && blue:load() == u8(0) && alpha:load() == u8(255));
    0
}
"#;
