use support::{pipeline, project::Project, shaders, toolchain};
mod support;

const SHADERS: &str = r#"
import { "$/graphics.resin", "$/shared.resin" };
type OptionalColor = None | Color;

@vertex_shader
fn vertex(index: i32) -> Vertex {
    let x: f32 = if (index == 1) { 3.0 } else { -1.0 };
    let y: f32 = if (index == 2) { 3.0 } else { -1.0 };
    Vertex {
        position = Position { x = x, y = y, z = 0.0, w = 1.0 },
        color = Color { r = (x + 1.0) * 0.5, g = (y + 1.0) * 0.5, b = 0.0, a = 1.0 },
    }
}

fn mask(color: Color) -> OptionalColor {
    if (color.r < 0.5) { None }
    else { Color { r = 1.0, g = 0.0, b = 0.0, a = 0.0 } }
}

@fragment_shader
fn fragment(color: Color, fallback: Ptr<Color>) -> OptionalColor {
    let result = mask(color);
    if (color.g < 0.5) { result }
    else {
        match (result) {
            Color(value) => { value },
            None => { fallback.* },
        }
    }
}
"#;

#[test]
fn optional_fragment_results_remain_values_on_the_host() {
    let source = format!(
        r#"export {{ main }}; {SHADERS}
        fn main() -> i32 | Err<_> {{
            let fallback = arc_ptr_alloc(Color {{ r = 0.0, g = 1.0, b = 0.0, a = 1.0 }})?;
            let absent = fragment(Color {{ r = 0.0, g = 0.0, b = 0.0, a = 1.0 }}, fallback:get());
            let recovered = fragment(Color {{ r = 0.0, g = 1.0, b = 0.0, a = 1.0 }}, fallback:get())!;
            let transparent = fragment(Color {{ r = 1.0, g = 0.0, b = 0.0, a = 1.0 }}, fallback:get())!;
            let discarded = match (absent) {{ None => {{ true }}, Color(value) => {{ false }} }};
            if (discarded && recovered.g == 1.0 && transparent.r == 1.0 && transparent.a == 0.0) {{ 0 }} else {{ 1 }}
        }}"#
    );
    let module = support::module(&source);
    let project = Project::new(&module, Some("main")).unwrap();
    assert!(project.generated.shaders().is_empty());
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn only_the_fragment_entry_interprets_none_as_discard() {
    let source = format!("export {{ vertex, fragment }}; {SHADERS}");
    let module = pipeline::source_module(&source).unwrap();
    let project = Project::new(&module, None).unwrap();
    for shader in project.generated.shaders() {
        let bytes = std::fs::read(shader.unoptimized_spirv()).unwrap();
        let entry = shaders::instructions(&bytes, 15).next().unwrap(); // OpEntryPoint
        let fragment = entry[0] == 4;
        let terminators = shaders::instructions(&bytes, 4416).count(); // OpTerminateInvocation
        assert_eq!(terminators, usize::from(fragment));
        if fragment {
            let words = bytes
                .chunks_exact(4)
                .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>();
            let mut current_function = None;
            let mut offset = 5;
            while offset < words.len() {
                let opcode = words[offset] & 0xffff;
                match opcode {
                    54 => current_function = Some(words[offset + 2]), // OpFunction
                    56 => current_function = None,                    // OpFunctionEnd
                    4416 => assert_eq!(current_function, Some(entry[1])),
                    _ => (),
                }
                offset += (words[offset] >> 16) as usize;
            }
            // Forced early tests could commit depth/stencil before discard.
            assert!(!shaders::instructions(&bytes, 16).any(|args| args[1] == 9));
        }
    }
    if let Some(optimizer) = shaders::optimizer() {
        project.build(&toolchain::spirv(&optimizer)).unwrap();
    }
}

#[test]
fn invalid_optional_fragment_results_are_rejected_at_declaration() {
    for (result, expression) in [
        ("None", "None"),
        ("f32 | None", "None"),
        ("Color | None | u32", "None"),
        ("Color | Err<u32>", "color"),
    ] {
        let source = format!(
            "import {{ \"$/graphics.resin\" }}; @fragment_shader fn bad(color: Color) -> {result} {{ {expression} }}"
        );
        let error = pipeline::source_module(&source).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid @fragment_shader signature"),
            "{error}"
        );
    }
}

#[test]
fn constant_and_fallible_optional_outputs_validate_before_and_after_optimization() {
    for body in [
        "None",
        "color",
        "let checked = u8(color.r); if (checked == 0) { None } else { color }",
    ] {
        let source = format!(
            "export {{ fragment }}; import {{ \"$/graphics.resin\" }}; @fragment_shader fn fragment(color: Color) -> Color | None {{ {body} }}"
        );
        let module = support::module(&source);
        let project = Project::new(&module, None).unwrap();
        shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
        if let Some(optimizer) = shaders::optimizer() {
            project.build(&toolchain::spirv(&optimizer)).unwrap();
        }
    }
}

#[cfg(feature = "gpu")]
fn gpu() -> Option<resin_runtime::ResinGpu> {
    use resin_runtime::{ResinGpu, ResinStatus};
    match ResinGpu::create() {
        Ok(gpu) => Some(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            eprintln!("skipping: no suitable Vulkan device");
            None
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
}

#[cfg(feature = "gpu")]
#[test]
fn discard_preserves_background_while_zero_alpha_and_recovered_none_write_color() {
    use resin_runtime::{ResinMemory, testing::lock_gpu};
    use resin_types::prelude::Stage;
    let Some(optimizer) = shaders::optimizer() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let source = format!("export {{ vertex, fragment }}; {SHADERS}");
    let module = pipeline::source_module(&source).unwrap();
    let project = Project::new(&module, None).unwrap();
    let built = project.build(&toolchain::spirv(&optimizer)).unwrap();
    let shader_bytes = |stage| {
        let shader = project
            .generated
            .shaders()
            .iter()
            .find(|shader| shader.stage() == stage)
            .unwrap();
        std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap()
    };
    let vertex = shader_bytes(Stage::Vertex);
    let fragment = shader_bytes(Stage::Fragment);
    // Root and readback allocations stay alive through synchronous submission.
    unsafe {
        let pipeline = gpu.create_graphics_pipeline(&vertex, &fragment).unwrap();
        let root = gpu.malloc(16, 4, ResinMemory::Default).unwrap();
        root.host_pointer()
            .cast::<[f32; 4]>()
            .write([0.0, 1.0, 0.0, 1.0]);
        let mut image = gpu.create_image(64, 64).unwrap();
        let pixels = gpu.malloc(64 * 64 * 4, 4, ResinMemory::Readback).unwrap();
        let mut commands = gpu.start_command_recording().unwrap();
        commands
            .begin_rendering(&mut image, [0.0, 0.0, 1.0, 1.0])
            .unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands.draw(root.device_pointer(), 3).unwrap();
        commands.end_rendering().unwrap();
        commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
        gpu.submit(commands).unwrap();
        for (index, pixel) in pixels.host_bytes().unwrap().chunks_exact(4).enumerate() {
            let (x, y) = (index % 64, index / 64);
            let expected = if x >= 32 {
                [255, 0, 0, 0]
            } else if y >= 32 {
                [0, 255, 0, 255]
            } else {
                [0, 0, 255, 255]
            };
            assert_eq!(pixel, expected, "pixel ({x}, {y})");
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
fn discard_fragments_example_projects_its_buffer_and_preserves_the_previous_draw() {
    let _lock = resin_runtime::testing::lock_gpu();
    let Some(gpu) = gpu() else { return };
    drop(gpu);
    let module = pipeline::host_entry(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/discard_fragments.resin"),
        "main",
    )
    .unwrap();
    let project = Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    let directory = tempfile::TempDir::new().unwrap();
    let output = std::process::Command::new(executable.path())
        .current_dir(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image =
        resin_runtime::image_read_png(directory.path().join("discard-fragments.png"), 4).unwrap();
    assert_eq!((image.width, image.height, image.channels), (512, 512, 4));
    for y in 0..8 {
        for x in 0..8 {
            let offset = ((y * 64 + 32) * 512 + x * 64 + 32) * 4;
            let pixel = &image.pixels[offset..offset + 4];
            assert_eq!(pixel[3], 255);
            if (x + y) % 2 == 0 {
                assert_eq!(pixel[0], 255, "opaque square ({x}, {y})");
                assert!(pixel[1] > 110 && pixel[2] < 25);
            } else {
                assert!(
                    pixel[0] < 20 && pixel[2] > 75,
                    "discarded square ({x}, {y})"
                );
            }
        }
    }
}
