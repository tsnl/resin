#![cfg(feature = "gpu")]

use resin::backend::glsl::{self, Stage};
use resin::toolchain::TempDir;
use resin_runtime::{
    ResinGpu, ResinMemory, ResinStatus, image_read_png, image_write_png, testing::lock_gpu,
};
use std::{path::Path, process::Command};

#[path = "support/shaders.rs"]
mod shaders;
mod support;

fn gpu() -> Option<ResinGpu> {
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

#[test]
fn typed_device_buffers_match_host_layout_and_preserve_bounds() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let module = support::module(
        r#"export { kernel, main };

        type Data = { marker: uint, wide: ulong, amount: float32 };
        type Payload = { tag: uint, data: Data, end: uint };
        type Params = { count: uint, values: Ptr<Payload>, tail: float32 };
        def at (values: Ptr<Payload>, index: uint) -> Ptr<Payload> = { values + index };
        def bump (p: Ptr<Payload>, index: uint) -> () = {
            var old = p.*;
            p.* := Payload {
                tag = old.tag + uint (1),
                data = Data { marker = index, wide = old.data.wide + ulong (4294967297), amount = old.data.amount + float32 (0.5) },
                end = old.end + uint (2)
            };
        };
        def kernel (index: uint, root: Ptr<Params>) -> () = {
            if (index < root.count) {
                var p = (at(root.values, index) + 1) + -1;
                bump(p - uint (0), index)
            } else { () }
        };
        def main () -> int = {
            var value = Payload { tag = uint (10), data = Data { marker = uint (99), wide = ulong (7), amount = float32 (1.25) }, end = uint (20) };
            var root = Params { count = uint (1), values = &value, tail = float32 (0.75) };
            kernel(uint (0), &root);
            kernel(uint (1), &root);
            if (value.tag == uint (11) && value.data.marker == uint (0) && value.data.wide == ulong (4294967304) && value.data.amount == float32 (1.75) && value.end == uint (22) && root.tail == float32 (0.75)) { 0 } else { 1 }
        };
    "#,
    );
    let c = resin::backend::c::emit(&module, "main").unwrap();
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let executable = temp
        .path()
        .join(format!("host-layout{}", std::env::consts::EXE_SUFFIX));
    let cc = std::env::var_os("CC").unwrap_or_else(|| resin::toolchain::DEFAULT_C_COMPILER.into());
    resin::toolchain::compile_c(&c, &executable, &cc).unwrap();
    assert!(Command::new(executable).status().unwrap().success());
    let glsl = glsl::emit(&module, "kernel", Stage::Compute).unwrap();
    let spv = resin::toolchain::compile_glsl(&glsl, Stage::Compute, &compiler).unwrap();
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Data {
        marker: u32,
        wide: u64,
        amount: f32,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Payload {
        tag: u32,
        data: Data,
        end: u32,
    }
    #[repr(C)]
    struct Params {
        count: u32,
        values: u64,
        tail: f32,
    }
    const COUNT: usize = 67;
    let initial = Payload {
        tag: 10,
        data: Data {
            marker: 99,
            wide: 7,
            amount: 1.25,
        },
        end: 20,
    };
    assert_eq!((size_of::<Payload>(), align_of::<Payload>()), (40, 8));
    // Both address spaces use live, aligned allocations; reads follow synchronous submission.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&spv).unwrap();
        let values = gpu
            .malloc(
                (COUNT + 1) * size_of::<Payload>(),
                align_of::<Payload>(),
                ResinMemory::Default,
            )
            .unwrap();
        let root = gpu
            .malloc(
                size_of::<Params>(),
                align_of::<Params>(),
                ResinMemory::Default,
            )
            .unwrap();
        std::slice::from_raw_parts_mut(values.host_pointer().cast::<Payload>(), COUNT + 1)
            .fill(initial);
        root.host_pointer().cast::<Params>().write(Params {
            count: COUNT as u32,
            values: values.device_pointer(),
            tail: 0.75,
        });
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands
            .dispatch(root.device_pointer(), (COUNT as u32).div_ceil(64), 1, 1)
            .unwrap();
        gpu.submit(commands).unwrap();
        let values = std::slice::from_raw_parts(values.host_pointer().cast::<Payload>(), COUNT + 1);
        for (index, value) in values[..COUNT].iter().enumerate() {
            assert_eq!(
                *value,
                Payload {
                    tag: 11,
                    data: Data {
                        marker: index as u32,
                        wide: 4294967304,
                        amount: 1.75
                    },
                    end: 22
                }
            );
        }
        assert_eq!(values[COUNT], initial);
        assert_eq!((*root.host_pointer().cast::<Params>()).tail, 0.75);
    }
}

#[test]
fn particles_compute_then_render_from_the_same_buffer() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/lib/particles.resin");
    let module = resin::ir::generate_program(&resin::ast::load(&source).unwrap()).unwrap();
    let compile = |stage: Stage| {
        let glsl = glsl::emit(&module, stage.entry(), stage).unwrap();
        resin::toolchain::compile_glsl(&glsl, stage, &compiler).unwrap()
    };
    let compute = compile(Stage::Compute);
    let vertex = compile(Stage::Vertex);
    let fragment = compile(Stage::Fragment);
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Particle {
        x: f32,
        y: f32,
        vx: f32,
        vy: f32,
        color: [f32; 4],
    }
    #[repr(C)]
    struct Params {
        count: u32,
        dt: f32,
        radius: f32,
        particles: u64,
    }
    let initial = Particle {
        x: -0.5,
        y: 0.0,
        vx: 1.0,
        vy: 0.0,
        color: [1.0, 0.0, 0.0, 1.0],
    };
    // Compute, draw, and readback share a recording; every resource outlives its submission.
    unsafe {
        let compute = gpu.create_compute_pipeline(&compute).unwrap();
        let graphics = gpu.create_graphics_pipeline(&vertex, &fragment).unwrap();
        let particles = gpu
            .malloc(
                2 * size_of::<Particle>(),
                align_of::<Particle>(),
                ResinMemory::Default,
            )
            .unwrap();
        let root = gpu
            .malloc(
                size_of::<Params>(),
                align_of::<Params>(),
                ResinMemory::Default,
            )
            .unwrap();
        std::slice::from_raw_parts_mut(particles.host_pointer().cast::<Particle>(), 2)
            .fill(initial);
        root.host_pointer().cast::<Params>().write(Params {
            count: 1,
            dt: 0.5,
            radius: 0.2,
            particles: particles.device_pointer(),
        });
        let mut image = gpu.create_image(64, 64).unwrap();
        let pixels = gpu.malloc(64 * 64 * 4, 4, ResinMemory::Readback).unwrap();
        for (expected_x, expected_vx) in [(0.0, 1.0), (0.5, 1.0), (1.0, -1.0), (0.5, -1.0)] {
            let mut commands = gpu.start_command_recording().unwrap();
            commands.set_pipeline(&compute).unwrap();
            commands.dispatch(root.device_pointer(), 1, 1, 1).unwrap();
            commands
                .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
                .unwrap();
            commands.set_pipeline(&graphics).unwrap();
            commands.draw(root.device_pointer(), 3).unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            gpu.submit(commands).unwrap();
            let values = std::slice::from_raw_parts(particles.host_pointer().cast::<Particle>(), 2);
            assert_eq!(values[0].x, expected_x);
            assert_eq!(values[0].vx, expected_vx);
            assert_eq!(values[1], initial);
            let pixels = pixels.host_bytes().unwrap();
            let sample_x = ((expected_x * 0.5 + 0.5) * 64.0) as usize;
            let offset = (32 * 64 + sample_x.min(63)) * 4;
            assert_eq!(&pixels[offset..offset + 4], &[255, 0, 0, 255]);
            assert_eq!(
                &pixels[(32 * 64 + 16) * 4..(32 * 64 + 16) * 4 + 4],
                &[0, 0, 0, 255]
            );
        }
    }
}

#[test]
fn fragment_shaders_read_typed_root_parameters() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/lib/triangle.resin");
    let mut ast = resin::ast::load(&source).unwrap();
    let file = &mut ast.modules.last_mut().unwrap().file;
    file.stmts.retain(|stmt| !matches!(&stmt.val, resin::ast::StmtKind::Function { name, .. } if name.val.as_ref() == "fragment"));
    file.stmts.extend(
        support::parse(
            "export { fragment }; def fragment (color: Color, root: Ptr<Color>) -> Color = { root.* };",
        )
        .stmts,
    );
    let module = resin::ir::generate_program(&ast).unwrap();
    let compile = |stage: Stage| {
        let glsl = glsl::emit(&module, stage.entry(), stage).unwrap();
        resin::toolchain::compile_glsl(&glsl, stage, &compiler).unwrap()
    };
    let vertex = compile(Stage::Vertex);
    let fragment = compile(Stage::Fragment);
    // The mapped root is updated only between completed submissions.
    unsafe {
        let pipeline = gpu.create_graphics_pipeline(&vertex, &fragment).unwrap();
        let root = gpu.malloc(16, 4, ResinMemory::Default).unwrap();
        let mut image = gpu.create_image(64, 64).unwrap();
        let pixels = gpu.malloc(64 * 64 * 4, 4, ResinMemory::Readback).unwrap();
        for color in [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]] {
            root.host_pointer().cast::<[f32; 4]>().write(color);
            let mut commands = gpu.start_command_recording().unwrap();
            commands
                .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
                .unwrap();
            commands.set_pipeline(&pipeline).unwrap();
            commands.draw(root.device_pointer(), 3).unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            gpu.submit(commands).unwrap();
            let pixels = pixels.host_bytes().unwrap();
            let center = (32 * 64 + 32) * 4;
            assert_eq!(
                &pixels[center..center + 4],
                color.map(|v| (v * 255.0) as u8)
            );
        }
    }
}

#[test]
fn shader_while_loops_execute_with_nested_and_zero_trip_iterations() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let module = support::module(
        "export { kernel }; def kernel (index: uint) -> uint = { var total = uint (0); var n = index; while (n > uint (0) && n <= index) { var j = uint (0); while (j < n) { total := total + uint (1); j := j + uint (1); }; n := n - uint (1); }; total };",
    );
    let glsl = resin::backend::glsl::emit(&module, "kernel", resin::backend::glsl::Stage::Compute)
        .unwrap();
    let spv =
        resin::toolchain::compile_glsl(&glsl, resin::backend::glsl::Stage::Compute, &compiler)
            .unwrap();
    #[repr(C)]
    struct Root {
        count: u32,
        pixels: u64,
    }
    const COUNT: u32 = 67;
    // All resources share one GPU and remain live until synchronous submission completes.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&spv).unwrap();
        let pixels = gpu
            .malloc((COUNT as usize + 1) * 4, 4, ResinMemory::Default)
            .unwrap();
        let root = gpu
            .malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default)
            .unwrap();
        std::slice::from_raw_parts_mut(pixels.host_pointer().cast::<u32>(), COUNT as usize + 1)
            .fill(u32::MAX);
        root.host_pointer().cast::<Root>().write(Root {
            count: COUNT,
            pixels: pixels.device_pointer(),
        });
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands
            .dispatch(root.device_pointer(), COUNT.div_ceil(64), 1, 1)
            .unwrap();
        gpu.submit(commands).unwrap();
        let values =
            std::slice::from_raw_parts(pixels.host_pointer().cast::<u32>(), COUNT as usize + 1);
        for (index, &value) in values[..COUNT as usize].iter().enumerate() {
            assert_eq!(
                value,
                (index * (index + 1) / 2) as u32,
                "invocation {index}"
            );
        }
        assert_eq!(values[COUNT as usize], u32::MAX);
    }
}

#[test]
fn ordinary_resin_programs_render_and_write_pngs() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(gpu) = gpu() else { return };
    drop(gpu);
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    for name in ["gradient", "triangle"] {
        let source = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(format!("{name}.resin"));
        let executable = temp
            .path()
            .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        let output = Command::new(env!("CARGO_BIN_EXE_resin"))
            .current_dir(temp.path())
            .arg(source)
            .arg("--glslc")
            .arg(&compiler)
            .arg("-o")
            .arg(&executable)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("wrote {name}.png\n")
        );

        // The copy is standalone: no compiler or source files are needed to run it.
        let output = Command::new(executable)
            .current_dir(temp.path())
            .env("GLSLC", "/missing/glslc")
            .env("CC", "/missing/cc")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let image = image_read_png(temp.path().join("gradient.png"), 4).unwrap();
    assert_eq!((image.width, image.height), (256, 256));
    assert_eq!(image.pixels.len(), 256 * 256 * 4);
    for (i, pixel) in image.pixels.chunks_exact(4).enumerate() {
        assert_eq!(pixel, &[i as u8, (i >> 8) as u8, 64, 255], "pixel {i}");
    }
    let image = image_read_png(temp.path().join("triangle.png"), 4).unwrap();
    let reference = image_read_png(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/resin-runtime/tests/hello_triangle.png"
        ),
        4,
    )
    .unwrap();
    assert_eq!(
        (image.width, image.height),
        (reference.width, reference.height)
    );
    assert_eq!(image.pixels.len(), reference.pixels.len());
    for (i, (&actual, &expected)) in image.pixels.iter().zip(&reference.pixels).enumerate() {
        let tolerance = u8::from(i % 4 != 3);
        assert!(
            actual.abs_diff(expected) <= tolerance,
            "channel {i}: {actual} != {expected}"
        );
    }
}

#[test]
fn invalid_images_do_not_replace_existing_files() {
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("existing.png");
    std::fs::write(&output, b"keep me").unwrap();
    assert!(image_write_png(&output, 2, 2, 4, &[]).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
}
