#![cfg(feature = "gpu")]

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
fn shader_while_loops_execute_with_nested_and_zero_trip_iterations() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    let Some(mut gpu) = gpu() else { return };
    let module = support::module(
        "kernel (index: uint) -> uint = { total = uint (0); n = index; while (n > uint (0) && n <= index) { j = uint (0); while (j < n) { total := total + uint (1); j := j + uint (1); }; n := n - uint (1); }; total };",
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
        let executable = temp.path().join(name);
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
            "/crates/resin-runtime/tests/hello_triangle.png"
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
