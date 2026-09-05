#![cfg(feature = "gpu")]

use resin::{
    gpu::{self, Pipeline},
    toolchain::TempDir,
};
use resin_runtime::{ResinGpu, ResinStatus, image_read_png, testing::lock_gpu};

#[path = "support/shaders.rs"]
mod shaders;
mod support;

#[test]
fn resin_programs_render_and_roundtrip_pngs() {
    let Some(compiler) = shaders::compiler() else {
        return;
    };
    let _lock = lock_gpu();
    match ResinGpu::create() {
        Ok(gpu) => drop(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            eprintln!("skipping: no suitable Vulkan device");
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }

    let gradient = support::module(include_str!("../examples/shaders/gradient.resin"));
    let image = gpu::render(&gradient, Pipeline::Compute, &compiler).unwrap();
    assert_eq!((image.width, image.height), (256, 256));
    assert_eq!(image.rgba.len(), 256 * 256 * 4);
    for (i, pixel) in image.rgba.chunks_exact(4).enumerate() {
        assert_eq!(pixel, &[i as u8, (i >> 8) as u8, 64, 255], "pixel {i}");
    }
    let temp = TempDir::new(&std::env::temp_dir()).unwrap();
    let output = temp.path().join("gradient.png");
    gpu::write_png(&image, &output).unwrap();
    assert_eq!(image_read_png(&output, 4).unwrap().pixels, image.rgba);

    let triangle = support::module(include_str!("../examples/shaders/triangle.resin"));
    let image = gpu::render(&triangle, Pipeline::Graphics, &compiler).unwrap();
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
    assert_eq!(image.rgba.len(), reference.pixels.len());
    for (i, (&actual, &expected)) in image.rgba.iter().zip(&reference.pixels).enumerate() {
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
    let image = gpu::Image {
        width: 2,
        height: 2,
        rgba: vec![],
    };
    assert!(gpu::write_png(&image, &output).is_err());
    assert_eq!(std::fs::read(&output).unwrap(), b"keep me");
    assert_eq!(std::fs::read_dir(temp.path()).unwrap().count(), 1);
}
