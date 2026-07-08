//! Render a real HuggingFace 3DGS checkpoint (the `luigi` scene from the
//! `dylanebert/3dgs` dataset submodule) and check tensor/reference parity.
//!
//! Skips silently when the submodule is not initialized
//! (`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1`).

use std::path::PathBuf;

use resin_dsl::Tensor;
use resin_gaussians::{
    chunked_renderer, load_ply, render, render_reference, Camera, GaussianCloud,
};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};

fn luigi_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/hf/dylanebert-3dgs/luigi/luigi.ply")
}

fn luigi_camera(width: usize, height: usize) -> Camera {
    // Object scan around the origin; COLMAP convention (y down).
    Camera::orbit(
        [0.0, 0.0, 0.0],
        2.2,
        20.0,
        -10.0,
        [0.0, -1.0, 0.0],
        50.0,
        width,
        height,
    )
}

#[test]
fn luigi_checkpoint_renders_and_matches_reference() {
    let path = luigi_path();
    if !path.exists() {
        eprintln!("skip: HF checkpoint submodule not initialized ({path:?})");
        return;
    }
    let full = load_ply(&path).expect("parse luigi.ply");
    assert_eq!(full.count(), 14526, "luigi gaussian count");

    // CPU-interpreter scale: every 30th gaussian, small image.
    let data = full.subsample(30);
    let camera = luigi_camera(32, 32);

    let n = data.count();
    let cloud = GaussianCloud {
        means: CpuTensor::from_f32(&[n, 3], &data.means_flat()),
        scales: CpuTensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: CpuTensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: CpuTensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: CpuTensor::from_f32(&[n], &data.opacities),
    };
    let cam = camera.clone();
    let f = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| render(cloud, &cam));
    let image = f.call(&cloud).expect("render luigi subsample").to_f32();

    let expected = render_reference(&data, &camera);
    let diff = image
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-3, "max abs diff vs reference: {diff}");
    assert!(
        expected.iter().any(|&v| v > 0.05),
        "checkpoint render came out blank"
    );
}

#[test]
fn chunked_renderer_matches_dense_on_checkpoint() {
    let path = luigi_path();
    if !path.exists() {
        eprintln!("skip: HF checkpoint submodule not initialized ({path:?})");
        return;
    }
    let data = load_ply(&path).expect("parse luigi.ply").subsample(40);
    let camera = luigi_camera(24, 24);
    let n = data.count();
    let cloud = GaussianCloud {
        means: CpuTensor::from_f32(&[n, 3], &data.means_flat()),
        scales: CpuTensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: CpuTensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: CpuTensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: CpuTensor::from_f32(&[n], &data.opacities),
    };

    // Dense path.
    let cam = camera.clone();
    let dense_fn = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| render(cloud, &cam));
    let dense = dense_fn.call(&cloud).unwrap().to_f32();

    // Chunked fold with a chunk size that does not divide n evenly.
    let chunked = chunked_renderer::<CpuJit>(CpuJit, n, 24, 24, 64)(&cloud, &camera);

    let diff = dense
        .iter()
        .zip(&chunked)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(diff < 1e-4, "chunked vs dense max abs diff: {diff}");
    assert!(dense.iter().any(|&v| v > 0.05), "blank image");
}
