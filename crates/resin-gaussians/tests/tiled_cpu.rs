//! Tiled renderer tests: parity with the dense path, graceful capacity
//! overflow, and differentiability of the whole tiled pipeline.
//!
//! Parity note: tiling crops each gaussian at its 3σ radius box (as real
//! tiled rasterizers do), while the dense path evaluates everywhere. A
//! gaussian with opacity `o` still has `alpha = o·e^{-4.5} ≈ 0.011·o` at the
//! box edge, so high-opacity scenes may differ from dense by up to ~0.01 per
//! pixel at gaussian fringes; low-opacity scenes (tail below the 1/255
//! threshold) match exactly.

use resin_dsl::{grad_wrt, Tensor};
use resin_gaussians::{
    gnomen_cloud, load_ply, render, render_tiled, Camera, CloudData, GaussianCloud, TiledConfig,
};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};

fn cloud_tensors(data: &CloudData) -> GaussianCloud<CpuTensor> {
    let n = data.count();
    GaussianCloud {
        means: CpuTensor::from_f32(&[n, 3], &data.means_flat()),
        scales: CpuTensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: CpuTensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: CpuTensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: CpuTensor::from_f32(&[n], &data.opacities),
    }
}

fn render_dense_cpu(data: &CloudData, camera: &Camera) -> Vec<f32> {
    let cam = camera.clone();
    let f = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| render(cloud, &cam));
    f.call(&cloud_tensors(data)).unwrap().to_f32()
}

fn render_tiled_cpu(data: &CloudData, camera: &Camera, cfg: TiledConfig) -> Vec<f32> {
    let cam = camera.clone();
    let (w, h) = (camera.width, camera.height);
    let f = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| {
        let (view, proj) = cam.matrix_constants();
        render_tiled(cloud, &view, &proj, w, h, &cfg)
    });
    f.call(&cloud_tensors(data)).unwrap().to_f32()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn tiled_matches_dense_up_to_radius_crop() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(16, 16);
    let cfg = TiledConfig {
        tile: 8,
        max_tiles_per_gaussian: 8,
        tile_capacity: 16,
    };
    let tiled = render_tiled_cpu(&data, &camera, cfg);
    let dense = render_dense_cpu(&data, &camera);
    let diff = max_abs_diff(&tiled, &dense);
    // Opacity 0.9 → fringe alpha ≈ 0.01 beyond the radius box.
    assert!(diff < 0.03, "tiled vs dense max abs diff: {diff}");
    assert!(dense.iter().any(|&v| v > 0.05), "blank image");
}

#[test]
fn tiled_matches_dense_exactly_at_low_opacity() {
    // Fringe alpha 0.3·e^{-4.5} ≈ 0.0033 < 1/255: both paths threshold it
    // away, so the radius crop changes nothing and parity is exact.
    let mut data = gnomen_cloud();
    data.opacities = vec![0.3; 3];
    let camera = Camera::gnomen_default(16, 16);
    let cfg = TiledConfig {
        tile: 8,
        max_tiles_per_gaussian: 8,
        tile_capacity: 16,
    };
    let diff = max_abs_diff(
        &render_tiled_cpu(&data, &camera, cfg),
        &render_dense_cpu(&data, &camera),
    );
    assert!(diff < 1e-5, "tiled vs dense max abs diff: {diff}");
}

#[test]
fn tiled_matches_dense_on_checkpoint_subsample() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../data/hf/dylanebert-3dgs/luigi/luigi.ply");
    if !path.exists() {
        eprintln!("skip: HF checkpoint submodule not initialized");
        return;
    }
    let data = load_ply(&path).unwrap().subsample(40);
    let camera = Camera::orbit(
        [0.0, 0.0, 0.0],
        2.2,
        20.0,
        -10.0,
        [0.0, 1.0, 0.0],
        50.0,
        24,
        24,
    );
    let cfg = TiledConfig {
        tile: 8,
        max_tiles_per_gaussian: 9,
        // Every gaussian may overlap one tile; capacity must cover n for
        // exact parity.
        tile_capacity: 384,
    };
    let tiled = render_tiled_cpu(&data, &camera, cfg);
    let dense = render_dense_cpu(&data, &camera);
    let diff = max_abs_diff(&tiled, &dense);
    // Checkpoint opacities go up to ~1.0 → fringe alpha up to ~0.011; a few
    // fringes may stack on one pixel.
    assert!(diff < 0.05, "tiled vs dense max abs diff: {diff}");
    assert!(dense.iter().any(|&v| v > 0.05), "blank image");
}

#[test]
fn capacity_overflow_degrades_gracefully() {
    // One-slot tiles: only the nearest instance per tile survives. The image
    // must stay finite and bounded — capacity pressure drops the deepest
    // contributions, it does not corrupt.
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(16, 16);
    let cfg = TiledConfig {
        tile: 8,
        max_tiles_per_gaussian: 4,
        tile_capacity: 1,
    };
    let image = render_tiled_cpu(&data, &camera, cfg);
    assert!(image.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
    // The nearest (red) gaussian must still be visible.
    assert!(image.iter().step_by(3).any(|&r| r > 0.05), "red vanished");
}

#[test]
fn gradients_flow_through_tiled_renderer() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(16, 16);
    let cfg = TiledConfig {
        tile: 8,
        max_tiles_per_gaussian: 8,
        tile_capacity: 16,
    };
    let cam = camera.clone();
    let f = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| {
        let (view, proj) = cam.matrix_constants();
        let image = render_tiled(cloud, &view, &proj, 16, 16, &cfg);
        let target = Tensor::constant_f32(&[16, 16, 3], &vec![0.25f32; 16 * 16 * 3]);
        let err = image - target;
        let loss = (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all();
        grad_wrt(&loss, cloud).unwrap()
    });
    let grads = f.call(&cloud_tensors(&data)).unwrap();
    for (name, t) in [
        ("means", &grads.means),
        ("scales", &grads.scales),
        ("colors", &grads.colors),
        ("opacities", &grads.opacities),
    ] {
        let v = t.to_f32();
        assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite grad");
        assert!(v.iter().any(|&x| x != 0.0), "{name}: all-zero grad");
    }
}

#[test]
#[should_panic(expected = "tile must divide width")]
fn rejects_non_dividing_tile() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(20, 16);
    let cfg = TiledConfig {
        tile: 8,
        ..TiledConfig::default()
    };
    render_tiled_cpu(&data, &camera, cfg);
}
