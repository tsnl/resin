//! Gaussian cloud containers: a `Tree`-shaped tensor struct for graphs and
//! backends, plus plain host data for reference rendering and demos.

use resin_macros::Tree;

/// Per-gaussian attributes. `T` is a DSL [`resin_dsl::Tensor`] at trace time
/// or a backend tensor at call time (the struct derives `Tree`, so it can be
/// a JIT parameter tree directly).
///
/// Shapes: `means`/`scales`/`colors` are `[N, 3]`, `quats` is `[N, 4]`
/// (w, x, y, z), `opacities` is `[N]`.
#[derive(Debug, Clone, Tree)]
pub struct GaussianCloud<T> {
    pub means: T,
    pub scales: T,
    pub quats: T,
    pub colors: T,
    pub opacities: T,
}

/// Host-side cloud data (row-major), the input to [`crate::render_reference`]
/// and the source for backend tensors in demos and tests.
#[derive(Debug, Clone, PartialEq)]
pub struct CloudData {
    pub means: Vec<[f32; 3]>,
    pub scales: Vec<[f32; 3]>,
    pub quats: Vec<[f32; 4]>,
    pub colors: Vec<[f32; 3]>,
    pub opacities: Vec<f32>,
}

impl CloudData {
    pub fn count(&self) -> usize {
        self.means.len()
    }

    pub fn means_flat(&self) -> Vec<f32> {
        self.means.iter().flatten().copied().collect()
    }
    pub fn scales_flat(&self) -> Vec<f32> {
        self.scales.iter().flatten().copied().collect()
    }
    pub fn quats_flat(&self) -> Vec<f32> {
        self.quats.iter().flatten().copied().collect()
    }
    pub fn colors_flat(&self) -> Vec<f32> {
        self.colors.iter().flatten().copied().collect()
    }
}

/// Three axis-aligned gaussians on −Z in front of a +Z-origin camera
/// (red near, green middle, blue far) — the fixed golden-test scene.
pub fn gnomen_cloud() -> CloudData {
    CloudData {
        means: vec![[0.0, 0.0, -2.0], [0.4, 0.0, -3.0], [-0.4, 0.0, -4.0]],
        scales: vec![[0.15, 0.15, 0.15], [0.12, 0.12, 0.12], [0.12, 0.12, 0.12]],
        quats: vec![
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        colors: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        opacities: vec![0.9, 0.9, 0.9],
    }
}
