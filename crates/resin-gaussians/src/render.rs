//! Untiled forward renderer: depth sort → gather → dense per-pixel alpha →
//! transmittance scan → weighted reduction. Pure graph composition; the
//! whole image formation is differentiable via `grad_wrt` with no
//! renderer-specific adjoints.

use resin_dsl::sort::argsort_f32;
use resin_dsl::{cumprod_exclusive, IndexKeyElement, Tensor};
use resin_macros::Tree;

use crate::camera::Camera;
use crate::cloud::GaussianCloud;
use crate::linalg::{col, sc};
use crate::preprocess::preprocess_view;

/// A renderable scene as one JIT parameter tree: cloud attributes plus
/// `[4, 4]` view/proj matrices. Compile a renderer over this once, then move
/// the camera (or the gaussians) every call with no recompilation.
#[derive(Debug, Clone, Tree)]
pub struct RenderScene<T> {
    pub cloud: GaussianCloud<T>,
    pub view: T,
    pub proj: T,
}

/// Render `cloud` through a host [`Camera`] (matrices baked as constants).
pub fn render(cloud: &GaussianCloud<Tensor>, camera: &Camera) -> Tensor {
    let (view, proj) = camera.matrix_constants();
    render_view(cloud, &view, &proj, camera.width, camera.height)
}

/// Render with the camera as graph inputs (`[4, 4]` view/proj tensors).
/// Compile once with the camera as a JIT parameter, then move it every call.
/// Output is an `[H, W, 3]` RGB image (black background, front-to-back alpha
/// compositing).
pub fn render_view(
    cloud: &GaussianCloud<Tensor>,
    view: &Tensor,
    proj: &Tensor,
    width: usize,
    height: usize,
) -> Tensor {
    let pre = preprocess_view(cloud, view, proj, width, height);
    let n = pre.depth.shape()[0];
    let (h, w) = (height, width);

    // Global front-to-back order. Culled gaussians keep zero alpha, so their
    // position in the order is irrelevant.
    let order = argsort_f32(&pre.depth);
    let mean_px = pre.mean_px.gather_rows(&order);
    let mean_py = pre.mean_py.gather_rows(&order);
    let conic0 = pre.conic[0].gather_rows(&order);
    let conic1 = pre.conic[1].gather_rows(&order);
    let conic2 = pre.conic[2].gather_rows(&order);
    let opacity = pre.opacity.gather_rows(&order);
    let valid = pre.valid.gather_rows(&order);
    let color = pre.color.gather_rows(&order); // [N, 3]

    // Pixel-center grids as [H, W] constants; per-gaussian attributes and the
    // grid broadcast to [N, H, W] as pure pitch-0 views.
    let (px_grid, py_grid) = pixel_center_grids(w, h);
    let per_g = |t: &Tensor| t.broadcast_to(&[n, h, w], &[0]);
    let per_px = |t: &Tensor| t.broadcast_to(&[n, h, w], &[1, 2]);

    let dx = per_px(&px_grid) - per_g(&mean_px);
    let dy = per_px(&py_grid) - per_g(&mean_py);
    let power = (dx.clone() * dx.clone() * per_g(&conic0)
        + dy.clone() * dy.clone() * per_g(&conic2))
        * sc(-0.5)
        - dx * dy * per_g(&conic1);

    // Reference semantics as masks: skip power > 0, clamp alpha to 0.99,
    // skip alpha < 1/255, and zero out culled gaussians.
    let a0 = (per_g(&opacity) * power.exp()).minimum(&sc(0.99));
    let contributes = power.cmp_le(&sc(0.0)) * a0.cmp_ge(&sc(1.0 / 255.0)) * per_g(&valid);
    let alpha = a0 * contributes;

    // Front-to-back transmittance: T_i = Π_{j<i} (1 − α_j) — a scan over the
    // gaussian axis.
    let transmittance = cumprod_exclusive(&(sc(1.0) - alpha.clone()), 0);
    let weight = alpha * transmittance; // [N, H, W]

    // image[y, x, c] = Σ_i weight[i, y, x] · color[i, c]
    let mut image: Option<Tensor> = None;
    for c in 0..3 {
        let channel = (weight.clone() * per_g(&col(&color, c)))
            .sum_axes(&[0])
            .squeeze(&[0]); // [H, W]
        let placed = channel.broadcast_to(&[h, w, 1], &[0, 1]).scatter_index(
            &[h, w, 3],
            &[
                IndexKeyElement::Slice(0..h),
                IndexKeyElement::Slice(0..w),
                IndexKeyElement::Single(c),
            ],
        );
        image = Some(match image {
            None => placed,
            Some(acc) => acc + placed,
        });
    }
    image.expect("three channels")
}

pub(crate) fn pixel_center_grids(w: usize, h: usize) -> (Tensor, Tensor) {
    let mut xs = Vec::with_capacity(h * w);
    let mut ys = Vec::with_capacity(h * w);
    for y in 0..h {
        for x in 0..w {
            xs.push(x as f32 + 0.5);
            ys.push(y as f32 + 0.5);
        }
    }
    (
        Tensor::constant_f32(&[h, w], &xs),
        Tensor::constant_f32(&[h, w], &ys),
    )
}
