//! Project 3D gaussians to screen space: 2D means (pixels), view depths,
//! conics (inverse 2D covariance), and a validity mask.
//!
//! Culling is a *mask*, not compaction: shapes stay static (`[N]`), and
//! culled gaussians simply contribute zero alpha downstream. Every branch of
//! the reference implementation becomes a `select`/compare mask here.

use resin_dsl::Tensor;

use crate::camera::Camera;
use crate::cloud::GaussianCloud;
use crate::linalg::{col, mat_elem, sc, scale_rot_to_cov3d, transform_point};

/// Screen-space per-gaussian attributes, all `[N]` except `color` (`[N, 3]`).
pub struct Preprocessed {
    /// 2D mean in pixel coordinates.
    pub mean_px: Tensor,
    pub mean_py: Tensor,
    /// View-space depth (positive in front of the camera).
    pub depth: Tensor,
    /// Conic (inverse 2D covariance): `[c0, c1, c2]` for
    /// `power = -½(c0·dx² + c2·dy²) − c1·dx·dy`.
    pub conic: [Tensor; 3],
    pub opacity: Tensor,
    /// Passthrough `[N, 3]` RGB.
    pub color: Tensor,
    /// 0/1 f32 mask: in front of the near plane, invertible 2D covariance,
    /// at least a pixel of extent, and overlapping the screen.
    pub valid: Tensor,
}

/// Constant-camera convenience wrapper over [`preprocess_view`].
pub fn preprocess(cloud: &GaussianCloud<Tensor>, camera: &Camera) -> Preprocessed {
    let (view, proj) = camera.matrix_constants();
    preprocess_view(cloud, &view, &proj, camera.width, camera.height)
}

/// Preprocess with the camera as *graph inputs*: `view` and `proj` are
/// `[4, 4]` tensors (parameters or constants). Passing them as JIT parameters
/// lets a compiled renderer move the camera every call with zero
/// recompilation — the basis for the interactive viewer and multi-view
/// training. Image dimensions stay compile-time (they fix output shapes).
pub fn preprocess_view(
    cloud: &GaussianCloud<Tensor>,
    view: &Tensor,
    proj: &Tensor,
    width: usize,
    height: usize,
) -> Preprocessed {
    assert_eq!(view.shape(), &[4, 4], "view must be a [4,4] tensor");
    assert_eq!(proj.shape(), &[4, 4], "proj must be a [4,4] tensor");
    let width = width as f32;
    let height = height as f32;

    let mx = col(&cloud.means, 0);
    let my = col(&cloud.means, 1);
    let mz = col(&cloud.means, 2);

    let cam = transform_point(view, &mx, &my, &mz);
    let clip = transform_point(proj, &cam[0], &cam[1], &cam[2]);

    let w_ok = clip[3].abs().cmp_gt(&sc(1e-8));
    let w_safe = w_ok.select(&clip[3], &sc(1.0));
    let ndc_x = clip[0].clone() / w_safe.clone();
    let ndc_y = clip[1].clone() / w_safe;
    let mean_px = (ndc_x * sc(0.5) + sc(0.5)) * sc(width);
    // NDC y is up; pixel rows count down.
    let mean_py = (sc(0.5) - ndc_y * sc(0.5)) * sc(height);

    let depth = cam[2].clone();
    let mut valid = depth.cmp_gt(&sc(1e-4));
    // Guarded depth for the divisions below (masked lanes compute garbage
    // that is later multiplied by zero — but must not divide by ~0).
    let tz = valid.select(&depth, &sc(1.0));

    // EWA projection of the 3D covariance to a 2D conic.
    let [c00, c01, c02, c11, c12, c22] = scale_rot_to_cov3d(&cloud.scales, &cloud.quats);
    let fx = mat_elem(proj, 0, 0);
    let fy = mat_elem(proj, 1, 1);
    let tz_inv = sc(1.0) / tz;
    let tz2_inv = tz_inv.clone() * tz_inv.clone();
    let j00 = tz_inv.clone() * fx.clone();
    let j02 = cam[0].clone() * tz2_inv.clone() * (-fx);
    let j11 = tz_inv * fy.clone();
    let j12 = cam[1].clone() * tz2_inv * (-fy);

    let t00 = j00.clone() * c00 + j02.clone() * c02.clone();
    let t01 = j00.clone() * c01 + j02.clone() * c12.clone();
    let t02 = j00.clone() * c02 + j02.clone() * c22;
    let t12 = j11.clone() * c11 + j12.clone() * c12;

    // Screen-space covariance with the classic +0.3 pixel dilation.
    let cov00 = t00 * j00.clone() + t02.clone() * j02.clone() + sc(0.3);
    let cov01 = t01.clone() * j00 + t12.clone() * j02;
    let cov11 = t01 * j11 + t12 * j12 + sc(0.3);

    let det = cov00.clone() * cov11.clone() - cov01.clone() * cov01.clone();
    valid = valid * det.cmp_gt(&sc(1e-12));
    let det_safe = valid.select(&det, &sc(1.0));
    let conic0 = cov11 / det_safe.clone();
    let conic1 = -cov01 / det_safe.clone();
    let conic2 = cov00 / det_safe;

    // Extent + screen-overlap culling (conic-based radius, matching the
    // scalar reference).
    let radius = (conic0.maximum(&conic2).sqrt() * sc(3.0)).ceil();
    valid = valid * radius.cmp_ge(&sc(1.0));
    valid = valid
        * (mean_px.clone() + radius.clone()).cmp_ge(&sc(0.0))
        * (mean_py.clone() + radius.clone()).cmp_ge(&sc(0.0))
        * (mean_px.clone() - radius.clone()).cmp_le(&sc(width))
        * (mean_py.clone() - radius).cmp_le(&sc(height));

    Preprocessed {
        mean_px,
        mean_py,
        depth,
        conic: [conic0, conic1, conic2],
        opacity: cloud.opacities.clone(),
        color: cloud.colors.clone(),
        valid,
    }
}
