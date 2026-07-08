//! Chunked forward renderer: bounded memory for real checkpoints.
//!
//! The dense renderer materializes `O(N·H·W)`, which is impossible for
//! checkpoint-scale clouds (100k gaussians × 800² ≈ hundreds of GB). This
//! module splits the work into two compiled programs and a *host-side fold*
//! — host Rust is the control language, so the loop lives there:
//!
//! 1. **Prep** (whole cloud, `[N]` tensors only): preprocess + global depth
//!    argsort + gather → sorted screen-space attributes.
//! 2. **Fold step** (per chunk of `K` sorted gaussians): alpha `[K, H, W]`,
//!    local transmittance scan, and composition into a running
//!    `(image [H,W,3], transmittance [H,W])` state carried between calls.
//!
//! Peak intermediate memory is `O(K·H·W)` — front-to-back compositing is
//! associative, so folding chunks in depth order is exact, not an
//! approximation. Still zero custom kernels and zero new autodiff rules;
//! the fold is forward-only (training uses the dense path at training
//! resolutions, or one chunk when it fits).

use resin_dsl::sort::argsort_f32;
use resin_dsl::{cumprod, shift_axis, IndexKeyElement, Tensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

use crate::camera::Camera;
use crate::cloud::GaussianCloud;
use crate::linalg::{col, sc};
use crate::preprocess::preprocess_view;
use crate::render::RenderScene;

/// Sorted screen-space attributes produced by the prep program.
#[derive(Debug, Clone, Tree)]
pub struct ScreenAttrs<T> {
    pub mean_px: T,
    pub mean_py: T,
    pub conic0: T,
    pub conic1: T,
    pub conic2: T,
    /// Opacity pre-multiplied by the validity mask (culled → 0).
    pub opacity: T,
    /// `[N, 3]` RGB, sorted like the rest.
    pub color: T,
}

/// Inputs of the per-chunk fold step.
#[derive(Debug, Clone, Tree)]
pub struct ChunkState<T> {
    pub attrs: ScreenAttrs<T>,
    /// Running composited image `[H, W, 3]`.
    pub image: T,
    /// Running transmittance `[H, W]`.
    pub transmittance: T,
}

/// Trace the prep graph: preprocess + depth sort + gather.
fn prep_graph(scene: &RenderScene<Tensor>, width: usize, height: usize) -> ScreenAttrs<Tensor> {
    let pre = preprocess_view(&scene.cloud, &scene.view, &scene.proj, width, height);
    let order = argsort_f32(&pre.depth);
    ScreenAttrs {
        mean_px: pre.mean_px.gather_rows(&order),
        mean_py: pre.mean_py.gather_rows(&order),
        conic0: pre.conic[0].gather_rows(&order),
        conic1: pre.conic[1].gather_rows(&order),
        conic2: pre.conic[2].gather_rows(&order),
        opacity: (pre.opacity * pre.valid).gather_rows(&order),
        color: pre.color.gather_rows(&order),
    }
}

/// Trace one fold step: blend `K` sorted gaussians into the running state.
fn fold_graph(state: &ChunkState<Tensor>, width: usize, height: usize) -> (Tensor, Tensor) {
    let a = &state.attrs;
    let k = a.mean_px.shape()[0];
    let (h, w) = (height, width);

    let (px_grid, py_grid) = crate::render::pixel_center_grids(w, h);
    let per_g = |t: &Tensor| t.broadcast_to(&[k, h, w], &[0]);
    let per_px = |t: &Tensor| t.broadcast_to(&[k, h, w], &[1, 2]);

    let dx = per_px(&px_grid) - per_g(&a.mean_px);
    let dy = per_px(&py_grid) - per_g(&a.mean_py);
    let power = (dx.clone() * dx.clone() * per_g(&a.conic0)
        + dy.clone() * dy.clone() * per_g(&a.conic2))
        * sc(-0.5)
        - dx * dy * per_g(&a.conic1);
    let a0 = (per_g(&a.opacity) * power.exp()).minimum(&sc(0.99));
    let contributes = power.cmp_le(&sc(0.0)) * a0.cmp_ge(&sc(1.0 / 255.0));
    let alpha = a0 * contributes;

    // Local transmittance within the chunk; one inclusive scan serves both
    // the per-gaussian prefix (shift) and the chunk total (last row).
    let one_minus = sc(1.0) - alpha.clone();
    let t_incl = cumprod(&one_minus, 0); // [K, H, W]
    let one = resin_dsl::Tensor::full(&[], 1.0, resin_dsl::ElementType::F32);
    let t_local = shift_axis(&t_incl, 0, 1, &one); // exclusive prefix
    let full_key = |single: usize| {
        [
            IndexKeyElement::Single(single),
            IndexKeyElement::Slice(0..h),
            IndexKeyElement::Slice(0..w),
        ]
    };
    let t_chunk = t_incl.index(&full_key(k - 1)).squeeze(&[0]); // [H, W]

    let weight = alpha * t_local; // [K, H, W]
    let t_in = &state.transmittance; // [H, W]
    let mut image = state.image.clone();
    for c in 0..3 {
        let channel = (weight.clone() * per_g(&col(&a.color, c)))
            .sum_axes(&[0])
            .squeeze(&[0])
            * t_in.clone(); // [H, W]
        let placed = channel.broadcast_to(&[h, w, 1], &[0, 1]).scatter_index(
            &[h, w, 3],
            &[
                IndexKeyElement::Slice(0..h),
                IndexKeyElement::Slice(0..w),
                IndexKeyElement::Single(c),
            ],
        );
        image = image + placed;
    }
    let t_out = t_in.clone() * t_chunk;
    (image, t_out)
}

/// Build a chunked renderer for clouds of `n` gaussians at a fixed image
/// size. Both programs compile on first call and are reused for every
/// subsequent frame (the camera is a parameter, not a constant).
///
/// The returned closure renders to a row-major `[H, W, 3]` buffer.
pub fn chunked_renderer<J: Jit>(
    jit: J,
    n: usize,
    width: usize,
    height: usize,
    chunk: usize,
) -> impl Fn(&GaussianCloud<J::Tensor>, &Camera) -> Vec<f32> {
    assert!(chunk >= 2, "chunk must be at least 2");
    let prep = jit
        .clone()
        .jit(move |scene: &RenderScene<Tensor>| prep_graph(scene, width, height));
    let fold = jit.jit(move |state: &ChunkState<Tensor>| {
        let (image, transmittance) = fold_graph(state, width, height);
        vec![image, transmittance]
    });

    move |cloud: &GaussianCloud<J::Tensor>, camera: &Camera| -> Vec<f32> {
        assert_eq!(camera.width, width, "camera/image width mismatch");
        assert_eq!(camera.height, height, "camera/image height mismatch");
        let scene = RenderScene {
            cloud: cloud.clone(),
            view: J::Tensor::from_f32(&[4, 4], &camera.view_flat()),
            proj: J::Tensor::from_f32(&[4, 4], &camera.proj_flat()),
        };
        let attrs = prep.call(&scene).expect("chunked prep");

        // Host-side densify of sorted attributes, padded to a whole number
        // of chunks (padding rows have zero opacity → no contribution).
        let padded = n.div_ceil(chunk) * chunk;
        let pad1 = |t: &J::Tensor| -> Vec<f32> {
            let mut v = t.to_f32();
            v.resize(padded, 0.0);
            v
        };
        let px = pad1(&attrs.mean_px);
        let py = pad1(&attrs.mean_py);
        let c0 = pad1(&attrs.conic0);
        let c1 = pad1(&attrs.conic1);
        let c2 = pad1(&attrs.conic2);
        let op = pad1(&attrs.opacity);
        let mut color = attrs.color.to_f32();
        color.resize(padded * 3, 0.0);

        let mut image = vec![0.0f32; height * width * 3];
        let mut transmittance = vec![1.0f32; height * width];
        for c in 0..padded / chunk {
            let range = c * chunk..(c + 1) * chunk;
            let state = ChunkState {
                attrs: ScreenAttrs {
                    mean_px: J::Tensor::from_f32(&[chunk], &px[range.clone()]),
                    mean_py: J::Tensor::from_f32(&[chunk], &py[range.clone()]),
                    conic0: J::Tensor::from_f32(&[chunk], &c0[range.clone()]),
                    conic1: J::Tensor::from_f32(&[chunk], &c1[range.clone()]),
                    conic2: J::Tensor::from_f32(&[chunk], &c2[range.clone()]),
                    opacity: J::Tensor::from_f32(&[chunk], &op[range.clone()]),
                    color: J::Tensor::from_f32(
                        &[chunk, 3],
                        &color[range.start * 3..range.end * 3],
                    ),
                },
                image: J::Tensor::from_f32(&[height, width, 3], &image),
                transmittance: J::Tensor::from_f32(&[height, width], &transmittance),
            };
            let out = fold.call(&state).expect("chunked fold");
            image = out[0].to_f32();
            transmittance = out[1].to_f32();
        }
        image
    }
}
