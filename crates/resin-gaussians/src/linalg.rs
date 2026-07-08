//! Batched quaternion / covariance linear algebra as `[N]` column tensor ops.
//!
//! Everything here is elementwise arithmetic over column *views* of `[N, k]`
//! attribute tensors — columns are strided accessors (no copies), and the
//! elementwise chains fuse in the IR, so "small linear algebra" costs a
//! handful of kernels for the whole cloud, not per-gaussian work.

use resin_dsl::{ElementType, IndexKeyElement, Tensor};

/// Scalar f32 constant (rank 0; broadcasts against any shape).
pub(crate) fn sc(value: f32) -> Tensor {
    Tensor::full(&[], value, ElementType::F32)
}

/// Column `i` of a rank-2 tensor as a `[N]` view.
pub(crate) fn col(t: &Tensor, i: usize) -> Tensor {
    let n = t.shape()[0];
    t.index(&[IndexKeyElement::Slice(0..n), IndexKeyElement::Single(i)])
        .squeeze(&[1])
}

/// Symmetric 3×3 covariance `R · diag(s²) · Rᵀ` from `[N, 3]` scales and
/// `[N, 4]` quaternions (w, x, y, z; normalized internally). Returns the six
/// unique entries `[c00, c01, c02, c11, c12, c22]`, each `[N]`.
pub fn scale_rot_to_cov3d(scales: &Tensor, quats: &Tensor) -> [Tensor; 6] {
    let (w, x, y, z) = normalized_quat_cols(quats);
    let r = rotmat_rows(&w, &x, &y, &z);

    let s0 = col(scales, 0);
    let s1 = col(scales, 1);
    let s2 = col(scales, 2);
    let v = [s0.clone() * s0, s1.clone() * s1, s2.clone() * s2];

    // cov_ab = Σ_k r[a][k] · s_k² · r[b][k]
    let entry = |a: usize, b: usize| -> Tensor {
        r[a][0].clone() * v[0].clone() * r[b][0].clone()
            + r[a][1].clone() * v[1].clone() * r[b][1].clone()
            + r[a][2].clone() * v[2].clone() * r[b][2].clone()
    };
    [
        entry(0, 0),
        entry(0, 1),
        entry(0, 2),
        entry(1, 1),
        entry(1, 2),
        entry(2, 2),
    ]
}

fn normalized_quat_cols(quats: &Tensor) -> (Tensor, Tensor, Tensor, Tensor) {
    let w = col(quats, 0);
    let x = col(quats, 1);
    let y = col(quats, 2);
    let z = col(quats, 3);
    let norm = (w.clone() * w.clone()
        + x.clone() * x.clone()
        + y.clone() * y.clone()
        + z.clone() * z.clone())
    .sqrt();
    (
        w / norm.clone(),
        x / norm.clone(),
        y / norm.clone(),
        z / norm,
    )
}

/// Rotation matrix rows from unit quaternion columns; `r[row][col]`, each `[N]`.
fn rotmat_rows(w: &Tensor, x: &Tensor, y: &Tensor, z: &Tensor) -> [[Tensor; 3]; 3] {
    let two = sc(2.0);
    let one = sc(1.0);
    let m = |a: &Tensor, b: &Tensor| a.clone() * b.clone() * two.clone();
    [
        [
            one.clone() + sc(-2.0) * (y.clone() * y.clone() + z.clone() * z.clone()),
            m(x, y) - m(w, z),
            m(x, z) + m(w, y),
        ],
        [
            m(x, y) + m(w, z),
            one.clone() + sc(-2.0) * (x.clone() * x.clone() + z.clone() * z.clone()),
            m(y, z) - m(w, x),
        ],
        [
            m(x, z) - m(w, y),
            m(y, z) + m(w, x),
            one.clone() + sc(-2.0) * (x.clone() * x.clone() + y.clone() * y.clone()),
        ],
    ]
}

/// `m · [x, y, z, 1]` for a host 4×4 matrix over `[N]` coordinate columns.
pub(crate) fn transform_point(
    m: &[[f32; 4]; 4],
    x: &Tensor,
    y: &Tensor,
    z: &Tensor,
) -> [Tensor; 4] {
    let row = |r: usize| -> Tensor {
        x.clone() * sc(m[r][0]) + y.clone() * sc(m[r][1]) + z.clone() * sc(m[r][2]) + sc(m[r][3])
    };
    [row(0), row(1), row(2), row(3)]
}
