//! Batched 3-vectors and quaternions built from tensor ops.
//!
//! The DSL cannot index into a tensor, so "the y column of an [N, 3] tensor"
//! is not directly expressible. Multiplying by a constant one-hot matrix does
//! the same job with an op the compiler has (matmul).
//!
//! A batch of vectors — N per world, across W independent worlds — is
//! therefore three `[N, W]` matrices, one per component. With components
//! held separately, every vector operation below is plain elementwise
//! arithmetic (and reverse-mode differentiable for free), and every world
//! rides along in the trailing axis: the same graph simulates one world or
//! thousands, with no per-world control flow.

use crate::dsl::Tensor;

/// Added to denominators so divisions by a vanishing quantity yield zero
/// instead of NaN (and stay differentiable).
pub const EPS: f32 = 1e-9;

/// Added under square roots: `sqrt` has an infinite derivative at exactly
/// zero, which would poison gradients of zero-length vectors.
const LENGTH_EPS: f32 = 1e-12;

/// Elementwise `min`, from the one nonlinearity the DSL has:
/// `min(a, b) = a - relu(a - b)`.
pub fn min(a: &Tensor, b: &Tensor) -> Tensor {
    a.clone() - (a.clone() - b.clone()).relu()
}

/// Branch-free indicator of `x > 0` for `x >= 0`: exactly 0 at 0, ~1 for
/// `x >> EPS`. Multiplying by a gate replaces `if` in a language without one.
pub fn gate(x: &Tensor) -> Tensor {
    x.clone() / (x.clone() + Tensor::scalar(EPS))
}

/// Smooth sign of `x`: ±1 away from zero, 0 at zero.
pub fn sign(x: &Tensor) -> Tensor {
    x.clone() / (x.abs() + Tensor::scalar(EPS))
}

/// One-hot selection matrix: `[rows.len(), n]` with `out[r, rows[r]] = 1`.
/// Left-multiplying gathers those rows; its transpose scatter-adds them back.
pub fn one_hot_rows(rows: &[usize], n: usize) -> Tensor {
    let mut values = vec![0.0; rows.len() * n];
    for (slot, &row) in rows.iter().enumerate() {
        values[slot * n + row] = 1.0;
    }
    Tensor::constant(&[rows.len(), n], &values)
}

/// Constant `[values.len(), 1]` column.
pub fn host_column(values: &[f32]) -> Tensor {
    Tensor::constant(&[values.len(), 1], values)
}

/// A one-hot `[dim]` vector marking one component slot.
fn component_mask(dim: usize, index: usize) -> Tensor {
    let mut values = vec![0.0; dim];
    values[index] = 1.0;
    Tensor::constant(&[dim], &values)
}

/// Pull component `index` out of packed `[W, N, dim]` rows as an `[N, W]`
/// matrix: mask with a one-hot, sum away the component axis, put worlds
/// last.
fn unpack_component(rows: &Tensor, dim: usize, index: usize) -> Tensor {
    (rows.clone() * component_mask(dim, index))
        .sum_axes(&[2])
        .squeeze(&[2])
        .transpose()
}

/// Put an `[N, W]` component matrix back at `index` of `[W, N, dim]` rows
/// (the other slots are zero, so packed components just add together).
fn pack_component(component: &Tensor, dim: usize, index: usize) -> Tensor {
    let (n, w) = (component.shape()[0], component.shape()[1]);
    component.transpose().broadcast_to(&[w, n, dim], &[0, 1]) * component_mask(dim, index)
}

/// A batch of 3D vectors, N per world across W worlds: three `[N, W]`
/// component matrices.
#[derive(Clone)]
pub struct Vec3s {
    pub x: Tensor,
    pub y: Tensor,
    pub z: Tensor,
}

impl Vec3s {
    /// Split packed `[W, N, 3]` rows into `[N, W]` components.
    pub fn from_rows(rows: &Tensor) -> Self {
        assert_eq!(
            rows.shape()[2..],
            [3],
            "expected [W, N, 3], got {:?}",
            rows.shape()
        );
        Vec3s {
            x: unpack_component(rows, 3, 0),
            y: unpack_component(rows, 3, 1),
            z: unpack_component(rows, 3, 2),
        }
    }

    /// Reassemble components into packed `[W, N, 3]` rows.
    pub fn to_rows(&self) -> Tensor {
        pack_component(&self.x, 3, 0)
            + pack_component(&self.y, 3, 1)
            + pack_component(&self.z, 3, 2)
    }

    /// Constant `[N, 1]` columns from one host vector per row — broadcast
    /// across worlds by elementwise ops.
    pub fn from_host(rows: &[[f32; 3]]) -> Self {
        let column = |axis: usize| host_column(&rows.iter().map(|v| v[axis]).collect::<Vec<_>>());
        Vec3s {
            x: column(0),
            y: column(1),
            z: column(2),
        }
    }

    pub fn zeros(rows: usize, worlds: usize) -> Self {
        Vec3s {
            x: Tensor::zeros(&[rows, worlds]),
            y: Tensor::zeros(&[rows, worlds]),
            z: Tensor::zeros(&[rows, worlds]),
        }
    }

    fn map(&self, f: impl Fn(&Tensor) -> Tensor) -> Self {
        Vec3s {
            x: f(&self.x),
            y: f(&self.y),
            z: f(&self.z),
        }
    }

    fn zip(&self, other: &Self, f: impl Fn(&Tensor, &Tensor) -> Tensor) -> Self {
        Vec3s {
            x: f(&self.x, &other.x),
            y: f(&self.y, &other.y),
            z: f(&self.z, &other.z),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a.clone() + b.clone())
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a.clone() - b.clone())
    }

    pub fn neg(&self) -> Self {
        self.map(|c| -c.clone())
    }

    /// Multiply every component by a broadcast-compatible tensor
    /// (`[N, 1]` or scalar).
    pub fn scale(&self, s: &Tensor) -> Self {
        self.map(|c| c.clone() * s.clone())
    }

    /// Componentwise product (used to apply a diagonal inertia tensor).
    pub fn mul_each(&self, other: &Self) -> Self {
        self.zip(other, |a, b| a.clone() * b.clone())
    }

    pub fn dot(&self, other: &Self) -> Tensor {
        self.x.clone() * other.x.clone()
            + self.y.clone() * other.y.clone()
            + self.z.clone() * other.z.clone()
    }

    pub fn cross(&self, other: &Self) -> Self {
        Vec3s {
            x: self.y.clone() * other.z.clone() - self.z.clone() * other.y.clone(),
            y: self.z.clone() * other.x.clone() - self.x.clone() * other.z.clone(),
            z: self.x.clone() * other.y.clone() - self.y.clone() * other.x.clone(),
        }
    }

    pub fn length(&self) -> Tensor {
        (self.dot(self) + Tensor::scalar(LENGTH_EPS)).sqrt()
    }

    /// Unit direction; smoothly zero for a (near-)zero vector.
    pub fn normalized(&self) -> Self {
        let len = self.length() + Tensor::scalar(EPS);
        self.map(|c| c.clone() / len.clone())
    }

    /// Gather batch rows through a one-hot `[C, N]` matrix.
    pub fn gather(&self, select: &Tensor) -> Self {
        self.map(|c| select.matmul(c))
    }

    /// Scatter-add `[C, 1]` columns back to `[N, 1]` through the transpose
    /// of the same one-hot matrix.
    pub fn scatter_add(&self, select: &Tensor) -> Self {
        self.map(|c| select.transpose().matmul(c))
    }
}

/// A batch of quaternions `w + xi + yj + zk`, N per world across W worlds:
/// four `[N, W]` component matrices. Packed rows are `(w, x, y, z)`;
/// rotations use unit quaternions.
#[derive(Clone)]
pub struct Quats {
    pub w: Tensor,
    pub x: Tensor,
    pub y: Tensor,
    pub z: Tensor,
}

impl Quats {
    /// Split packed `[W, N, 4]` rows into `[N, W]` components.
    pub fn from_rows(rows: &Tensor) -> Self {
        assert_eq!(
            rows.shape()[2..],
            [4],
            "expected [W, N, 4], got {:?}",
            rows.shape()
        );
        Quats {
            w: unpack_component(rows, 4, 0),
            x: unpack_component(rows, 4, 1),
            y: unpack_component(rows, 4, 2),
            z: unpack_component(rows, 4, 3),
        }
    }

    pub fn to_rows(&self) -> Tensor {
        pack_component(&self.w, 4, 0)
            + pack_component(&self.x, 4, 1)
            + pack_component(&self.y, 4, 2)
            + pack_component(&self.z, 4, 3)
    }

    /// The imaginary part `(x, y, z)`.
    pub fn vector(&self) -> Vec3s {
        Vec3s {
            x: self.x.clone(),
            y: self.y.clone(),
            z: self.z.clone(),
        }
    }

    pub fn conjugate(&self) -> Self {
        Quats {
            w: self.w.clone(),
            x: -self.x.clone(),
            y: -self.y.clone(),
            z: -self.z.clone(),
        }
    }

    /// Hamilton product `self ⊗ rhs` (compose rotations: apply `rhs` first).
    pub fn mul(&self, rhs: &Self) -> Self {
        let (w1, x1, y1, z1) = (&self.w, &self.x, &self.y, &self.z);
        let (w2, x2, y2, z2) = (&rhs.w, &rhs.x, &rhs.y, &rhs.z);
        Quats {
            w: w1.clone() * w2.clone()
                - x1.clone() * x2.clone()
                - y1.clone() * y2.clone()
                - z1.clone() * z2.clone(),
            x: w1.clone() * x2.clone() + x1.clone() * w2.clone() + y1.clone() * z2.clone()
                - z1.clone() * y2.clone(),
            y: w1.clone() * y2.clone() - x1.clone() * z2.clone()
                + y1.clone() * w2.clone()
                + z1.clone() * x2.clone(),
            z: w1.clone() * z2.clone() + x1.clone() * y2.clone() - y1.clone() * x2.clone()
                + z1.clone() * w2.clone(),
        }
    }

    /// Rotate vectors by unit quaternions, expanded to avoid two full
    /// products: `t = 2 (q_v × v); v' = v + w t + q_v × t`.
    pub fn rotate(&self, v: &Vec3s) -> Vec3s {
        let qv = self.vector();
        let t = qv.cross(v).scale(&Tensor::scalar(2.0));
        v.add(&t.scale(&self.w)).add(&qv.cross(&t))
    }

    /// Rotate by the inverse (for unit quaternions, the conjugate).
    pub fn rotate_inv(&self, v: &Vec3s) -> Vec3s {
        self.conjugate().rotate(v)
    }

    pub fn normalized(&self) -> Self {
        let len = (self.w.clone() * self.w.clone()
            + self.x.clone() * self.x.clone()
            + self.y.clone() * self.y.clone()
            + self.z.clone() * self.z.clone()
            + Tensor::scalar(LENGTH_EPS))
        .sqrt();
        Quats {
            w: self.w.clone() / len.clone(),
            x: self.x.clone() / len.clone(),
            y: self.y.clone() / len.clone(),
            z: self.z.clone() / len,
        }
    }

    /// Rotate each quaternion by a small rotation vector `dtheta` (radians
    /// times axis): `q' = normalize(q + ½ (0, dθ) ⊗ q)`. First-order, exact
    /// direction — both time integration (`dθ = h ω`) and the solver's
    /// orientation corrections go through here.
    pub fn rotated_by(&self, dtheta: &Vec3s) -> Self {
        let half = dtheta.scale(&Tensor::scalar(0.5));
        let dq = Quats {
            w: half.x.zeros_like(),
            x: half.x.clone(),
            y: half.y.clone(),
            z: half.z.clone(),
        }
        .mul(self);
        Quats {
            w: self.w.clone() + dq.w,
            x: self.x.clone() + dq.x,
            y: self.y.clone() + dq.y,
            z: self.z.clone() + dq.z,
        }
        .normalized()
    }

    pub fn gather(&self, select: &Tensor) -> Self {
        Quats {
            w: select.matmul(&self.w),
            x: select.matmul(&self.x),
            y: select.matmul(&self.y),
            z: select.matmul(&self.z),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tree;
    use crate::jit::{Array, CpuJit, Jit};

    #[derive(Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    fn host_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    /// Rotate `v` by unit quaternion `q = (w, x, y, z)`, the textbook way.
    fn host_rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        let qv = [q[1], q[2], q[3]];
        let t = host_cross(qv, v).map(|c| 2.0 * c);
        let u = host_cross(qv, t);
        [
            v[0] + q[0] * t[0] + u[0],
            v[1] + q[0] * t[1] + u[1],
            v[2] + q[0] * t[2] + u[2],
        ]
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() <= tol, "got {actual:?}, want {expected:?}");
        }
    }

    #[test]
    fn cross_matches_reference_per_world() {
        // Two worlds, one vector each: each world computes its own cross.
        let f = CpuJit.jit(|p: &Pair<Tensor>| {
            Vec3s::from_rows(&p.a)
                .cross(&Vec3s::from_rows(&p.b))
                .to_rows()
        });
        let a = [[1.0, 2.0, 3.0], [-0.5, 0.25, 4.0]];
        let b = [[-2.0, 0.5, 1.0], [3.0, -1.0, 0.5]];
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[2, 1, 3], &a.concat()),
                b: Array::from_f32(&[2, 1, 3], &b.concat()),
            })
            .unwrap();
        let expected: Vec<f32> = (0..2).flat_map(|i| host_cross(a[i], b[i])).collect();
        assert_close(out.data(), &expected, 1e-5);
    }

    #[test]
    fn quaternion_rotation_matches_reference() {
        // One world, two quaternion/vector rows.
        let f = CpuJit.jit(|p: &Pair<Tensor>| {
            Quats::from_rows(&p.a)
                .normalized()
                .rotate(&Vec3s::from_rows(&p.b))
                .to_rows()
        });
        let quats = [[0.9f32, 0.1, -0.3, 0.2], [0.5, 0.5, 0.5, 0.5]];
        let vecs = [[1.0f32, -2.0, 0.5], [0.0, 1.0, 0.0]];
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[1, 2, 4], &quats.concat()),
                b: Array::from_f32(&[1, 2, 3], &vecs.concat()),
            })
            .unwrap();
        let expected: Vec<f32> = (0..2)
            .flat_map(|i| {
                let q = quats[i];
                let len = q.iter().map(|c| c * c).sum::<f32>().sqrt();
                host_rotate(q.map(|c| c / len), vecs[i])
            })
            .collect();
        assert_close(out.data(), &expected, 1e-4);
    }

    #[test]
    fn quaternion_product_composes_rotations() {
        // Rotating by q1 ⊗ q2 must equal rotating by q2, then by q1.
        let f = CpuJit.jit(|p: &Pair<Tensor>| {
            let q1 = Quats::from_rows(&p.a).normalized();
            let q2 = Quats::from_rows(&p.b).normalized();
            let v = Vec3s::from_host(&[[0.3, -1.0, 2.0]]);
            Pair {
                a: q1.mul(&q2).rotate(&v).to_rows(),
                b: q1.rotate(&q2.rotate(&v)).to_rows(),
            }
        });
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[1, 1, 4], &[0.8, -0.2, 0.4, 0.1]),
                b: Array::from_f32(&[1, 1, 4], &[0.3, 0.9, -0.1, 0.2]),
            })
            .unwrap();
        assert_close(out.a.data(), out.b.data(), 1e-4);
    }

    #[test]
    fn gather_and_scatter_add_move_rows() {
        let f = CpuJit.jit(|rows: &Tensor| {
            let v = Vec3s::from_rows(rows);
            let select = one_hot_rows(&[2, 0, 2], 3); // rows 2, 0, 2 of 3
            Pair {
                a: v.gather(&select).to_rows(),
                b: v.gather(&select).scatter_add(&select).to_rows(),
            }
        });
        let rows = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let out = f
            .call(&Array::from_f32(&[1, 3, 3], &rows.concat()))
            .unwrap();
        assert_close(
            out.a.data(),
            &[7.0, 8.0, 9.0, 1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
            1e-6,
        );
        // Row 2 was gathered twice, so it scatters back doubled; row 1 never.
        assert_close(
            out.b.data(),
            &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 14.0, 16.0, 18.0],
            1e-6,
        );
    }

    #[test]
    fn min_and_gate_are_branch_free() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| Pair {
            a: min(&p.a, &p.b),
            b: gate(&p.a),
        });
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[3], &[2.0, 0.0, 5.0]),
                b: Array::from_f32(&[3], &[3.0, 1.0, 4.0]),
            })
            .unwrap();
        assert_close(out.a.data(), &[2.0, 0.0, 4.0], 1e-6);
        assert_close(out.b.data(), &[1.0, 0.0, 1.0], 1e-6);
    }
}
