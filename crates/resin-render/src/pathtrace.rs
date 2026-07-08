//! Monte-Carlo path tracer composed from graph ops around `trace_rays`
//! (`docs/hw-nodes.md`).
//!
//! The node resolves visibility only; everything else is differentiable
//! graph code: per-bounce shading is a gather over per-triangle albedo /
//! emission tables masked by the hit flag, so gradients flow to materials
//! (never through visibility, whose adjoint is zero). The bounce loop is
//! unrolled in Rust — the host is the control language — and randomness
//! enters as *input* tensors: cosine-hemisphere local directions are sampled
//! on the host (the DSL has no trig) and rotated into the surface frame in
//! the graph. [`render_reference`] is a scalar tracer with deliberately
//! identical semantics consuming the identical random stream, for golden
//! tests.

use resin_core::hw::trace_channel;
use resin_dsl::{trace_rays, ElementType, IndexKeyElement, Tensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

use crate::camera::Camera;
use crate::scene::Scene;

/// Self-intersection guard for every traced ray.
const T_MIN: f32 = 1e-4;
/// Effectively-infinite far plane.
const T_MAX: f32 = 1e4;
/// Offset of bounce origins along the surface normal.
const NORMAL_OFFSET: f32 = 1e-3;
/// Guard for normalizing near-degenerate cross products.
const LENGTH_EPS: f32 = 1e-20;

#[derive(Debug, Clone, Copy)]
pub struct PathTracerConfig {
    /// Path length: number of `trace_rays` invocations per sample.
    pub bounces: usize,
    /// Seed of the (deterministic) per-sample random streams.
    pub seed: u64,
}

impl Default for PathTracerConfig {
    fn default() -> Self {
        Self {
            bounces: 3,
            seed: 0,
        }
    }
}

//
// Graph-side [N, 1] column vectors
//

/// A `[N, 3]` quantity as three `[N, 1]` columns. Everything stays rank-2:
/// mixing `[N, 1]` with `[N]` under trailing-align broadcasting would
/// produce `[N, N]`.
#[derive(Clone)]
struct Cols {
    x: Tensor,
    y: Tensor,
    z: Tensor,
}

impl Cols {
    fn split(rows3: &Tensor) -> Self {
        let n = rows3.shape()[0];
        let col = |c: usize| {
            rows3.index(&[IndexKeyElement::Slice(0..n), IndexKeyElement::Single(c)])
        };
        Self {
            x: col(0),
            y: col(1),
            z: col(2),
        }
    }

    /// Reassemble `[N, 3]` (for `trace_rays` inputs and the output image).
    fn join(&self, n: usize) -> Tensor {
        let embed = |t: &Tensor, c: usize| {
            t.scatter_index(
                &[n, 3],
                &[IndexKeyElement::Slice(0..n), IndexKeyElement::Single(c)],
            )
        };
        embed(&self.x, 0) + embed(&self.y, 1) + embed(&self.z, 2)
    }

    fn add(&self, rhs: &Cols) -> Cols {
        Cols {
            x: self.x.clone() + rhs.x.clone(),
            y: self.y.clone() + rhs.y.clone(),
            z: self.z.clone() + rhs.z.clone(),
        }
    }

    fn sub(&self, rhs: &Cols) -> Cols {
        Cols {
            x: self.x.clone() - rhs.x.clone(),
            y: self.y.clone() - rhs.y.clone(),
            z: self.z.clone() - rhs.z.clone(),
        }
    }

    /// Scale by a `[N, 1]` factor.
    fn scale(&self, factor: &Tensor) -> Cols {
        Cols {
            x: self.x.clone() * factor.clone(),
            y: self.y.clone() * factor.clone(),
            z: self.z.clone() * factor.clone(),
        }
    }

    fn dot(&self, rhs: &Cols) -> Tensor {
        self.x.clone() * rhs.x.clone()
            + self.y.clone() * rhs.y.clone()
            + self.z.clone() * rhs.z.clone()
    }

    fn cross(&self, rhs: &Cols) -> Cols {
        Cols {
            x: self.y.clone() * rhs.z.clone() - self.z.clone() * rhs.y.clone(),
            y: self.z.clone() * rhs.x.clone() - self.x.clone() * rhs.z.clone(),
            z: self.x.clone() * rhs.y.clone() - self.y.clone() * rhs.x.clone(),
        }
    }

    /// Normalize with a degeneracy guard (mirrored in the reference).
    fn normalize(&self) -> Cols {
        let eps = self.x.full_like_scalar(LENGTH_EPS);
        let inv = eps.ones() / self.dot(self).maximum(&eps).sqrt();
        self.scale(&inv)
    }

    /// Per-component `mask ? on_true : self` with a `[N, 1]` 0/1 mask.
    fn select_where(&self, mask: &Tensor, on_true: &Cols) -> Cols {
        Cols {
            x: mask.select(&on_true.x, &self.x),
            y: mask.select(&on_true.y, &self.y),
            z: mask.select(&on_true.z, &self.z),
        }
    }

    fn neg(&self) -> Cols {
        Cols {
            x: -self.x.clone(),
            y: -self.y.clone(),
            z: -self.z.clone(),
        }
    }
}

/// Shape-matched constant helpers for `[N, 1]` columns.
trait ColConst {
    fn full_like_scalar(&self, value: f32) -> Tensor;
    fn ones(&self) -> Tensor;
}

impl ColConst for Tensor {
    fn full_like_scalar(&self, value: f32) -> Tensor {
        Tensor::full(self.shape(), value, ElementType::F32)
    }

    fn ones(&self) -> Tensor {
        self.full_like_scalar(1.0)
    }
}

/// Trace `locals.len()` bounces and return the `[N, 3]` radiance estimate.
///
/// All arguments are graph tensors; pass `albedo` / `emission` as JIT
/// *parameters* to differentiate the image w.r.t. materials (visibility is
/// piecewise constant, so gradients flow through the material gathers only).
/// `locals` holds one `[N, 3]` cosine-hemisphere local direction per bounce
/// (see [`render`] for host-side sampling); it must be non-empty.
pub fn radiance_graph(
    vertices: &Tensor,
    triangles: &Tensor,
    albedo: &Tensor,
    emission: &Tensor,
    origins: &Tensor,
    directions: &Tensor,
    locals: &[Tensor],
) -> Tensor {
    assert!(!locals.is_empty(), "radiance_graph needs at least one bounce");
    let n = origins.shape()[0];
    let slice_n = IndexKeyElement::Slice(0..n);

    let t_min = Tensor::full(&[n], T_MIN, ElementType::F32);
    let t_max = Tensor::full(&[n], T_MAX, ElementType::F32);
    let mut o = origins.clone();
    let mut d = directions.clone();
    let mut radiance = Tensor::zeros(&[n, 3], ElementType::F32);
    let mut throughput = Tensor::full(&[n, 3], 1.0, ElementType::F32);

    for (bounce, local) in locals.iter().enumerate() {
        let hits = trace_rays(&o, &d, &t_min, &t_max, vertices, triangles);
        let col =
            |c: usize| hits.index(&[slice_n.clone(), IndexKeyElement::Single(c)]);
        let hit = col(trace_channel::HIT); // [N, 1] 0/1
        let t = col(trace_channel::T);
        // Miss keys are 0 — in-bounds for the gathers, masked by `hit`.
        let prim = col(trace_channel::PRIM).squeeze(&[1]).cast(ElementType::U32);

        // Lambertian shading under cosine-weighted sampling: the estimator
        // weight per bounce is exactly the albedo. Missed rays gather row 0
        // but `hit` zeroes both the emitted light and all future throughput.
        // The broadcast to [N, 3] is explicit — implicit trailing-align
        // broadcasting carries no adjoint, and this product is on the
        // gradient path to the material tables.
        let hit3 = hit.broadcast_to(&[n, 3], &[0, 1]);
        radiance = radiance + throughput.clone() * (emission.gather_rows(&prim) * hit3.clone());
        throughput = throughput * (albedo.gather_rows(&prim) * hit3);

        if bounce + 1 == locals.len() {
            break;
        }

        // Geometric normal from the hit triangle's corners, flipped against
        // the incoming direction.
        let tri = triangles.gather_rows(&prim); // [N, 3] U32
        let corner = |k: usize| {
            let ids = tri
                .index(&[slice_n.clone(), IndexKeyElement::Single(k)])
                .squeeze(&[1]);
            Cols::split(&vertices.gather_rows(&ids))
        };
        let p0 = corner(0);
        let edge1 = corner(1).sub(&p0);
        let edge2 = corner(2).sub(&p0);
        let d_cols = Cols::split(&d);
        let normal = edge1.cross(&edge2).normalize();
        let backface = normal.dot(&d_cols).cmp_gt(&t.full_like_scalar(0.0));
        let normal = normal.select_where(&backface, &normal.neg());

        // Next ray: offset hit point, local cosine direction rotated by a
        // branchless orthonormal basis around the normal (helper axis picked
        // away from the normal).
        let point = Cols::split(&o).add(&d_cols.scale(&t));
        let new_o = point.add(&normal.scale(&t.full_like_scalar(NORMAL_OFFSET)));

        let use_z = normal.z.abs().cmp_lt(&t.full_like_scalar(0.9));
        let zero = t.full_like_scalar(0.0);
        let helper = Cols {
            x: use_z.select(&zero, &t.full_like_scalar(1.0)),
            y: zero.clone(),
            z: use_z.select(&t.full_like_scalar(1.0), &zero),
        };
        let tangent = helper.cross(&normal).normalize();
        let bitangent = normal.cross(&tangent);
        let local = Cols::split(local);
        let new_d = tangent
            .scale(&local.x)
            .add(&bitangent.scale(&local.y))
            .add(&normal.scale(&local.z));

        o = new_o.join(n);
        d = new_d.join(n);
    }

    radiance
}

//
// Host-side sampling (shared by `render` and `render_reference`)
//

/// xorshift64* — tiny, deterministic, good enough for Monte-Carlo tests.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Avoid the all-zeros fixed point.
        Self(seed | 1)
    }

    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Cosine-hemisphere direction in the local (tangent, bitangent, normal)
/// frame from two uniforms.
fn cosine_local(u1: f32, u2: f32) -> [f32; 3] {
    let phi = std::f32::consts::TAU * u1;
    let r = u2.sqrt();
    [r * phi.cos(), r * phi.sin(), (1.0 - u2).max(0.0).sqrt()]
}

/// One bounce worth of local directions for `n` rays (`[n * 3]`, row-major).
/// Both the jitted renderer and the scalar reference consume the stream in
/// this exact order.
fn sample_locals(rng: &mut Rng, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 3);
    for _ in 0..n {
        let u1 = rng.next_f32();
        let u2 = rng.next_f32();
        out.extend(cosine_local(u1, u2));
    }
    out
}

//
// Backend renderer
//

#[derive(Tree)]
struct PathTraceIn<T> {
    origins: T,
    directions: T,
    albedo: T,
    emission: T,
    /// One `[N, 3]` tensor per bounce. Random data must be a *parameter*:
    /// the compile cache is keyed on parameter shapes only, so per-sample
    /// values baked as constants would be stale from the second call on.
    locals: Vec<T>,
}

/// Path-trace `scene` from `camera` on any [`Jit`] backend, averaging `spp`
/// samples. Returns linear RGB `[H * W * 3]`, row-major.
pub fn render<J: Jit>(
    jit: J,
    scene: &Scene,
    camera: &Camera,
    spp: usize,
    config: &PathTracerConfig,
) -> Vec<f32> {
    let n = camera.width * camera.height;
    if spp == 0 || config.bounces == 0 {
        return vec![0.0; n * 3];
    }

    let (origins, directions) = camera.primary_rays();
    let triangle_count = scene.triangle_count();
    let vertices = scene.vertices.clone();
    let triangles = scene.triangles.clone();
    let f = jit.jit(move |input: &PathTraceIn<Tensor>| {
        // Geometry is fixed per jitted function; materials and randoms are
        // parameters (materials so the same graph differentiates).
        let v = Tensor::constant_f32(&[vertices.len() / 3, 3], &vertices);
        let t = Tensor::constant_u32(&[triangles.len() / 3, 3], &triangles);
        radiance_graph(
            &v,
            &t,
            &input.albedo,
            &input.emission,
            &input.origins,
            &input.directions,
            &input.locals,
        )
    });

    let mut rng = Rng::new(config.seed);
    let mut acc = vec![0.0f32; n * 3];
    for _ in 0..spp {
        let locals: Vec<J::Tensor> = (0..config.bounces)
            .map(|_| J::Tensor::from_f32(&[n, 3], &sample_locals(&mut rng, n)))
            .collect();
        // Same leaf shapes every sample → one compile, cached invokes.
        let out = f
            .call(&PathTraceIn {
                origins: J::Tensor::from_f32(&[n, 3], &origins),
                directions: J::Tensor::from_f32(&[n, 3], &directions),
                albedo: J::Tensor::from_f32(&[triangle_count, 3], &scene.albedo),
                emission: J::Tensor::from_f32(&[triangle_count, 3], &scene.emission),
                locals,
            })
            .expect("path trace sample");
        for (slot, value) in acc.iter_mut().zip(out.to_f32()) {
            *slot += value;
        }
    }
    let scale = 1.0 / spp as f32;
    for slot in &mut acc {
        *slot *= scale;
    }
    acc
}

//
// Scalar reference
//

type Vec3 = [f32; 3];

fn sub3(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross3(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot3(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Same guarded normalization as the graph (`x * 1/sqrt(max(x·x, eps))`).
fn normalize3(v: Vec3) -> Vec3 {
    let inv = 1.0 / dot3(v, v).max(LENGTH_EPS).sqrt();
    [v[0] * inv, v[1] * inv, v[2] * inv]
}

/// Möller–Trumbore exactly as in the CPU `trace_rays` executor.
fn intersect_triangle(o: Vec3, d: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, f32, f32)> {
    let e1 = sub3(v1, v0);
    let e2 = sub3(v2, v0);
    let p = cross3(d, e2);
    let det = dot3(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = sub3(o, v0);
    let u = dot3(s, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross3(s, e1);
    let v = dot3(d, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot3(e2, q) * inv_det;
    Some((t, u, v))
}

/// Closest hit `(t, prim)` with the CPU executor's exact accept rule
/// (strictly closer wins; first triangle wins ties).
fn closest_hit(scene: &Scene, o: Vec3, d: Vec3) -> Option<(f32, usize)> {
    let clamp = scene.vertex_count().saturating_sub(1);
    let vertex = |i: usize| -> Vec3 {
        let i = i.min(clamp) * 3;
        [
            scene.vertices[i],
            scene.vertices[i + 1],
            scene.vertices[i + 2],
        ]
    };
    let mut best: Option<(f32, usize)> = None;
    for (prim, tri) in scene.triangles.chunks_exact(3).enumerate() {
        if let Some((t, _, _)) = intersect_triangle(
            o,
            d,
            vertex(tri[0] as usize),
            vertex(tri[1] as usize),
            vertex(tri[2] as usize),
        ) && (T_MIN..=T_MAX).contains(&t)
            && best.map(|(bt, _)| t < bt).unwrap_or(true)
        {
            best = Some((t, prim));
        }
    }
    best
}

/// Scalar path tracer with semantics identical to [`radiance_graph`] and the
/// same random stream as [`render`] — the golden oracle. Returns linear RGB
/// `[H * W * 3]`.
pub fn render_reference(
    scene: &Scene,
    camera: &Camera,
    spp: usize,
    config: &PathTracerConfig,
) -> Vec<f32> {
    let n = camera.width * camera.height;
    if spp == 0 || config.bounces == 0 {
        return vec![0.0; n * 3];
    }

    let (flat_origins, flat_directions) = camera.primary_rays();
    let ray3 = |flat: &[f32], i: usize| -> Vec3 { [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]] };

    let mut rng = Rng::new(config.seed);
    let mut acc = vec![0.0f32; n * 3];
    for _ in 0..spp {
        // Locals are drawn per bounce for all rays up front, matching the
        // tensor order in `render` (dead rays still consume their draws).
        let locals: Vec<Vec<f32>> = (0..config.bounces)
            .map(|_| sample_locals(&mut rng, n))
            .collect();

        let mut origins: Vec<Vec3> = (0..n).map(|i| ray3(&flat_origins, i)).collect();
        let mut directions: Vec<Vec3> = (0..n).map(|i| ray3(&flat_directions, i)).collect();
        let mut throughput = vec![[1.0f32; 3]; n];
        let mut alive = vec![true; n];

        for (bounce, bounce_locals) in locals.iter().enumerate() {
            for ray in 0..n {
                // A ray that missed has zero throughput forever; its graph
                // twin keeps tracing but contributes exactly nothing.
                if !alive[ray] {
                    continue;
                }
                let o = origins[ray];
                let d = directions[ray];
                let Some((t, prim)) = closest_hit(scene, o, d) else {
                    alive[ray] = false;
                    continue;
                };

                for c in 0..3 {
                    acc[ray * 3 + c] += throughput[ray][c] * scene.emission[prim * 3 + c];
                    throughput[ray][c] *= scene.albedo[prim * 3 + c];
                }

                if bounce + 1 == config.bounces {
                    continue;
                }

                let vertex = |slot: usize| -> Vec3 {
                    let i = scene.triangles[prim * 3 + slot] as usize * 3;
                    [
                        scene.vertices[i],
                        scene.vertices[i + 1],
                        scene.vertices[i + 2],
                    ]
                };
                let p0 = vertex(0);
                let mut normal = normalize3(cross3(sub3(vertex(1), p0), sub3(vertex(2), p0)));
                if dot3(normal, d) > 0.0 {
                    normal = [-normal[0], -normal[1], -normal[2]];
                }

                let helper: Vec3 = if normal[2].abs() < 0.9 {
                    [0.0, 0.0, 1.0]
                } else {
                    [1.0, 0.0, 0.0]
                };
                let tangent = normalize3(cross3(helper, normal));
                let bitangent = cross3(normal, tangent);
                let local = [
                    bounce_locals[ray * 3],
                    bounce_locals[ray * 3 + 1],
                    bounce_locals[ray * 3 + 2],
                ];
                for c in 0..3 {
                    origins[ray][c] = o[c] + d[c] * t + normal[c] * NORMAL_OFFSET;
                    directions[ray][c] =
                        tangent[c] * local[0] + bitangent[c] * local[1] + normal[c] * local[2];
                }
            }
        }
    }

    let scale = 1.0 / spp as f32;
    for slot in &mut acc {
        *slot *= scale;
    }
    acc
}
