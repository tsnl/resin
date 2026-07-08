//! Deferred shading over the `rasterize` visibility buffer
//! (`docs/hw-nodes.md`): the node resolves per-pixel visibility only; the
//! vertex transform ([`clip_transform`]) and shading ([`shade_lambert`]) are
//! ordinary graph code, differentiable w.r.t. materials.

use resin_core::hw::{raster_channel, RASTER_PIXEL_WIDTH};
use resin_dsl::{rasterize, ElementType, IndexKeyElement, Tensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

use crate::camera::Camera;
use crate::scene::Scene;

/// Per-triangle material parameters of [`render`] (`[T, 3]` each).
#[derive(Tree)]
pub struct Materials<T> {
    pub albedo: T,
    pub emission: T,
}

/// Lift world-space positions `[V, 3]` to homogeneous clip space `[V, 4]`
/// with a row-major world→clip matrix (see [`Camera::view_proj`]):
/// `clip = positions @ A + b` where `A[c][r] = m[r][c]` is the linear part
/// and `b` the translation column (the implicit homogeneous 1).
pub fn clip_transform(world_positions: &Tensor, view_proj: &[f32; 16]) -> Tensor {
    let rows = world_positions.shape()[0];
    let mut linear = [0.0f32; 12];
    for c in 0..3 {
        for r in 0..4 {
            linear[c * 4 + r] = view_proj[r * 4 + c];
        }
    }
    let translation: Vec<f32> = (0..4).map(|r| view_proj[r * 4 + 3]).collect();
    let a = Tensor::constant_f32(&[3, 4], &linear);
    // Explicit broadcast: its adjoint reduces, keeping the transform
    // differentiable w.r.t. positions too.
    let b = Tensor::constant_f32(&[1, 4], &translation).broadcast_to(&[rows, 4], &[0, 1]);
    world_positions.matmul(&a) + b
}

/// Rasterize `vertices` `[V, 3]` / `triangles` `[T, 3]` as seen through
/// `camera` into an `[H, W, 4]` visibility buffer (image size from the
/// camera; channels in [`raster_channel`]).
pub fn visibility(vertices: &Tensor, triangles: &Tensor, camera: &Camera) -> Tensor {
    let clip = clip_transform(vertices, &camera.view_proj());
    rasterize(&clip, triangles, camera.height, camera.width)
}

/// Flat-Lambert deferred shading of a visibility buffer → `[H, W, 3]` linear
/// RGB: gather per-triangle `albedo` / `emission` `[T, 3]` by prim id (both
/// may be graph parameters — gradients flow through the gathers), light with
/// the face normal as `albedo · (ambient + (1 − ambient)·|n·l|) + emission`
/// (|·| because without backface culling normal signs are arbitrary), and
/// mask background pixels by the hit channel. Degenerate triangles yield NaN
/// normals.
pub fn shade_lambert(
    vis: &Tensor,
    vertices: &Tensor,
    triangles: &Tensor,
    albedo: &Tensor,
    emission: &Tensor,
    light_dir: [f32; 3],
    ambient: f32,
) -> Tensor {
    let (h, w) = (vis.shape()[0], vis.shape()[1]);
    let n = h * w;
    let flat = vis.reshape(&[n, RASTER_PIXEL_WIDTH]);
    // Per-pixel scalars stay [N, 1] until the explicit [N, 3] broadcast.
    let column = |t: &Tensor, k: usize| {
        t.index(&[IndexKeyElement::Slice(0..n), IndexKeyElement::Single(k)])
    };
    let prim = column(&flat, raster_channel::PRIM)
        .squeeze(&[1])
        .cast(ElementType::U32);
    let hit = column(&flat, raster_channel::HIT);

    // Face normal from the gathered triangle corners.
    let ids = triangles.gather_rows(&prim);
    let corner = |k: usize| vertices.gather_rows(&column(&ids, k).squeeze(&[1]));
    let v0 = corner(0);
    let e1 = corner(1) - v0.clone();
    let e2 = corner(2) - v0;
    let cross = |i: usize, j: usize| {
        column(&e1, i) * column(&e2, j) - column(&e1, j) * column(&e2, i)
    };
    let (nx, ny, nz) = (cross(1, 2), cross(2, 0), cross(0, 1));
    // Guarded like the path tracer's normalize: a degenerate triangle (or the
    // background gathering triangle 0) must not divide by zero — NaN survives
    // the hit-mask multiply (NaN·0 = NaN) and would poison the image.
    let eps = Tensor::full(&[n, 1], 1e-20, ElementType::F32);
    let norm = (nx.clone() * nx.clone() + ny.clone() * ny.clone() + nz.clone() * nz.clone())
        .maximum(&eps)
        .sqrt();

    let inv_len = 1.0 / (light_dir.iter().map(|d| d * d).sum::<f32>()).sqrt();
    let scalar = |v: f32| Tensor::full(&[], v, ElementType::F32);
    let light = |axis: usize| scalar(light_dir[axis] * inv_len);
    let n_dot_l = (nx * light(0) + ny * light(1) + nz * light(2)) / norm;
    let shade = scalar(ambient) + scalar(1.0 - ambient) * n_dot_l.abs();

    // Explicit [N, 1] → [N, 3] broadcasts: the reverse pass reduces only
    // through `Broadcast` nodes, and these sit on the material grad path.
    let per_pixel = |t: &Tensor| t.broadcast_to(&[n, 3], &[0, 1]);
    let rgb = (albedo.gather_rows(&prim) * per_pixel(&shade)
        + emission.gather_rows(&prim))
        * per_pixel(&hit);
    rgb.reshape(&[h, w, 3])
}

/// Render `scene` through `camera` with flat Lambert shading on `jit`,
/// returning linear RGB `[H·W·3]`. Albedo and emission enter the graph as
/// [`Materials`] *parameters*, so the traced program stays differentiable
/// w.r.t. materials.
pub fn render<J: Jit>(
    jit: J,
    scene: &Scene,
    camera: &Camera,
    light_dir: [f32; 3],
    ambient: f32,
) -> Vec<f32> {
    let triangle_count = scene.triangle_count();
    let vertex_data = scene.vertices.clone();
    let triangle_data = scene.triangles.clone();
    let camera = *camera;
    let f = jit.jit(move |materials: &Materials<Tensor>| {
        let vertices = Tensor::constant_f32(&[vertex_data.len() / 3, 3], &vertex_data);
        let triangles = Tensor::constant_u32(&[triangle_data.len() / 3, 3], &triangle_data);
        let vis = visibility(&vertices, &triangles, &camera);
        shade_lambert(
            &vis,
            &vertices,
            &triangles,
            &materials.albedo,
            &materials.emission,
            light_dir,
            ambient,
        )
    });
    let image = f
        .call(&Materials {
            albedo: J::Tensor::from_f32(&[triangle_count, 3], &scene.albedo),
            emission: J::Tensor::from_f32(&[triangle_count, 3], &scene.emission),
        })
        .expect("raster render");
    image.to_f32()
}
