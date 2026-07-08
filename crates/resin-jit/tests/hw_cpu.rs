//! End-to-end CPU tests for the fixed-function hardware nodes
//! (`trace_rays` / `rasterize`) and their graph composition.

#![cfg(feature = "cpu")]

use resin_core::hw::{raster_channel, trace_channel, RASTER_PIXEL_WIDTH, TRACE_HIT_WIDTH};
use resin_dsl::{rasterize, trace_rays, ElementType, IndexKeyElement, Tensor};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

#[derive(Tree)]
struct RayIn<T> {
    origins: T,
    directions: T,
}

/// Two triangles: prim 0 in the z=1 plane (unit right triangle at the
/// origin), prim 1 a far quad-half at z=3 behind it covering more area.
fn scene() -> (Tensor, Tensor) {
    let vertices = Tensor::constant_f32(
        &[7, 3],
        &[
            // prim 0 (z = 1)
            0.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            0.0, 1.0, 1.0, //
            // prim 1 (z = 3), large
            -2.0, -2.0, 3.0, //
            4.0, -2.0, 3.0, //
            -2.0, 4.0, 3.0, //
            // unused vertex (checks indexing is real)
            9.0, 9.0, 9.0,
        ],
    );
    let triangles = Tensor::constant_u32(&[2, 3], &[0, 1, 2, 3, 4, 5]);
    (vertices, triangles)
}

fn trace(origins: &[f32], directions: &[f32], t_max: f32) -> Vec<f32> {
    let n = origins.len() / 3;
    let jit = CpuJit;
    let t_max_owned = t_max;
    let f = jit.jit(move |rays: &RayIn<Tensor>| {
        let (vertices, triangles) = scene();
        let t_min = Tensor::full(&[rays.origins.shape()[0]], 1e-4, ElementType::F32);
        let t_max = Tensor::full(&[rays.origins.shape()[0]], t_max_owned, ElementType::F32);
        trace_rays(
            &rays.origins,
            &rays.directions,
            &t_min,
            &t_max,
            &vertices,
            &triangles,
        )
    });
    let out = f
        .call(&RayIn {
            origins: CpuTensor::from_f32(&[n, 3], origins),
            directions: CpuTensor::from_f32(&[n, 3], directions),
        })
        .expect("trace");
    assert_eq!(out.shape(), &[n, TRACE_HIT_WIDTH]);
    out.to_f32()
}

#[test]
fn trace_hits_nearest_triangle() {
    // Ray through both triangles: must report prim 0 at t = 1.
    let hits = trace(&[0.25, 0.25, 0.0], &[0.0, 0.0, 1.0], 1e4);
    assert_eq!(hits[trace_channel::HIT], 1.0);
    assert!((hits[trace_channel::T] - 1.0).abs() < 1e-5, "t = {}", hits[0]);
    assert_eq!(hits[trace_channel::PRIM], 0.0);
    assert!((hits[trace_channel::U] - 0.25).abs() < 1e-5);
    assert!((hits[trace_channel::V] - 0.25).abs() < 1e-5);
}

#[test]
fn trace_beside_near_triangle_hits_far_one() {
    // Misses prim 0 (outside its extent) but hits the large far prim 1.
    let hits = trace(&[0.9, 0.9, 0.0], &[0.0, 0.0, 1.0], 1e4);
    assert_eq!(hits[trace_channel::HIT], 1.0);
    assert!((hits[trace_channel::T] - 3.0).abs() < 1e-5);
    assert_eq!(hits[trace_channel::PRIM], 1.0);
}

#[test]
fn trace_miss_reports_zero_record() {
    let hits = trace(&[10.0, 10.0, 0.0], &[0.0, 0.0, 1.0], 1e4);
    assert_eq!(&hits[..], &[0.0; TRACE_HIT_WIDTH]);
}

#[test]
fn trace_t_max_makes_shadow_rays() {
    // Same ray as trace_hits_nearest_triangle, but t_max = 0.5 stops short.
    let hits = trace(&[0.25, 0.25, 0.0], &[0.0, 0.0, 1.0], 0.5);
    assert_eq!(hits[trace_channel::HIT], 0.0);
}

#[test]
fn trace_unnormalized_direction_scales_t() {
    let hits = trace(&[0.25, 0.25, 0.0], &[0.0, 0.0, 2.0], 1e4);
    assert!((hits[trace_channel::T] - 0.5).abs() < 1e-5);
}

#[test]
fn trace_backface_hits_without_culling() {
    // Approach prim 0 from behind (+z looking down -z).
    let hits = trace(&[0.25, 0.25, 2.0], &[0.0, 0.0, -1.0], 1e4);
    assert_eq!(hits[trace_channel::HIT], 1.0);
    assert_eq!(hits[trace_channel::PRIM], 0.0);
    assert!((hits[trace_channel::T] - 1.0).abs() < 1e-5);
}

#[derive(Tree)]
struct PosIn<T> {
    positions: T,
}

fn raster_call(positions: &[f32], triangles: &[u32], h: usize, w: usize) -> Vec<f32> {
    let v = positions.len() / 4;
    let tris = triangles.to_vec();
    let jit = CpuJit;
    let f = jit.jit(move |input: &PosIn<Tensor>| {
        let triangles = Tensor::constant_u32(&[tris.len() / 3, 3], &tris);
        rasterize(&input.positions, &triangles, h, w)
    });
    let out = f
        .call(&PosIn {
            positions: CpuTensor::from_f32(&[v, 4], positions),
        })
        .expect("rasterize");
    assert_eq!(out.shape(), &[h, w, RASTER_PIXEL_WIDTH]);
    out.to_f32()
}

fn pixel(image: &[f32], w: usize, r: usize, c: usize) -> &[f32] {
    let base = (r * w + c) * RASTER_PIXEL_WIDTH;
    &image[base..base + RASTER_PIXEL_WIDTH]
}

#[test]
fn raster_covers_expected_pixels() {
    // Full-viewport-lower-left triangle in NDC (w = 1): covers below the
    // anti-diagonal. 4x4 image.
    let positions = [
        -1.0, -1.0, 0.5, 1.0, //
        1.0, -1.0, 0.5, 1.0, //
        -1.0, 1.0, 0.5, 1.0,
    ];
    let img = raster_call(&positions, &[0, 1, 2], 4, 4);
    // Bottom-left pixel (row 3, col 0) is deep inside.
    let p = pixel(&img, 4, 3, 0);
    assert_eq!(p[raster_channel::HIT], 1.0);
    assert_eq!(p[raster_channel::PRIM], 0.0);
    // Top-right pixel (row 0, col 3) is outside.
    let q = pixel(&img, 4, 0, 3);
    assert_eq!(q, &[0.0; RASTER_PIXEL_WIDTH]);
    // Barycentrics sum to <= 1 and are non-negative wherever hit.
    for r in 0..4 {
        for c in 0..4 {
            let p = pixel(&img, 4, r, c);
            if p[raster_channel::HIT] == 1.0 {
                let (u, v) = (p[raster_channel::U], p[raster_channel::V]);
                assert!(u >= -1e-6 && v >= -1e-6 && u + v <= 1.0 + 1e-6);
            }
        }
    }
}

#[test]
fn raster_depth_test_prefers_near_triangle() {
    // Two full-screen-ish triangles; prim 1 nearer (smaller z).
    let positions = [
        // prim 0, z = 0.8
        -1.0, -1.0, 0.8, 1.0, //
        3.0, -1.0, 0.8, 1.0, //
        -1.0, 3.0, 0.8, 1.0, //
        // prim 1, z = 0.2
        -1.0, -1.0, 0.2, 1.0, //
        3.0, -1.0, 0.2, 1.0, //
        -1.0, 3.0, 0.2, 1.0,
    ];
    let img = raster_call(&positions, &[0, 1, 2, 3, 4, 5], 2, 2);
    for r in 0..2 {
        for c in 0..2 {
            let p = pixel(&img, 2, r, c);
            assert_eq!(p[raster_channel::HIT], 1.0);
            assert_eq!(p[raster_channel::PRIM], 1.0, "near triangle wins at ({r},{c})");
        }
    }
}

#[test]
fn raster_draw_order_breaks_ties_toward_first() {
    // Same depth: first triangle drawn survives the strict '<' test.
    let positions = [
        -1.0, -1.0, 0.5, 1.0, //
        3.0, -1.0, 0.5, 1.0, //
        -1.0, 3.0, 0.5, 1.0, //
        -1.0, -1.0, 0.5, 1.0, //
        3.0, -1.0, 0.5, 1.0, //
        -1.0, 3.0, 0.5, 1.0,
    ];
    let img = raster_call(&positions, &[0, 1, 2, 3, 4, 5], 2, 2);
    assert_eq!(pixel(&img, 2, 0, 0)[raster_channel::PRIM], 0.0);
}

#[test]
fn raster_perspective_correct_barycentrics() {
    // A triangle with strongly differing vertex w. At the pixel nearest a
    // vertex, that vertex's barycentric weight should dominate.
    let positions = [
        -0.9, -0.9, 0.5, 1.0, // v0, near
        0.9, -0.9, 0.5, 4.0, // v1, far (larger w)
        -0.9, 0.9, 0.5, 1.0, // v2
    ];
    let img = raster_call(&positions, &[0, 1, 2], 8, 8);
    // Pixel near v0 (bottom-left, row 7 col 0): w = 1-u-v should dominate.
    let p = pixel(&img, 8, 7, 0);
    if p[raster_channel::HIT] == 1.0 {
        let w0 = 1.0 - p[raster_channel::U] - p[raster_channel::V];
        assert!(w0 > 0.8, "vertex-0 weight = {w0}");
    }
}

/// Full composition: slice hit records with `index`, gather per-triangle
/// colors by prim id, mask by hit — the "shading in the graph" pattern —
/// and check gradients flow to the color table (not the geometry).
#[test]
fn trace_then_gather_shades_and_differentiates() {
    #[derive(Tree)]
    struct P<T> {
        colors: T,
    }

    let jit = CpuJit;
    let f = jit.jit(move |p: &P<Tensor>| {
        let (vertices, triangles) = scene();
        let origins =
            Tensor::constant_f32(&[2, 3], &[0.25, 0.25, 0.0, 0.9, 0.9, 0.0]);
        let directions =
            Tensor::constant_f32(&[2, 3], &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let t_min = Tensor::full(&[2], 1e-4, ElementType::F32);
        let t_max = Tensor::full(&[2], 1e4, ElementType::F32);
        let hits = trace_rays(&origins, &directions, &t_min, &t_max, &vertices, &triangles);
        let prim = hits
            .index(&[
                IndexKeyElement::Slice(0..2),
                IndexKeyElement::Single(trace_channel::PRIM),
            ])
            .squeeze(&[1])
            .cast(ElementType::U32);
        let hit = hits.index(&[
            IndexKeyElement::Slice(0..2),
            IndexKeyElement::Single(trace_channel::HIT),
        ]);
        let shaded = p.colors.gather_rows(&prim) * hit;
        let loss = shaded.sum_axes(&[0, 1]).squeeze_all();
        let grads = resin_dsl::grad_wrt(&loss, &p.colors).expect("grad");
        vec![loss, grads]
    });

    let out = f
        .call(&P {
            colors: CpuTensor::from_f32(&[2, 1], &[3.0, 5.0]),
        })
        .expect("run");
    // Ray 0 sees color[0] = 3, ray 1 sees color[1] = 5.
    assert!((out[0].scalar_f32() - 8.0).abs() < 1e-5);
    // Each color is seen by exactly one ray: d loss / d color = 1.
    assert_eq!(out[1].to_f32(), vec![1.0, 1.0]);
}

/// Reshape as a view: flatten a visibility buffer, gather per-pixel colors
/// by prim id, and reassemble — the deferred-shading skeleton.
#[test]
fn reshape_flattens_visibility_buffer_for_gather() {
    let jit = CpuJit;
    let f = jit.jit(move |input: &PosIn<Tensor>| {
        let triangles = Tensor::constant_u32(&[1, 3], &[0, 1, 2]);
        let vis = rasterize(&input.positions, &triangles, 2, 2);
        let flat = vis.reshape(&[4, RASTER_PIXEL_WIDTH]);
        let prim = flat
            .index(&[
                IndexKeyElement::Slice(0..4),
                IndexKeyElement::Single(raster_channel::PRIM),
            ])
            .squeeze(&[1])
            .cast(ElementType::U32);
        let hit = flat.index(&[
            IndexKeyElement::Slice(0..4),
            IndexKeyElement::Single(raster_channel::HIT),
        ]);
        let colors = Tensor::constant_f32(&[1, 3], &[0.25, 0.5, 0.75]);
        let shaded = colors.gather_rows(&prim) * hit;
        shaded.reshape(&[2, 2, 3])
    });
    // Full-screen triangle: every pixel center covered.
    let positions = [
        -3.0, -3.0, 0.5, 1.0, //
        3.0, -3.0, 0.5, 1.0, //
        0.0, 3.0, 0.5, 1.0,
    ];
    let out = f
        .call(&PosIn {
            positions: CpuTensor::from_f32(&[3, 4], &positions),
        })
        .expect("deferred shade");
    assert_eq!(out.shape(), &[2, 2, 3]);
    for pixel in out.to_f32().chunks_exact(3) {
        assert_eq!(pixel, &[0.25, 0.5, 0.75]);
    }
}
