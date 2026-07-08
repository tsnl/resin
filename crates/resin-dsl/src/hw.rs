//! Fixed-function hardware nodes: ray tracing and rasterization.
//!
//! Both nodes are *visibility only*: they resolve which primitive is seen
//! (along a ray, or through a pixel center) and at what barycentric
//! coordinates. Everything that looks like shading — interpolating
//! attributes, looking up materials, lighting — composes in the graph from
//! [`Tensor::gather_rows`], [`Tensor::index`], and elementwise math, which is
//! where differentiability lives. The nodes themselves are piecewise
//! constant and propagate no gradient (`docs/hw-nodes.md`).

use resin_core::hw::{RASTER_PIXEL_WIDTH, TRACE_HIT_WIDTH};

use crate::tensor::{ElementType, Tensor, TensorInner, TensorKind};

/// Trace one closest-hit ray per row against triangle geometry.
///
/// - `origins`, `directions`: `[N, 3]` F32 (directions need not be normalized;
///   `t` is measured in units of `directions`' length),
/// - `t_min`, `t_max`: `[N]` F32 per-ray interval (use a small positive
///   `t_min` to avoid self-intersection; a finite `t_max` makes shadow rays),
/// - `vertices`: `[V, 3]` F32, `triangles`: `[T, 3]` U32 vertex indices.
///
/// Returns `[N, 5]` F32 hit records `(t, u, v, prim, hit)` — channel indices
/// in [`resin_core::hw::trace_channel`]. On miss, `t = 0`, `prim = 0` (safe
/// for gathers) and `hit = 0`. `prim` is an exact integer-valued f32; use
/// `.cast(ElementType::U32)` to build gather keys. Backends run this on
/// ray-tracing hardware where available, else a compute/CPU fallback; results
/// agree up to watertightness at shared triangle edges.
pub fn trace_rays(
    origins: &Tensor,
    directions: &Tensor,
    t_min: &Tensor,
    t_max: &Tensor,
    vertices: &Tensor,
    triangles: &Tensor,
) -> Tensor {
    let n = check_rows3(origins, "trace_rays origins");
    let n2 = check_rows3(directions, "trace_rays directions");
    assert_eq!(n, n2, "trace_rays: origins and directions row counts differ");
    for (t, name) in [(t_min, "t_min"), (t_max, "t_max")] {
        assert_eq!(
            t.element_type(),
            ElementType::F32,
            "trace_rays {name} must be F32"
        );
        assert_eq!(
            t.shape(),
            &[n],
            "trace_rays {name} must have shape [N] matching the rays"
        );
    }
    check_geometry(vertices, triangles, "trace_rays");

    Tensor::from(TensorInner {
        element_type: ElementType::F32,
        shape: Box::from([n, TRACE_HIT_WIDTH]),
        kind: TensorKind::TraceRays {
            origins: origins.clone(),
            directions: directions.clone(),
            t_min: t_min.clone(),
            t_max: t_max.clone(),
            vertices: vertices.clone(),
            triangles: triangles.clone(),
        },
    })
}

/// Rasterize triangles into an `[H, W, 4]` visibility buffer.
///
/// - `clip_positions`: `[V, 4]` F32 clip-space positions (the "vertex shader"
///   is ordinary graph code upstream of this node),
/// - `triangles`: `[T, 3]` U32 vertex indices.
///
/// Each pixel of the result holds `(prim, hit, u, v)` — channel indices in
/// [`resin_core::hw::raster_channel`] — for the depth-nearest triangle
/// covering the pixel center, with perspective-correct barycentrics
/// (`w = 1 - u - v` for vertex 0). Background pixels are all zero. Pixel
/// `(r, c)` samples NDC `x = (c+0.5)/W·2−1`, `y = 1−(r+0.5)/H·2` (row 0 =
/// top); depth is NDC `z ∈ [0, 1]`, closer wins; no backface culling.
pub fn rasterize(
    clip_positions: &Tensor,
    triangles: &Tensor,
    height: usize,
    width: usize,
) -> Tensor {
    assert_eq!(
        clip_positions.element_type(),
        ElementType::F32,
        "rasterize clip_positions must be F32"
    );
    assert!(
        clip_positions.shape().len() == 2 && clip_positions.shape()[1] == 4,
        "rasterize clip_positions must have shape [V, 4], got {:?}",
        clip_positions.shape()
    );
    check_triangles(triangles, "rasterize");
    assert!(height > 0 && width > 0, "rasterize: image must be non-empty");

    Tensor::from(TensorInner {
        element_type: ElementType::F32,
        shape: Box::from([height, width, RASTER_PIXEL_WIDTH]),
        kind: TensorKind::Rasterize {
            clip_positions: clip_positions.clone(),
            triangles: triangles.clone(),
        },
    })
}

fn check_rows3(t: &Tensor, what: &str) -> usize {
    assert_eq!(t.element_type(), ElementType::F32, "{what} must be F32");
    assert!(
        t.shape().len() == 2 && t.shape()[1] == 3,
        "{what} must have shape [N, 3], got {:?}",
        t.shape()
    );
    t.shape()[0]
}

fn check_triangles(triangles: &Tensor, ctx: &str) {
    assert_eq!(
        triangles.element_type(),
        ElementType::U32,
        "{ctx} triangles must be U32 vertex indices"
    );
    assert!(
        triangles.shape().len() == 2 && triangles.shape()[1] == 3,
        "{ctx} triangles must have shape [T, 3], got {:?}",
        triangles.shape()
    );
}

fn check_geometry(vertices: &Tensor, triangles: &Tensor, ctx: &str) {
    assert_eq!(
        vertices.element_type(),
        ElementType::F32,
        "{ctx} vertices must be F32"
    );
    assert!(
        vertices.shape().len() == 2 && vertices.shape()[1] == 3,
        "{ctx} vertices must have shape [V, 3], got {:?}",
        vertices.shape()
    );
    check_triangles(triangles, ctx);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> (Tensor, Tensor) {
        let vertices =
            Tensor::constant_f32(&[3, 3], &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let triangles = Tensor::constant_u32(&[1, 3], &[0, 1, 2]);
        (vertices, triangles)
    }

    #[test]
    fn trace_rays_shape_and_kind() {
        let (vertices, triangles) = scene();
        let origins = Tensor::constant_f32(&[2, 3], &[0.0; 6]);
        let directions = Tensor::constant_f32(&[2, 3], &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let t_min = Tensor::full(&[2], 1e-3, ElementType::F32);
        let t_max = Tensor::full(&[2], 1e3, ElementType::F32);
        let hits = trace_rays(&origins, &directions, &t_min, &t_max, &vertices, &triangles);
        assert_eq!(hits.shape(), &[2, TRACE_HIT_WIDTH]);
        assert_eq!(hits.element_type(), ElementType::F32);
        assert!(matches!(hits.kind(), TensorKind::TraceRays { .. }));
    }

    #[test]
    #[should_panic(expected = "triangles must be U32")]
    fn trace_rays_rejects_f32_triangles() {
        let vertices = Tensor::constant_f32(&[3, 3], &[0.0; 9]);
        let triangles = Tensor::constant_f32(&[1, 3], &[0.0; 3]);
        let origins = Tensor::constant_f32(&[1, 3], &[0.0; 3]);
        let directions = Tensor::constant_f32(&[1, 3], &[0.0, 0.0, 1.0]);
        let t = Tensor::full(&[1], 0.0, ElementType::F32);
        trace_rays(&origins, &directions, &t, &t, &vertices, &triangles);
    }

    #[test]
    fn rasterize_shape_and_kind() {
        let positions = Tensor::constant_f32(&[3, 4], &[0.0; 12]);
        let triangles = Tensor::constant_u32(&[1, 3], &[0, 1, 2]);
        let vis = rasterize(&positions, &triangles, 4, 6);
        assert_eq!(vis.shape(), &[4, 6, RASTER_PIXEL_WIDTH]);
        assert!(matches!(vis.kind(), TensorKind::Rasterize { .. }));
    }
}
