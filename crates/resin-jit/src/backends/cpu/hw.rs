//! CPU reference executors for the hardware nodes.
//!
//! Deliberately simple — brute-force ray/triangle intersection and a
//! bounding-box scanline rasterizer — so they can serve as the golden oracle
//! for the GPU implementations (`docs/hw-nodes.md`).

use resin_core::hw::{RASTER_PIXEL_WIDTH, TRACE_HIT_WIDTH};
use resin_core::Accessor;
use resin_ir::{IrRasterizeKernel, IrTraceRaysKernel};

use super::exec::{read_f32_view, read_u32_view, write_f32};
use crate::error::RunError;

type Vec3 = [f32; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Möller–Trumbore without backface culling. Returns `(t, u, v)`.
fn intersect_triangle(o: Vec3, d: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> Option<(f32, f32, f32)> {
    let e1 = sub(v1, v0);
    let e2 = sub(v2, v0);
    let p = cross(d, e2);
    let det = dot(e1, p);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    let s = sub(o, v0);
    let u = dot(s, p) * inv_det;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = cross(s, e1);
    let v = dot(d, q) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = dot(e2, q) * inv_det;
    Some((t, u, v))
}

fn read_vec3(
    storage: &[Vec<u8>],
    arg: &(usize, Accessor),
    row: u32,
) -> Result<Vec3, RunError> {
    let (buf, acc) = arg;
    Ok([
        read_f32_view(&storage[*buf], acc, &[row, 0])?,
        read_f32_view(&storage[*buf], acc, &[row, 1])?,
        read_f32_view(&storage[*buf], acc, &[row, 2])?,
    ])
}

/// Triangle vertex indices, clamped in-bounds (mirrors gather semantics).
fn triangle_indices(
    storage: &[Vec<u8>],
    arg: &(usize, Accessor),
    tri: u32,
    vertex_count: u32,
) -> Result<[u32; 3], RunError> {
    let (buf, acc) = arg;
    let max = vertex_count.saturating_sub(1);
    Ok([
        read_u32_view(&storage[*buf], acc, &[tri, 0])?.min(max),
        read_u32_view(&storage[*buf], acc, &[tri, 1])?.min(max),
        read_u32_view(&storage[*buf], acc, &[tri, 2])?.min(max),
    ])
}

pub(super) fn run_trace_rays(
    kernel: &IrTraceRaysKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    let n = kernel.ray_count();
    let vertex_count = kernel.vertex_count();
    let triangle_count = kernel.triangle_count();
    let (origins, directions, t_min, t_max, vertices, triangles) =
        (&args[0], &args[1], &args[2], &args[3], &args[4], &args[5]);

    let expected = n as usize * TRACE_HIT_WIDTH * 4;
    if storage[out_index].len() != expected {
        return Err(RunError::BufferSizeMismatch {
            expected,
            got: storage[out_index].len(),
        });
    }

    for ray in 0..n {
        let o = read_vec3(storage, origins, ray)?;
        let d = read_vec3(storage, directions, ray)?;
        let lo = read_f32_view(&storage[t_min.0], &t_min.1, &[ray])?;
        let hi = read_f32_view(&storage[t_max.0], &t_max.1, &[ray])?;

        let mut best: Option<(f32, f32, f32, u32)> = None;
        for tri in 0..triangle_count {
            let [i0, i1, i2] = triangle_indices(storage, triangles, tri, vertex_count)?;
            let v0 = read_vec3(storage, vertices, i0)?;
            let v1 = read_vec3(storage, vertices, i1)?;
            let v2 = read_vec3(storage, vertices, i2)?;
            if let Some((t, u, v)) = intersect_triangle(o, d, v0, v1, v2)
                && t >= lo
                && t <= hi
                && best.map(|(bt, ..)| t < bt).unwrap_or(true)
            {
                best = Some((t, u, v, tri));
            }
        }

        let record = match best {
            Some((t, u, v, prim)) => [t, u, v, prim as f32, 1.0],
            None => [0.0; TRACE_HIT_WIDTH],
        };
        let base = ray as usize * TRACE_HIT_WIDTH;
        for (k, value) in record.iter().enumerate() {
            write_f32(
                &mut storage[out_index][(base + k) * 4..(base + k + 1) * 4],
                *value,
            );
        }
    }
    Ok(())
}

pub(super) fn run_rasterize(
    kernel: &IrRasterizeKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    let (h, w) = (kernel.height() as usize, kernel.width() as usize);
    let vertex_count = kernel.vertex_count();
    let triangle_count = kernel.triangle_count();
    let (positions, triangles) = (&args[0], &args[1]);

    let expected = h * w * RASTER_PIXEL_WIDTH * 4;
    if storage[out_index].len() != expected {
        return Err(RunError::BufferSizeMismatch {
            expected,
            got: storage[out_index].len(),
        });
    }

    // Output starts zeroed (validated clear_output_before_dispatch).
    let mut depth = vec![f32::INFINITY; h * w];

    for tri in 0..triangle_count {
        let [i0, i1, i2] = triangle_indices(storage, triangles, tri, vertex_count)?;
        let mut clip = [[0.0f32; 4]; 3];
        for (slot, idx) in [i0, i1, i2].into_iter().enumerate() {
            for (k, value) in clip[slot].iter_mut().enumerate() {
                *value =
                    read_f32_view(&storage[positions.0], &positions.1, &[idx, k as u32])?;
            }
        }
        // v1 has no near-plane clipping: skip triangles that cross or sit
        // behind w = 0 (test scenes keep geometry in front of the camera).
        if clip.iter().any(|p| p[3] <= 0.0) {
            continue;
        }
        // Screen space: x right, y down, pixel centers at (c + 0.5, r + 0.5).
        let screen: Vec<[f32; 3]> = clip
            .iter()
            .map(|p| {
                let inv_w = 1.0 / p[3];
                let ndc = [p[0] * inv_w, p[1] * inv_w, p[2] * inv_w];
                [
                    (ndc[0] + 1.0) * 0.5 * w as f32,
                    (1.0 - ndc[1]) * 0.5 * h as f32,
                    ndc[2],
                ]
            })
            .collect();

        let edge = |a: &[f32; 3], b: &[f32; 3], px: f32, py: f32| -> f32 {
            (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        };
        let area = edge(&screen[0], &screen[1], screen[2][0], screen[2][1]);
        if area.abs() < 1e-12 {
            continue;
        }

        let min_x = screen.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min);
        let max_x = screen.iter().map(|p| p[0]).fold(f32::NEG_INFINITY, f32::max);
        let min_y = screen.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min);
        let max_y = screen.iter().map(|p| p[1]).fold(f32::NEG_INFINITY, f32::max);
        let c0 = (min_x - 0.5).floor().max(0.0) as usize;
        let c1 = ((max_x - 0.5).ceil() as isize).clamp(0, w as isize - 1) as usize;
        let r0 = (min_y - 0.5).floor().max(0.0) as usize;
        let r1 = ((max_y - 0.5).ceil() as isize).clamp(0, h as isize - 1) as usize;
        if min_x > w as f32 || max_x < 0.0 || min_y > h as f32 || max_y < 0.0 {
            continue;
        }

        for r in r0..=r1 {
            for c in c0..=c1 {
                let px = c as f32 + 0.5;
                let py = r as f32 + 0.5;
                // Normalizing by the signed area keeps λ ∈ [0,1] inside the
                // triangle for both windings (no backface culling).
                let l0 = edge(&screen[1], &screen[2], px, py) / area;
                let l1 = edge(&screen[2], &screen[0], px, py) / area;
                let l2 = edge(&screen[0], &screen[1], px, py) / area;
                if l0 < 0.0 || l1 < 0.0 || l2 < 0.0 {
                    continue;
                }
                // NDC depth is linear in screen space.
                let z = l0 * screen[0][2] + l1 * screen[1][2] + l2 * screen[2][2];
                if !(0.0..=1.0).contains(&z) {
                    continue;
                }
                let pixel = r * w + c;
                if z >= depth[pixel] {
                    continue;
                }
                depth[pixel] = z;
                // Perspective-correct barycentrics from screen-space ones.
                let pw0 = l0 / clip[0][3];
                let pw1 = l1 / clip[1][3];
                let pw2 = l2 / clip[2][3];
                let inv_sum = 1.0 / (pw0 + pw1 + pw2);
                let record = [tri as f32, 1.0, pw1 * inv_sum, pw2 * inv_sum];
                let base = pixel * RASTER_PIXEL_WIDTH;
                for (k, value) in record.iter().enumerate() {
                    write_f32(
                        &mut storage[out_index][(base + k) * 4..(base + k + 1) * 4],
                        *value,
                    );
                }
            }
        }
    }
    Ok(())
}
