//! GPU `rasterize` parity against the CPU reference (`docs/hw-nodes.md`).
//!
//! Coverage rules legitimately differ right at triangle edges (fill
//! conventions, watertightness), so only pixels whose 3×3 neighborhood has a
//! uniform `(prim, hit)` in BOTH images are compared — on those the record
//! must match exactly (`prim`, `hit`) / tightly (`u`, `v`). Tests skip
//! gracefully when no device is available.

#![cfg(all(feature = "cpu", any(feature = "wgpu", feature = "vulkan")))]

use resin_core::hw::{raster_channel, RASTER_PIXEL_WIDTH};
use resin_dsl::{rasterize, Tensor};
use resin_jit::backends::cpu::CpuJit;
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

#[derive(Tree)]
struct PosIn<T> {
    positions: T,
}

// The >90%-compared requirement needs the skipped 1-pixel edge bands to stay
// under 10% of the image, which bounds the visible edge length; 128² keeps
// the scene interesting while staying trivial for lavapipe (one small draw).
const H: usize = 128;
const W: usize = 128;

/// Triangles with varying depth and w (all w > 0): a near-full-screen far
/// backdrop (edges mostly outside the viewport), a nearer overlapping one,
/// one with strongly differing vertex w (exercises perspective-correct
/// barycentrics), and one crossing the viewport edge (exercises clipping vs.
/// the CPU's bbox clamp). A sliver of background survives top-right.
fn scene() -> (Vec<f32>, Vec<u32>) {
    #[rustfmt::skip]
    let positions = vec![
        // prim 0: backdrop at z = 0.9, hypotenuse clipping one corner
        -1.2, -1.2, 0.9, 1.0,
        3.0, -1.2, 0.9, 1.0,
        -1.2, 3.0, 0.9, 1.0,
        // prim 1: nearer, overlapping the backdrop
        -0.55, -0.5, 0.4, 1.0,
        0.25, -0.35, 0.4, 1.0,
        -0.15, 0.3, 0.4, 1.0,
        // prim 2: strong per-vertex w variation (NDC kept in range by
        // pre-multiplying each corner by its w)
        0.35 * 1.0, -0.6 * 1.0, 0.30 * 1.0, 1.0,
        0.85 * 2.5, -0.1 * 2.5, 0.60 * 2.5, 2.5,
        0.45 * 1.5, 0.55 * 1.5, 0.20 * 1.5, 1.5,
        // prim 3: crosses the top-right viewport corner
        0.7, 0.6, 0.5, 1.0,
        1.3, 0.9, 0.5, 1.0,
        0.8, 1.2, 0.5, 1.0,
    ];
    let triangles = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    (positions, triangles)
}

/// Backdrop + one small triangle: nearly no edge pixels, so it passes the
/// compared-ratio bar even at tiny odd sizes (which exist to exercise
/// row-stride handling in the readback paths).
fn stride_scene() -> (Vec<f32>, Vec<u32>) {
    #[rustfmt::skip]
    let positions = vec![
        -1.2, -1.2, 0.9, 1.0,
        4.0, -1.2, 0.9, 1.0,
        -1.2, 4.0, 0.9, 1.0,
        -0.2, -0.15, 0.4, 1.0,
        0.25, -0.05, 0.4, 1.0,
        0.0, 0.2, 0.4, 1.0,
    ];
    let triangles = vec![0, 1, 2, 3, 4, 5];
    (positions, triangles)
}

fn raster_with<J: Jit>(
    jit: J,
    positions: &[f32],
    triangles: &[u32],
    h: usize,
    w: usize,
) -> Vec<f32> {
    let tris = triangles.to_vec();
    let f = jit.jit(move |input: &PosIn<Tensor>| {
        let triangles = Tensor::constant_u32(&[tris.len() / 3, 3], &tris);
        rasterize(&input.positions, &triangles, h, w)
    });
    let out = f
        .call(&PosIn {
            positions: J::Tensor::from_f32(&[positions.len() / 4, 4], positions),
        })
        .expect("rasterize");
    assert_eq!(out.shape(), &[h, w, RASTER_PIXEL_WIDTH]);
    out.to_f32()
}

/// Pixels whose 3×3 in-image neighborhood carries one `(prim, hit)` value.
fn interior_mask(image: &[f32], h: usize, w: usize) -> Vec<bool> {
    let key = |r: usize, c: usize| {
        let base = (r * w + c) * RASTER_PIXEL_WIDTH;
        (
            image[base + raster_channel::PRIM],
            image[base + raster_channel::HIT],
        )
    };
    (0..h * w)
        .map(|i| {
            let (r, c) = (i / w, i % w);
            let center = key(r, c);
            (r.saturating_sub(1)..=(r + 1).min(h - 1)).all(|rr| {
                (c.saturating_sub(1)..=(c + 1).min(w - 1)).all(|cc| key(rr, cc) == center)
            })
        })
        .collect()
}

fn assert_parity(cpu: &[f32], gpu: &[f32], h: usize, w: usize) {
    let cpu_interior = interior_mask(cpu, h, w);
    let gpu_interior = interior_mask(gpu, h, w);
    let mut compared = 0usize;
    for i in 0..h * w {
        if !(cpu_interior[i] && gpu_interior[i]) {
            continue;
        }
        compared += 1;
        let a = &cpu[i * RASTER_PIXEL_WIDTH..(i + 1) * RASTER_PIXEL_WIDTH];
        let b = &gpu[i * RASTER_PIXEL_WIDTH..(i + 1) * RASTER_PIXEL_WIDTH];
        let at = (i / w, i % w);
        assert_eq!(
            a[raster_channel::PRIM],
            b[raster_channel::PRIM],
            "prim at {at:?}: cpu {a:?} vs gpu {b:?}"
        );
        assert_eq!(
            a[raster_channel::HIT],
            b[raster_channel::HIT],
            "hit at {at:?}: cpu {a:?} vs gpu {b:?}"
        );
        for channel in [raster_channel::U, raster_channel::V] {
            assert!(
                (a[channel] - b[channel]).abs() < 1e-3,
                "channel {channel} at {at:?}: cpu {a:?} vs gpu {b:?}"
            );
        }
    }
    assert!(
        compared * 10 > h * w * 9,
        "only {compared}/{} pixels away from edges in both images",
        h * w
    );
}

fn parity_case<J: Jit>(jit: J, positions: &[f32], triangles: &[u32], h: usize, w: usize) {
    let cpu = raster_with(CpuJit, positions, triangles, h, w);
    let gpu = raster_with(jit, positions, triangles, h, w);
    assert_parity(&cpu, &gpu, h, w);
}

#[cfg(feature = "wgpu")]
mod wgpu_parity {
    use super::*;
    use resin_jit::backends::wgpu::WgpuJit;

    fn skip() -> bool {
        !resin_jit::backends::wgpu::shared_context_available()
    }

    #[test]
    fn wgpu_rasterize_matches_cpu() {
        if skip() {
            eprintln!("skip wgpu_rasterize_matches_cpu: no GPU adapter");
            return;
        }
        let (positions, triangles) = scene();
        parity_case(WgpuJit, &positions, &triangles, H, W);
    }

    /// Width whose row bytes are not 256-aligned (the padded readback path).
    #[test]
    fn wgpu_rasterize_matches_cpu_unaligned_width() {
        if skip() {
            eprintln!("skip wgpu_rasterize_matches_cpu_unaligned_width: no GPU adapter");
            return;
        }
        let (positions, triangles) = stride_scene();
        parity_case(WgpuJit, &positions, &triangles, 17, 33);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan_parity {
    use super::*;
    use resin_jit::backends::vulkan::VulkanJit;

    fn skip() -> bool {
        !resin_jit::backends::vulkan::shared_context_available()
    }

    #[test]
    fn vulkan_rasterize_matches_cpu() {
        if skip() {
            eprintln!("skip vulkan_rasterize_matches_cpu: no Vulkan device");
            return;
        }
        let (positions, triangles) = scene();
        parity_case(VulkanJit, &positions, &triangles, H, W);
    }

    #[test]
    fn vulkan_rasterize_matches_cpu_odd_size() {
        if skip() {
            eprintln!("skip vulkan_rasterize_matches_cpu_odd_size: no Vulkan device");
            return;
        }
        let (positions, triangles) = stride_scene();
        parity_case(VulkanJit, &positions, &triangles, 17, 33);
    }
}
