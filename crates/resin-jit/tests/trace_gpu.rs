//! GPU ↔ CPU parity for `trace_rays` on randomized scenes, for both the
//! hardware ray-query path (chosen automatically on ray-tracing devices —
//! lavapipe included) and the compute-BVH fallback (forced via
//! `RESIN_TRACE_FORCE_FALLBACK`). Watertightness differs between
//! implementations at shared triangle edges, so a small fraction of rays may
//! legitimately disagree; agreeing rays must match tightly.
//!
//! BVH-internal tests (host traversal vs brute force, kernel-text naga
//! validation) live as unit tests in `backends/hwtrace.rs` — the builder is
//! crate-private.

#![cfg(feature = "cpu")]

use std::sync::Mutex;

use resin_core::hw::TRACE_HIT_WIDTH;
use resin_dsl::{trace_rays, ElementType, Tensor};
use resin_jit::backends::cpu::CpuJit;
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

/// Serializes tests that touch the process-global fallback override, so a
/// forced-fallback run cannot leak into a concurrently running hardware one.
static TRACE_MODE: Mutex<()> = Mutex::new(());

/// Forces the compute fallback while alive. Hold the [`TRACE_MODE`] lock.
struct ForceFallback;

impl ForceFallback {
    fn new() -> Self {
        // SAFETY: guarded by TRACE_MODE, no other thread in this test binary
        // reads or writes the environment concurrently.
        unsafe { std::env::set_var("RESIN_TRACE_FORCE_FALLBACK", "1") };
        Self
    }
}

impl Drop for ForceFallback {
    fn drop(&mut self) {
        // SAFETY: as in `new`.
        unsafe { std::env::remove_var("RESIN_TRACE_FORCE_FALLBACK") };
    }
}

fn lock_trace_mode() -> std::sync::MutexGuard<'static, ()> {
    TRACE_MODE.lock().unwrap_or_else(|e| e.into_inner())
}

#[derive(Tree)]
struct Rays<T> {
    origins: T,
    directions: T,
}

struct SceneData {
    vertices: Vec<f32>,
    triangles: Vec<u32>,
}

/// Deterministic xorshift* for scene/ray generation.
struct Rng(u64);

impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }

    fn in_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

fn random_scene(rng: &mut Rng, tris: usize) -> SceneData {
    let mut vertices = Vec::new();
    let mut triangles = Vec::new();
    for tri in 0..tris {
        let center = [
            rng.in_range(-1.0, 1.0),
            rng.in_range(-1.0, 1.0),
            rng.in_range(-1.0, 1.0),
        ];
        for _ in 0..3 {
            for &c in &center {
                vertices.push(c + rng.in_range(-0.4, 0.4));
            }
        }
        triangles.extend([tri as u32 * 3, tri as u32 * 3 + 1, tri as u32 * 3 + 2]);
    }
    SceneData {
        vertices,
        triangles,
    }
}

/// Rays from a shell around the scene toward random interior points, with
/// unnormalized directions (t is measured in direction lengths).
fn random_rays(rng: &mut Rng, count: usize) -> (Vec<f32>, Vec<f32>) {
    let mut origins = Vec::new();
    let mut directions = Vec::new();
    for _ in 0..count {
        let o = [
            rng.in_range(-2.5, 2.5),
            rng.in_range(-2.5, 2.5),
            rng.in_range(-2.5, 2.5),
        ];
        let target = [
            rng.in_range(-1.0, 1.0),
            rng.in_range(-1.0, 1.0),
            rng.in_range(-1.0, 1.0),
        ];
        origins.extend(o);
        directions.extend([target[0] - o[0], target[1] - o[1], target[2] - o[2]]);
    }
    (origins, directions)
}

fn trace_on<J: Jit>(jit: J, scene: &SceneData, origins: &[f32], directions: &[f32]) -> Vec<f32> {
    let n = origins.len() / 3;
    let vertices = scene.vertices.clone();
    let triangles = scene.triangles.clone();
    let f = jit.jit(move |rays: &Rays<Tensor>| {
        let v = Tensor::constant_f32(&[vertices.len() / 3, 3], &vertices);
        let t = Tensor::constant_u32(&[triangles.len() / 3, 3], &triangles);
        let n = rays.origins.shape()[0];
        let t_min = Tensor::full(&[n], 1e-4, ElementType::F32);
        let t_max = Tensor::full(&[n], 1e4, ElementType::F32);
        trace_rays(&rays.origins, &rays.directions, &t_min, &t_max, &v, &t)
    });
    let out = f
        .call(&Rays {
            origins: J::Tensor::from_f32(&[n, 3], origins),
            directions: J::Tensor::from_f32(&[n, 3], directions),
        })
        .expect("trace_rays run");
    assert_eq!(out.shape(), &[n, TRACE_HIT_WIDTH]);
    out.to_f32()
}

/// Rays that resolve the same `(hit, prim)` must agree on t/u/v within
/// 1e-3; the rest (watertightness at shared edges) must stay under 2%.
fn assert_parity(cpu: &[f32], gpu: &[f32], label: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{label}: record count");
    let n = cpu.len() / TRACE_HIT_WIDTH;
    let mut disagreements = 0usize;
    for ray in 0..n {
        let c = &cpu[ray * TRACE_HIT_WIDTH..(ray + 1) * TRACE_HIT_WIDTH];
        let g = &gpu[ray * TRACE_HIT_WIDTH..(ray + 1) * TRACE_HIT_WIDTH];
        if c[3] != g[3] || c[4] != g[4] {
            disagreements += 1;
            continue;
        }
        for channel in 0..3 {
            assert!(
                (c[channel] - g[channel]).abs() < 1e-3,
                "{label}: ray {ray} channel {channel}: cpu {} vs gpu {}",
                c[channel],
                g[channel],
            );
        }
    }
    let allowed = (n as f32 * 0.02).ceil() as usize;
    assert!(
        disagreements <= allowed,
        "{label}: {disagreements}/{n} rays disagree (allowed {allowed})"
    );
}

fn parity_on<J: Jit>(jit: J, label: &str) {
    let mut rng = Rng(0x5eed_0001);
    for round in 0..3 {
        let scene = random_scene(&mut rng, 50);
        let (origins, directions) = random_rays(&mut rng, 200);
        let cpu = trace_on(CpuJit, &scene, &origins, &directions);
        let hit_count = cpu.chunks_exact(TRACE_HIT_WIDTH).filter(|r| r[4] == 1.0).count();
        assert!(hit_count > 20, "{label}: degenerate scene {round} ({hit_count} hits)");
        let gpu = trace_on(jit.clone(), &scene, &origins, &directions);
        assert_parity(&cpu, &gpu, &format!("{label} round {round}"));
    }
}

fn zero_triangles_on<J: Jit>(jit: J, label: &str) {
    let scene = SceneData {
        vertices: vec![0.0; 9],
        triangles: Vec::new(),
    };
    let mut rng = Rng(0x5eed_0002);
    let (origins, directions) = random_rays(&mut rng, 33);
    let gpu = trace_on(jit, &scene, &origins, &directions);
    assert!(
        gpu.iter().all(|&v| v == 0.0),
        "{label}: empty scene must miss every ray"
    );
}

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;
    use resin_jit::backends::wgpu::{shared_context_available, WgpuJit};

    #[test]
    fn wgpu_matches_cpu_on_random_scenes() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip wgpu_matches_cpu_on_random_scenes: no adapter");
            return;
        }
        parity_on(WgpuJit, "wgpu-auto");
    }

    #[test]
    fn wgpu_fallback_matches_cpu_on_random_scenes() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip wgpu_fallback_matches_cpu_on_random_scenes: no adapter");
            return;
        }
        let _force = ForceFallback::new();
        parity_on(WgpuJit, "wgpu-fallback");
    }

    #[test]
    fn wgpu_zero_triangles_miss() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip wgpu_zero_triangles_miss: no adapter");
            return;
        }
        zero_triangles_on(WgpuJit, "wgpu");
    }
}

#[cfg(feature = "vulkan")]
mod vulkan_tests {
    use super::*;
    use resin_jit::backends::vulkan::{shared_context_available, VulkanJit};

    #[test]
    fn vulkan_matches_cpu_on_random_scenes() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip vulkan_matches_cpu_on_random_scenes: no Vulkan device");
            return;
        }
        parity_on(VulkanJit, "vulkan-auto");
    }

    #[test]
    fn vulkan_fallback_matches_cpu_on_random_scenes() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip vulkan_fallback_matches_cpu_on_random_scenes: no Vulkan device");
            return;
        }
        let _force = ForceFallback::new();
        parity_on(VulkanJit, "vulkan-fallback");
    }

    #[test]
    fn vulkan_zero_triangles_miss() {
        let _lock = lock_trace_mode();
        if !shared_context_available() {
            eprintln!("skip vulkan_zero_triangles_miss: no Vulkan device");
            return;
        }
        zero_triangles_on(VulkanJit, "vulkan");
    }
}
