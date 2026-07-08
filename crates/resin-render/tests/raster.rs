//! Deferred-raster tests: the vertex transform against the host camera, a
//! Cornell-box CPU golden render, GPU image parity, and gradient flow to
//! materials (`docs/hw-nodes.md`).

use resin_core::hw::{raster_channel, RASTER_PIXEL_WIDTH};
use resin_dsl::{grad_wrt, Tensor};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;
use resin_render::raster::{clip_transform, shade_lambert, visibility};
use resin_render::{cornell_box, Camera, Scene};

const LIGHT_DIR: [f32; 3] = [0.3, -1.0, 0.25];
const AMBIENT: f32 = 0.25;

#[derive(Tree)]
struct PosIn<T> {
    positions: T,
}

#[derive(Tree)]
struct AlbedoIn<T> {
    albedo: T,
}

#[test]
fn clip_transform_matches_camera_to_clip() {
    let scene = cornell_box();
    let camera = Camera::cornell_default(24, 16);
    let view_proj = camera.view_proj();
    let f = CpuJit.jit(move |input: &PosIn<Tensor>| clip_transform(&input.positions, &view_proj));
    let clip = f
        .call(&PosIn {
            positions: CpuTensor::from_f32(&[scene.vertex_count(), 3], &scene.vertices),
        })
        .expect("clip transform")
        .to_f32();

    let host = camera.to_clip(&scene.vertices);
    assert_eq!(clip.len(), host.len());
    for (i, (a, b)) in clip.iter().zip(host.iter()).enumerate() {
        assert!((a - b).abs() < 1e-4, "clip[{i}]: graph {a} vs host {b}");
    }
}

/// Visibility + shaded image through one jitted function, materials as
/// parameters (mirrors `raster::render` but also returns the visibility
/// buffer so parity tests can mask triangle edges).
fn render_with_vis<J: Jit>(jit: J, scene: &Scene, camera: &Camera) -> (Vec<f32>, Vec<f32>) {
    #[derive(Tree)]
    struct Materials<T> {
        albedo: T,
        emission: T,
    }

    let triangle_count = scene.triangle_count();
    let vertex_data = scene.vertices.clone();
    let triangle_data = scene.triangles.clone();
    let camera = *camera;
    let f = jit.jit(move |materials: &Materials<Tensor>| {
        let vertices = Tensor::constant_f32(&[vertex_data.len() / 3, 3], &vertex_data);
        let triangles = Tensor::constant_u32(&[triangle_data.len() / 3, 3], &triangle_data);
        let vis = visibility(&vertices, &triangles, &camera);
        let rgb = shade_lambert(
            &vis,
            &vertices,
            &triangles,
            &materials.albedo,
            &materials.emission,
            LIGHT_DIR,
            AMBIENT,
        );
        vec![vis, rgb]
    });
    let out = f
        .call(&Materials {
            albedo: J::Tensor::from_f32(&[triangle_count, 3], &scene.albedo),
            emission: J::Tensor::from_f32(&[triangle_count, 3], &scene.emission),
        })
        .expect("raster render");
    (out[0].to_f32(), out[1].to_f32())
}

fn mean_channel(image: &[f32], w: usize, columns: std::ops::Range<usize>, channel: usize) -> f32 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (pixel, rgb) in image.chunks_exact(3).enumerate() {
        if columns.contains(&(pixel % w)) {
            sum += rgb[channel];
            count += 1;
        }
    }
    sum / count as f32
}

#[test]
fn cornell_cpu_render_has_expected_layout() {
    let (h, w) = (48, 48);
    let scene = cornell_box();
    let camera = Camera::cornell_default(w, h);
    let image =
        resin_render::raster::render(CpuJit, &scene, &camera, LIGHT_DIR, AMBIENT);
    assert_eq!(image.len(), h * w * 3);

    // Red wall on the left, green wall on the right.
    let left_red = mean_channel(&image, w, 0..w / 3, 0);
    let left_green = mean_channel(&image, w, 0..w / 3, 1);
    assert!(left_red > left_green, "left third: red {left_red} vs green {left_green}");
    let right_red = mean_channel(&image, w, w - w / 3..w, 0);
    let right_green = mean_channel(&image, w, w - w / 3..w, 1);
    assert!(
        right_green > right_red,
        "right third: red {right_red} vs green {right_green}"
    );

    // The emissive ceiling panel (emission 15) is visible from below.
    assert!(
        image.iter().any(|&v| v > 5.0),
        "no bright emissive pixels found"
    );

    // Flat shading varies across differently oriented white surfaces:
    // moderate-luminance (non-emissive, lit) pixels are not all one value.
    let lit: Vec<f32> = image
        .chunks_exact(3)
        .map(|rgb| rgb[0] + rgb[1] + rgb[2])
        .filter(|&lum| lum > 0.1 && lum < 3.0)
        .collect();
    let mean = lit.iter().sum::<f32>() / lit.len() as f32;
    let variance = lit.iter().map(|l| (l - mean) * (l - mean)).sum::<f32>() / lit.len() as f32;
    assert!(
        variance.sqrt() > 0.05,
        "expected shading variation, stddev = {}",
        variance.sqrt()
    );
}

/// Pixels whose 3×3 in-image neighborhood carries one `(prim, hit)` value.
fn interior_mask(vis: &[f32], h: usize, w: usize) -> Vec<bool> {
    let key = |r: usize, c: usize| {
        let base = (r * w + c) * RASTER_PIXEL_WIDTH;
        (
            vis[base + raster_channel::PRIM],
            vis[base + raster_channel::HIT],
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

/// Compare shaded images away from triangle edges (coverage rules differ
/// there); shading itself is flat per triangle, so interior pixels must
/// agree tightly.
fn assert_image_parity<J: Jit>(jit: J) {
    let (h, w) = (48, 48);
    let scene = cornell_box();
    let camera = Camera::cornell_default(w, h);
    let (cpu_vis, cpu_rgb) = render_with_vis(CpuJit, &scene, &camera);
    let (gpu_vis, gpu_rgb) = render_with_vis(jit, &scene, &camera);

    let cpu_interior = interior_mask(&cpu_vis, h, w);
    let gpu_interior = interior_mask(&gpu_vis, h, w);
    let mut compared = 0usize;
    let mut total_diff = 0.0f32;
    for i in 0..h * w {
        if !(cpu_interior[i] && gpu_interior[i]) {
            continue;
        }
        compared += 1;
        for channel in 0..3 {
            let a = cpu_rgb[i * 3 + channel];
            let b = gpu_rgb[i * 3 + channel];
            total_diff += (a - b).abs();
            assert!(
                (a - b).abs() < 0.02,
                "channel {channel} at ({}, {}): cpu {a} vs gpu {b}",
                i / w,
                i % w
            );
        }
    }
    assert!(
        compared * 2 > h * w,
        "only {compared}/{} pixels away from edges in both images",
        h * w
    );
    let mean_diff = total_diff / (compared * 3) as f32;
    assert!(mean_diff < 0.02, "mean abs diff {mean_diff}");
}

#[test]
fn wgpu_render_matches_cpu() {
    if !resin_jit::backends::wgpu::shared_context_available() {
        eprintln!("skip wgpu_render_matches_cpu: no GPU adapter");
        return;
    }
    assert_image_parity(resin_jit::backends::wgpu::WgpuJit);
}

#[test]
fn vulkan_render_matches_cpu() {
    if !resin_jit::backends::vulkan::shared_context_available() {
        eprintln!("skip vulkan_render_matches_cpu: no Vulkan device");
        return;
    }
    assert_image_parity(resin_jit::backends::vulkan::VulkanJit);
}

#[test]
fn albedo_gradients_flow_through_shading() {
    let (h, w) = (24, 24);
    let scene = cornell_box();
    let camera = Camera::cornell_default(w, h);
    let triangle_count = scene.triangle_count();
    let vertex_data = scene.vertices.clone();
    let triangle_data = scene.triangles.clone();
    let emission_data = scene.emission.clone();

    let f = CpuJit.jit(move |p: &AlbedoIn<Tensor>| {
        let vertices = Tensor::constant_f32(&[vertex_data.len() / 3, 3], &vertex_data);
        let triangles = Tensor::constant_u32(&[triangle_data.len() / 3, 3], &triangle_data);
        let emission = Tensor::constant_f32(&[emission_data.len() / 3, 3], &emission_data);
        let vis = visibility(&vertices, &triangles, &camera);
        let image = shade_lambert(
            &vis,
            &vertices,
            &triangles,
            &p.albedo,
            &emission,
            LIGHT_DIR,
            AMBIENT,
        );
        let loss = image.sum_axes(&[0, 1, 2]).squeeze_all();
        let grads = grad_wrt(&loss, &p.albedo).expect("differentiable materials");
        vec![loss, grads]
    });

    let out = f
        .call(&AlbedoIn {
            albedo: CpuTensor::from_f32(&[triangle_count, 3], &scene.albedo),
        })
        .expect("grad render");
    assert!(out[0].scalar_f32() > 0.0, "image should not be black");
    let grads = out[1].to_f32();
    assert_eq!(grads.len(), triangle_count * 3);
    assert!(grads.iter().all(|g| g.is_finite()), "non-finite gradient");
    // d loss / d albedo = Σ shade·hit over the triangle's pixels: nonzero for
    // every visible triangle.
    assert!(grads.iter().any(|&g| g > 0.0), "no gradient reached albedo");
}
