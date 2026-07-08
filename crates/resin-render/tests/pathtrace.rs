//! Path tracer tests: CPU golden vs the scalar reference, GPU image parity,
//! and differentiability w.r.t. materials (`docs/hw-nodes.md`). Sizes stay
//! tiny — lavapipe executes the GPU paths in software.

use resin_dsl::{grad_wrt, Tensor};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;
use resin_render::pathtrace::{radiance_graph, render, render_reference, PathTracerConfig};
use resin_render::{cornell_box, Camera, Scene};

fn config() -> PathTracerConfig {
    PathTracerConfig {
        bounces: 3,
        seed: 42,
    }
}

/// The jitted CPU renderer and the scalar reference share semantics and the
/// random stream, so they agree to float noise.
#[test]
fn cpu_render_matches_reference() {
    let scene = cornell_box();
    let camera = Camera::cornell_default(8, 8);
    let config = config();
    let jitted = render(CpuJit, &scene, &camera, 2, &config);
    let reference = render_reference(&scene, &camera, 2, &config);
    assert_eq!(jitted.len(), reference.len());
    let mut worst = 0.0f32;
    for (index, (a, b)) in jitted.iter().zip(&reference).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "value {index}: jit {a} vs reference {b} (diff {diff})"
        );
        worst = worst.max(diff);
    }
    assert!(
        reference.iter().any(|&v| v > 0.0),
        "image must not be black (worst diff {worst})"
    );
}

/// Color bleeding sanity: the red wall tints the left of the image, the
/// green wall the right.
#[test]
fn cornell_box_tints_left_red_and_right_green() {
    let (w, h) = (16, 16);
    let scene = cornell_box();
    let camera = Camera::cornell_default(w, h);
    let image = render(CpuJit, &scene, &camera, 4, &config());

    let mean_channel = |cols: std::ops::Range<usize>, channel: usize| -> f32 {
        let mut sum = 0.0;
        let mut count = 0;
        for row in 0..h {
            for col in cols.clone() {
                sum += image[(row * w + col) * 3 + channel];
                count += 1;
            }
        }
        sum / count as f32
    };

    let left_red = mean_channel(0..w / 3, 0);
    let left_green = mean_channel(0..w / 3, 1);
    let right_red = mean_channel(w - w / 3..w, 0);
    let right_green = mean_channel(w - w / 3..w, 1);
    assert!(left_red > 0.0, "left third is lit");
    assert!(
        left_red > left_green,
        "left third: red {left_red} vs green {left_green}"
    );
    assert!(
        right_green > right_red,
        "right third: green {right_green} vs red {right_red}"
    );
}

/// GPU renders differ from CPU only where hardware watertightness picks a
/// different triangle at shared edges, so image means stay close.
fn assert_image_parity<J: Jit>(jit: J, label: &str) {
    let scene = cornell_box();
    let camera = Camera::cornell_default(16, 16);
    let config = PathTracerConfig {
        bounces: 2,
        seed: 7,
    };
    let cpu = render(CpuJit, &scene, &camera, 2, &config);
    let gpu = render(jit, &scene, &camera, 2, &config);
    let mean_abs = cpu
        .iter()
        .zip(&gpu)
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / cpu.len() as f32;
    assert!(
        gpu.iter().any(|&v| v > 0.0),
        "{label}: image must not be black"
    );
    assert!(mean_abs < 2e-3, "{label}: mean abs diff {mean_abs}");
}

#[test]
fn wgpu_image_matches_cpu() {
    use resin_jit::backends::wgpu::{shared_context_available, WgpuJit};
    if !shared_context_available() {
        eprintln!("skip wgpu_image_matches_cpu: no adapter");
        return;
    }
    assert_image_parity(WgpuJit, "wgpu");
}

#[test]
fn vulkan_image_matches_cpu() {
    use resin_jit::backends::vulkan::{shared_context_available, VulkanJit};
    if !shared_context_available() {
        eprintln!("skip vulkan_image_matches_cpu: no Vulkan device");
        return;
    }
    assert_image_parity(VulkanJit, "vulkan");
}

//
// Differentiability
//

#[derive(Tree)]
struct GradIn<T> {
    albedo: T,
    locals: Vec<T>,
}

/// Jitted `(loss, dloss/dalbedo)` for a fixed scene/rays, with albedo and
/// the per-bounce locals as parameters.
fn loss_and_grad(
    scene: &Scene,
    origins: Vec<f32>,
    directions: Vec<f32>,
    albedo: &[f32],
    locals: &[Vec<f32>],
) -> (f32, Vec<f32>) {
    let n = origins.len() / 3;
    let triangle_count = scene.triangle_count();
    let vertices = scene.vertices.clone();
    let triangles = scene.triangles.clone();
    let emission = scene.emission.clone();

    let f = CpuJit.jit(move |input: &GradIn<Tensor>| {
        let v = Tensor::constant_f32(&[vertices.len() / 3, 3], &vertices);
        let t = Tensor::constant_u32(&[triangles.len() / 3, 3], &triangles);
        let e = Tensor::constant_f32(&[emission.len() / 3, 3], &emission);
        let o = Tensor::constant_f32(&[origins.len() / 3, 3], &origins);
        let d = Tensor::constant_f32(&[directions.len() / 3, 3], &directions);
        let radiance = radiance_graph(&v, &t, &input.albedo, &e, &o, &d, &input.locals);
        let loss = radiance.sum_axes(&[0, 1]).squeeze_all();
        let grad = grad_wrt(&loss, &input.albedo).expect("differentiable in albedo");
        vec![loss, grad]
    });

    let out = f
        .call(&GradIn {
            albedo: CpuTensor::from_f32(&[triangle_count, 3], albedo),
            locals: locals
                .iter()
                .map(|l| CpuTensor::from_f32(&[n, 3], l))
                .collect(),
        })
        .expect("loss + grad");
    (out[0].scalar_f32(), out[1].to_f32())
}

/// Two-triangle floor lit by an emissive panel; rays point straight down and
/// bounce straight up (fixed locals), making the loss linear in the floor
/// albedo.
fn lit_floor_scene() -> Scene {
    let mut scene = Scene::empty();
    scene.push_quad(
        [
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
        ],
        [0.8, 0.5, 0.3],
        [0.0, 0.0, 0.0],
    );
    scene.push_quad(
        [
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        [0.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    );
    scene
}

fn down_rays() -> (Vec<f32>, Vec<f32>) {
    let mut origins = Vec::new();
    let mut directions = Vec::new();
    for (x, z) in [(-0.3, -0.3), (0.3, -0.3), (-0.3, 0.3), (0.3, 0.3)] {
        origins.extend([x, 0.5, z]);
        directions.extend([0.0, -1.0, 0.0]);
    }
    (origins, directions)
}

#[test]
fn albedo_gradients_are_finite_and_nonzero() {
    let scene = cornell_box();
    let camera = Camera::cornell_default(8, 8);
    let (origins, directions) = camera.primary_rays();
    let n = origins.len() / 3;
    // Fixed but non-axis-aligned locals for two bounces.
    let locals: Vec<Vec<f32>> = (0..2)
        .map(|bounce| {
            (0..n)
                .flat_map(|ray| {
                    let spread = 0.1 + 0.05 * ((ray + bounce) % 7) as f32;
                    let up = (1.0f32 - 2.0 * spread * spread).max(0.0).sqrt();
                    [spread, spread, up]
                })
                .collect()
        })
        .collect();
    let (loss, grad) = loss_and_grad(
        &scene,
        origins,
        directions,
        &scene.albedo,
        &locals,
    );
    assert!(loss.is_finite() && loss > 0.0, "loss = {loss}");
    assert!(grad.iter().all(|g| g.is_finite()), "finite gradients");
    assert!(
        grad.iter().any(|&g| g != 0.0),
        "some gradient reaches the albedo table"
    );
}

#[test]
fn albedo_gradient_matches_finite_difference() {
    let scene = lit_floor_scene();
    let (origins, directions) = down_rays();
    let n = origins.len() / 3;
    // Bounce straight along the normal; the second bounce's draw is unused
    // (last bounce) but the graph still takes it as a parameter.
    let straight_up: Vec<f32> = vec![[0.0f32, 0.0, 1.0]; n].concat();
    let locals = vec![straight_up.clone(), straight_up];

    let (_, grad) = loss_and_grad(
        &scene,
        origins.clone(),
        directions.clone(),
        &scene.albedo,
        &locals,
    );

    // Central difference on the red channel of floor triangle 0.
    let step = 1e-3;
    let channel = 0;
    let mut plus = scene.albedo.clone();
    plus[channel] += step;
    let mut minus = scene.albedo.clone();
    minus[channel] -= step;
    let (loss_plus, _) = loss_and_grad(&scene, origins.clone(), directions.clone(), &plus, &locals);
    let (loss_minus, _) = loss_and_grad(&scene, origins, directions, &minus, &locals);
    let numeric = (loss_plus - loss_minus) / (2.0 * step);
    let analytic = grad[channel];
    assert!(
        numeric.abs() > 1e-3,
        "finite difference must see the channel (numeric {numeric})"
    );
    let relative = (analytic - numeric).abs() / numeric.abs();
    assert!(
        relative < 0.2,
        "grad {analytic} vs finite difference {numeric} (relative {relative})"
    );
}
