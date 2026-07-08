//! End-to-end 3DGS tests on the CPU backend: golden parity against the
//! scalar reference, ordering/culling behavior, and differentiability of the
//! full composed renderer.

use resin_dsl::{grad_wrt, Tensor};
use resin_gaussians::{
    gnomen_cloud, render, render_reference, Camera, CloudData, GaussianCloud,
};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};

fn cloud_tensors(data: &CloudData) -> GaussianCloud<CpuTensor> {
    let n = data.count();
    GaussianCloud {
        means: CpuTensor::from_f32(&[n, 3], &data.means_flat()),
        scales: CpuTensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: CpuTensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: CpuTensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: CpuTensor::from_f32(&[n], &data.opacities),
    }
}

fn render_cpu(data: &CloudData, camera: &Camera) -> Vec<f32> {
    let camera = camera.clone();
    let f = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| render(cloud, &camera));
    f.call(&cloud_tensors(data)).unwrap().to_f32()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn golden_matches_scalar_reference() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(16, 16);
    let got = render_cpu(&data, &camera);
    let expected = render_reference(&data, &camera);
    assert_eq!(got.len(), 16 * 16 * 3);
    let diff = max_abs_diff(&got, &expected);
    assert!(diff < 1e-4, "max abs diff {diff}");
    // The scene must actually put ink on the canvas.
    assert!(expected.iter().any(|&v| v > 0.05), "blank golden image");
}

#[test]
fn golden_matches_on_non_square_image() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(24, 12);
    let diff = max_abs_diff(&render_cpu(&data, &camera), &render_reference(&data, &camera));
    assert!(diff < 1e-4, "max abs diff {diff}");
}

#[test]
fn depth_sort_puts_near_gaussian_in_front() {
    // Two gaussians on the view axis: red near, green far — listed in
    // back-to-front order so a missing sort would show green.
    let data = CloudData {
        means: vec![[0.0, 0.0, -4.0], [0.0, 0.0, -2.0]],
        scales: vec![[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
        quats: vec![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        colors: vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        opacities: vec![0.99, 0.99],
    };
    let camera = Camera::gnomen_default(9, 9);
    let image = render_cpu(&data, &camera);
    let center = (4 * 9 + 4) * 3;
    let (r, g) = (image[center], image[center + 1]);
    assert!(
        r > 5.0 * g.max(1e-6),
        "near red gaussian must dominate: r={r} g={g}"
    );
    // And the composed renderer agrees with the reference on this scene too.
    let diff = max_abs_diff(&image, &render_reference(&data, &camera));
    assert!(diff < 1e-4, "max abs diff {diff}");
}

#[test]
fn gaussian_behind_camera_is_culled() {
    let mut data = gnomen_cloud();
    // A blinding white gaussian *behind* the camera (+Z).
    data.means.push([0.0, 0.0, 3.0]);
    data.scales.push([0.5, 0.5, 0.5]);
    data.quats.push([1.0, 0.0, 0.0, 0.0]);
    data.colors.push([1.0, 1.0, 1.0]);
    data.opacities.push(0.99);

    let camera = Camera::gnomen_default(12, 12);
    let with_ghost = render_cpu(&data, &camera);
    let without = render_cpu(&gnomen_cloud(), &camera);
    let diff = max_abs_diff(&with_ghost, &without);
    assert!(diff < 1e-5, "culled gaussian changed the image by {diff}");
}

/// MSE between a rendered image and a target rendered from different colors.
fn color_loss_graph(
    cloud: &GaussianCloud<Tensor>,
    camera: &Camera,
    target: &[f32],
) -> Tensor {
    let (h, w) = (camera.height, camera.width);
    let image = render(cloud, camera);
    let target = Tensor::constant_f32(&[h, w, 3], target);
    let err = image - target;
    let count = (h * w * 3) as f32;
    (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all()
        * Tensor::full(&[], 1.0 / count, resin_dsl::ElementType::F32)
}

#[test]
fn autodiff_matches_finite_differences_on_color() {
    // Color enters image formation linearly (downstream of every mask), so
    // central differences agree tightly with the composed adjoints.
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(8, 8);
    let target = render_reference(
        &CloudData {
            colors: vec![[0.3, 0.7, 0.2], [0.8, 0.1, 0.4], [0.2, 0.2, 0.9]],
            ..gnomen_cloud()
        },
        &camera,
    );

    // Autodiff gradient w.r.t. colors.
    let cam = camera.clone();
    let t2 = target.clone();
    let grad_fn = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| {
        let loss = color_loss_graph(cloud, &cam, &t2);
        grad_wrt(&loss, &cloud.colors).unwrap()
    });
    let grad = grad_fn.call(&cloud_tensors(&data)).unwrap().to_f32();

    // Finite differences on each color component of gaussian 0.
    let cam = camera.clone();
    let t2 = target.clone();
    let loss_fn =
        CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| color_loss_graph(cloud, &cam, &t2));
    let eps = 1e-3f32;
    for (comp, &ad) in grad.iter().enumerate().take(3) {
        let mut plus = data.clone();
        plus.colors[0][comp] += eps;
        let mut minus = data.clone();
        minus.colors[0][comp] -= eps;
        let lp = loss_fn.call(&cloud_tensors(&plus)).unwrap().scalar_f32();
        let lm = loss_fn.call(&cloud_tensors(&minus)).unwrap().scalar_f32();
        let fd = (lp - lm) / (2.0 * eps);
        assert!(
            (fd - ad).abs() <= 1e-3 * fd.abs().max(1e-3),
            "component {comp}: finite-diff {fd} vs autodiff {ad}"
        );
    }
}

#[test]
fn gradients_flow_to_all_attributes() {
    let data = gnomen_cloud();
    let camera = Camera::gnomen_default(8, 8);
    let target = vec![0.25f32; 8 * 8 * 3];

    let cam = camera.clone();
    let grad_fn = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| {
        let loss = color_loss_graph(cloud, &cam, &target);
        grad_wrt(&loss, cloud).unwrap()
    });
    let grads = grad_fn.call(&cloud_tensors(&data)).unwrap();

    let nonzero_finite = |t: &CpuTensor, name: &str| {
        let v = t.to_f32();
        assert!(v.iter().all(|x| x.is_finite()), "{name}: non-finite grad");
        assert!(v.iter().any(|&x| x != 0.0), "{name}: all-zero grad");
    };
    nonzero_finite(&grads.means, "means");
    nonzero_finite(&grads.scales, "scales");
    nonzero_finite(&grads.colors, "colors");
    nonzero_finite(&grads.opacities, "opacities");
    // Quats of the isotropic gnomen gaussians sit at a symmetry point, so we
    // only require finiteness there.
    assert!(
        grads.quats.to_f32().iter().all(|x| x.is_finite()),
        "quats: non-finite grad"
    );
}

#[test]
fn sgd_on_colors_decreases_loss() {
    let camera = Camera::gnomen_default(8, 8);
    let target_data = gnomen_cloud();
    let target = render_reference(&target_data, &camera);

    // Start from wrong colors; optimize colors only.
    let mut data = gnomen_cloud();
    data.colors = vec![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]];

    let cam = camera.clone();
    let t2 = target.clone();
    let step_fn = CpuJit.jit(move |cloud: &GaussianCloud<Tensor>| {
        let loss = color_loss_graph(cloud, &cam, &t2);
        let grad_colors = grad_wrt(&loss, &cloud.colors).unwrap();
        vec![loss, grad_colors]
    });

    let lr = 200.0f32;
    let mut losses = Vec::new();
    for _ in 0..5 {
        let out = step_fn.call(&cloud_tensors(&data)).unwrap();
        let loss = out[0].scalar_f32();
        let grad = out[1].to_f32();
        losses.push(loss);
        for i in 0..data.colors.len() {
            for c in 0..3 {
                data.colors[i][c] -= lr * grad[i * 3 + c];
            }
        }
    }
    assert!(
        losses.last().unwrap() < &(0.5 * losses[0]),
        "loss did not decrease: {losses:?}"
    );
}
