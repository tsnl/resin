//! The training story, end to end: **`grad()` on the forward function is the
//! whole backward pass.** One compiled step program (camera as data),
//! unconstrained raw parameters, host SGD loop — no renderer-specific
//! adjoint code anywhere.

use resin_dsl::{grad_wrt, Tensor};
use resin_gaussians::{
    activate, gnomen_cloud, mse, render_reference, render_view, sgd_step, Camera,
    RawGaussianCloud, SgdRates, TrainStep,
};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

/// Params of one training step: raw cloud + camera matrices (data, so one
/// compiled program serves every view).
#[derive(Debug, Clone, Tree)]
struct TrainScene<T> {
    raw: RawGaussianCloud<T>,
    view: T,
    proj: T,
    target: T,
}

fn orbit_cameras(count: usize, side: usize) -> Vec<Camera> {
    (0..count)
        .map(|i| {
            Camera::orbit(
                [0.0, 0.0, -2.8],
                2.8,
                (i as f32) * 30.0 - 30.0,
                -5.0,
                [0.0, 1.0, 0.0],
                60.0,
                side,
                side,
            )
        })
        .collect()
}

#[test]
fn multi_view_training_recovers_scene() {
    let side = 12usize;
    let cameras = orbit_cameras(3, side);
    let scene_truth = gnomen_cloud();
    let targets: Vec<Vec<f32>> = cameras
        .iter()
        .map(|c| render_reference(&scene_truth, c))
        .collect();

    // Perturbed init: gray colors, nudged positions, wrong opacity.
    let mut init = gnomen_cloud();
    for m in &mut init.means {
        m[0] += 0.05;
        m[1] -= 0.04;
    }
    init.colors = vec![[0.5, 0.5, 0.5]; 3];
    init.opacities = vec![0.6; 3];
    let mut raw =
        RawGaussianCloud::<Vec<f32>>::from_cloud_data(&init).to_tensors::<CpuTensor>(3);

    // The training step: activate -> render -> mse, then grad_wrt on the raw
    // parameter subtree. This is the entire "backward implementation".
    let step = CpuJit.jit(move |scene: &TrainScene<Tensor>| {
        let cloud = activate(&scene.raw);
        let image = render_view(&cloud, &scene.view, &scene.proj, side, side);
        let err = image - scene.target.clone();
        let count = (side * side * 3) as f32;
        let loss = (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all()
            * Tensor::full(&[], 1.0 / count, resin_dsl::ElementType::F32);
        TrainStep {
            grads: grad_wrt(&loss, &scene.raw).unwrap(),
            loss,
        }
    });

    // 3DGS convention: positions step far more gently than logits.
    let rates = SgdRates {
        means: 5.0,
        log_scales: 50.0,
        quats: 10.0,
        color_logits: 800.0,
        opacity_logits: 800.0,
    };
    let mut first_epoch = 0.0f32;
    let mut last_epoch = 0.0f32;
    for epoch in 0..40 {
        let mut epoch_loss = 0.0f32;
        for (camera, target) in cameras.iter().zip(&targets) {
            let out = step
                .call(&TrainScene {
                    raw: raw.clone(),
                    view: CpuTensor::from_f32(&[4, 4], &camera.view_flat()),
                    proj: CpuTensor::from_f32(&[4, 4], &camera.proj_flat()),
                    target: CpuTensor::from_f32(&[side, side, 3], target),
                })
                .unwrap();
            epoch_loss += out.loss.scalar_f32();
            sgd_step(&mut raw, &out.grads, &rates);
        }
        if epoch == 0 {
            first_epoch = epoch_loss;
        }
        last_epoch = epoch_loss;
        assert!(epoch_loss.is_finite(), "loss diverged at epoch {epoch}");
    }
    assert!(
        last_epoch < 0.05 * first_epoch,
        "training did not converge: first epoch {first_epoch}, last epoch {last_epoch}"
    );
}

#[test]
fn raw_parameter_gradients_match_finite_differences() {
    // Through activation (sigmoid) AND the full renderer.
    let side = 6usize;
    let camera = Camera::gnomen_default(side, side);
    let target = render_reference(&gnomen_cloud(), &camera);

    let mut init = gnomen_cloud();
    init.colors = vec![[0.4, 0.6, 0.5]; 3];
    let raw_host = RawGaussianCloud::<Vec<f32>>::from_cloud_data(&init);

    let cam = camera.clone();
    let t2 = target.clone();
    let loss_fn = CpuJit.jit(move |raw: &RawGaussianCloud<Tensor>| {
        let image = render_view(
            &activate(raw),
            &cam.matrix_constants().0,
            &cam.matrix_constants().1,
            side,
            side,
        );
        mse(&image, &t2)
    });
    let cam = camera.clone();
    let t2 = target.clone();
    let grad_fn = CpuJit.jit(move |raw: &RawGaussianCloud<Tensor>| {
        let image = render_view(
            &activate(raw),
            &cam.matrix_constants().0,
            &cam.matrix_constants().1,
            side,
            side,
        );
        grad_wrt(&mse(&image, &t2), raw).unwrap()
    });

    let grads = grad_fn.call(&raw_host.to_tensors::<CpuTensor>(3)).unwrap();
    let ad = grads.opacity_logits.to_f32()[0];

    let eps = 1e-3f32;
    let mut plus = raw_host.clone();
    plus.opacity_logits[0] += eps;
    let mut minus = raw_host.clone();
    minus.opacity_logits[0] -= eps;
    let lp = loss_fn
        .call(&plus.to_tensors::<CpuTensor>(3))
        .unwrap()
        .scalar_f32();
    let lm = loss_fn
        .call(&minus.to_tensors::<CpuTensor>(3))
        .unwrap()
        .scalar_f32();
    let fd = (lp - lm) / (2.0 * eps);
    assert!(
        (fd - ad).abs() <= 2e-2 * fd.abs().max(1e-4),
        "opacity_logit[0]: finite-diff {fd} vs autodiff {ad}"
    );
}
