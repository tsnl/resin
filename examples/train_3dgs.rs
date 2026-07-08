//! 3DGS training demo: recover a scene from rendered target views.
//!
//! The backward pass is `grad_wrt` on the forward renderer — there is no
//! renderer-specific adjoint code. One step program is compiled once; the
//! camera and target are data.
//!
//! ```sh
//! cargo run --example train_3dgs --release -- --epochs 60
//! ```

use std::io::Write;

use resin::dsl::{grad_wrt, ElementType, Tensor};
use resin::gaussians::{
    activate, gnomen_cloud, render_reference, render_view, sgd_step, Camera, RawGaussianCloud,
    SgdRates, TrainStep,
};
use resin::jit::backends::cpu::{CpuJit, CpuTensor};
use resin::jit::{ConcreteTensor, Jit};
use resin::macros::Tree;

#[derive(Debug, Clone, Tree)]
struct TrainScene<T> {
    raw: RawGaussianCloud<T>,
    view: T,
    proj: T,
    target: T,
}

fn main() {
    let mut epochs = 60usize;
    let mut side = 16usize;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--epochs" => epochs = args.next().unwrap().parse().expect("--epochs <int>"),
            "--size" => side = args.next().unwrap().parse().expect("--size <pixels>"),
            other => panic!("unknown argument {other}"),
        }
    }

    // Ground truth and target views.
    let truth = gnomen_cloud();
    let cameras: Vec<Camera> = (0..4)
        .map(|i| {
            Camera::orbit(
                [0.0, 0.0, -2.8],
                2.8,
                (i as f32) * 25.0 - 37.5,
                -5.0,
                [0.0, 1.0, 0.0],
                60.0,
                side,
                side,
            )
        })
        .collect();
    let targets: Vec<Vec<f32>> = cameras
        .iter()
        .map(|c| render_reference(&truth, c))
        .collect();

    // Broken init: gray, displaced, half-transparent.
    let mut init = gnomen_cloud();
    for m in &mut init.means {
        m[0] += 0.06;
        m[1] -= 0.05;
    }
    init.colors = vec![[0.5, 0.5, 0.5]; 3];
    init.opacities = vec![0.5; 3];
    let mut raw = RawGaussianCloud::<Vec<f32>>::from_cloud_data(&init).to_tensors::<CpuTensor>(3);

    // The whole training machinery: forward + grad_wrt, jitted once.
    let step = CpuJit.jit(move |scene: &TrainScene<Tensor>| {
        let cloud = activate(&scene.raw);
        let image = render_view(&cloud, &scene.view, &scene.proj, side, side);
        let err = image - scene.target.clone();
        let loss = (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all()
            * Tensor::full(&[], 1.0 / (side * side * 3) as f32, ElementType::F32);
        TrainStep {
            grads: grad_wrt(&loss, &scene.raw).unwrap(),
            loss,
        }
    });

    let rates = SgdRates {
        means: 5.0,
        log_scales: 50.0,
        quats: 10.0,
        color_logits: 800.0,
        opacity_logits: 800.0,
    };
    for epoch in 0..epochs {
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
        if epoch % 5 == 0 || epoch + 1 == epochs {
            println!("epoch {epoch:>3}  loss {:.6}", epoch_loss / cameras.len() as f32);
        }
    }

    // Render the recovered scene next to the target from the first view.
    let cam = cameras[0].clone();
    let render_fn = CpuJit.jit(move |raw: &RawGaussianCloud<Tensor>| {
        let (view, proj) = cam.matrix_constants();
        render_view(&activate(raw), &view, &proj, side, side)
    });
    let recovered = render_fn.call(&raw).unwrap().to_f32();
    write_ppm("train_3dgs_target.ppm", side, side, &targets[0]);
    write_ppm("train_3dgs_recovered.ppm", side, side, &recovered);
    println!("wrote train_3dgs_target.ppm / train_3dgs_recovered.ppm");
}

fn write_ppm(path: &str, width: usize, height: usize, rgb: &[f32]) {
    let mut out = Vec::with_capacity(rgb.len() + 32);
    out.extend_from_slice(format!("P6\n{width} {height}\n255\n").as_bytes());
    out.extend(
        rgb.iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8),
    );
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(&out))
        .expect("write ppm");
}
