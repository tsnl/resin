//! Benchmark harness: composed-graph costs for scan / sort / render /
//! render+grad, plus the chunked checkpoint renderer.
//!
//! For each case we report the *program shape* (dispatch count and total
//! buffer bytes after optimization — backend-independent) and wall-clock
//! timings: first call (trace + compile + run) and steady-state calls
//! (trace + cache hit + run).
//!
//! ```sh
//! cargo run --example bench_3dgs --release -- --backend cpu
//! cargo run --example bench_3dgs --release -- --backend wgpu --reps 20
//! cargo run --example bench_3dgs --release -- --backend cpu --luigi
//! ```

use std::time::Instant;

use resin::dsl::{cumsum, grad_wrt, ElementType, Tensor};
use resin::dsl::sort::argsort_f32;
use resin::gaussians::{
    chunked_renderer, gnomen_cloud, load_ply, render, render_view, Camera, CloudData,
    GaussianCloud, RenderScene,
};
use resin::jit::{ConcreteTensor, Jit};

struct Options {
    backend: String,
    reps: usize,
    luigi: bool,
}

fn main() {
    let mut opts = Options {
        backend: "cpu".into(),
        reps: 5,
        luigi: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => opts.backend = args.next().expect("--backend <cpu|wgpu>"),
            "--reps" => opts.reps = args.next().unwrap().parse().expect("--reps <int>"),
            "--luigi" => opts.luigi = true,
            other => panic!("unknown argument {other}"),
        }
    }

    println!(
        "{:<38} {:>9} {:>12} {:>12} {:>12}",
        "case", "dispatch", "buffer MB", "first ms", "steady ms"
    );

    match opts.backend.as_str() {
        #[cfg(feature = "cpu")]
        "cpu" => run_all(resin::jit::backends::cpu::CpuJit, &opts),
        #[cfg(feature = "wgpu")]
        "wgpu" => {
            assert!(
                resin::jit::backends::wgpu::shared_context_available(),
                "no GPU adapter"
            );
            run_all(resin::jit::backends::wgpu::WgpuJit, &opts)
        }
        other => panic!("backend {other} not available in this build"),
    }
}

fn run_all<J: Jit>(jit: J, opts: &Options) {
    // --- scan -----------------------------------------------------------
    for n in [4_096usize, 65_536] {
        let stats = ir_stats(&[("x", vec![n], ElementType::F32)], |params| {
            vec![cumsum(&params[0], 0)]
        });
        let f = jit.clone().jit(|x: &Tensor| cumsum(x, 0));
        let x = J::Tensor::from_f32(&[n], &vec![1.0; n]);
        let (first, steady) = time_calls(opts.reps, || {
            f.call(&x).unwrap();
        });
        report(&format!("cumsum n={n}"), &stats, first, steady);
    }

    // --- argsort ---------------------------------------------------------
    for n in [4_096usize, 16_384] {
        let stats = ir_stats(&[("k", vec![n], ElementType::F32)], |params| {
            vec![argsort_f32(&params[0])]
        });
        let f = jit.clone().jit(|k: &Tensor| argsort_f32(k));
        let keys: Vec<f32> = (0..n).map(|i| ((i * 2_654_435_761) % n) as f32 - n as f32 / 2.0).collect();
        let k = J::Tensor::from_f32(&[n], &keys);
        let (first, steady) = time_calls(opts.reps, || {
            f.call(&k).unwrap();
        });
        report(&format!("argsort_f32 n={n} (32-bit)"), &stats, first, steady);
    }

    // --- dense render ----------------------------------------------------
    for (n, side) in [(256usize, 64usize), (1_024, 64)] {
        let data = synthetic_cloud(n);
        let camera = Camera::gnomen_default(side, side);
        let stats = render_stats(n, side, side);
        let f = {
            let (w, h) = (side, side);
            jit.clone().jit(move |scene: &RenderScene<Tensor>| {
                render_view(&scene.cloud, &scene.view, &scene.proj, w, h)
            })
        };
        let scene = scene_tensors::<J>(&data, &camera);
        let (first, steady) = time_calls(opts.reps, || {
            f.call(&scene).unwrap();
        });
        report(
            &format!("render dense N={n} {side}x{side}"),
            &stats,
            first,
            steady,
        );
    }

    // --- forward + gradient (training step shape) ------------------------
    {
        let (n, side) = (64usize, 32usize);
        let data = synthetic_cloud(n);
        let camera = Camera::gnomen_default(side, side);
        let target = vec![0.2f32; side * side * 3];
        let stats = ir_stats(
            &[
                ("means", vec![n, 3], ElementType::F32),
                ("scales", vec![n, 3], ElementType::F32),
                ("quats", vec![n, 4], ElementType::F32),
                ("colors", vec![n, 3], ElementType::F32),
                ("opacities", vec![n], ElementType::F32),
            ],
            |params| {
                let cloud = GaussianCloud {
                    means: params[0].clone(),
                    scales: params[1].clone(),
                    quats: params[2].clone(),
                    colors: params[3].clone(),
                    opacities: params[4].clone(),
                };
                let image = render(&cloud, &Camera::gnomen_default(side, side));
                let t = Tensor::constant_f32(&[side, side, 3], &vec![0.2f32; side * side * 3]);
                let err = image - t;
                let loss = (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all();
                let grads = grad_wrt(&loss, &cloud).unwrap();
                vec![
                    loss,
                    grads.means,
                    grads.scales,
                    grads.quats,
                    grads.colors,
                    grads.opacities,
                ]
            },
        );
        let cam = camera.clone();
        let f = jit.clone().jit(move |cloud: &GaussianCloud<Tensor>| {
            let image = render(cloud, &cam);
            let t = Tensor::constant_f32(&[side, side, 3], &target);
            let err = image - t;
            let loss = (err.clone() * err).sum_axes(&[0, 1, 2]).squeeze_all();
            let grads = grad_wrt(&loss, cloud).unwrap();
            vec![
                loss,
                grads.means,
                grads.scales,
                grads.quats,
                grads.colors,
                grads.opacities,
            ]
        });
        let cloud = cloud_tensors::<J>(&data);
        let (first, steady) = time_calls(opts.reps, || {
            f.call(&cloud).unwrap();
        });
        report(
            &format!("render+grad(all) N={n} {side}x{side}"),
            &stats,
            first,
            steady,
        );
    }

    // --- chunked checkpoint ------------------------------------------------
    if opts.luigi {
        let path = std::path::Path::new("data/hf/dylanebert-3dgs/luigi/luigi.ply");
        if !path.exists() {
            eprintln!("--luigi: submodule not initialized, skipping");
            return;
        }
        let data = load_ply(path).unwrap();
        let n = data.count();
        let (side, chunk) = (96usize, 512usize);
        let camera = Camera::orbit(
            [0.0, 0.0, 0.0],
            2.2,
            20.0,
            -10.0,
            [0.0, -1.0, 0.0],
            50.0,
            side,
            side,
        );
        let cloud = cloud_tensors::<J>(&data);
        let renderer = chunked_renderer::<J>(jit.clone(), n, side, side, chunk);
        let (first, steady) = time_calls(opts.reps.min(3), || {
            renderer(&cloud, &camera);
        });
        let stats = Stats {
            dispatches: 0,
            buffer_bytes: 0,
        };
        report(
            &format!("luigi chunked N={n} {side}x{side} K={chunk}"),
            &stats,
            first,
            steady,
        );
    }
}

// --- helpers -------------------------------------------------------------

struct Stats {
    dispatches: usize,
    buffer_bytes: u64,
}

/// Program shape after optimization: trace the graph over placeholder
/// parameters, lower, optimize, count.
fn ir_stats(
    params: &[(&str, Vec<usize>, ElementType)],
    trace: impl Fn(&[Tensor]) -> Vec<Tensor>,
) -> Stats {
    let placeholders: Vec<Tensor> = params
        .iter()
        .map(|(_, shape, etype)| Tensor::parameter(shape, *etype))
        .collect();
    let sinks = trace(&placeholders);
    let program = resin::jit::lower_for_tests(&placeholders, &sinks).expect("lower");
    let optimized = resin::ir::optimize(program);
    let buffer_bytes: u64 = optimized
        .buffers
        .iter()
        .map(|b| {
            let count: u64 = if b.shape.is_empty() {
                1
            } else {
                b.shape.iter().map(|&d| u64::from(d)).product()
            };
            count * b.element_type.nbytes() as u64
        })
        .sum();
    Stats {
        dispatches: optimized.queue.len(),
        buffer_bytes,
    }
}

fn render_stats(n: usize, w: usize, h: usize) -> Stats {
    ir_stats(
        &[
            ("means", vec![n, 3], ElementType::F32),
            ("scales", vec![n, 3], ElementType::F32),
            ("quats", vec![n, 4], ElementType::F32),
            ("colors", vec![n, 3], ElementType::F32),
            ("opacities", vec![n], ElementType::F32),
            ("view", vec![4, 4], ElementType::F32),
            ("proj", vec![4, 4], ElementType::F32),
        ],
        |params| {
            let cloud = GaussianCloud {
                means: params[0].clone(),
                scales: params[1].clone(),
                quats: params[2].clone(),
                colors: params[3].clone(),
                opacities: params[4].clone(),
            };
            vec![render_view(&cloud, &params[5], &params[6], w, h)]
        },
    )
}

fn time_calls(reps: usize, mut call: impl FnMut()) -> (f64, f64) {
    let t0 = Instant::now();
    call();
    let first = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    for _ in 0..reps {
        call();
    }
    let steady = t1.elapsed().as_secs_f64() * 1e3 / reps.max(1) as f64;
    (first, steady)
}

fn report(name: &str, stats: &Stats, first: f64, steady: f64) {
    println!(
        "{:<38} {:>9} {:>12.2} {:>12.1} {:>12.1}",
        name,
        stats.dispatches,
        stats.buffer_bytes as f64 / 1e6,
        first,
        steady
    );
}

fn synthetic_cloud(n: usize) -> CloudData {
    // Deterministic pseudo-random cloud in front of the camera.
    let mut rng = 0x9E3779B9u32;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        (rng as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    let mut data = gnomen_cloud();
    data.means.clear();
    data.scales.clear();
    data.quats.clear();
    data.colors.clear();
    data.opacities.clear();
    for _ in 0..n {
        data.means.push([next() * 1.2, next() * 1.2, -2.5 + next()]);
        data.scales
            .push([0.02 + next().abs() * 0.05; 3]);
        data.quats.push([1.0, next() * 0.2, next() * 0.2, next() * 0.2]);
        data.colors
            .push([next().abs(), next().abs(), next().abs()]);
        data.opacities.push(0.3 + next().abs() * 0.6);
    }
    data
}

fn cloud_tensors<J: Jit>(data: &CloudData) -> GaussianCloud<J::Tensor> {
    let n = data.count();
    GaussianCloud {
        means: J::Tensor::from_f32(&[n, 3], &data.means_flat()),
        scales: J::Tensor::from_f32(&[n, 3], &data.scales_flat()),
        quats: J::Tensor::from_f32(&[n, 4], &data.quats_flat()),
        colors: J::Tensor::from_f32(&[n, 3], &data.colors_flat()),
        opacities: J::Tensor::from_f32(&[n], &data.opacities),
    }
}

fn scene_tensors<J: Jit>(data: &CloudData, camera: &Camera) -> RenderScene<J::Tensor> {
    RenderScene {
        cloud: cloud_tensors::<J>(data),
        view: J::Tensor::from_f32(&[4, 4], &camera.view_flat()),
        proj: J::Tensor::from_f32(&[4, 4], &camera.proj_flat()),
    }
}
