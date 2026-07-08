//! MNIST MLP training (Python `demo_mnist.py` analogue).
//!
//! Traces forward + MSE loss + reverse-mode grad + SGD into one jitted train step.
//! The training loop is backend-agnostic; only `main` picks CPU vs wgpu.
//!
//! Usage:
//!   cargo run --example train_mnist --features cpu
//!   cargo run --example train_mnist --features cpu -- 3
//!   cargo run --example train_mnist --features cpu -- --backend cpu 2
//!   cargo run --example train_mnist --features wgpu -- --backend wgpu 2
//!   cargo run --example train_mnist --features cpu -- --quick 2

use std::env;
use std::path::PathBuf;

use resin::dataset::{BatchIndices, MnistDataset, IMG_WH, NUM_CLS};
use resin::dsl::{grad_wrt, ElementType, Tensor};
use resin::jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

const LR: f32 = 1e-3;
const SEED: u64 = 0;
const BATCH_SIZE: usize = 64;
const HIDDEN: usize = 128;

#[derive(Clone, Copy)]
struct RunConfig {
    epochs: usize,
    batch_size: usize,
    hidden: usize,
    max_batches: Option<usize>,
}

impl RunConfig {
    /// Full MNIST settings (matches Python `demo_mnist.py`).
    fn full(epochs: usize) -> Self {
        Self {
            epochs,
            batch_size: BATCH_SIZE,
            hidden: HIDDEN,
            max_batches: None,
        }
    }

    /// Tiny smoke config for CI / quick iteration.
    fn quick(epochs: usize) -> Self {
        Self {
            epochs,
            batch_size: 8,
            hidden: 32,
            max_batches: Some(4),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackendKind {
    Cpu,
    Wgpu,
}

#[derive(Tree, Clone)]
struct LinearParams<T> {
    weight: T,
    bias: T,
}

#[derive(Tree, Clone)]
struct MlpParams<T> {
    layer0: LinearParams<T>,
    layer1: LinearParams<T>,
    layer2: LinearParams<T>,
}

#[derive(Tree, Clone)]
struct TrainStepIn<T> {
    xs: T,
    ys: T,
    model: MlpParams<T>,
}

#[derive(Tree, Clone)]
struct TrainStepOut<T> {
    loss: T,
    new_model: MlpParams<T>,
}

fn linear(x: &Tensor, layer: &LinearParams<Tensor>) -> Tensor {
    let out = x.matmul(&layer.weight);
    out.clone() + layer.bias.broadcast_to(out.shape(), &[1])
}

fn forward(model: &MlpParams<Tensor>, x: &Tensor) -> Tensor {
    let h0 = linear(x, &model.layer0).relu();
    let h1 = linear(&h0, &model.layer1).relu();
    linear(&h1, &model.layer2)
}

fn mean_all(tensor: &Tensor) -> Tensor {
    // `sum_axes` is keepdims (matches Python resin). Reduce every axis, then
    // squeeze unit dims to a true scalar so `grad_wrt` accepts the loss.
    let count = tensor.shape().iter().product::<usize>() as f32;
    let rank = tensor.shape().len();
    let reduced = if rank == 0 {
        tensor.clone()
    } else {
        let axes: Vec<usize> = (0..rank).collect();
        tensor.sum_axes(&axes).squeeze_all()
    };
    reduced / Tensor::full(&[], count, tensor.element_type())
}

fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.clone() - target.clone();
    mean_all(&(diff.clone() * diff))
}

fn sgd_step(model: &MlpParams<Tensor>, grads: &MlpParams<Tensor>, lr: f32) -> MlpParams<Tensor> {
    let lr = Tensor::full(&[], lr, ElementType::F32);
    MlpParams {
        layer0: LinearParams {
            weight: model.layer0.weight.clone() - grads.layer0.weight.clone() * lr.clone(),
            bias: model.layer0.bias.clone() - grads.layer0.bias.clone() * lr.clone(),
        },
        layer1: LinearParams {
            weight: model.layer1.weight.clone() - grads.layer1.weight.clone() * lr.clone(),
            bias: model.layer1.bias.clone() - grads.layer1.bias.clone() * lr.clone(),
        },
        layer2: LinearParams {
            weight: model.layer2.weight.clone() - grads.layer2.weight.clone() * lr.clone(),
            bias: model.layer2.bias.clone() - grads.layer2.bias.clone() * lr,
        },
    }
}

fn trace_train_step(step: &TrainStepIn<Tensor>) -> TrainStepOut<Tensor> {
    let pred = forward(&step.model, &step.xs);
    let loss = mse_loss(&pred, &step.ys);
    let grads = grad_wrt(&loss, &step.model).expect("differentiable train step");
    TrainStepOut {
        loss,
        new_model: sgd_step(&step.model, &grads, LR),
    }
}

/// Match Python `random.uniform(-0.1, 0.1)` for every element (weights and biases).
fn random_model<J: Jit>(hidden: usize, seed: u64) -> MlpParams<J::Tensor> {
    let mut rng = seed;
    fn layer<T: ConcreteTensor>(
        in_dim: usize,
        out_dim: usize,
        rng: &mut u64,
    ) -> LinearParams<T> {
        let w: Vec<f32> = (0..in_dim * out_dim).map(|_| next_uniform(rng)).collect();
        let b: Vec<f32> = (0..out_dim).map(|_| next_uniform(rng)).collect();
        LinearParams {
            weight: T::from_f32(&[in_dim, out_dim], &w),
            bias: T::from_f32(&[out_dim], &b),
        }
    }
    MlpParams {
        layer0: layer(IMG_WH, hidden, &mut rng),
        layer1: layer(hidden, hidden, &mut rng),
        layer2: layer(hidden, NUM_CLS, &mut rng),
    }
}

/// LCG unit float in [0, 1), then map to [-0.1, 0.1).
fn next_uniform(rng: &mut u64) -> f32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u = ((*rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32));
    u * 0.2 - 0.1
}

fn mnist_cache_dir() -> PathBuf {
    env::var_os("RESIN_MNIST_DIR")
        .map(PathBuf::from)
        .or_else(|| {
            env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/resin/mnist"))
        })
        .unwrap_or_else(|| PathBuf::from("target/mnist"))
}

/// Backend-agnostic train loop (Python epoch printout).
fn train_mnist<J: Jit>(jit: J, config: RunConfig) {
    let cache = mnist_cache_dir();
    // Match older Rust/Python demo loading lines.
    eprintln!("loading MNIST train from {} …", cache.display());
    let dataset = MnistDataset::load("train", &cache).expect("load MNIST");
    eprintln!("loaded {} samples", dataset.n);

    let train = jit.jit(trace_train_step);
    let mut model = random_model::<J>(config.hidden, SEED);
    let mut batches = BatchIndices::new(dataset.n, config.batch_size, SEED, true);

    for epoch in 0..config.epochs {
        batches.rewind(SEED + u64::from(epoch as u32));
        // Python: one line per epoch = loss of the last full batch.
        let mut loss_value = f32::NAN;
        let mut batch_idx = 0usize;

        while let Some(indices) = batches.next_batch() {
            if config
                .max_batches
                .is_some_and(|limit| batch_idx >= limit)
            {
                break;
            }

            let (xs, ys) = dataset.batch_f32(indices);
            let step = TrainStepIn {
                xs: J::Tensor::from_f32(&[config.batch_size, IMG_WH], &xs),
                ys: J::Tensor::from_f32(&[config.batch_size, NUM_CLS], &ys),
                model: model.clone(),
            };
            let out = train.call(&step).expect("train step");
            loss_value = out.loss.scalar_f32();
            model = out.new_model;
            batch_idx += 1;
        }

        eprintln!("epoch {epoch}: loss={loss_value:.6}");
    }
}

fn parse_args() -> (RunConfig, BackendKind) {
    let mut epochs: Option<usize> = None;
    let mut quick = false;
    let mut backend = BackendKind::Cpu;
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => {
                quick = true;
                if let Some(n) = args.next().and_then(|s| s.parse().ok()) {
                    epochs = Some(n);
                }
            }
            "--full" => {
                quick = false;
                if let Some(n) = args.next().and_then(|s| s.parse().ok()) {
                    epochs = Some(n);
                }
            }
            "--backend" => {
                let value = args
                    .next()
                    .unwrap_or_else(|| panic!("--backend requires cpu or wgpu"));
                backend = match value.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "wgpu" | "gpu" => BackendKind::Wgpu,
                    other => panic!("unknown backend {other:?}; expected cpu or wgpu"),
                };
            }
            flag if flag.starts_with('-') => {
                panic!("unknown flag {flag}; try --backend cpu|wgpu, --quick [N], or EPOCHS");
            }
            epochs_str => {
                epochs = Some(epochs_str.parse().unwrap_or_else(|_| {
                    panic!("invalid epochs {epochs_str:?}");
                }));
            }
        }
    }

    let epochs = epochs.unwrap_or(2);
    let config = if quick {
        RunConfig::quick(epochs)
    } else {
        RunConfig::full(epochs)
    };
    (config, backend)
}

fn main() {
    let (config, backend) = parse_args();

    match backend {
        #[cfg(feature = "cpu")]
        BackendKind::Cpu => {
            use resin::jit::backends::cpu::CpuJit;
            train_mnist(CpuJit, config);
        }
        #[cfg(not(feature = "cpu"))]
        BackendKind::Cpu => {
            panic!("CPU backend requested but resin was built without the `cpu` feature");
        }
        #[cfg(feature = "wgpu")]
        BackendKind::Wgpu => {
            use resin::jit::backends::wgpu::WgpuJit;
            train_mnist(WgpuJit, config);
        }
        #[cfg(not(feature = "wgpu"))]
        BackendKind::Wgpu => {
            panic!("wgpu backend requested but resin was built without the `wgpu` feature");
        }
    }
}
