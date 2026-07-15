//! MNIST MLP training.
//!
//! One jitted train step (forward + MSE + reverse-mode grad + SGD). Compile
//! once from abstract parameter shapes; minibatches collate on the host,
//! upload batch leaves, and keep the model as device arrays. Loss is read
//! with `.scalar()` once per epoch. Backend-agnostic; `main` picks CPU/wgpu.
//!
//! Usage:
//!   cargo run --example train_mnist
//!   cargo run --example train_mnist -- 3
//!   cargo run --example train_mnist -- --backend wgpu 2
//!   cargo run --example train_mnist -- --quick 2

use std::env;

use resin::Tree;
use resin::dsl::{Tensor, grad_wrt};
use resin::jit::{DeviceArray, HostArray, Jit};
use resin_extras::dataset::mnist::{self, Mnist};
use resin_extras::dataset::{Dataset, Split};
use resin_extras::sampler::{IndexSampler, Sampler};

const LR: f32 = 1e-3;
const SEED: u64 = 0;

#[derive(Clone, Copy)]
struct RunConfig {
    epochs: usize,
    batch_size: usize,
    hidden: usize,
    max_batches: Option<usize>,
}

impl RunConfig {
    fn full(epochs: usize) -> Self {
        Self {
            epochs,
            batch_size: 64,
            hidden: 128,
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

fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.clone() - target.clone();
    (diff.clone() * diff).mean_all()
}

fn sgd_step(model: &MlpParams<Tensor>, grads: &MlpParams<Tensor>, lr: f32) -> MlpParams<Tensor> {
    let lr = Tensor::scalar(lr);
    let step = |layer: &LinearParams<Tensor>, grad: &LinearParams<Tensor>| LinearParams {
        weight: layer.weight.clone() - grad.weight.clone() * lr.clone(),
        bias: layer.bias.clone() - grad.bias.clone() * lr.clone(),
    };
    MlpParams {
        layer0: step(&model.layer0, &grads.layer0),
        layer1: step(&model.layer1, &grads.layer1),
        layer2: step(&model.layer2, &grads.layer2),
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

/// Abstract inputs for one train step — shapes only, for [`JittedFn::compile`].
fn train_step_shapes(batch_size: usize, hidden: usize) -> TrainStepIn<Tensor> {
    let layer = |in_dim: usize, out_dim: usize| LinearParams {
        weight: Tensor::parameter(&[in_dim, out_dim]),
        bias: Tensor::parameter(&[out_dim]),
    };
    TrainStepIn {
        xs: Tensor::parameter(&[batch_size, mnist::IMG_WH]),
        ys: Tensor::parameter(&[batch_size, mnist::CLASSES.len()]),
        model: MlpParams {
            layer0: layer(mnist::IMG_WH, hidden),
            layer1: layer(hidden, hidden),
            layer2: layer(hidden, mnist::CLASSES.len()),
        },
    }
}

fn random_model(hidden: usize, seed: u64) -> MlpParams<HostArray> {
    let mut rng = seed;
    let mut layer = |in_dim: usize, out_dim: usize| LinearParams {
        weight: HostArray::from_f32(
            &[in_dim, out_dim],
            &(0..in_dim * out_dim)
                .map(|_| next_uniform(&mut rng))
                .collect::<Vec<_>>(),
        ),
        bias: HostArray::from_f32(
            &[out_dim],
            &(0..out_dim)
                .map(|_| next_uniform(&mut rng))
                .collect::<Vec<_>>(),
        ),
    };
    MlpParams {
        layer0: layer(mnist::IMG_WH, hidden),
        layer1: layer(hidden, hidden),
        layer2: layer(hidden, mnist::CLASSES.len()),
    }
}

/// LCG unit float in [0, 1), mapped to [-0.1, 0.1).
fn next_uniform(rng: &mut u64) -> f32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    let u = ((*rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32));
    u * 0.2 - 0.1
}

/// Host collate: keys → normalized images + one-hot labels (train-step policy).
fn collate_batch(dataset: &Mnist, keys: &[usize]) -> (HostArray, HostArray) {
    let b = keys.len();
    let mut xs = vec![0f32; b * mnist::IMG_WH];
    let mut ys = vec![0f32; b * mnist::CLASSES.len()];
    for (row, key) in keys.iter().enumerate() {
        let ex = dataset.get(key);
        let x_base = row * mnist::IMG_WH;
        for (j, &px) in ex.image.iter().enumerate() {
            xs[x_base + j] = px as f32 / 255.0;
        }
        ys[row * mnist::CLASSES.len() + ex.label as usize] = 1.0;
    }
    (
        HostArray::from_f32(&[b, mnist::IMG_WH], &xs),
        HostArray::from_f32(&[b, mnist::CLASSES.len()], &ys),
    )
}

fn train_mnist<J: Jit>(jit: J, config: RunConfig) {
    eprintln!("loading MNIST train …");
    let dataset = Mnist::load(Split::Train).expect("load MNIST");
    eprintln!("loaded {} samples", dataset.len());

    // Clone: `jit()` consumes self; we still need `upload` on the original.
    let train = jit.clone().jit(trace_train_step);
    // Trace once from shapes — no dataset/sampler involvement.
    train
        .compile(&train_step_shapes(config.batch_size, config.hidden))
        .expect("compile train step");

    // Model stays device-local across minibatches; loss is `.scalar()`'d once per epoch.
    let mut model = jit
        .upload_tree(&random_model(config.hidden, SEED))
        .expect("upload model");
    let mut sampler = IndexSampler::new(dataset.len(), config.batch_size, SEED, true);

    for epoch in 0..config.epochs {
        sampler.reset(SEED + epoch as u64);
        let mut last_loss: Option<J::Array> = None;
        let mut batch_index = 0;
        let t0 = std::time::Instant::now();

        while let Some(keys) = sampler.next_batch_keys() {
            if config.max_batches.is_some_and(|limit| batch_index >= limit) {
                break;
            }
            let (xs, ys) = collate_batch(&dataset, keys);
            // Same shapes → cache hit; upload batch leaves + invoke only.
            let out = train
                .call(&TrainStepIn {
                    xs: jit.upload(&xs).expect("upload xs"),
                    ys: jit.upload(&ys).expect("upload ys"),
                    model,
                })
                .expect("train step");
            model = out.new_model;
            last_loss = Some(out.loss);
            batch_index += 1;
        }

        let loss = last_loss
            .map(|l| l.scalar().expect("loss scalar"))
            .unwrap_or(f32::NAN);
        eprintln!(
            "epoch {epoch}: loss={loss:.6}  {batch_index} steps in {:.2?} (last-batch loss; host once/epoch)",
            t0.elapsed()
        );
    }
}

fn parse_args() -> (RunConfig, String) {
    let mut epochs: Option<usize> = None;
    let mut quick = false;
    let mut backend = "cpu".to_string();
    let mut args = env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--quick" => quick = true,
            "--full" => quick = false,
            "--backend" => {
                backend = args.next().expect("--backend requires cpu or wgpu");
            }
            flag if flag.starts_with('-') => {
                panic!("unknown flag {flag}; try --backend cpu|wgpu, --quick, or EPOCHS");
            }
            n => epochs = Some(n.parse().unwrap_or_else(|_| panic!("invalid epochs {n:?}"))),
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
    match backend.as_str() {
        "cpu" => train_mnist(resin::jit::CpuJit, config),
        #[cfg(feature = "wgpu")]
        "wgpu" | "gpu" => train_mnist(resin::jit::WgpuJit::default(), config),
        other => panic!("unknown backend {other:?} (was resin built with that feature?)"),
    }
}
