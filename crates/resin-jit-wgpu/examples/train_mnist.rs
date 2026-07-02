//! MNIST MLP training (Python `demo_mnist` analogue).
//!
//! Compile-time train step: trace forward + loss + backward + SGD into one graph.
//! [`resin_dataset::MinibatchLoader`] yields [`Minibatch`] each step; model buffers are
//! separate GPU-resident state gathered from the pipeline once at setup.

use std::path::PathBuf;

use resin_core::Tree;
use resin_core::F4;
use resin_dataset::{Minibatch, MinibatchLoader, MnistDataset, IMG_WH, NUM_CLS};
use resin_dsl::{param, View};
use resin_grad::grad_wrt;
use resin_jit_wgpu::{compile_open, DeviceConfig, Session, SessionWriter, WgpuBuffer, WgpuSession};
use resin_nn::{
    classifier_layers, cross_entropy, forward_probs, initialize_tree, mean, sgd_tree, InitMethod,
    Layer,
};

const LR: f32 = 1e-3;
const HIDDEN: u32 = 128;
const N_HIDDEN: usize = 2;

const BATCH_SIZE: u32 = 64;
const DEFAULT_EPOCHS: u32 = 10_000;

// --- compile-time train step trees ---

/// Graph inputs at trace time (`View` leaves registered by [`Tree`] path at compile).
#[derive(Clone, Tree)]
struct TrainStepIn<T> {
    minibatch: Minibatch<T>,
    model: Vec<Layer<T>>,
}

impl TrainStepIn<View> {
    fn new(model: &[Layer<View>], batch_size: u32, image_wh: u32, num_cls: u32) -> Self {
        Self {
            minibatch: Minibatch {
                xs: param([batch_size, image_wh], F4),
                ys: param([batch_size, num_cls], F4),
            },
            model: model.to_vec(),
        }
    }

    fn trace(&self) -> Result<TrainStepOut<View>, String> {
        let y_hats = forward_probs(&self.model, &self.minibatch.xs)?;
        let loss = mean(&cross_entropy(&y_hats, &self.minibatch.ys)?)?;
        let grads = grad_wrt(&loss, &self.model)?;
        let new_model = sgd_tree(&self.model, &grads, LR);
        Ok(TrainStepOut { loss, new_model })
    }
}

#[derive(Clone, Tree)]
struct TrainStepOut<T> {
    loss: T,
    new_model: Vec<Layer<T>>,
}

fn init_model_on_gpu(
    session: &WgpuSession,
    template: &Vec<Layer<View>>,
    buffers: &Vec<Layer<WgpuBuffer>>,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let init = initialize_tree(
        template,
        InitMethod::He {
            seed: seed.to_le_bytes().into(),
        },
    );
    for ((_, bytes), (_, buf)) in init.flatten().zip(buffers.flatten()) {
        session.write_buffer(buf, 0, bytes)?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_EPOCHS);
    let seed: u64 = 0;

    let dataset = load_mnist()?;
    eprintln!("loaded {} MNIST train samples", dataset.n);

    let model = classifier_layers(IMG_WH as u32, NUM_CLS as u32, N_HIDDEN, HIDDEN, true);
    let inputs = TrainStepIn::new(&model, BATCH_SIZE, IMG_WH as u32, NUM_CLS as u32);
    let outputs = inputs.trace()?;

    let mut pipe = compile_open(&inputs, &outputs, DeviceConfig::default(), None)?.create()?;
    let params = pipe.param_tree(&inputs)?;
    let session = pipe.session().clone();

    init_model_on_gpu(&session, &inputs.model, &params.model, seed)?;

    let mut loader =
        MinibatchLoader::new(params.minibatch, dataset.n, BATCH_SIZE as usize, seed, true);
    let writer = SessionWriter::new(&session);
    let model_bufs = params.model;
    let loss_buf = pipe.sink_tree(&outputs)?.loss;

    for epoch in 0..epochs {
        loader.rewind(seed + u64::from(epoch));
        let mut loss = f32::NAN;
        let mut epoch_done = None;
        while let Some(minibatch) = loader.next_minibatch(&dataset, &writer)? {
            let step_in = TrainStepIn {
                minibatch,
                model: model_bufs.clone(),
            };
            pipe.submit(&step_in)?;
            let outs = pipe.sink_tree(&outputs)?;
            pipe.copy_tree(&outs.new_model, &model_bufs)?;
            epoch_done = Some(pipe.flush_binds()?);
        }
        if let Some(done) = epoch_done {
            done.wait();
            loss = session.read_buffer_f32(&loss_buf)?;
        }
        eprintln!("epoch {epoch}: loss={loss:.6}");
    }

    Ok(())
}

fn load_mnist() -> Result<MnistDataset, Box<dyn std::error::Error>> {
    let cache = std::env::var_os("RESIN_MNIST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join(".cache/resin/mnist"))
                .unwrap_or_else(|| PathBuf::from("target/mnist"))
        });
    eprintln!("loading MNIST from {} …", cache.display());
    Ok(MnistDataset::load("train", &cache)?)
}
