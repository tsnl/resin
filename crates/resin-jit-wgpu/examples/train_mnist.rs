//! MNIST MLP training (Python `demo_mnist` analogue).

use std::path::PathBuf;

use resin_core::{join_param_path, ParamTree, ParamTreePathElement, F4};
use resin_dataset::{BatchIndices, MnistDataset, IMG_WH, NUM_CLS};
use resin_dsl::param;
use resin_grad::grad_wrt;
use resin_ir::IrProgram;
use resin_jit_wgpu::{create_interp, AdmitProgram, Interp, InterpConfig};
use resin_nn::{cross_entropy, mean, sgd_tree, Mlp};

const BATCH_SIZE: u32 = 64;
// Python demo used LR=1e-3 with uniform(±0.1); He init allows the same LR (or
// higher) without dead ReLUs. Keep epoch-end last-batch logging like Python.
const LR: f32 = 1e-3;
const HIDDEN: u32 = 128;
const N_HIDDEN: usize = 2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let epochs: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);
    let seed: u64 = 0;

    let cache = std::env::var_os("RESIN_MNIST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| dirs_next_cache().unwrap_or_else(|| PathBuf::from("target/mnist")));
    eprintln!("loading MNIST train from {} …", cache.display());
    let dataset = MnistDataset::load("train", &cache)?;
    eprintln!("loaded {} samples", dataset.n);

    let mlp = Mlp::new(IMG_WH as u32, NUM_CLS as u32, N_HIDDEN, HIDDEN, true);
    let xs = param([BATCH_SIZE, IMG_WH as u32], F4, "xs");
    let ys = param([BATCH_SIZE, NUM_CLS as u32], F4, "ys");

    let y_hats = mlp.forward(xs.clone())?;
    let losses = cross_entropy(&y_hats, &ys)?;
    let loss = mean(&losses)?;

    // Clone leaves for tree walks (same Arc nodes).
    let grads = grad_wrt(&loss, clone_mlp_views(&mlp))?;
    let new_model = sgd_tree(clone_mlp_views(&mlp), grads, LR);

    let mut ir = IrProgram::new();
    ir.register_param("xs", &xs)?;
    ir.register_param("ys", &ys)?;
    ir.register_param_tree(&mlp, "model")?;
    ir.build_sink("loss", &loss)?;
    ir.build_sink_tree("new_model", &new_model)?;
    ir.seal_params()?;
    let artifact = resin_jit_wgpu::build_wgpu_program(&ir, None);

    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;

    // He/Xavier uniform: weights ~ U(-sqrt(6/fan_in), +sqrt(6/fan_in)), bias 0.
    // (Python used uniform(±0.1) on all leaves; that often starves ReLU grads.)
    let mut rng = seed;
    for (path, view) in mlp.flatten() {
        let name = join_param_path("model", &path);
        let bytes = init_param_bytes(&path, view.shape(), &mut rng);
        interp.write_param(program_id, &name, &bytes)?;
    }

    let batch_size = BATCH_SIZE as usize;
    for epoch in 0..epochs {
        let mut batches = BatchIndices::new(dataset.n, batch_size, seed + u64::from(epoch), true);
        let mut loss_value = f32::NAN;
        while let Some(indices) = batches.next_batch() {
            let (xs_f, ys_f) = dataset.batch_f32(indices);
            interp.write_param(program_id, "xs", &f32_slice_as_bytes(&xs_f))?;
            interp.write_param(program_id, "ys", &f32_slice_as_bytes(&ys_f))?;
            interp.run(program_id)?;

            let loss_bytes = interp.read_sink(program_id, "loss")?;
            loss_value = f32::from_le_bytes(loss_bytes[..4].try_into()?);

            // Commit new_model → model (host copies via read/write).
            for (path, _) in mlp.flatten() {
                let sink_name = join_param_path("new_model", &path);
                let param_name = join_param_path("model", &path);
                let bytes = interp.read_sink(program_id, &sink_name)?;
                interp.write_param(program_id, &param_name, &bytes)?;
            }
        }
        // Python: one line per epoch = loss of the last full batch.
        eprintln!("epoch {epoch}: loss={loss_value:.6}");
    }

    Ok(())
}

fn clone_mlp_views(mlp: &Mlp<resin_dsl::View>) -> Mlp<resin_dsl::View> {
    Mlp {
        layers: mlp
            .layers
            .iter()
            .map(|l| resin_nn::Linear {
                weight: l.weight.clone(),
                bias: l.bias.clone(),
            })
            .collect(),
    }
}

/// He/Xavier-style uniform init. Weight layout is `[out, in]` so `fan_in = shape[1]`.
fn init_param_bytes(path: &[ParamTreePathElement], shape: &[u32], rng: &mut u64) -> Vec<u8> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut out = Vec::with_capacity(n * 4);
    let is_bias = matches!(
        path.last(),
        Some(ParamTreePathElement::Name(s)) if s.as_ref() == "bias"
    );
    if is_bias {
        for _ in 0..n {
            out.extend_from_slice(&0f32.to_le_bytes());
        }
        return out;
    }
    let fan_in = *shape.last().unwrap_or(&1) as f32;
    let scale = (6.0 / fan_in).sqrt();
    for _ in 0..n {
        let u = next_unit(rng); // [0, 1)
        let v = (u * 2.0 - 1.0) * scale;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn next_unit(rng: &mut u64) -> f32 {
    *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32))
}

fn f32_slice_as_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dirs_next_cache() -> Option<PathBuf> {
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache/resin/mnist"))
}
