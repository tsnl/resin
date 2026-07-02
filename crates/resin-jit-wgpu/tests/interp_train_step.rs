//! T5: short train-step contracts on fixed synthetic batches (no download).
//!
//! See `tests/README.md`. S1–S4 pin graph SGD, multi-step stability, multi-batch,
//! and re-run integrity.

#[path = "interp_helpers.rs"]
mod interp_helpers;

use std::collections::BTreeMap;

use interp_helpers::{f32_bytes, max_abs_diff};
use resin_core::{format_param_path, join_param_path, ParamTree, ParamTreePathElement, F4};
use resin_dsl::param;
use resin_grad::grad_wrt;
use resin_ir::IrProgram;
use resin_jit_wgpu::{build_wgpu_program, create_interp, AdmitProgram, Interp, InterpConfig};
use resin_nn::{cross_entropy, mean, sgd_tree, Linear, Mlp};

const LR: f32 = 1e-3;
const CE_FLOOR: f32 = 16.0; // −ln(1e−7) ≈ 16.118; must not collapse here

fn clone_mlp(mlp: &Mlp<resin_dsl::View>) -> Mlp<resin_dsl::View> {
    Mlp {
        layers: mlp
            .layers
            .iter()
            .map(|l| Linear {
                weight: l.weight.clone(),
                bias: l.bias.clone(),
            })
            .collect(),
    }
}

fn clone_linear(l: &Linear<resin_dsl::View>) -> Linear<resin_dsl::View> {
    Linear {
        weight: l.weight.clone(),
        bias: l.bias.clone(),
    }
}

/// Deterministic He-ish init for weight paths; zeros for bias.
fn init_leaf(path: &[ParamTreePathElement], shape: &[u32], rng: &mut u64) -> Vec<f32> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let is_bias = matches!(
        path.last(),
        Some(ParamTreePathElement::Name(s)) if s.as_ref() == "bias"
    );
    if is_bias {
        return vec![0.0; n];
    }
    let fan_in = *shape.last().unwrap_or(&1) as f32;
    let scale = (2.0 / fan_in).sqrt();
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = ((*rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32));
        out.push((u * 2.0 - 1.0) * scale);
    }
    out
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_sink(
    interp: &impl Interp,
    pid: resin_jit_wgpu::ProgramId,
    sink: &str,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    Ok(bytes_to_f32(&interp.read_sink(pid, sink)?))
}

struct TrainProg {
    pid: resin_jit_wgpu::ProgramId,
    interp: resin_jit_wgpu::WgpuInterp,
    /// path (e.g. `layers.0.weight`) → host values
    weights: BTreeMap<String, Vec<f32>>,
    xs: Vec<f32>,
    ys: Vec<f32>,
}

impl TrainProg {
    fn admit_mlp(
        mlp: &Mlp<resin_dsl::View>,
        xs_shape: [u32; 2],
        ys_shape: [u32; 2],
        with_new_model: bool,
        rng_seed: u64,
        xs: Vec<f32>,
        ys: Vec<f32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let xs_v = param(xs_shape, F4, "xs");
        let ys_v = param(ys_shape, F4, "ys");
        let loss = mean(&cross_entropy(&mlp.forward(xs_v.clone())?, &ys_v)?)?;
        let grads = grad_wrt(&loss, clone_mlp(mlp))?;
        let new_model = sgd_tree(clone_mlp(mlp), clone_mlp(&grads), LR);

        let mut ir = IrProgram::new();
        ir.register_param("xs", &xs_v)?;
        ir.register_param("ys", &ys_v)?;
        ir.register_param_tree(mlp, "model")?;
        ir.build_sink("loss", &loss)?;
        ir.build_sink_tree("grad", &grads)?;
        if with_new_model {
            ir.build_sink_tree("new_model", &new_model)?;
        }
        ir.seal_params()?;
        let artifact = build_wgpu_program(&ir, None);
        let mut interp = create_interp(InterpConfig::default())?;
        let pid = interp.admit_program(artifact)?;

        let mut rng = rng_seed;
        let mut weights = BTreeMap::new();
        for (path, view) in mlp.flatten() {
            let vals = init_leaf(&path, view.shape(), &mut rng);
            let name = join_param_path("model", &path);
            interp.write_param(pid, &name, &f32_bytes(&vals))?;
            weights.insert(format_param_path(&path), vals);
        }
        interp.write_param(pid, "xs", &f32_bytes(&xs))?;
        interp.write_param(pid, "ys", &f32_bytes(&ys))?;

        Ok(Self {
            pid,
            interp,
            weights,
            xs,
            ys,
        })
    }

    fn write_weights_and_batch(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        for (path, w) in &self.weights {
            let name = format!("model.{path}");
            self.interp.write_param(self.pid, &name, &f32_bytes(w))?;
        }
        self.interp
            .write_param(self.pid, "xs", &f32_bytes(&self.xs))?;
        self.interp
            .write_param(self.pid, "ys", &f32_bytes(&self.ys))?;
        Ok(())
    }

    fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        self.interp.run(self.pid)?;
        Ok(())
    }

    fn loss(&self) -> Result<f32, Box<dyn std::error::Error>> {
        Ok(read_sink(&self.interp, self.pid, "loss")?[0])
    }

    fn host_sgd_step(&mut self) -> Result<f32, Box<dyn std::error::Error>> {
        self.write_weights_and_batch()?;
        self.run()?;
        let loss = self.loss()?;
        for (path, w) in self.weights.iter_mut() {
            let g = read_sink(&self.interp, self.pid, &format!("grad.{path}"))?;
            for i in 0..w.len() {
                w[i] -= LR * g[i];
            }
        }
        Ok(loss)
    }
}

/// Fixed 4×3 input, 4×2 one-hot labels (tiny, fully specified).
fn synth_batch_a() -> (Vec<f32>, Vec<f32>) {
    let xs = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 1.0, 0.0,
    ];
    let ys = vec![
        1.0, 0.0, // class 0
        0.0, 1.0, // class 1
        1.0, 0.0, //
        0.0, 1.0,
    ];
    (xs, ys)
}

fn synth_batch_b() -> (Vec<f32>, Vec<f32>) {
    let xs = vec![
        0.5, 0.5, 0.0, //
        0.0, 0.5, 0.5, //
        1.0, 0.0, 1.0, //
        0.2, 0.3, 0.4,
    ];
    let ys = vec![
        0.0, 1.0, //
        1.0, 0.0, //
        0.0, 1.0, //
        1.0, 0.0,
    ];
    (xs, ys)
}

/// S1: graph `new_model = w - lr*g` matches host `w - lr*g` on every leaf.
#[test]
fn s1_graph_sgd_matches_host_step() {
    let (xs, ys) = synth_batch_a();
    // in=3, out=2, one hidden of 4
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], true, 42, xs, ys).expect("gpu");
    prog.run().expect("run");

    let mut max_diff = 0f32;
    for (path, w_host) in &prog.weights {
        let g = read_sink(&prog.interp, prog.pid, &format!("grad.{path}")).expect("grad");
        let nm =
            read_sink(&prog.interp, prog.pid, &format!("new_model.{path}")).expect("new_model");
        for i in 0..w_host.len() {
            let host_new = w_host[i] - LR * g[i];
            max_diff = max_diff.max((nm[i] - host_new).abs());
        }
    }
    assert!(
        max_diff < 1e-5,
        "graph SGD vs host w-lr*g max_diff={max_diff}"
    );
}

/// S2: same batch, K host-SGD steps — loss finite, not CE-floor, final < init − ε.
#[test]
fn s2_same_batch_multi_step_loss_decreases() {
    let (xs, ys) = synth_batch_a();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], false, 7, xs, ys).expect("gpu");

    let mut losses = Vec::new();
    for _ in 0..20 {
        let l = prog.host_sgd_step().expect("step");
        assert!(l.is_finite(), "loss not finite: {l}");
        assert!(
            l < CE_FLOOR,
            "collapsed toward CE floor (log-eps): loss={l}"
        );
        losses.push(l);
    }
    let init = losses[0];
    let final_l = *losses.last().unwrap();
    assert!(
        final_l < init - 1e-3,
        "expected loss drop on same batch: init={init} final={final_l} losses={losses:?}"
    );
}

/// S3: two different batches, one step each — both losses finite (no blow-up).
#[test]
fn s3_two_batches_one_step_each_finite() {
    let (xs_a, ys_a) = synth_batch_a();
    let (xs_b, ys_b) = synth_batch_b();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], false, 11, xs_a, ys_a).expect("gpu");

    let l0 = prog.host_sgd_step().expect("batch a");
    assert!(l0.is_finite() && l0 < CE_FLOOR, "batch a loss={l0}");

    prog.xs = xs_b;
    prog.ys = ys_b;
    let l1 = prog.host_sgd_step().expect("batch b");
    assert!(l1.is_finite() && l1 < CE_FLOOR, "batch b loss={l1}");
}

/// S4: after multi-step training, restore step-0 weights and re-run — same loss as step 0.
#[test]
fn s4_restore_init_weights_rerun_matches_step0_loss() {
    let (xs, ys) = synth_batch_a();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], false, 99, xs, ys).expect("gpu");
    let w0 = prog.weights.clone();

    prog.write_weights_and_batch().expect("write");
    prog.run().expect("run0");
    let loss0 = prog.loss().expect("loss0");

    for _ in 0..10 {
        let _ = prog.host_sgd_step().expect("step");
    }
    // Corrupt weights then restore
    prog.weights = w0;
    prog.write_weights_and_batch().expect("restore");
    prog.run().expect("rerun");
    let loss_r = prog.loss().expect("loss_r");
    assert!(
        (loss_r - loss0).abs() < 1e-5,
        "rerun after restore: loss0={loss0} loss_r={loss_r} (interp/global corruption?)"
    );
}

/// Extra: single Linear (no hidden) graph SGD — smaller surface than MLP.
#[test]
fn s1b_linear_graph_sgd_matches_host() {
    let (xs, ys) = synth_batch_a();
    let layer = Linear::new(3, 2, true);
    let xs_v = param([4, 3], F4, "xs");
    let ys_v = param([4, 2], F4, "ys");
    let logits = layer.forward(&xs_v).unwrap();
    let loss =
        mean(&cross_entropy(&resin_nn::softmax(&logits, &[1]).unwrap(), &ys_v).unwrap()).unwrap();
    let grads = grad_wrt(&loss, clone_linear(&layer)).unwrap();
    let new_model = sgd_tree(clone_linear(&layer), clone_linear(&grads), LR);

    let mut ir = IrProgram::new();
    ir.register_param("xs", &xs_v).unwrap();
    ir.register_param("ys", &ys_v).unwrap();
    ir.register_param_tree(&layer, "model").unwrap();
    ir.build_sink("loss", &loss).unwrap();
    ir.build_sink_tree("grad", &grads).unwrap();
    ir.build_sink_tree("new_model", &new_model).unwrap();
    ir.seal_params().unwrap();
    let artifact = build_wgpu_program(&ir, None);
    let mut interp = create_interp(InterpConfig::default()).expect("interp");
    let pid = interp.admit_program(artifact).expect("admit");

    let mut rng = 3u64;
    let mut host_w = BTreeMap::new();
    for (path, view) in layer.flatten() {
        let vals = init_leaf(&path, view.shape(), &mut rng);
        let name = join_param_path("model", &path);
        interp.write_param(pid, &name, &f32_bytes(&vals)).unwrap();
        host_w.insert(format_param_path(&path), vals);
    }
    interp.write_param(pid, "xs", &f32_bytes(&xs)).unwrap();
    interp.write_param(pid, "ys", &f32_bytes(&ys)).unwrap();
    interp.run(pid).unwrap();

    for (path, w) in &host_w {
        let g = read_sink(&interp, pid, &format!("grad.{path}")).unwrap();
        let nm = read_sink(&interp, pid, &format!("new_model.{path}")).unwrap();
        let mut host_new = w.clone();
        for i in 0..host_new.len() {
            host_new[i] -= LR * g[i];
        }
        let d = max_abs_diff(&nm, &host_new);
        assert!(d < 1e-5, "{path}: graph vs host SGD max_diff={d}");
    }
}
