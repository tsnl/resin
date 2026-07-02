//! T5: short train-step contracts on fixed synthetic batches (no download).

#[path = "interp_helpers.rs"]
mod interp_helpers;

use std::collections::BTreeMap;

use interp_helpers::{f32_bytes, max_abs_diff};
use resin_core::{format_param_path, join_param_path, ParamTree, ParamTreePathElement, F4};
use resin_dsl::param;
use resin_grad::grad_wrt;
use resin_ir::IrProgram;
use resin_jit_wgpu::{build_wgpu_program, DeviceConfig, DeviceContext, Pipeline, PipelineFactory};
use resin_nn::{cross_entropy, mean, sgd_tree, Linear, Mlp};

const LR: f32 = 1e-3;
const CE_FLOOR: f32 = 16.0;

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

struct TrainProg {
    pipe: Pipeline,
    /// dotted path without `model.` prefix for weights map keys used with grad.
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
        let ctx = DeviceContext::from_config(&DeviceConfig::default())?;
        let factory = PipelineFactory::from_program(ctx, artifact)?;
        let pipe = factory.create()?;

        let mut rng = rng_seed;
        let mut weights = BTreeMap::new();
        for (path, view) in mlp.flatten() {
            let vals = init_leaf(&path, view.shape(), &mut rng);
            weights.insert(format_param_path(&path), vals);
        }

        Ok(Self {
            pipe,
            weights,
            xs,
            ys,
        })
    }

    fn run_once(&mut self) -> Result<BTreeMap<String, Vec<u8>>, Box<dyn std::error::Error>> {
        let xs_b = f32_bytes(&self.xs);
        let ys_b = f32_bytes(&self.ys);
        let weight_bytes: Vec<(String, Vec<u8>)> = self
            .weights
            .iter()
            .map(|(path, vals)| (format!("model.{path}"), f32_bytes(vals)))
            .collect();
        let mut inputs: Vec<(&str, &[u8])> = vec![("xs", xs_b.as_slice()), ("ys", ys_b.as_slice())];
        for (n, b) in &weight_bytes {
            inputs.push((n.as_str(), b.as_slice()));
        }
        Ok(self.pipe.call(inputs)?)
    }

    fn host_sgd_step(&mut self) -> Result<f32, Box<dyn std::error::Error>> {
        let outs = self.run_once()?;
        let loss = bytes_to_f32(&outs["loss"])[0];
        for (path, w) in self.weights.iter_mut() {
            let g = bytes_to_f32(&outs[&format!("grad.{path}")]);
            for i in 0..w.len() {
                w[i] -= LR * g[i];
            }
        }
        Ok(loss)
    }
}

fn synth_batch_a() -> (Vec<f32>, Vec<f32>) {
    let xs = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let ys = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    (xs, ys)
}

fn synth_batch_b() -> (Vec<f32>, Vec<f32>) {
    let xs = vec![0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 1.0, 0.2, 0.3, 0.4];
    let ys = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    (xs, ys)
}

#[test]
fn s1_graph_sgd_matches_host_step() {
    let (xs, ys) = synth_batch_a();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], true, 42, xs, ys).expect("gpu");
    let outs = prog.run_once().expect("run");

    let mut max_diff = 0f32;
    for (path, w_host) in &prog.weights {
        let g = bytes_to_f32(&outs[&format!("grad.{path}")]);
        let nm = bytes_to_f32(&outs[&format!("new_model.{path}")]);
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

#[test]
fn s2_same_batch_multi_step_loss_decreases() {
    let (xs, ys) = synth_batch_a();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], false, 7, xs, ys).expect("gpu");
    let mut losses = Vec::new();
    for _ in 0..20 {
        let l = prog.host_sgd_step().expect("step");
        assert!(l.is_finite(), "loss not finite: {l}");
        assert!(l < CE_FLOOR, "collapsed toward CE floor: loss={l}");
        losses.push(l);
    }
    let init = losses[0];
    let final_l = *losses.last().unwrap();
    assert!(
        final_l < init - 1e-3,
        "expected loss drop: init={init} final={final_l}"
    );
}

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

#[test]
fn s4_restore_init_weights_rerun_matches_step0_loss() {
    let (xs, ys) = synth_batch_a();
    let mlp = Mlp::new(3, 2, 1, 4, true);
    let mut prog = TrainProg::admit_mlp(&mlp, [4, 3], [4, 2], false, 99, xs, ys).expect("gpu");
    let w0 = prog.weights.clone();
    let loss0 = bytes_to_f32(&prog.run_once().expect("run0")["loss"])[0];
    for _ in 0..10 {
        let _ = prog.host_sgd_step().expect("step");
    }
    prog.weights = w0;
    let loss_r = bytes_to_f32(&prog.run_once().expect("rerun")["loss"])[0];
    assert!(
        (loss_r - loss0).abs() < 1e-5,
        "rerun after restore: loss0={loss0} loss_r={loss_r}"
    );
}

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
    let ctx = DeviceContext::from_config(&DeviceConfig::default()).unwrap();
    let factory = PipelineFactory::from_program(ctx, artifact).unwrap();
    let mut pipe = factory.create().unwrap();

    let mut rng = 3u64;
    let mut host_w = BTreeMap::new();
    let mut inputs_owned: Vec<(String, Vec<u8>)> =
        vec![("xs".into(), f32_bytes(&xs)), ("ys".into(), f32_bytes(&ys))];
    for (path, view) in layer.flatten() {
        let vals = init_leaf(&path, view.shape(), &mut rng);
        let name = join_param_path("model", &path);
        inputs_owned.push((name, f32_bytes(&vals)));
        host_w.insert(format_param_path(&path), vals);
    }
    let input_refs: Vec<(&str, &[u8])> = inputs_owned
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice()))
        .collect();
    let outs = pipe.call(input_refs).unwrap();

    for (path, w) in &host_w {
        let g = bytes_to_f32(&outs[&format!("grad.{path}")]);
        let nm = bytes_to_f32(&outs[&format!("new_model.{path}")]);
        let mut host_new = w.clone();
        for i in 0..host_new.len() {
            host_new[i] -= LR * g[i];
        }
        let d = max_abs_diff(&nm, &host_new);
        assert!(d < 1e-5, "{path}: graph vs host SGD max_diff={d}");
    }
}
