//! T5: short train-step contracts on fixed synthetic batches (no download).

#[path = "interp_helpers.rs"]
mod interp_helpers;

use std::collections::BTreeMap;

use interp_helpers::{
    cross_entropy, f32_bytes, forward_probs, max_abs_diff, mean, mlp_classifier, softmax, Layer,
    Linear,
};
use resin_core::{format_param_path, join_param_path, named, Tree, F4};
use resin_dsl::param;
use resin_grad::grad_wrt;
use resin_jit_wgpu::{compile_open, DeviceConfig, Pipeline};
use resin_nn::{initialize_tree, sgd_tree, InitMethod};

const LR: f32 = 1e-3;
const CE_FLOOR: f32 = 16.0;

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
        mlp: &Vec<Layer<resin_dsl::View>>,
        xs_shape: [u32; 2],
        ys_shape: [u32; 2],
        with_new_model: bool,
        rng_seed: u64,
        xs: Vec<f32>,
        ys: Vec<f32>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let xs_v = param(xs_shape, F4);
        let ys_v = param(ys_shape, F4);
        let loss = mean(&cross_entropy(&forward_probs(mlp, &xs_v)?, &ys_v)?)?;
        let grads = grad_wrt(&loss, mlp)?;
        let new_model = sgd_tree(mlp, &grads, LR);

        let inputs = (
            named("xs", xs_v),
            named("ys", ys_v),
            named("model", mlp.to_vec()),
        );
        let factory = if with_new_model {
            compile_open(
                &inputs,
                &(
                    named("loss", loss),
                    named("grad", grads),
                    named("new_model", new_model),
                ),
                DeviceConfig::default(),
                None,
            )?
        } else {
            compile_open(
                &inputs,
                &(named("loss", loss), named("grad", grads)),
                DeviceConfig::default(),
                None,
            )?
        };
        let pipe = factory.create()?;

        let mut weights = BTreeMap::new();
        for (path, bytes) in initialize_tree(
            mlp,
            InitMethod::He {
                seed: rng_seed.to_le_bytes().into(),
            },
        )
        .flatten()
        {
            weights.insert(format_param_path(&path), bytes_to_f32(bytes));
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
    let mlp = mlp_classifier(3, 2, 1, 4, true);
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
    let mlp = mlp_classifier(3, 2, 1, 4, true);
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
    let mlp = mlp_classifier(3, 2, 1, 4, true);
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
    let mlp = mlp_classifier(3, 2, 1, 4, true);
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
    let xs_v = param([4, 3], F4);
    let ys_v = param([4, 2], F4);
    let logits = layer.forward(&xs_v).unwrap();
    let loss = mean(&cross_entropy(&softmax(&logits, &[1]).unwrap(), &ys_v).unwrap()).unwrap();
    let grads = grad_wrt(&loss, &layer).unwrap();
    let new_model = sgd_tree(&layer, &grads, LR);

    let factory = compile_open(
        &(
            named("xs", xs_v),
            named("ys", ys_v),
            named("model", layer.clone()),
        ),
        &(
            named("loss", loss),
            named("grad", grads),
            named("new_model", new_model),
        ),
        DeviceConfig::default(),
        None,
    )
    .unwrap();
    let mut pipe = factory.create().unwrap();

    let mut host_w = BTreeMap::new();
    let mut inputs_owned: Vec<(String, Vec<u8>)> =
        vec![("xs".into(), f32_bytes(&xs)), ("ys".into(), f32_bytes(&ys))];
    for (path, bytes) in initialize_tree(
        &layer,
        InitMethod::He {
            seed: 3u64.to_le_bytes().into(),
        },
    )
    .flatten()
    {
        let vals = bytes_to_f32(bytes);
        host_w.insert(format_param_path(&path), vals.clone());
        inputs_owned.push((join_param_path("model", &path), f32_bytes(&vals)));
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
