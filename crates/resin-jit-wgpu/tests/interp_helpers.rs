//! Shared helpers for GPU interpreter integration tests.
#![allow(dead_code, unused_imports)]

use resin_core::{ElementType, F4};

use resin_dsl::{const_bytes, param, View};
use resin_jit_wgpu::{compile_named_open, DeviceConfig, Pipeline};
pub use resin_nn::{classifier_layers, cross_entropy, forward_probs, mean, softmax, Layer, Linear};

/// Compile `out` as the sole sink, create a pipeline instance, write params, run, read `out`.
pub fn run_graph(
    out: &View,
    params: &[(&View, &[f32])],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut named_params = Vec::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        named_params.push((name, *view));
    }
    let param_refs: Vec<(&str, &View)> =
        named_params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let factory = compile_named_open(&param_refs, &[("out", out)], DeviceConfig::default(), None)?;
    let mut pipe = factory.create()?;
    let inputs: Vec<(String, Vec<u8>)> = params
        .iter()
        .enumerate()
        .map(|(i, (view, data))| {
            let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
            (name, f32_bytes(data))
        })
        .collect();
    let input_refs: Vec<(&str, &[u8])> = inputs
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice()))
        .collect();
    let outs = pipe.call(input_refs)?;
    Ok(bytes_to_f32(&outs["out"]))
}

/// Run the same pipeline instance twice with the same inputs.
pub fn run_graph_twice(
    out: &View,
    params: &[(&View, &[f32])],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut named_params = Vec::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        named_params.push((name, *view));
    }
    let param_refs: Vec<(&str, &View)> =
        named_params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let factory = compile_named_open(&param_refs, &[("out", out)], DeviceConfig::default(), None)?;
    let mut pipe = factory.create()?;

    let write_and_run = |pipe: &mut Pipeline| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let inputs: Vec<(String, Vec<u8>)> = params
            .iter()
            .enumerate()
            .map(|(i, (view, data))| {
                let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
                (name, f32_bytes(data))
            })
            .collect();
        let input_refs: Vec<(&str, &[u8])> = inputs
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        let outs = pipe.call(input_refs)?;
        Ok(bytes_to_f32(&outs["out"]))
    };

    let first = write_and_run(&mut pipe)?;
    let second = write_and_run(&mut pipe)?;
    Ok((first, second))
}

fn param_name(view: &View) -> Option<String> {
    use resin_dsl::NodeKind;
    match &view.node_ref.kind {
        NodeKind::Param => None,
        _ => None,
    }
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

pub fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tol || (x.is_nan() && y.is_nan()))
}

pub fn f4_param(shape: &[u32]) -> View {
    param(shape.to_vec(), F4)
}

/// Classifier layer stack for tests (alias of [`classifier_layers`]).
pub fn mlp_classifier(
    in_dim: u32,
    out_dim: u32,
    n_hidden: usize,
    hidden_dim: u32,
    bias: bool,
) -> Vec<Layer<resin_dsl::View>> {
    classifier_layers(in_dim, out_dim, n_hidden, hidden_dim, bias)
}

pub fn f4_const(data: &[f32]) -> View {
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();
    const_bytes(
        [data.len() as u32],
        ElementType::F4,
        bytes.into_boxed_slice(),
    )
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

type GradMap = std::collections::BTreeMap<String, Vec<f32>>;

pub fn run_loss_grads(
    loss: &View,
    grad_sinks: &[(&str, &View)],
    params: &[(&View, &[f32])],
) -> Result<(f32, GradMap), Box<dyn std::error::Error>> {
    let mut named_params = Vec::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        named_params.push((name, *view));
    }
    let param_refs: Vec<(&str, &View)> =
        named_params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let mut sinks: Vec<(&str, &View)> = vec![("loss", loss)];
    sinks.extend(grad_sinks.iter().copied());
    let factory = compile_named_open(&param_refs, &sinks, DeviceConfig::default(), None)?;
    let mut pipe = factory.create()?;
    let inputs: Vec<(String, Vec<u8>)> = params
        .iter()
        .enumerate()
        .map(|(i, (view, data))| {
            let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
            (name, f32_bytes(data))
        })
        .collect();
    let input_refs: Vec<(&str, &[u8])> = inputs
        .iter()
        .map(|(n, b)| (n.as_str(), b.as_slice()))
        .collect();
    let outs = pipe.call(input_refs)?;
    let loss_v = bytes_to_f32(&outs["loss"]);
    let mut grads = GradMap::new();
    for (name, _) in grad_sinks {
        grads.insert((*name).to_string(), bytes_to_f32(&outs[*name]));
    }
    Ok((loss_v[0], grads))
}

pub fn finite_diff_param(
    loss: &View,
    params: &[(&View, Vec<f32>)],
    param_index: usize,
    perturb_index: usize,
    eps: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut named_params = Vec::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        named_params.push((name, *view));
    }
    let param_refs: Vec<(&str, &View)> =
        named_params.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    let factory = compile_named_open(
        &param_refs,
        &[("loss", loss)],
        DeviceConfig::default(),
        None,
    )?;
    let mut pipe = factory.create()?;

    let run_with = |pipe: &mut Pipeline,
                    data: &[(&View, Vec<f32>)]|
     -> Result<f32, Box<dyn std::error::Error>> {
        let inputs: Vec<(String, Vec<u8>)> = data
            .iter()
            .enumerate()
            .map(|(i, (view, vals))| {
                let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
                (name, f32_bytes(vals))
            })
            .collect();
        let input_refs: Vec<(&str, &[u8])> = inputs
            .iter()
            .map(|(n, b)| (n.as_str(), b.as_slice()))
            .collect();
        let outs = pipe.call(input_refs)?;
        Ok(bytes_to_f32(&outs["loss"])[0])
    };

    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    plus[param_index].1[perturb_index] += eps;
    minus[param_index].1[perturb_index] -= eps;
    let lp = run_with(&mut pipe, &plus)?;
    let lm = run_with(&mut pipe, &minus)?;
    Ok((lp - lm) / (2.0 * eps))
}
