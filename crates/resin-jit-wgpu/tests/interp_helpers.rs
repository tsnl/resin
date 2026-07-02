//! Shared helpers for GPU interpreter integration tests.
//!
//! Included via `#[path]` into multiple test binaries; not every helper is used
//! in every file.
#![allow(dead_code)]

use resin_core::{ElementType, F4};
use resin_dsl::{const_bytes, param, View};
use resin_ir::IrProgram;
use resin_jit_wgpu::{
    build_wgpu_program, create_interp, AdmitProgram, BufferId, Interp, InterpConfig, ProgramId,
};

/// Compile `out` as the sole sink, admit, write params, run, read `out` as `f32`s.
pub fn run_graph(
    out: &View,
    params: &[(&View, &[f32])],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut ir = IrProgram::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        ir.register_param(name, view)?;
    }
    ir.build_sink("out", out)?;
    ir.seal_params()?;
    let artifact = build_wgpu_program(&ir, None);

    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;

    for (view, data) in params {
        let name = param_name(view).expect("param view");
        interp.write_param(program_id, &name, &f32_bytes(data))?;
    }
    interp.run(program_id)?;
    Ok(bytes_to_f32(&interp.read_sink(program_id, "out")?))
}

/// Run the same admitted program twice with the same inputs; return both outputs.
pub fn run_graph_twice(
    out: &View,
    params: &[(&View, &[f32])],
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn std::error::Error>> {
    let mut ir = IrProgram::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        ir.register_param(name, view)?;
    }
    ir.build_sink("out", out)?;
    ir.seal_params()?;
    let artifact = build_wgpu_program(&ir, None);

    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;

    let write_params =
        |interp: &mut resin_jit_wgpu::WgpuInterp| -> Result<(), Box<dyn std::error::Error>> {
            for (view, data) in params {
                let name = param_name(view).expect("param view");
                interp.write_param(program_id, &name, &f32_bytes(data))?;
            }
            Ok(())
        };

    write_params(&mut interp)?;
    interp.run(program_id)?;
    let first = bytes_to_f32(&interp.read_sink(program_id, "out")?);

    write_params(&mut interp)?;
    interp.run(program_id)?;
    let second = bytes_to_f32(&interp.read_sink(program_id, "out")?);

    Ok((first, second))
}

fn param_name(view: &View) -> Option<String> {
    use resin_dsl::NodeKind;
    match &view.node_ref.kind {
        NodeKind::Param(p) => Some(p.name.to_string()),
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

pub fn f4_param(shape: &[u32], name: &str) -> View {
    param(shape.to_vec().into_boxed_slice(), F4, name)
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

/// Admit `loss` + named grad sinks; write params; run once; return loss and grads by sink name.
pub fn run_loss_grads(
    loss: &View,
    grad_sinks: &[(&str, &View)],
    params: &[(&View, &[f32])],
) -> Result<(f32, GradMap), Box<dyn std::error::Error>> {
    let mut ir = IrProgram::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        ir.register_param(name, view)?;
    }
    ir.build_sink("loss", loss)?;
    for (name, view) in grad_sinks {
        ir.build_sink(*name, view)?;
    }
    ir.seal_params()?;
    let artifact = build_wgpu_program(&ir, None);

    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;

    for (view, data) in params {
        let name = param_name(view).expect("param view");
        interp.write_param(program_id, &name, &f32_bytes(data))?;
    }
    interp.run(program_id)?;

    let loss_v = bytes_to_f32(&interp.read_sink(program_id, "loss")?);
    let mut grads = GradMap::new();
    for (name, _) in grad_sinks {
        grads.insert(
            (*name).to_string(),
            bytes_to_f32(&interp.read_sink(program_id, name)?),
        );
    }
    Ok((loss_v[0], grads))
}

/// Central finite difference of `loss` w.r.t. one entry of a named param buffer.
pub fn finite_diff_param(
    loss: &View,
    params: &[(&View, Vec<f32>)],
    perturb_name: &str,
    perturb_index: usize,
    eps: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let mut ir = IrProgram::new();
    for (i, (view, _)) in params.iter().enumerate() {
        let name = param_name(view).unwrap_or_else(|| format!("p{i}"));
        ir.register_param(name, view)?;
    }
    ir.build_sink("loss", loss)?;
    ir.seal_params()?;
    let artifact = build_wgpu_program(&ir, None);
    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;

    let write_all = |interp: &mut resin_jit_wgpu::WgpuInterp,
                     data: &[(&View, Vec<f32>)]|
     -> Result<(), Box<dyn std::error::Error>> {
        for (view, vals) in data {
            let name = param_name(view).expect("param");
            interp.write_param(program_id, &name, &f32_bytes(vals))?;
        }
        Ok(())
    };

    let mut plus = params.to_vec();
    let mut minus = params.to_vec();
    for (view, vals) in &mut plus {
        if param_name(view).as_deref() == Some(perturb_name) {
            vals[perturb_index] += eps;
        }
    }
    for (view, vals) in &mut minus {
        if param_name(view).as_deref() == Some(perturb_name) {
            vals[perturb_index] -= eps;
        }
    }

    write_all(&mut interp, &plus)?;
    interp.run(program_id)?;
    let lp = bytes_to_f32(&interp.read_sink(program_id, "loss")?)[0];
    write_all(&mut interp, &minus)?;
    interp.run(program_id)?;
    let lm = bytes_to_f32(&interp.read_sink(program_id, "loss")?)[0];
    Ok((lp - lm) / (2.0 * eps))
}

// Silence unused import when this file is compiled as its own test crate.
#[allow(dead_code)]
fn _use_program_id(_: ProgramId, _: BufferId) {}
