//! Compile DSL sinks/params to a [`WgpuProgram`] with stable param names.

use std::collections::BTreeMap;

use resin_core::ParamTree;
use resin_dsl::View;
use resin_ir::IrProgram;

use crate::{build_wgpu_program, WgpuProgram, WgslKernelConfig};

#[derive(Debug, Clone)]
pub struct ParamEntry {
    pub name: String,
    pub shape: Box<[u32]>,
    pub buffer_index: usize,
}

#[derive(Debug, Clone)]
pub struct CompiledProgram {
    pub artifact: WgpuProgram,
    pub ir: IrProgram,
    pub manifest: Vec<ParamEntry>,
}

pub fn compile_program(
    sinks: &[(impl AsRef<str>, &View)],
    params: &[(impl AsRef<str>, &View)],
    config: Option<WgslKernelConfig>,
) -> Result<CompiledProgram, String> {
    let mut ir = IrProgram::new();
    for (name, view) in params {
        ir.register_param(name.as_ref(), view)?;
    }
    for (name, view) in sinks {
        ir.build_sink(name.as_ref(), view)?;
    }
    ir.seal_params()?;
    finish(ir, config)
}

pub fn compile_with_tree<M: ParamTree<Leaf = View>>(
    sink_name: &str,
    sink: &View,
    param_prefix: &str,
    params: &M,
    config: Option<WgslKernelConfig>,
) -> Result<CompiledProgram, String> {
    let mut ir = IrProgram::new();
    ir.register_param_tree(params, param_prefix)?;
    ir.build_sink(sink_name, sink)?;
    ir.seal_params()?;
    finish(ir, config)
}

fn finish(ir: IrProgram, config: Option<WgslKernelConfig>) -> Result<CompiledProgram, String> {
    let artifact = build_wgpu_program(&ir, config);
    let manifest = ir
        .param_buffers
        .iter()
        .map(|(name, buffer_index)| ParamEntry {
            name: name.clone(),
            shape: ir.buffers[*buffer_index].shape.clone(),
            buffer_index: *buffer_index,
        })
        .collect();
    Ok(CompiledProgram {
        artifact,
        ir,
        manifest,
    })
}

pub fn named_param_indices(program: &WgpuProgram) -> BTreeMap<String, usize> {
    program.param_buffers.clone()
}
