//! Compile DSL sinks/params to a [`WgpuProgram`] with stable param names.

use std::collections::BTreeMap;

use resin_core::ParamTree;
use resin_dsl::View;
use resin_ir::{IrProgram, IrProgramBuilder};
use resin_jit_wgpu::{build_wgpu_program, WgpuProgram, WgslKernelConfig};

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
    let mut builder = IrProgramBuilder::new();
    for (name, view) in params {
        builder.register_param(name.as_ref(), view)?;
    }
    for (name, view) in sinks {
        builder.build_sink(name.as_ref(), view)?;
    }
    let ir = builder.finish()?;
    let artifact = build_wgpu_program(&ir, config);
    let manifest = ir
        .param_buffers
        .iter()
        .map(|(name, buffer_index)| {
            let shape = ir.buffers[*buffer_index].shape.clone();
            ParamEntry {
                name: name.clone(),
                shape,
                buffer_index: *buffer_index,
            }
        })
        .collect();
    Ok(CompiledProgram {
        artifact,
        ir,
        manifest,
    })
}

/// Compile with a `ParamTree` of params under a single prefix and one scalar/tensor sink.
pub fn compile_with_tree<M: ParamTree<Leaf = View>>(
    sink_name: &str,
    sink: &View,
    param_prefix: &str,
    params: &M,
    config: Option<WgslKernelConfig>,
) -> Result<CompiledProgram, String> {
    let mut builder = IrProgramBuilder::new();
    builder.register_param_tree(params, param_prefix)?;
    builder.build_sink(sink_name, sink)?;
    let ir = builder.finish()?;
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

pub use resin_jit_wgpu::param_buffer_index;
