//! Naive IR → [`WgpuProgram`] lowering (WGSL text + buffer/view tables).

use std::collections::BTreeMap;

use resin_core::Tree;
use resin_ir::{BufferRef, BufferViewRef, IrKernel, IrProgram};

use super::error::WgpuLowerError;
use super::program::{
    WgpuBufferSpec, WgpuBufferViewSpec, WgpuDispatch, WgpuPipelineSpec, WgpuProgram,
    WgpuRasterStep, WgpuStep, WgpuTraceStep,
};
use crate::backends::wgsl::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};

pub fn lower_ir_to_wgpu<P, S>(
    program: &IrProgram<P, S>,
    config: Option<WgslKernelConfig>,
) -> Result<WgpuProgram, WgpuLowerError>
where
    P: Tree<BufferRef>,
    S: Tree<BufferViewRef>,
{
    program.validate()?;
    let config = config.unwrap_or_default();

    let buffers: Vec<WgpuBufferSpec> = program
        .buffers
        .iter()
        .map(|b| {
            let count: u64 = if b.shape.is_empty() {
                1
            } else {
                b.shape.iter().map(|&d| u64::from(d)).product()
            };
            let nbytes = count * b.element_type.nbytes() as u64;
            WgpuBufferSpec {
                shape: b.shape.clone(),
                nbytes,
                init: b.init.clone(),
                readonly: b.readonly,
            }
        })
        .collect();

    let buffer_views: Vec<WgpuBufferViewSpec> = program
        .buffer_views
        .iter()
        .map(|v| WgpuBufferViewSpec {
            buffer_index: v.buffer_index.index(),
            offset: v.accessor.offset,
            shape: v.accessor.shape.clone(),
            pitch: v.accessor.pitch.clone(),
        })
        .collect();

    // Dedup pipelines by WGSL text (naive: no IR fusion, just string reuse).
    let mut pipelines: Vec<WgpuPipelineSpec> = Vec::new();
    let mut pipeline_key_to_index: BTreeMap<(String, [u32; 3], u32, bool), usize> = BTreeMap::new();

    let mut queue = Vec::with_capacity(program.queue.len());
    for dispatch in &program.queue {
        let arg_buffer = |arg: usize| -> usize {
            buffer_views[dispatch.arg_view_indices[arg].index()].buffer_index
        };
        match &dispatch.kernel {
            IrKernel::TraceRays(k) => {
                queue.push(WgpuStep::TraceRays(WgpuTraceStep {
                    arg_buffer_indices: std::array::from_fn(arg_buffer),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                    ray_count: k.ray_count(),
                    vertex_count: k.vertex_count(),
                    triangle_count: k.triangle_count(),
                }));
            }
            IrKernel::Rasterize(k) => {
                queue.push(WgpuStep::Rasterize(WgpuRasterStep {
                    positions_buffer_index: arg_buffer(0),
                    triangles_buffer_index: arg_buffer(1),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                    height: k.height(),
                    width: k.width(),
                    triangle_count: k.triangle_count(),
                }));
            }
            kernel => {
                let wgsl = emit_wgsl_for_kernel(kernel, &config)?;
                let dispatch_size = dispatch_size_for_kernel(kernel, &config);
                let num_arg_bindings = kernel.arg_accessors().len() as u32;
                let clear = kernel.clear_output_before_dispatch();
                let key = (wgsl.clone(), dispatch_size, num_arg_bindings, clear);
                let pipeline_index = *pipeline_key_to_index.entry(key).or_insert_with(|| {
                    let idx = pipelines.len();
                    pipelines.push(WgpuPipelineSpec {
                        wgsl,
                        entry_point: "main".into(),
                        dispatch_size,
                        num_arg_bindings,
                        clear_output_before_dispatch: clear,
                    });
                    idx
                });

                queue.push(WgpuStep::Compute(WgpuDispatch {
                    pipeline_index,
                    arg_view_indices: dispatch
                        .arg_view_indices
                        .iter()
                        .map(|v| v.index())
                        .collect(),
                    output_buffer_index: dispatch.output_buffer_index.index(),
                }));
            }
        }
    }

    let mut param_buffer_indices = Vec::new();
    program.params.for_each_leaf(|buffer| {
        param_buffer_indices.push(buffer.index());
    });
    let mut sink_view_indices = Vec::new();
    program.sinks.for_each_leaf(|view| {
        sink_view_indices.push(view.index());
    });

    Ok(WgpuProgram {
        buffers,
        buffer_views,
        pipelines,
        queue,
        param_buffer_indices,
        sink_view_indices,
    })
}
