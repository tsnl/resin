//! Lower [`resin_ir::IrProgram`] to [`WgpuProgram`].
//!
//! This is **not** the same as WGSL codegen: it builds the buffer/pipeline/queue
//! tables. Kernel source strings come from [`crate::codegen`]. Kept separate so
//! those two responsibilities stay reviewable; they could live in one file if we
//! prefer fewer modules.

use std::collections::BTreeMap;

use resin_core::ElementType as CoreElementType;
use resin_ir::{IrKernel, IrProgram};

use crate::codegen::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};
use crate::program::{
    ElementType, WgpuAccessorSpec, WgpuBufferSpec, WgpuBufferViewSpec, WgpuComputePipelineSpec,
    WgpuDispatch, WgpuProgram, WgpuQueueOp,
};

pub fn build_wgpu_program(program: &IrProgram, config: Option<WgslKernelConfig>) -> WgpuProgram {
    let config = config.unwrap_or_default();

    let buffers: Vec<WgpuBufferSpec> = program
        .buffers
        .iter()
        .map(|b| WgpuBufferSpec {
            shape: b.shape.to_vec(),
            etype: map_etype(b.etype),
            init: b.init.as_ref().map(|x| x.to_vec()),
            readonly: b.readonly,
        })
        .collect();

    let buffer_views: Vec<WgpuBufferViewSpec> = program
        .buffer_views
        .iter()
        .map(|v| WgpuBufferViewSpec {
            buffer_index: v.buffer_index,
            accessor: WgpuAccessorSpec {
                offset: v.accessor.offset,
                shape: v.accessor.shape.to_vec(),
                pitch: v.accessor.pitch.to_vec(),
            },
        })
        .collect();

    // Deduplicate pipelines by WGSL text + dispatch metadata.
    let mut pipelines: Vec<WgpuComputePipelineSpec> = Vec::new();
    let mut pipeline_key_to_index: BTreeMap<(String, [u32; 3], u32, bool), usize> = BTreeMap::new();

    let mut queue = Vec::with_capacity(program.queue.len());
    for dispatch in &program.queue {
        let kernel = &dispatch.kernel;
        let wgsl = emit_wgsl_for_kernel(kernel, &config);
        let dispatch_size = dispatch_size_for_kernel(kernel, &config);
        let num_arg_bindings = kernel_arg_accessors(kernel).len() as u32;
        let clear = kernel_clear_output(kernel);
        let key = (wgsl.clone(), dispatch_size, num_arg_bindings, clear);
        let pipeline_index = *pipeline_key_to_index.entry(key).or_insert_with(|| {
            let idx = pipelines.len();
            pipelines.push(WgpuComputePipelineSpec {
                wgsl,
                entry_point: "main".into(),
                dispatch_size,
                num_arg_bindings,
                clear_output_before_dispatch: clear,
            });
            idx
        });

        queue.push(WgpuQueueOp::Dispatch(WgpuDispatch {
            pipeline_index,
            arg_buffer_view_indices: dispatch.arg_view_indices.clone(),
            output_buffer_index: dispatch.output_buffer_index,
        }));
    }

    let param_buffers: BTreeMap<String, usize> = program.param_buffers.iter().cloned().collect();
    let sinks: BTreeMap<String, usize> = program.sinks.iter().cloned().collect();

    WgpuProgram {
        param_buffers,
        sinks,
        queue,
        buffers,
        buffer_views,
        pipelines,
    }
}

pub fn param_buffer_index(program: &WgpuProgram, name: &str) -> Option<usize> {
    program.param_buffers.get(name).copied()
}

fn map_etype(e: CoreElementType) -> ElementType {
    match e {
        CoreElementType::F4 => ElementType::F4,
        CoreElementType::F2 => ElementType::F2,
        CoreElementType::U4 => ElementType::U4,
    }
}

fn kernel_arg_accessors(kernel: &IrKernel) -> &[resin_core::Accessor] {
    match kernel {
        IrKernel::ElementwiseRpn(k) => &k.arg_accessors,
        IrKernel::Matmul(k) => &k.arg_accessors,
        IrKernel::Reduction(k) => &k.arg_accessors,
        IrKernel::Remap(k) => &k.arg_accessors,
    }
}

fn kernel_clear_output(kernel: &IrKernel) -> bool {
    match kernel {
        IrKernel::ElementwiseRpn(k) => k.clear_output_before_dispatch,
        IrKernel::Matmul(k) => k.clear_output_before_dispatch,
        IrKernel::Reduction(k) => k.clear_output_before_dispatch,
        IrKernel::Remap(k) => k.clear_output_before_dispatch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use resin_core::F4;
    use resin_dsl::param;
    use resin_ir::IrProgram;

    #[test]
    fn lower_add_emits_pipeline() {
        let a = param([4], F4, "a");
        let b = param([4], F4, "b");
        let c = &a + &b;
        let mut ir = IrProgram::new();
        ir.register_param("a", &a).unwrap();
        ir.register_param("b", &b).unwrap();
        ir.build_sink("out", &c).unwrap();
        ir.seal_params().unwrap();
        let prog = build_wgpu_program(&ir, None);
        assert_eq!(prog.pipelines.len(), 1);
        assert!(prog.pipelines[0].wgsl.contains("eval_rpn_expr"));
        assert_eq!(prog.queue.len(), 1);
        assert!(prog.param_buffers.contains_key("a"));
        assert!(prog.sinks.contains_key("out"));
    }
}
