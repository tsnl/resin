//! WebGPU backend: IR → WGSL lowering, dispatched through wgpu.
//!
//! **Contract:** `program` is already backend-ready — i.e. it has been through
//! [`crate::ir::layout::prepare_for_backend`] (dead-elim + arena pack). Layout
//! is not this backend's job. Shaders bind one heap per dtype and address
//! logical tensors via view offsets.

mod codegen;
mod runtime;

pub use codegen::KernelConfig;

use super::{Array, Error, Jit};
use crate::ir::{BufferRef, Program};

/// Whether a GPU adapter is available (for tests / graceful skip).
pub fn gpu_available() -> bool {
    runtime::shared_context().is_ok()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WgpuJit {
    pub config: KernelConfig,
}

/// Lowered artifact: the IR program plus one WGSL pipeline per distinct shader.
#[derive(Debug, Clone)]
pub struct WgpuProgram {
    pub ir: Program,
    pub pipelines: Vec<PipelineSpec>,
    /// Pipeline index for each dispatch in `ir.queue`.
    pub pipeline_of: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub wgsl: String,
    pub workgroups: [u32; 3],
    /// Arena buffers bound by this shader, in `@binding` order (matches codegen).
    pub heap_buffers: Vec<BufferRef>,
}

/// Emit WGSL for a single kernel (used by validation tests).
pub fn emit_wgsl_for_dispatch(
    program: &Program,
    dispatch_index: usize,
    config: &KernelConfig,
) -> String {
    codegen::emit_dispatch(program, &program.queue[dispatch_index], config)
}

/// Binding order for heaps used by `dispatch` (must match codegen).
pub fn heap_buffers_for(program: &Program, dispatch: &crate::ir::Dispatch) -> Vec<BufferRef> {
    let out_e = program.buffer(program.view(dispatch.output).buffer).element_type;
    let mut etypes = vec![out_e];
    for &arg in &dispatch.args {
        let e = program.buffer(program.view(arg).buffer).element_type;
        if !etypes.contains(&e) {
            etypes.push(e);
        }
    }
    // Map etype → the arena BufferRef in the packed program (unique per etype).
    etypes
        .into_iter()
        .map(|e| {
            program
                .buffers
                .iter()
                .enumerate()
                .find(|(_, b)| b.element_type == e)
                .map(|(i, _)| BufferRef(i))
                .expect("heap etype present")
        })
        .collect()
}

impl Jit for WgpuJit {
    type Artifact = WgpuProgram;

    fn lower(&self, program: &Program) -> Result<WgpuProgram, Error> {
        program.validate()?;
        ensure_fits_u32(program)?;
        let config = self.config;

        let mut pipelines: Vec<PipelineSpec> = Vec::new();
        let mut pipeline_of = Vec::with_capacity(program.queue.len());
        for dispatch in &program.queue {
            let wgsl = codegen::emit_dispatch(program, dispatch, &config);
            let heap_buffers = heap_buffers_for(program, dispatch);
            // Identical WGSL implies identical bind layout; reuse it.
            let index = pipelines
                .iter()
                .position(|p| p.wgsl == wgsl)
                .unwrap_or_else(|| {
                    pipelines.push(PipelineSpec {
                        wgsl,
                        workgroups: codegen::workgroups_for(program, dispatch, &config),
                        heap_buffers: heap_buffers.clone(),
                    });
                    pipelines.len() - 1
                });
            pipeline_of.push(index);
        }

        Ok(WgpuProgram { ir: program.clone(), pipelines, pipeline_of })
    }

    fn invoke(
        &self,
        program: &WgpuProgram,
        params: &[&Array],
        outputs: &mut [&mut Array],
    ) -> Result<(), Error> {
        if params.len() != program.ir.params.len() || outputs.len() != program.ir.sinks.len() {
            return Err(Error::LeafCount);
        }
        let ctx = runtime::shared_context()?;
        runtime::run(ctx, program, params, outputs)
    }
}

/// WGSL address math is u32; reject programs that could overflow it.
fn ensure_fits_u32(program: &Program) -> Result<(), Error> {
    let too_big = program.buffers.iter().any(|b| b.len() > u32::MAX as usize)
        || program.views.iter().any(|v| {
            let a = &v.accessor;
            a.offset > u32::MAX as usize
                || a.shape.iter().any(|&d| d > u32::MAX as usize)
                || a.pitch.iter().any(|&p| p > u32::MAX as usize)
        });
    if too_big {
        return Err(Error::Wgpu("program does not fit u32 address space".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Tensor;
    use crate::jit::lower::lower;

    #[test]
    fn lower_dedups_identical_shaders() {
        use crate::ir::layout::prepare_for_backend;

        let a = Tensor::parameter(&[4]);
        // Two structurally identical adds in a row.
        let out = (a.clone() + a.clone()) + (a.clone() + a.clone());
        let program = prepare_for_backend(lower(&a, &out).unwrap());
        let artifact = WgpuJit::default().lower(&program).unwrap();
        assert_eq!(artifact.pipeline_of.len(), program.queue.len());
        assert!(
            artifact.pipelines.len() <= program.queue.len(),
            "pipelines={} queue={}",
            artifact.pipelines.len(),
            program.queue.len()
        );
        assert!(artifact.pipelines[0].wgsl.contains("heap_f32"));
        assert!(artifact.pipelines[0].heap_buffers.len() <= 2);
    }
}

#[cfg(test)]
mod e2e_tests {
    use super::*;
    use crate::dsl::Tensor;

    #[derive(resin_macros::Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    fn no_gpu() -> bool {
        if gpu_available() {
            false
        } else {
            eprintln!("skip: no GPU adapter");
            true
        }
    }

    #[test]
    fn wgpu_add_runs() {
        if no_gpu() {
            return;
        }
        let f = WgpuJit::default().jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
                b: Array::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
            })
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn wgpu_matmul_runs() {
        if no_gpu() {
            return;
        }
        let f = WgpuJit::default().jit(|p: &Pair<Tensor>| p.a.matmul(&p.b));
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                b: Array::from_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            })
            .unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn wgpu_sum_squeeze_runs() {
        if no_gpu() {
            return;
        }
        let f = WgpuJit::default().jit(|x: &Tensor| x.sum_axes(&[0, 1]).squeeze_all());
        let out = f
            .call(&Array::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        assert!((out.scalar() - 21.0).abs() < 1e-5);
    }
}
