//! WebGPU backend: naive IR → WGSL lowering, dispatched through wgpu.

mod codegen;
mod runtime;

pub use codegen::KernelConfig;

use super::{Array, Error, Jit};
use crate::ir::Program;

/// Whether a GPU adapter is available (for tests / graceful skip).
pub fn gpu_available() -> bool {
    runtime::shared_context().is_ok()
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct WgpuJit {
    pub config: KernelConfig,
}

impl WgpuJit {
    /// Emit tiled kernels with `chromium_experimental_subgroup_matrix`,
    /// decomposed into `mma`-sized (8 or 16) multiply-accumulates. The
    /// shaders only run on runtimes that implement the extension (Dawn /
    /// Chrome); use [`WgpuJit::lower`] to extract the WGSL.
    pub fn with_subgroup_matrix(mma: usize) -> Self {
        Self {
            config: KernelConfig { subgroup_matrix_size: Some(mma), ..KernelConfig::default() },
        }
    }
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
            let args: Vec<_> = dispatch
                .args
                .iter()
                .map(|&r| &program.view(r).accessor)
                .collect();
            let out = &program.view(dispatch.output).accessor;
            let wgsl = codegen::emit(&dispatch.kernel, &args, out, &config)?;
            // Identical WGSL implies identical dispatch geometry; reuse it.
            let index = pipelines
                .iter()
                .position(|p| p.wgsl == wgsl)
                .unwrap_or_else(|| {
                    pipelines.push(PipelineSpec {
                        wgsl,
                        workgroups: codegen::workgroups(&dispatch.kernel, out, &config),
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
            let a = v.accessor.strided();
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
        let a = Tensor::parameter(&[4]);
        // Two structurally identical adds in a row.
        let out = (a.clone() + a.clone()) + (a.clone() + a.clone());
        let program = lower(&a, &out).unwrap();
        let artifact = WgpuJit::default().lower(&program).unwrap();
        assert_eq!(artifact.pipeline_of.len(), program.queue.len());
        assert!(artifact.pipelines.len() < program.queue.len());
        assert!(artifact.pipelines[0].wgsl.contains("@compute"));
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
