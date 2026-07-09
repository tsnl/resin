//! WebGPU backend: IR → WGSL lowering, dispatched through wgpu.
//!
//! **Contract:** `program` is already backend-ready — i.e. it has been through
//! [`crate::ir::layout::prepare_for_backend`] (dead-elim + arena pack). Layout
//! is not this backend's job.
//!
//! ## Bind groups
//!
//! Every kernel shares one **program-wide** bind-group layout:
//! `@binding(i)` ↔ `program.buffers[i]` (the packed arenas). One bind group
//! covers all pipelines for a given invoke.
//!
//! ## Pipelines
//!
//! Compiled compute pipelines live on the [`WgpuProgram`] artifact (built in
//! [`WgpuJit::lower`]), so repeated [`Jit::invoke`] calls reuse them. The
//! shape-keyed compile cache on [`crate::jit::JittedFn`] holds the artifact.

mod codegen;
mod runtime;

pub use codegen::KernelConfig;

use std::sync::Arc;

use super::{Array, Error, Jit};
use crate::ir::Program;

/// Whether a GPU adapter is available (for tests / graceful skip).
pub fn gpu_available() -> bool {
    runtime::shared_context().is_ok()
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WgpuJit {
    pub config: KernelConfig,
}

/// One compiled compute pipeline plus its dispatch geometry.
#[derive(Debug, Clone)]
pub struct CompiledPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub workgroups: [u32; 3],
    /// Source WGSL (for tests / debugging).
    pub wgsl: String,
}

/// GPU objects shared across clones of a [`WgpuProgram`] (via [`Arc`]).
#[derive(Debug)]
pub struct WgpuGpuState {
    /// Deduped pipelines for this program.
    pub pipelines: Vec<CompiledPipeline>,
    /// Shared by every pipeline: `@binding(i)` = arena `buffers[i]`.
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Lowered artifact: IR + compiled GPU state.
///
/// Cheap to clone (`gpu` is [`Arc`]); the jitted-fn compile cache stores these.
#[derive(Debug, Clone)]
pub struct WgpuProgram {
    pub ir: Program,
    pub gpu: Arc<WgpuGpuState>,
    /// `ir.queue[i]` uses `gpu.pipelines[pipeline_of[i]]`.
    pub pipeline_of: Vec<usize>,
}

/// Emit WGSL for a single kernel (used by validation tests).
pub fn emit_wgsl_for_dispatch(
    program: &Program,
    dispatch_index: usize,
    config: &KernelConfig,
) -> String {
    codegen::emit_dispatch(program, &program.queue[dispatch_index], config)
}

impl Jit for WgpuJit {
    type Artifact = WgpuProgram;

    fn lower(&self, program: &Program) -> Result<WgpuProgram, Error> {
        program.validate()?;
        ensure_fits_u32(program)?;
        let ctx = runtime::shared_context()?;
        let config = self.config;

        // Shared bind-group layout: one storage buffer per IR arena, binding = index.
        let bgl_entries: Vec<wgpu::BindGroupLayoutEntry> = program
            .buffers
            .iter()
            .enumerate()
            .map(|(i, _)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("resin-arena-bgl"),
                    entries: &bgl_entries,
                });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resin-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let mut pipelines: Vec<CompiledPipeline> = Vec::new();
        let mut pipeline_of = Vec::with_capacity(program.queue.len());
        for dispatch in &program.queue {
            let wgsl = codegen::emit_dispatch(program, dispatch, &config);
            let workgroups = codegen::workgroups_for(program, dispatch, &config);
            let index = pipelines.iter().position(|p| p.wgsl == wgsl).unwrap_or_else(|| {
                let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("resin-shader"),
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(wgsl.clone())),
                });
                let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("resin-pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });
                pipelines.push(CompiledPipeline { pipeline, workgroups, wgsl });
                pipelines.len() - 1
            });
            pipeline_of.push(index);
        }

        Ok(WgpuProgram {
            ir: program.clone(),
            gpu: Arc::new(WgpuGpuState { pipelines, bind_group_layout }),
            pipeline_of,
        })
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
    use crate::ir::layout::prepare_for_backend;
    use crate::jit::lower::lower;

    #[test]
    fn lower_compiles_pipelines_and_shared_layout() {
        if !gpu_available() {
            eprintln!("skip: no GPU");
            return;
        }
        let a = Tensor::parameter(&[4]);
        let out = (a.clone() + a.clone()) + (a.clone() + a.clone());
        let program = prepare_for_backend(lower(&a, &out).unwrap());
        let artifact = WgpuJit::default().lower(&program).unwrap();
        assert_eq!(artifact.pipeline_of.len(), program.queue.len());
        assert!(!artifact.gpu.pipelines.is_empty());
        // Shared layout: every shader declares all program arenas.
        for p in &artifact.gpu.pipelines {
            assert!(p.wgsl.contains("heap_f32"), "{}", p.wgsl);
            assert!(p.wgsl.contains("@binding(0)"), "{}", p.wgsl);
        }
        // Binding count matches arena count.
        assert_eq!(program.buffers.len(), 1);
    }

    #[test]
    fn shared_bind_layout_declares_all_heaps() {
        if !gpu_available() {
            eprintln!("skip: no GPU");
            return;
        }
        use crate::dsl::ScatterOp;

        let x = Tensor::parameter(&[4]);
        let indices = Tensor::constant_u32(&[4], &[0, 1, 0, 2]);
        let out = x.scatter_rows(&indices, 3, ScatterOp::Add);
        let program = prepare_for_backend(lower(&x, &out).unwrap());
        let artifact = WgpuJit::default().lower(&program).unwrap();
        // Expect plain f32, plain u32, atomic f32 (order depends on pack).
        let n_arenas = program.buffers.len();
        assert!(n_arenas >= 2);
        for p in &artifact.gpu.pipelines {
            for i in 0..n_arenas {
                assert!(
                    p.wgsl.contains(&format!("@binding({i})")),
                    "missing binding {i} in {}",
                    p.wgsl
                );
            }
        }
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

    #[test]
    fn wgpu_repeated_invoke_reuses_compiled_pipelines() {
        if no_gpu() {
            return;
        }
        let f = WgpuJit::default().jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let input = Pair {
            a: Array::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
            b: Array::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
        };
        let out1 = f.call(&input).unwrap();
        let out2 = f.call(&input).unwrap();
        assert_eq!(out1.data(), out2.data());
        assert_eq!(out1.data(), &[11.0, 22.0, 33.0, 44.0]);
    }
}
