use resin_core::Tree;
use resin_dsl::Tensor as DslTensor;
use resin_ir::{BufferRef, BufferViewRef, IrProgram};

use super::error::VulkanLowerError;
use super::lower::lower_ir_to_vulkan;
use super::program::VulkanProgram;
use super::runtime::{run_program, shared_context};
use super::tensor::VulkanTensor;
use crate::{Jit, RunError};

/// Native Vulkan execution backend (shared WGSL kernels → SPIR-V via naga).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct VulkanJit;

impl Jit for VulkanJit {
    type Tensor = VulkanTensor;
    type Artifact = VulkanProgram;
    type LowerError = VulkanLowerError;

    fn allocate_like(&self, traced: &DslTensor) -> VulkanTensor {
        let _ = self;
        <VulkanTensor as crate::ConcreteTensor>::zeros(traced.shape(), traced.element_type())
    }

    fn lower<P, S>(&self, program: &IrProgram<P, S>) -> Result<VulkanProgram, VulkanLowerError>
    where
        P: Tree<BufferRef> + Send + Sync,
        S: Tree<BufferViewRef> + Send + Sync,
    {
        let _ = self;
        lower_ir_to_vulkan(program, None)
    }

    fn invoke<P, O>(
        &self,
        program: &VulkanProgram,
        params: &P,
        outputs: &mut O::Mapped<VulkanTensor>,
    ) -> Result<(), RunError>
    where
        P: Tree<VulkanTensor>,
        O: Tree<DslTensor>,
    {
        let _ = self;
        let ctx = shared_context().map_err(|e| RunError::Vulkan(e.to_string()))?;

        let mut param_owned: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut param_iter = program.param_buffer_indices.iter();
        params.for_each_leaf(|tensor| {
            let buffer_index = *param_iter.next().expect("param leaf count mismatch");
            param_owned.push((buffer_index, tensor.bytes().to_vec()));
        });
        if param_iter.next().is_some() {
            return Err(RunError::ParamLeafMismatch);
        }

        let param_refs: Vec<(usize, &[u8])> = param_owned
            .iter()
            .map(|(i, b)| (*i, b.as_slice()))
            .collect();

        // Staging host buffers for densified sink readback (order = leaf walk).
        let mut sink_bufs: Vec<(usize, Vec<u8>)> = Vec::new();
        let mut sink_iter = program.sink_view_indices.iter();
        outputs.for_each_leaf(|tensor| {
            let view_index = *sink_iter.next().expect("sink leaf count mismatch");
            sink_bufs.push((view_index, vec![0u8; tensor.bytes().len()]));
        });
        if sink_iter.next().is_some() {
            return Err(RunError::SinkLeafMismatch);
        }

        let mut sink_refs: Vec<(usize, &mut [u8])> = sink_bufs
            .iter_mut()
            .map(|(i, b)| (*i, b.as_mut_slice()))
            .collect();

        run_program(ctx, program, &param_refs, &mut sink_refs)
            .map_err(|e| RunError::Vulkan(e.to_string()))?;

        let mut out_i = 0;
        outputs.for_each_leaf_mut(|tensor| {
            tensor.bytes_mut().copy_from_slice(&sink_bufs[out_i].1);
            out_i += 1;
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use resin_core::{Accessor, ElementOperator, F4, UnaryElementOperator};
    use resin_ir::{
        BufferRef, BufferViewRef, ElementRpnExpr, IrBuffer, IrBufferView, IrDispatch,
        IrElementwiseRpnKernel, IrKernel, IrProgram, RpnAtom,
    };

    use super::*;

    #[test]
    fn lower_minimal_program_to_spirv() {
        let shape: Box<[u32]> = Box::from([4]);
        let program = IrProgram::<BufferRef, BufferViewRef> {
            params: BufferRef::new(0),
            sinks: BufferViewRef::new(0),
            queue: vec![IrDispatch {
                kernel: IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                    arg_accessors: vec![Accessor::dense(shape.clone(), 0)],
                    arg_element_types: vec![F4],
                    element_type: F4,
                    shape: shape.clone(),
                    rpn_expr: ElementRpnExpr {
                        atoms: vec![
                            RpnAtom::Arg(0),
                            RpnAtom::Op(ElementOperator::Unary(UnaryElementOperator::Neg)),
                        ],
                    },
                    clear_output_before_dispatch: false,
                }),
                arg_view_indices: vec![BufferViewRef::new(0)],
                output_buffer_index: BufferRef::new(1),
            }],
            buffers: vec![
                IrBuffer {
                    shape: shape.clone(),
                    element_type: F4,
                    init: None,
                    readonly: false,
                },
                IrBuffer {
                    shape,
                    element_type: F4,
                    init: None,
                    readonly: false,
                },
            ],
            buffer_views: vec![IrBufferView {
                buffer_index: BufferRef::new(0),
                accessor: Accessor::dense([4], 0),
            }],
        };

        let artifact = VulkanJit.lower(&program).unwrap();
        assert_eq!(artifact.dispatch_count(), 1);
        assert_eq!(artifact.buffer_count(), 2);
        assert_eq!(artifact.pipelines.len(), 1);
        // SPIR-V magic number.
        assert_eq!(artifact.pipelines[0].spirv[0], 0x0723_0203);
    }
}

#[cfg(test)]
mod e2e_tests {
    use resin_dsl::Tensor;
    use resin_macros::Tree;

    use super::VulkanJit;
    use crate::backends::vulkan::tensor::VulkanTensor;
    use crate::jit::Jit;
    use crate::tensor::ConcreteTensor;

    #[derive(Tree)]
    struct In<T> {
        x: T,
    }

    #[derive(Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    fn skip_if_no_vulkan() -> bool {
        crate::backends::vulkan::runtime::shared_context().is_err()
    }

    #[test]
    fn vulkan_add_runs() {
        if skip_if_no_vulkan() {
            eprintln!("skip vulkan_add_runs: no Vulkan device");
            return;
        }
        let jit = VulkanJit;
        let f = jit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let p = Pair {
            a: VulkanTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
            b: VulkanTensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
        };
        let out = f.call(&p).expect("vulkan add");
        assert_eq!(out.to_f32(), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn vulkan_matmul_runs() {
        if skip_if_no_vulkan() {
            eprintln!("skip vulkan_matmul_runs: no Vulkan device");
            return;
        }
        let jit = VulkanJit;
        let f = jit.jit(|p: &Pair<Tensor>| p.a.matmul(&p.b));
        let p = Pair {
            a: VulkanTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            b: VulkanTensor::from_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        };
        let out = f.call(&p).expect("vulkan matmul");
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.to_f32(), vec![1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn vulkan_sum_squeeze_runs() {
        if skip_if_no_vulkan() {
            eprintln!("skip vulkan_sum_squeeze_runs: no Vulkan device");
            return;
        }
        let jit = VulkanJit;
        let f = jit.jit(|input: &In<Tensor>| input.x.sum_axes(&[0, 1]).squeeze_all());
        let x = VulkanTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = f.call(&In { x }).expect("vulkan sum");
        assert!((out.scalar_f32() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn vulkan_gather_scatter_runs() {
        if skip_if_no_vulkan() {
            eprintln!("skip vulkan_gather_scatter_runs: no Vulkan device");
            return;
        }
        let jit = VulkanJit;
        let f = jit.jit(|input: &In<Tensor>| {
            let indices = Tensor::constant_u32(&[3], &[2, 0, 2]);
            input.x.gather_rows(&indices)
        });
        let x = VulkanTensor::from_f32(&[3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = f.call(&In { x }).expect("vulkan gather");
        assert_eq!(out.to_f32(), vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
    }
}
