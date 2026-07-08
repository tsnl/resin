use resin_core::Tree;
use resin_dsl::Tensor as DslTensor;
use resin_ir::{BufferRef, BufferViewRef, IrProgram};

use super::exec::{init_storage, read_sink_bytes, run_program, write_param_bytes};
use super::program::CpuProgram;
use super::tensor::CpuTensor;
use crate::{Jit, RunError};

/// CPU interpreter execution backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuJit;

impl Jit for CpuJit {
    type Tensor = CpuTensor;
    type Artifact = CpuProgram;
    type LowerError = super::error::CpuLowerError;

    fn allocate_like(&self, traced: &DslTensor) -> CpuTensor {
        let _ = self;
        <CpuTensor as crate::ConcreteTensor>::zeros(traced.shape(), traced.element_type())
    }

    fn lower<P, S>(&self, program: &IrProgram<P, S>) -> Result<CpuProgram, super::error::CpuLowerError>
    where
        P: Tree<BufferRef> + Send + Sync,
        S: Tree<BufferViewRef> + Send + Sync,
    {
        program.validate()?;
        let mut param_buffer_indices = Vec::new();
        program.params.for_each_leaf(|buffer| {
            param_buffer_indices.push(buffer.index());
        });
        let mut sink_view_indices = Vec::new();
        program.sinks.for_each_leaf(|view| {
            sink_view_indices.push(view.index());
        });
        Ok(CpuProgram {
            buffers: program.buffers.clone(),
            buffer_views: program.buffer_views.clone(),
            queue: program.queue.clone(),
            param_buffer_indices,
            sink_view_indices,
        })
    }

    fn invoke<P, O>(
        &self,
        program: &CpuProgram,
        params: &P,
        outputs: &mut O::Mapped<CpuTensor>,
    ) -> Result<(), RunError>
    where
        P: Tree<CpuTensor>,
        O: Tree<DslTensor>,
    {
        let _ = self;
        let mut storage = init_storage(program)?;

        let mut param_iter = program.param_buffer_indices.iter();
        params.for_each_leaf(|tensor| {
            let buffer_index = *param_iter.next().expect("param leaf count mismatch");
            write_param_bytes(program, &mut storage, buffer_index, tensor.bytes())
                .expect("param write");
        });
        if param_iter.next().is_some() {
            return Err(RunError::ParamLeafMismatch);
        }

        run_program(program, &mut storage)?;

        let mut sink_iter = program.sink_view_indices.iter();
        outputs.for_each_leaf_mut(|tensor| {
            let view_index = *sink_iter.next().expect("sink leaf count mismatch");
            let bytes = read_sink_bytes(program, &storage, view_index).expect("sink read");
            tensor.bytes_mut().copy_from_slice(&bytes);
        });
        if sink_iter.next().is_some() {
            return Err(RunError::SinkLeafMismatch);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use resin_core::{Accessor, F4};
    use resin_ir::{BufferRef, BufferViewRef, IrBuffer, IrBufferView, IrProgram};

    use super::*;

    #[test]
    fn lower_empty_program() {
        let program = IrProgram::<BufferRef, BufferViewRef> {
            params: BufferRef::new(0),
            sinks: BufferViewRef::new(0),
            queue: vec![],
            buffers: vec![IrBuffer {
                shape: Box::from([2, 3]),
                element_type: F4,
                init: None,
                readonly: true,
            }],
            buffer_views: vec![IrBufferView {
                buffer_index: BufferRef::new(0),
                accessor: Accessor::dense([2, 3], 0),
            }],
        };

        let artifact = CpuJit.lower(&program).unwrap();
        assert_eq!(artifact.dispatch_count(), 0);
        assert_eq!(artifact.buffer_count(), 1);
        assert_eq!(artifact.buffer_view_count(), 1);
    }
}
#[cfg(test)]
mod e2e_tests {
    use resin_dsl::{ElementType, Tensor};
    use resin_macros::Tree;

    use super::CpuJit;
    use crate::backends::cpu::tensor::CpuTensor;
    use crate::jit::Jit;
    use crate::tensor::ConcreteTensor;

    #[derive(Tree)]
    struct In<T> {
        x: T,
    }

    #[test]
    fn sum_axes_then_squeeze_runs_on_cpu() {
        let jit = CpuJit;
        let f = jit.jit(|input: &In<Tensor>| {
            input.x.sum_axes(&[0, 1]).squeeze_all()
        });
        let x = CpuTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let out = f.call(&In { x }).unwrap();
        assert!((out.scalar_f32() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn scalar_mul_weight_runs_on_cpu() {
        let jit = CpuJit;
        let f = jit.jit(|input: &In<Tensor>| {
            let lr = Tensor::full(&[], 0.5, ElementType::F32);
            input.x.clone() * lr
        });
        let x = CpuTensor::from_f32(&[2, 2], &[2.0, 4.0, 6.0, 8.0]);
        let out = f.call(&In { x }).unwrap();
        assert_eq!(out.to_f32(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[derive(Tree)]
    struct LinearIn<T> {
        x: T,
        w: T,
        bias: T,
    }

    #[test]
    fn bias_add_broadcast_view_runs_on_cpu() {
        // y = x @ W + bias, bias broadcast without materializing.
        let jit = CpuJit;
        let f = jit.jit(|input: &LinearIn<Tensor>| {
            let y = input.x.matmul(&input.w);
            y.clone() + input.bias.broadcast_to(y.shape(), &[1])
        });
        let step = LinearIn {
            x: CpuTensor::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
            w: CpuTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            bias: CpuTensor::from_f32(&[3], &[10.0, 20.0, 30.0]),
        };
        let out = f.call(&step).unwrap();
        // row0 = [1,4] @ W? x is 2x2 identity-ish: row0=[1,0] -> first row of W + bias
        assert_eq!(out.to_f32(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn matmul_transpose_view_runs_on_cpu() {
        #[derive(Tree)]
        struct Pair<T> {
            a: T,
            b: T,
        }
        let jit = CpuJit;
        let f = jit.jit(|p: &Pair<Tensor>| p.a.matmul(&p.b.transpose()));
        let p = Pair {
            a: CpuTensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            b: CpuTensor::from_f32(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        };
        // b.T is [3,2]; a @ b.T is [2,2] → [[1,2],[4,5]]
        let out = f.call(&p).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.to_f32(), vec![1.0, 2.0, 4.0, 5.0]);
    }
}
