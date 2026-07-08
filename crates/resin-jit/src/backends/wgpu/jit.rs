use resin_core::Tree;
use resin_dsl::Tensor as DslTensor;
use resin_ir::{BufferRef, BufferViewRef, IrProgram};

use super::program::WgpuProgram;
use super::tensor::WgpuTensor;
use crate::{Jit, RunError};

/// WebGPU execution backend.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct WgpuJit;

impl Jit for WgpuJit {
    type Tensor = WgpuTensor;
    type Artifact = WgpuProgram;
    type LowerError = super::error::WgpuLowerError;

    fn allocate_like(&self, traced: &DslTensor) -> WgpuTensor {
        let _ = self;
        <WgpuTensor as crate::ConcreteTensor>::zeros(traced.shape(), traced.element_type())
    }

    fn lower<P, S>(&self, program: &IrProgram<P, S>) -> Result<WgpuProgram, super::error::WgpuLowerError>
    where
        P: Tree<BufferRef> + Send + Sync,
        S: Tree<BufferViewRef> + Send + Sync,
    {
        program.validate()?;
        Ok(WgpuProgram {
            dispatch_count: program.queue.len(),
            buffer_count: program.buffers.len(),
            buffer_view_count: program.buffer_views.len(),
        })
    }

    fn invoke<P, O>(
        &self,
        _artifact: &WgpuProgram,
        _params: &P,
        _outputs: &mut O::Mapped<WgpuTensor>,
    ) -> Result<(), RunError>
    where
        P: Tree<WgpuTensor>,
        O: Tree<DslTensor>,
    {
        let _ = self;
        Err(RunError::NotImplemented)
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
    fn lower_minimal_program() {
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
                output_buffer_index: BufferRef::new(0),
            }],
            buffers: vec![IrBuffer {
                shape,
                element_type: F4,
                init: None,
                readonly: false,
            }],
            buffer_views: vec![IrBufferView {
                buffer_index: BufferRef::new(0),
                accessor: Accessor::dense([4], 0),
            }],
        };

        let artifact = WgpuJit.lower(&program).unwrap();
        assert_eq!(artifact.dispatch_count, 1);
        assert_eq!(artifact.buffer_count, 1);
        assert_eq!(artifact.buffer_view_count, 1);
    }
}