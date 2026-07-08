use resin_core::Tree;
use resin_dsl::{ElementType, Tensor as DslTensor};
use resin_ir::{BufferRef, BufferViewRef, IrProgram};

use crate::jitted::JittedFn;
use crate::tensor::ConcreteTensor;

/// Execution backend for jitted functions.
pub trait Jit: Send + Sync + Clone + 'static {
    /// Concrete tensor type supplied by this backend at call time.
    type Tensor: ConcreteTensor;

    /// Lowered program kept inside the JIT cache (not exposed to callers).
    #[doc(hidden)]
    type Artifact: Clone + Send + Sync;

    #[doc(hidden)]
    type LowerError: std::error::Error + Send + Sync + 'static;

    /// Bind `f` to this backend, producing a callable that traces, compiles, and runs.
    fn jit<P, O, F>(self, f: F) -> JittedFn<Self, P, O, F>
    where
        P: Tree<Self::Tensor> + Send + Sync,
        O: Tree<DslTensor> + Send + Sync,
        F: Fn(&P::Mapped<DslTensor>) -> O + Send + Sync + 'static,
        P::Mapped<DslTensor>: Send + Sync + Tree<DslTensor>,
        <P::Mapped<DslTensor> as Tree<DslTensor>>::Mapped<BufferRef>: Tree<BufferRef> + Send + Sync,
        O::Mapped<BufferViewRef>: Tree<BufferViewRef> + Send + Sync,
    {
        JittedFn::new(self, f)
    }

    /// Allocate a zero-filled concrete tensor.
    fn zeros(&self, shape: &[usize], element_type: ElementType) -> Self::Tensor {
        let _ = self;
        Self::Tensor::zeros(shape, element_type)
    }

    /// Lift a concrete tensor to a trace-time DSL parameter.
    fn trace_parameter(&self, tensor: &Self::Tensor) -> DslTensor {
        let _ = self;
        DslTensor::parameter(tensor.shape(), tensor.element_type())
    }

    /// Allocate an output tensor with the shape and dtype of a traced value.
    fn allocate_like(&self, traced: &DslTensor) -> Self::Tensor;

    #[doc(hidden)]
    fn lower<P, S>(&self, program: &IrProgram<P, S>) -> Result<Self::Artifact, Self::LowerError>
    where
        P: Tree<BufferRef> + Send + Sync,
        S: Tree<BufferViewRef> + Send + Sync;

    #[doc(hidden)]
    fn invoke<P, O>(
        &self,
        artifact: &Self::Artifact,
        params: &P,
        outputs: &mut O::Mapped<Self::Tensor>,
    ) -> Result<(), crate::error::RunError>
    where
        P: Tree<Self::Tensor>,
        O: Tree<DslTensor>;
}