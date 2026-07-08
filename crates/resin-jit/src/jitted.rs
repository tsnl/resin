use std::marker::PhantomData;

use resin_core::Tree;
use resin_dsl::Tensor as DslTensor;
use resin_ir::{BufferRef, BufferViewRef};

use crate::cache::{CacheKey, CompileCache};
use crate::error::JitError;
use crate::jit::Jit;
use crate::pipeline;

/// A function bound to a [`Jit`] backend.
///
/// Call with a PyTree of [`Jit::Tensor`] leaves. Each call traces the bound
/// function, compiles on cache miss, then invokes the lowered program.
pub struct JittedFn<J: Jit, P, O, F> {
    jit: J,
    f: F,
    cache: CompileCache<J::Artifact>,
    _io: PhantomData<(P, O)>,
}

impl<J, P, O, F> JittedFn<J, P, O, F>
where
    J: Jit,
    P: Tree<J::Tensor> + Send + Sync,
    O: Tree<DslTensor> + Send + Sync,
    F: Fn(&P::Mapped<DslTensor>) -> O,
    P::Mapped<DslTensor>: Send + Sync + Tree<DslTensor>,
    <P::Mapped<DslTensor> as Tree<DslTensor>>::Mapped<BufferRef>: Tree<BufferRef> + Send + Sync,
    O::Mapped<BufferViewRef>: Tree<BufferViewRef> + Send + Sync,
{
    pub(crate) fn new(jit: J, f: F) -> Self {
        Self {
            jit,
            f,
            cache: CompileCache::new(),
            _io: PhantomData,
        }
    }

    /// Trace, compile (if needed), and run on concrete input tensors.
    pub fn call(&self, params: &P) -> Result<O::Mapped<J::Tensor>, JitError> {
        let dsl_params = params.map(|leaf| self.jit.trace_parameter(leaf));
        let traced_out = (self.f)(&dsl_params);
        let mut outputs = traced_out.map(|leaf| self.jit.allocate_like(leaf));

        let cache_key = CacheKey::from_params(params);
        let artifact = self
            .cache
            .get_or_insert_with(cache_key, || {
                pipeline::compile::<J, P, O>(&self.jit, &dsl_params, &traced_out)
            })
            .map_err(JitError::Compile)?;

        self.jit
            .invoke::<P, O>(&artifact, params, &mut outputs)
            .map_err(JitError::Run)?;

        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use resin_dsl::{ElementType, Tensor as DslTensor};
    use resin_ir::{BufferRef, BufferViewRef, IrProgram};
    use resin_macros::Tree;

    use super::*;
    use crate::error::RunError;
    use crate::jit::Jit;
    use crate::tensor::ConcreteTensor;

    #[derive(Clone)]
    struct TestTensor {
        shape: Box<[usize]>,
    }

    impl ConcreteTensor for TestTensor {
        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn element_type(&self) -> ElementType {
            ElementType::F32
        }

        fn zeros(shape: &[usize], _element_type: ElementType) -> Self {
            Self {
                shape: shape.into(),
            }
        }

        fn from_f32(shape: &[usize], values: &[f32]) -> Self {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            Self {
                shape: shape.into(),
            }
        }

        fn to_f32(&self) -> Vec<f32> {
            let n = self.shape.iter().product::<usize>().max(1);
            vec![0.0; if self.shape.is_empty() { 1 } else { n }]
        }

        fn from_u32(shape: &[usize], values: &[u32]) -> Self {
            assert_eq!(shape.iter().product::<usize>(), values.len());
            Self {
                shape: shape.into(),
            }
        }

        fn to_u32(&self) -> Vec<u32> {
            let n = self.shape.iter().product::<usize>().max(1);
            vec![0; if self.shape.is_empty() { 1 } else { n }]
        }
    }

    impl TestTensor {
        fn scalar() -> Self {
            Self {
                shape: Box::from([]),
            }
        }
    }

    #[derive(Debug, Clone)]
    struct CountingJit {
        lowers: Arc<AtomicUsize>,
    }

    impl CountingJit {
        fn new(lowers: Arc<AtomicUsize>) -> Self {
            Self { lowers }
        }
    }

    impl Jit for CountingJit {
        type Tensor = TestTensor;
        type Artifact = usize;
        type LowerError = std::convert::Infallible;

        fn allocate_like(&self, traced: &DslTensor) -> Self::Tensor {
            let _ = self;
            TestTensor {
                shape: traced.shape().into(),
            }
        }

        fn lower<P, S>(
            &self,
            _program: &IrProgram<P, S>,
        ) -> Result<Self::Artifact, Self::LowerError>
        where
            P: Tree<BufferRef> + Send + Sync,
            S: Tree<BufferViewRef> + Send + Sync,
        {
            self.lowers.fetch_add(1, Ordering::SeqCst);
            Ok(self.lowers.load(Ordering::SeqCst))
        }

        fn invoke<P, O>(
            &self,
            _artifact: &Self::Artifact,
            _params: &P,
            _outputs: &mut O::Mapped<Self::Tensor>,
        ) -> Result<(), RunError>
        where
            P: Tree<Self::Tensor>,
            O: Tree<DslTensor>,
        {
            Ok(())
        }
    }

    #[derive(Tree)]
    struct In<T> {
        x: T,
    }

    #[test]
    fn call_compiles_and_invokes() {
        let lowers = Arc::new(AtomicUsize::new(0));
        let f = CountingJit::new(lowers.clone()).jit(|x: &In<DslTensor>| x.x.clone() + x.x.clone());
        let params = In { x: TestTensor::scalar() };

        f.call(&params).unwrap();
        assert_eq!(lowers.load(Ordering::SeqCst), 1);

        f.call(&params).unwrap();
        assert_eq!(lowers.load(Ordering::SeqCst), 1, "second call should hit compile cache");
    }

    #[test]
    fn jitted_fn_traces_on_each_call_before_compile() {
        let trace_count = Arc::new(AtomicUsize::new(0));
        let trace_count_cb = trace_count.clone();
        let f = CountingJit::new(Arc::new(AtomicUsize::new(0))).jit(move |x: &In<DslTensor>| {
            trace_count_cb.fetch_add(1, Ordering::SeqCst);
            x.x.clone()
        });
        let params = In { x: TestTensor::scalar() };

        let _ = f.call(&params);
        let _ = f.call(&params);
        assert_eq!(trace_count.load(Ordering::SeqCst), 2);
    }
}