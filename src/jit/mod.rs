//! JAX-style JIT: bind a trace function to a backend, call it with arrays.
//!
//! ```no_run
//! # use resin::{Tree, dsl::Tensor, jit::{Array, CpuJit, Jit}};
//! let double = CpuJit.jit(|x: &Tensor| x.clone() + x.clone());
//! let out = double.call(&Array::from_f32(&[2], &[1.0, 2.0])).unwrap();
//! ```
//!
//! Each call traces the function over parameter placeholders, compiles on
//! cache miss (keyed by parameter shapes + element types), and invokes the
//! lowered program.

pub mod cpu;
pub mod lower;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use cpu::CpuJit;
pub use lower::lower;
#[cfg(feature = "wgpu")]
pub use wgpu::WgpuJit;

use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Mutex;

use crate::dsl::Tensor;
use crate::ir;
use crate::ops::ElementType;
use crate::tree::Tree;

/// Compile-cache key: the parameter leaves' shapes and element types.
type ShapeKey = Vec<(Box<[usize]>, ElementType)>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot lower tensor kind: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Ir(#[from] ir::Error),
    #[error("expected {expected} elements, got {got}")]
    Size { expected: usize, got: usize },
    #[error("param/output leaf count does not match compiled program")]
    LeafCount,
    #[error("element type mismatch: expected {expected:?}, got {got:?}")]
    ElementType { expected: ElementType, got: ElementType },
    #[error("wgpu: {0}")]
    Wgpu(String),
}

/// Host array payload (typed).
#[derive(Debug, Clone, PartialEq)]
pub enum ArrayData {
    F32(Box<[f32]>),
    U32(Box<[u32]>),
}

impl ArrayData {
    pub fn len(&self) -> usize {
        match self {
            ArrayData::F32(v) => v.len(),
            ArrayData::U32(v) => v.len(),
        }
    }

    pub fn element_type(&self) -> ElementType {
        match self {
            ArrayData::F32(_) => ElementType::F32,
            ArrayData::U32(_) => ElementType::U32,
        }
    }
}

/// A concrete array: the value type jitted functions are called with.
#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    shape: Box<[usize]>,
    data: ArrayData,
}

impl Array {
    pub fn zeros(shape: &[usize]) -> Self {
        Self::zeros_typed(shape, ElementType::F32)
    }

    pub fn zeros_typed(shape: &[usize], element_type: ElementType) -> Self {
        let n = ir::element_count(shape);
        let data = match element_type {
            ElementType::F32 => ArrayData::F32(vec![0.0; n].into()),
            ElementType::U32 => ArrayData::U32(vec![0; n].into()),
        };
        Self { shape: shape.into(), data }
    }

    /// Row-major f32 values.
    pub fn from_f32(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(
            ir::element_count(shape),
            values.len(),
            "from_f32: shape {shape:?} does not hold {} values",
            values.len()
        );
        Self {
            shape: shape.into(),
            data: ArrayData::F32(values.into()),
        }
    }

    /// Row-major u32 values.
    pub fn from_u32(shape: &[usize], values: &[u32]) -> Self {
        assert_eq!(
            ir::element_count(shape),
            values.len(),
            "from_u32: shape {shape:?} does not hold {} values",
            values.len()
        );
        Self {
            shape: shape.into(),
            data: ArrayData::U32(values.into()),
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn element_type(&self) -> ElementType {
        self.data.element_type()
    }

    /// Row-major f32 elements (panics if not f32).
    pub fn data(&self) -> &[f32] {
        match &self.data {
            ArrayData::F32(v) => v,
            ArrayData::U32(_) => panic!("Array::data requires f32"),
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        match &mut self.data {
            ArrayData::F32(v) => v,
            ArrayData::U32(_) => panic!("Array::data_mut requires f32"),
        }
    }

    pub fn data_u32(&self) -> &[u32] {
        match &self.data {
            ArrayData::U32(v) => v,
            ArrayData::F32(_) => panic!("Array::data_u32 requires u32"),
        }
    }

    pub fn data_u32_mut(&mut self) -> &mut [u32] {
        match &mut self.data {
            ArrayData::U32(v) => v,
            ArrayData::F32(_) => panic!("Array::data_u32_mut requires u32"),
        }
    }

    /// Clone as a dense f32 Vec (panics if not f32).
    pub fn to_f32(&self) -> Vec<f32> {
        self.data().to_vec()
    }

    /// Clone as a dense u32 Vec (panics if not u32).
    pub fn to_u32(&self) -> Vec<u32> {
        self.data_u32().to_vec()
    }

    pub fn as_data(&self) -> &ArrayData {
        &self.data
    }

    pub fn as_data_mut(&mut self) -> &mut ArrayData {
        &mut self.data
    }

    /// The single element of a rank-0 f32 array.
    pub fn scalar(&self) -> f32 {
        assert!(self.shape.is_empty(), "scalar() requires rank 0, got {:?}", self.shape);
        self.data()[0]
    }
}

/// An execution backend: lowers an IR program to an artifact and runs it.
///
/// `invoke` receives parameter and output leaves in the same order as
/// [`ir::Program::params`] / [`ir::Program::sinks`]. Output arrays are
/// pre-allocated with the traced shapes; the backend fills their data.
pub trait Jit: Clone {
    type Artifact: Clone;

    fn lower(&self, program: &ir::Program) -> Result<Self::Artifact, Error>;

    fn invoke(
        &self,
        artifact: &Self::Artifact,
        params: &[&Array],
        outputs: &mut [&mut Array],
    ) -> Result<(), Error>;

    /// Bind `f` to this backend, producing a callable that traces, compiles,
    /// and runs.
    fn jit<P, O, F>(self, f: F) -> JittedFn<Self, P, O, F> {
        JittedFn { jit: self, f, cache: Mutex::new(HashMap::new()), _io: PhantomData }
    }
}

/// A traceable function bound to a [`Jit`] backend. Parameters are a [`Tree`]
/// of [`Array`]s; the trace function sees the same tree of [`Tensor`]s.
pub struct JittedFn<J: Jit, P, O, F> {
    jit: J,
    f: F,
    cache: Mutex<HashMap<ShapeKey, J::Artifact>>,
    _io: PhantomData<fn(&P) -> O>,
}

impl<J, P, O, F> JittedFn<J, P, O, F>
where
    J: Jit,
    P: Tree<Array>,
    O: Tree<Tensor>,
    F: Fn(&P::Mapped<Tensor>) -> O,
{
    /// Trace, compile (if this shape signature is new), and run.
    pub fn call(&self, params: &P) -> Result<O::Mapped<Array>, Error> {
        let traced_params = params.map(|array| {
            Tensor::parameter_typed(array.shape(), array.element_type())
        });
        let traced_out = (self.f)(&traced_params);
        let mut outputs =
            traced_out.map(|tensor| Array::zeros_typed(tensor.shape(), tensor.element_type()));

        let key: ShapeKey = params
            .leaves()
            .iter()
            .map(|a| (a.shape.clone(), a.element_type()))
            .collect();
        let artifact = {
            let mut cache = self.cache.lock().expect("compile cache poisoned");
            match cache.get(&key) {
                Some(artifact) => artifact.clone(),
                None => {
                    // optimize? → layout (dead-elim + arena pack) → backend.
                    // Layout always runs, even when opt is a no-op.
                    let program = ir::layout::prepare_for_backend(ir::optimize::optimize(
                        lower::lower(&traced_params, &traced_out)?,
                    ));
                    let artifact = self.jit.lower(&program)?;
                    cache.insert(key, artifact.clone());
                    artifact
                }
            }
        };

        self.jit.invoke(&artifact, &params.leaves(), &mut outputs.leaves_mut())?;
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Counts lowers; runs nothing.
    #[derive(Clone)]
    struct CountingJit {
        lowers: Arc<AtomicUsize>,
    }

    impl Jit for CountingJit {
        type Artifact = ();

        fn lower(&self, _program: &ir::Program) -> Result<(), Error> {
            self.lowers.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn invoke(
            &self,
            _artifact: &(),
            _params: &[&Array],
            _outputs: &mut [&mut Array],
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    #[derive(resin_macros::Tree)]
    struct In<T> {
        x: T,
    }

    #[test]
    fn call_compiles_once_per_shape_signature() {
        let lowers = Arc::new(AtomicUsize::new(0));
        let f = CountingJit { lowers: lowers.clone() }
            .jit(|input: &In<Tensor>| input.x.clone() + input.x.clone());

        f.call(&In { x: Array::zeros(&[]) }).unwrap();
        f.call(&In { x: Array::zeros(&[]) }).unwrap();
        assert_eq!(lowers.load(Ordering::SeqCst), 1, "same shapes hit the cache");

        f.call(&In { x: Array::zeros(&[3]) }).unwrap();
        assert_eq!(lowers.load(Ordering::SeqCst), 2, "new shapes recompile");
    }

    #[test]
    fn call_traces_every_time() {
        let traces = Arc::new(AtomicUsize::new(0));
        let traces_in_f = traces.clone();
        let f = CountingJit { lowers: Arc::new(AtomicUsize::new(0)) }.jit(
            move |input: &In<Tensor>| {
                traces_in_f.fetch_add(1, Ordering::SeqCst);
                input.x.clone()
            },
        );
        f.call(&In { x: Array::zeros(&[]) }).unwrap();
        f.call(&In { x: Array::zeros(&[]) }).unwrap();
        assert_eq!(traces.load(Ordering::SeqCst), 2);
    }
}
