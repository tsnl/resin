//! JAX-style JIT: bind a trace function to a backend, call it with arrays.
//!
//! ```no_run
//! # use resin::{Tree, dsl::Tensor, jit::{DeviceArray, HostArray, CpuJit, Jit}};
//! let double = CpuJit.jit(|x: &Tensor| x.clone() + x.clone());
//! let out = double
//!     .call_host(&HostArray::from_f32(&[2], &[1.0, 2.0]))
//!     .unwrap();
//! assert_eq!(out.host().unwrap().data(), &[2.0, 4.0]);
//! ```
//!
//! Each call traces the function over parameter placeholders, compiles on
//! cache miss (keyed by parameter shapes + element types), and invokes the
//! lowered program. Use [`JittedFn::compile`] to trace from abstract
//! [`Tensor::parameter`] trees (shapes only) without a dummy batch or run.
//!
//! ## Host vs device arrays
//!
//! - [`HostArray`]: process memory — staging for IO, tests, and dataloaders.
//! - [`Jit::Array`]: resident storage for a backend. On CPU this *is*
//!   [`HostArray`]; on wgpu it is a GPU buffer ([`wgpu::Array`]).
//! - [`Jit::upload`] / [`Jit::upload_tree`] move host → device.
//! - [`DeviceArray::host`] / [`DeviceArray::scalar`] read back (may sync GPU).
//! - [`JittedFn::call`] takes device arrays; [`JittedFn::call_host`] uploads
//!   a host tree then calls (handy for demos and all-host pipelines).

pub mod cpu;
#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use cpu::CpuJit;
#[cfg(feature = "wgpu")]
pub use wgpu::{Array, WgpuJit};

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
    #[error(transparent)]
    Ir(#[from] ir::Error),
    #[error("expected {expected} elements, got {got}")]
    Size { expected: usize, got: usize },
    #[error("param/output leaf count does not match compiled program")]
    LeafCount,
    #[error("element type mismatch: expected {expected:?}, got {got:?}")]
    ElementType {
        expected: ElementType,
        got: ElementType,
    },
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

/// Process-memory array: staging for IO, tests, and dataloaders.
///
/// On [`CpuJit`], this is also [`Jit::Array`] (compute and staging share
/// storage). On wgpu, upload into [`wgpu::Array`] before [`JittedFn::call`],
/// or use [`JittedFn::call_host`].
#[derive(Debug, Clone, PartialEq)]
pub struct HostArray {
    shape: Box<[usize]>,
    data: ArrayData,
}

impl HostArray {
    pub fn zeros(shape: &[usize]) -> Self {
        Self::zeros_typed(shape, ElementType::F32)
    }

    pub fn zeros_typed(shape: &[usize], element_type: ElementType) -> Self {
        let n = ir::element_count(shape);
        let data = match element_type {
            ElementType::F32 => ArrayData::F32(vec![0.0; n].into()),
            ElementType::U32 => ArrayData::U32(vec![0; n].into()),
        };
        Self {
            shape: shape.into(),
            data,
        }
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
            ArrayData::U32(_) => panic!("HostArray::data requires f32"),
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        match &mut self.data {
            ArrayData::F32(v) => v,
            ArrayData::U32(_) => panic!("HostArray::data_mut requires f32"),
        }
    }

    pub fn data_u32(&self) -> &[u32] {
        match &self.data {
            ArrayData::U32(v) => v,
            ArrayData::F32(_) => panic!("HostArray::data_u32 requires u32"),
        }
    }

    pub fn data_u32_mut(&mut self) -> &mut [u32] {
        match &mut self.data {
            ArrayData::U32(v) => v,
            ArrayData::F32(_) => panic!("HostArray::data_u32_mut requires u32"),
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
        assert!(
            self.shape.is_empty(),
            "scalar() requires rank 0, got {:?}",
            self.shape
        );
        self.data()[0]
    }
}

/// Backend-resident array: shape/etype always; bytes live on that backend's device.
///
/// [`host`](DeviceArray::host) materializes a [`HostArray`]. On wgpu that is
/// the only path that waits for GPU completion.
pub trait DeviceArray: Clone {
    fn shape(&self) -> &[usize];
    fn element_type(&self) -> ElementType;
    fn host(&self) -> Result<HostArray, Error>;

    /// Rank-0 f32 value (downloads to host first; may sync the GPU).
    fn scalar(&self) -> Result<f32, Error> {
        Ok(self.host()?.scalar())
    }
}

impl DeviceArray for HostArray {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn element_type(&self) -> ElementType {
        self.data.element_type()
    }

    fn host(&self) -> Result<HostArray, Error> {
        Ok(self.clone())
    }
}

/// An execution backend: lowers an IR program to an artifact and runs it.
///
/// `invoke` receives parameter and output leaves in the same order as
/// [`ir::Program::params`] / [`ir::Program::sinks`]. Outputs are allocated with
/// [`Jit::alloc`] (traced shapes); the backend fills them.
pub trait Jit: Clone {
    type Artifact: Clone;
    /// Device-resident array type for this backend.
    type Array: DeviceArray;

    fn lower(&self, program: &ir::Program) -> Result<Self::Artifact, Error>;

    /// Empty output leaf (zeros on host; uninitialized storage buffer on GPU).
    fn alloc(&self, shape: &[usize], element_type: ElementType) -> Result<Self::Array, Error>;

    /// Copy a [`HostArray`] onto this backend's device (no GPU wait on wgpu).
    fn upload(&self, host: &HostArray) -> Result<Self::Array, Error>;

    /// Upload every leaf of a host tree. Models and large params that should
    /// stay device-local across steps use this once, then [`JittedFn::call`].
    fn upload_tree<T: Tree<HostArray>>(
        &self,
        tree: &T,
    ) -> Result<T::Mapped<Self::Array>, Error> {
        tree.try_map(&mut |h| self.upload(h))
    }

    fn invoke(
        &self,
        artifact: &Self::Artifact,
        params: &[&Self::Array],
        outputs: &mut [&mut Self::Array],
    ) -> Result<(), Error>;

    /// Bind `f` to this backend, producing a callable that traces, compiles,
    /// and runs. `P` is the **abstract** parameter tree (`Tensor` leaves) that
    /// `f` receives; [`JittedFn::call`] takes the same tree of device arrays.
    fn jit<P, O, F>(self, f: F) -> JittedFn<Self, P, O, F>
    where
        P: Tree<Tensor>,
        O: Tree<Tensor>,
    {
        JittedFn {
            jit: self,
            f,
            cache: Mutex::new(HashMap::new()),
            _io: PhantomData,
        }
    }
}

/// Output leaf shapes (and etypes) cached with the compiled artifact so we can
/// allocate device outputs without re-tracing.
type OutShapes<O> = <O as Tree<Tensor>>::Mapped<(Box<[usize]>, ElementType)>;

struct Compiled<J: Jit, O: Tree<Tensor>> {
    artifact: J::Artifact,
    out_shapes: OutShapes<O>,
}

/// A traceable function bound to a [`Jit`] backend.
///
/// `P` is the abstract parameter tree (`Tensor` leaves — usually
/// [`Tensor::parameter`] placeholders). [`call`](Self::call) takes the same
/// tree of device-resident arrays (`P::Mapped<J::Array>`).
///
/// **Trace once per shape signature** (like the benches): compile is cached, and
/// subsequent `call`s only allocate outputs + `invoke`. Prefer
/// [`compile`](Self::compile) to warm that cache from shapes alone.
pub struct JittedFn<J: Jit, P, O, F>
where
    O: Tree<Tensor>,
{
    jit: J,
    f: F,
    cache: Mutex<HashMap<ShapeKey, Compiled<J, O>>>,
    _io: PhantomData<fn(&P) -> O>,
}

impl<J, P, O, F> JittedFn<J, P, O, F>
where
    J: Jit,
    P: Tree<Tensor>,
    O: Tree<Tensor>,
    F: Fn(&P) -> O,
    // Array tree for params has the same structure as `P`; mapping leaves to
    // `Tensor` recovers `P` (needed when `call` builds placeholders from arrays).
    P::Mapped<J::Array>: Tree<J::Array, Mapped<Tensor> = P>,
    // Shape skeleton has the same tree shape as `O`; mapping leaves yields `O::Mapped<Array>`.
    OutShapes<O>: Clone + Tree<(Box<[usize]>, ElementType), Mapped<J::Array> = O::Mapped<J::Array>>,
{
    /// Trace, lower, and compile for `params`' shapes without running.
    ///
    /// Leaves should be abstract ([`Tensor::parameter`] / `parameter_typed`);
    /// only shape and element type are used. Later [`call`]s with matching
    /// shapes hit the cache — no dummy dataset batch required.
    pub fn compile(&self, params: &P) -> Result<(), Error> {
        let key = shape_key_tensors(params);
        let mut cache = self.cache.lock().expect("compile cache poisoned");
        if cache.contains_key(&key) {
            return Ok(());
        }
        let compiled = self.trace_compile(params)?;
        cache.insert(key, compiled);
        Ok(())
    }

    /// Run the jitted function with device-resident inputs.
    ///
    /// First call for a parameter shape signature traces, lowers, optimizes,
    /// lays out, and compiles (unless [`compile`] already did). Later calls
    /// with the same shapes only allocate outputs + invoke.
    ///
    /// Returns device-local outputs. Call [`.host()`](DeviceArray::host) or
    /// [`.scalar()`](DeviceArray::scalar) on a leaf only when you need bytes
    /// on the CPU (that may sync the GPU).
    pub fn call(&self, params: &P::Mapped<J::Array>) -> Result<O::Mapped<J::Array>, Error> {
        let key = shape_key_arrays(params);

        let (artifact, out_shapes) = {
            let mut cache = self.cache.lock().expect("compile cache poisoned");
            if let Some(entry) = cache.get(&key) {
                (entry.artifact.clone(), entry.out_shapes.clone())
            } else {
                let abstract_params = params
                    .map(|array| Tensor::parameter_typed(array.shape(), array.element_type()));
                let compiled = self.trace_compile(&abstract_params)?;
                let artifact = compiled.artifact.clone();
                let out_shapes = compiled.out_shapes.clone();
                cache.insert(key, compiled);
                (artifact, out_shapes)
            }
        };

        let mut outputs = out_shapes
            .try_map(&mut |(shape, etype)| self.jit.alloc(shape.as_ref(), *etype))?;
        self.jit
            .invoke(&artifact, &params.leaves(), &mut outputs.leaves_mut())?;
        Ok(outputs)
    }

    /// Upload every host leaf, then [`call`]. Convenience for demos and
    /// pipelines that keep nothing resident across steps.
    pub fn call_host(&self, params: &P::Mapped<HostArray>) -> Result<O::Mapped<J::Array>, Error>
    where
        P::Mapped<HostArray>: Tree<HostArray, Mapped<J::Array> = P::Mapped<J::Array>>,
    {
        let device = self.jit.upload_tree(params)?;
        self.call(&device)
    }

    fn trace_compile(&self, params: &P) -> Result<Compiled<J, O>, Error> {
        let traced_out = (self.f)(params);
        let out_shapes = traced_out.map(|t| (Box::<[usize]>::from(t.shape()), t.element_type()));
        // lower (dense sinks) → optimize? → layout (dead-elim + arena pack) → backend.
        let program = ir::layout::prepare_for_backend(ir::optimize::optimize(ir::lower(
            params,
            &traced_out,
        )?));
        let artifact = self.jit.lower(&program)?;
        Ok(Compiled {
            artifact,
            out_shapes,
        })
    }
}

fn shape_key_tensors<P: Tree<Tensor>>(params: &P) -> ShapeKey {
    params
        .leaves()
        .iter()
        .map(|t| (Box::<[usize]>::from(t.shape()), t.element_type()))
        .collect()
}

fn shape_key_arrays<A: DeviceArray, T: Tree<A>>(params: &T) -> ShapeKey {
    params
        .leaves()
        .iter()
        .map(|a| (Box::<[usize]>::from(a.shape()), a.element_type()))
        .collect()
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
        type Array = HostArray;

        fn lower(&self, _program: &ir::Program) -> Result<(), Error> {
            self.lowers.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn alloc(&self, shape: &[usize], element_type: ElementType) -> Result<HostArray, Error> {
            Ok(HostArray::zeros_typed(shape, element_type))
        }

        fn upload(&self, host: &HostArray) -> Result<HostArray, Error> {
            Ok(host.clone())
        }

        fn invoke(
            &self,
            _artifact: &(),
            _params: &[&HostArray],
            _outputs: &mut [&mut HostArray],
        ) -> Result<(), Error> {
            Ok(())
        }
    }

    #[derive(resin_macros::Tree, Clone)]
    struct In<T> {
        x: T,
    }

    #[test]
    fn call_compiles_once_per_shape_signature() {
        let lowers = Arc::new(AtomicUsize::new(0));
        let f = CountingJit {
            lowers: lowers.clone(),
        }
        .jit(|input: &In<Tensor>| input.x.clone() + input.x.clone());

        f.call(&In {
            x: HostArray::zeros(&[]),
        })
        .unwrap();
        f.call(&In {
            x: HostArray::zeros(&[]),
        })
        .unwrap();
        assert_eq!(
            lowers.load(Ordering::SeqCst),
            1,
            "same shapes hit the cache"
        );

        f.call(&In {
            x: HostArray::zeros(&[3]),
        })
        .unwrap();
        assert_eq!(lowers.load(Ordering::SeqCst), 2, "new shapes recompile");
    }

    #[test]
    fn call_traces_once_per_shape_signature() {
        let traces = Arc::new(AtomicUsize::new(0));
        let traces_in_f = traces.clone();
        let f = CountingJit {
            lowers: Arc::new(AtomicUsize::new(0)),
        }
        .jit(move |input: &In<Tensor>| {
            traces_in_f.fetch_add(1, Ordering::SeqCst);
            input.x.clone()
        });
        f.call(&In {
            x: HostArray::zeros(&[]),
        })
        .unwrap();
        f.call(&In {
            x: HostArray::zeros(&[]),
        })
        .unwrap();
        assert_eq!(traces.load(Ordering::SeqCst), 1, "same shapes: no re-trace");

        f.call(&In {
            x: HostArray::zeros(&[3]),
        })
        .unwrap();
        assert_eq!(traces.load(Ordering::SeqCst), 2, "new shapes re-trace");
    }

    #[test]
    fn compile_from_shapes_then_call_hits_cache() {
        let lowers = Arc::new(AtomicUsize::new(0));
        let traces = Arc::new(AtomicUsize::new(0));
        let traces_in_f = traces.clone();
        let f = CountingJit {
            lowers: lowers.clone(),
        }
        .jit(move |input: &In<Tensor>| {
            traces_in_f.fetch_add(1, Ordering::SeqCst);
            input.x.clone() + input.x.clone()
        });

        f.compile(&In {
            x: Tensor::parameter(&[4]),
        })
        .unwrap();
        assert_eq!(traces.load(Ordering::SeqCst), 1);
        assert_eq!(lowers.load(Ordering::SeqCst), 1);

        f.call(&In {
            x: HostArray::zeros(&[4]),
        })
        .unwrap();
        assert_eq!(traces.load(Ordering::SeqCst), 1, "call reuses compile");
        assert_eq!(lowers.load(Ordering::SeqCst), 1, "no second lower");
    }

    #[test]
    fn call_host_uploads_then_runs() {
        let f = CountingJit {
            lowers: Arc::new(AtomicUsize::new(0)),
        }
        .jit(|input: &In<Tensor>| input.x.clone());
        f.call_host(&In {
            x: HostArray::from_f32(&[2], &[1.0, 2.0]),
        })
        .unwrap();
    }
}
