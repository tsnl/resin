//! Resin: a small, portable compiler for tensor programs.
//!
//! The pipeline has three stages:
//!
//! 1. [`dsl`] — trace an expression graph of [`dsl::Tensor`]s (with
//!    reverse-mode autodiff via [`dsl::grad_wrt`]).
//! 2. [`ir`] — lower the graph to a flat queue of kernel dispatches over
//!    buffers and strided views; optional optimize; then
//!    [`ir::layout::prepare_for_backend`] (dead-elim + arena pack).
//! 3. [`jit`] — run the program on a backend: [`jit::CpuJit`] interprets it,
//!    [`jit::WgpuJit`] emits WGSL and dispatches through wgpu.
//!
//! Structured inputs and outputs are [`Tree`]s (pytrees): derive `Tree` on a
//! struct of tensors and pass it straight to a jitted function.
//!
//! ```no_run
//! use resin::{Tree, dsl::Tensor, jit::{DeviceValue, HostArray, CpuJit, Jit}};
//!
//! #[derive(Tree)]
//! struct Pair<T> {
//!     a: T,
//!     b: T,
//! }
//!
//! let add = CpuJit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
//! let out = add
//!     .call(&Pair {
//!         a: HostArray::from_f32(&[2], &[1.0, 2.0]),
//!         b: HostArray::from_f32(&[2], &[10.0, 20.0]),
//!     })
//!     .unwrap()
//!     .host()
//!     .unwrap();
//! assert_eq!(out.data(), &[11.0, 22.0]);
//! ```
//!
//! Element types are f32 (default) and u32 (indices, masks, bit packing).

// Let the `Tree` derive refer to this crate as `resin` from within itself.
extern crate self as resin;

pub mod dataset;
pub mod dsl;
pub mod ir;
pub mod jit;
pub mod ops;
pub mod tree;

pub use dsl::{IndexKeyElement, ScatterOp};
pub use ops::ElementType;
pub use resin_macros::Tree;
pub use tree::Tree;

#[cfg(test)]
mod tests {
    use crate::Tree;
    use crate::dsl::Tensor;
    use crate::jit::{DeviceValue, HostArray, CpuJit, Jit};

    #[derive(Tree)]
    struct Inputs<T> {
        a: T,
        b: T,
    }

    #[test]
    fn cpu_jit_add() {
        let add = CpuJit.jit(|inputs: &Inputs<Tensor>| inputs.a.clone() + inputs.b.clone());
        let out = add
            .call(&Inputs {
                a: HostArray::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
                b: HostArray::from_f32(&[2, 2], &[10.0, 20.0, 30.0, 40.0]),
            })
            .expect("cpu jit add")
            .host()
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn wgpu_jit_add() {
        use crate::jit::WgpuJit;

        // Skip when the machine has no GPU adapter (CI without Metal/Vulkan).
        if !crate::jit::wgpu::gpu_available() {
            eprintln!("skip wgpu_jit_add: no GPU");
            return;
        }

        let jit = WgpuJit::default();
        let add = jit.jit(|inputs: &Inputs<Tensor>| inputs.a.clone() + inputs.b.clone());
        let out = add
            .call(&Inputs {
                a: jit.upload(&HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0])).unwrap(),
                b: jit.upload(&HostArray::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0])).unwrap(),
            })
            .expect("wgpu jit add")
            .host()
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    }
}
