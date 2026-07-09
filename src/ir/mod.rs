//! Intermediate representation: kernels over buffers and views.
//!
//! Layout:
//!
//! - [`program`] — buffers, views, dispatches, kernels, validation; elementwise
//!   bodies live in [`program::expr`]
//! - [`accessor`] — strided addressing (broadcast / transpose / squeeze)
//! - [`optimize`] — IR→IR passes (kernel fusion, tiling, …)
//!
//! Invariants (enforced by [`Program::validate`]):
//!
//! - **Dense kernel outputs** — every dispatch writes through
//!   [`Accessor::Dense`]. Args may be non-dense; see [`program`].

mod accessor;
pub mod optimize;
mod program;

pub use accessor::{Accessor, Strided, dense_pitch, element_count};
pub use program::{
    Buffer, BufferRef, BufferView, BufferViewRef, Dispatch, Error, Expr, Kernel, Program,
};
