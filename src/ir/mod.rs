//! Intermediate representation: kernels over buffers and views.
//!
//! Layout:
//!
//! - [`lower`] — DSL tensor graph → [`Program`] (includes sink densification)
//! - [`program`] — buffers, views, dispatches, kernels, validation; elementwise
//!   bodies live in [`program::expr`]
//! - [`accessor`] — strided addressing (broadcast / transpose / squeeze)
//! - [`remap`] — gather / scatter kernel descriptors
//! - [`optimize`] — optional IR→IR rewrites (kernel fusion, …)
//! - [`layout`] — final backend prep (dead-elim + arena packing); always runs
//!
//! Pipeline: `lower → optimize? → layout::prepare_for_backend → jit`.
//!
//! Invariants (enforced by [`Program::validate`]):
//!
//! - **Dense kernel outputs** — every dispatch writes a C-contiguous
//!   [`Accessor`] ([`Accessor::is_dense`]). Args may be non-dense; see
//!   [`program`].
//! - **Dense sinks** — [`lower`] densifies strided sinks up-front so backends
//!   can copy each out as one block, and so optimize can see those copies.

mod accessor;
pub mod layout;
mod lower;
pub mod optimize;
mod program;
mod remap;

pub use accessor::{Accessor, dense_pitch, element_count};
pub use lower::lower;
pub use program::{
    Buffer, BufferData, BufferRef, BufferView, BufferViewRef, Dispatch, Error, Expr, Kernel,
    Program,
};
pub use remap::RemapInfo;
