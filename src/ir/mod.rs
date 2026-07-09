//! Intermediate representation: kernels over buffers and views.
//!
//! Layout:
//!
//! - [`program`] — buffers, views, dispatches, kernels, validation; elementwise
//!   bodies live in [`program::expr`]
//! - [`accessor`] — strided addressing (broadcast / transpose / squeeze)
//! - [`optimize`] — IR→IR passes (kernel fusion, tiling, …)

mod accessor;
pub mod optimize;
mod program;

pub use accessor::{Accessor, dense_pitch, element_count};
pub use program::{
    Buffer, BufferRef, BufferView, BufferViewRef, Dispatch, Element, Error, Expr, Kernel, Program,
    TILE, TILE_LANES,
};
