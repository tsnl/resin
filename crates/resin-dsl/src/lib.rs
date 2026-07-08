use resin_core::*;

mod debug_print;
pub mod grad;
pub mod prelude;
pub mod scan;
pub mod sort;
mod tensor;

pub use debug_print::{debug_print, debug_str, dedent, refcount};
pub use grad::{grad, grad_wrt, GradError};
pub use scan::{
    cumprod, cumprod_exclusive, cumsum, cumsum_exclusive, scan, scan_exclusive, shift_axis,
};
pub use sort::{argsort_f32, argsort_u32, float_sort_key};
pub use tensor::{ElementOperator, ElementType, IndexKeyElement, ScatterOp, Tensor, TensorKind};
