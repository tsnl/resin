use resin_core::*;

mod debug_print;
pub mod grad;
pub mod prelude;
mod tensor;

pub use debug_print::{debug_print, debug_str, dedent, refcount};
pub use grad::{GradError, grad};
