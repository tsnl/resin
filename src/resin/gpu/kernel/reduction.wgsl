//! Performs multi-axis reduction operations on the GPU using WebGPU.
//!
//! Single-axis reduction:
//!     output[P..., 0, S...] := Σi input[P..., i, S...]
//!
//! Multi-axis reduction:
//!     output[B..., 0, 0] := Σi Σj input[B..., i, j]
//!
//! We can enumerate (i, j) using a single flat index that winds through the R₁ and R₂
//! dimensions:
//!     output[B..., 0, 0] := Σk input[B..., k / R₂, k % R₂]
//!
//! This generalizes to arbitrary numbers of dimensions.

// TODO: WIP
