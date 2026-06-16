//! Performs multi-axis reduction operations on the GPU using WebGPU.
//!
//! Single-axis reduction along the last axis is straightforward enough to understand.
//! E.g. consider a tensor with shape (B..., R) that gets reduced along its last
//! dimension to (B..., 1). To perform this reduction, we can simply iterate over the
//! R dimension i \in [0, R), summing up elements into an accumulator.
//!     output[B..., 0] := Σi input[B..., i]
//!
//! Single-axis reduction along any axis but the last axis is also straightforward if
//! viewed as a permutation + a last-axis reduction. For (P..., R, S...) shape, we can
//! write
//!     output[P..., 0, S...] := Σi input[P..., i, S...]
//!
//! To handle multi-axis reduction, we can view it as a sequence of single-axis
//! reductions. For (B..., R₁, R₂) shape reduced along the last two axes, we can write:
//!     output[B..., 0, 0] := Σi Σj input[B..., i, j]
//! We can enumerate (i, j) using a single flat index that winds through the R₁ and R₂
//! dimensions:
//!     output[B..., 0, 0] := Σk input[B..., k / R₂, k % R₂]
//! This generalizes to arbitrary numbers of dimensions.

// TODO: WIP
