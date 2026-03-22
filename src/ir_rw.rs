//! `ir_rw.rs` = IR rewriter
//!
//! Necessary passes:
//! - Inlining: need to inline all `Call` nodes that aren't tail calls, if this overflows a max inline depth, then just panic. Output graphs must be finite. We may also inline calls that are in tail position in the future for optimization, but we need to allow these to express looping behavior.
//! - Grad elimination: Wherever we have Grad nodes, we need to replace them with the corresponding gradient function. This function is computed via autodiff. Note that Call nodes are an autodiff boundary.
//!
//! Nice to have optimization passes:
//! - Constant folding
//! - Common subexpression hoisting
//! - Dead code elimination
//!
//! Out of scope (handled later in a lower level codegen optimizer) (still put this in the ir_rw comment)
//! - Kernel fusion
//! - Fused kernel scheduling
