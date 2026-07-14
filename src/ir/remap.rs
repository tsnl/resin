//! Data-movement kernel descriptors (gather / scatter).
//!
//! One remap kernel covers data movement pure accessor views cannot express:
//! dynamically indexed row permutations and strided region writes (pad/embed).
//! Slicing, broadcast, and transpose stay views.

use super::Accessor;
use crate::ops::AssocOp;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RemapInfo {
    /// `out[i, tail…] = source[indices[i], tail…]`.
    ///
    /// Args: `[source, indices]`; indices are rank-1 U32, one per output row.
    /// Out-of-range indices clamp to the last source row. Threads iterate the
    /// output shape.
    GatherRows,

    /// `out[indices[i], tail…] ⊕= source[i, tail…]` into a cleared output.
    ///
    /// Args: `[source, indices]`; indices are rank-1 U32, one per source row.
    /// Out-of-range indices are dropped. `operator: None` overwrites
    /// (unordered for duplicate indices); `Some(Add)` accumulates.
    /// Threads iterate the source shape.
    ScatterRows { operator: Option<AssocOp> },

    /// `out[accessor(coords)] = source[coords]` into a cleared output
    /// (pad/embed a region). Args: `[source]`; `accessor.shape` must equal the
    /// source shape and address the output buffer. Threads iterate the source
    /// shape.
    ScatterView { accessor: Accessor },
}

impl RemapInfo {
    pub fn name(&self) -> &'static str {
        match self {
            RemapInfo::GatherRows => "gather_rows",
            RemapInfo::ScatterRows { .. } => "scatter_rows",
            RemapInfo::ScatterView { .. } => "scatter_view",
        }
    }

    /// Scatter variants write sparsely and require a cleared output.
    pub fn is_scatter(&self) -> bool {
        matches!(
            self,
            RemapInfo::ScatterRows { .. } | RemapInfo::ScatterView { .. }
        )
    }
}
