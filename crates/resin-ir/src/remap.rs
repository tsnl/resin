//! Data-movement kernel descriptors (gather / scatter).
//!
//! One general remap kernel covers all data movement that pure accessor views
//! cannot express: dynamically indexed row permutations (gather/scatter along
//! axis 0) and strided region writes (pad/embed). Everything else — slicing,
//! broadcast, transpose — stays a view.

use resin_core::{Accessor, BinaryAssocElementOperator};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RemapInfo {
    /// `out[i, tail…] = source[indices[i], tail…]`.
    ///
    /// Args: `[source, indices]`; indices are rank-1 U4, one per output row.
    /// Out-of-range indices clamp to the last source row. Threads iterate the
    /// output shape.
    GatherRows,

    /// `out[indices[i], tail…] ⊕= source[i, tail…]` into a cleared output.
    ///
    /// Args: `[source, indices]`; indices are rank-1 U4, one per source row.
    /// Out-of-range indices are dropped. `operator: None` overwrites
    /// (unordered for duplicate indices); `Some(Add)` accumulates atomically.
    /// Threads iterate the source shape.
    ScatterRows {
        operator: Option<BinaryAssocElementOperator>,
    },

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
