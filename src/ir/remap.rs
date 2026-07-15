//! Data-movement kernel descriptors (gather / scatter).
//!
//! One remap kernel covers data movement pure accessor views cannot express:
//! dynamically indexed row permutations and static affine scatter/gather
//! (pad/embed/crop). Slicing, broadcast, permute, and unfold stay views.

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

    /// `out[map(coords)] = source[coords]` into a cleared output (pad/embed).
    ///
    /// Args: `[source]`; `map.shape` equals the source shape and addresses the
    /// **output** buffer. Threads iterate the source shape.
    ScatterView { map: Accessor },

    /// `out[coords] = source[decode(map(coords))]` (OOR → zero).
    ///
    /// Args: `[source]`; `map.shape` equals the output shape. `map` yields a
    /// **dense-logical linear index** into `source`'s shape (then addressed
    /// through the source view, so broadcast sources work). Indices past
    /// `element_count(source.shape)` read as zero. Prefer [`ScatterView`] for
    /// padding (unsigned maps cannot express negative offsets).
    GatherView { map: Accessor },
}

impl RemapInfo {
    pub fn name(&self) -> &'static str {
        match self {
            RemapInfo::GatherRows => "gather_rows",
            RemapInfo::ScatterRows { .. } => "scatter_rows",
            RemapInfo::ScatterView { .. } => "scatter_view",
            RemapInfo::GatherView { .. } => "gather_view",
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
