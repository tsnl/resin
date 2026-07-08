use resin_core::Tree;
use serde::{Deserialize, Serialize};

use crate::error::IrError;

/// Index into an [`IrProgram`](crate::program::IrProgram)'s `buffers` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferRef(pub usize);

/// Index into an [`IrProgram`](crate::program::IrProgram)'s `buffer_views` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferViewRef(pub usize);

impl BufferRef {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

impl BufferViewRef {
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    pub const fn index(self) -> usize {
        self.0
    }
}

pub(crate) trait RefIndex {
    fn index(self) -> usize;
}

impl RefIndex for BufferRef {
    fn index(self) -> usize {
        self.0
    }
}

impl RefIndex for BufferViewRef {
    fn index(self) -> usize {
        self.0
    }
}

pub(crate) fn validate_tree_indices<T, R: RefIndex + Copy>(
    tree: &T,
    len: usize,
    kind: &'static str,
) -> Result<(), IrError>
where
    T: Tree<R>,
{
    let mut error = None;
    tree.for_each_leaf(|r| {
        let index = r.index();
        if error.is_none() && index >= len {
            error = Some(IrError::TreeIndexOutOfRange { kind, index, len });
        }
    });
    if let Some(err) = error {
        return Err(err);
    }
    Ok(())
}