//! Identity-keyed [`Node`] handles.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::sync::Arc;

use crate::node::Node;

/// Identity-keyed handle to a [`Node`]. Equality / hashing / ordering use the
/// allocation address (same as Python `eq=False` nodes), not structural fields.
#[derive(Clone)]
pub struct NodeRef(Arc<Node>);

impl std::fmt::Debug for NodeRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("NodeRef").field(&self.as_ptr()).finish()
    }
}

impl NodeRef {
    pub fn new(node: Node) -> Self {
        Self(Arc::new(node))
    }

    pub fn from_arc(arc: Arc<Node>) -> Self {
        Self(arc)
    }

    pub fn as_arc(&self) -> &Arc<Node> {
        &self.0
    }

    pub fn into_arc(self) -> Arc<Node> {
        self.0
    }

    pub fn as_ptr(&self) -> *const Node {
        Arc::as_ptr(&self.0)
    }
}

impl Deref for NodeRef {
    type Target = Node;

    fn deref(&self) -> &Node {
        &self.0
    }
}

impl PartialEq for NodeRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for NodeRef {}

impl Hash for NodeRef {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ptr().hash(state);
    }
}

impl PartialOrd for NodeRef {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NodeRef {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ptr().cmp(&other.as_ptr())
    }
}
