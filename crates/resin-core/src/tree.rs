use super::*;

pub trait Tree<T> {
    fn flatten(&self, dir: List<TreePathPart>) -> impl Iterator<Item = TreeLeaf<T>>;
    fn unflatten(leaves: impl Iterator<Item = TreeLeaf<T>>) -> Self;
}

pub struct TreeLeaf<T> {
    path: List<TreePathPart>,
    leaf: T,
}

pub enum TreePathPart {
    Field(&'static str),
    Index(usize),
}
