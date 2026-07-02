use std::collections::BTreeMap;

use crate::{format_param_path, path_name, path_starts_with, path_tail, TreePath, TreePathElement};

/// Peekable leaf stream in [`crate::Tree::flatten`] visit order.
pub struct TreeLeaves<L, I> {
    iter: I,
    peek: Option<(TreePath, L)>,
}

impl<L, I> TreeLeaves<L, I>
where
    I: Iterator<Item = (TreePath, L)>,
{
    pub fn new(leaves: impl IntoIterator<Item = (TreePath, L), IntoIter = I>) -> Self {
        Self {
            iter: leaves.into_iter(),
            peek: None,
        }
    }

    pub fn peek_path(&mut self) -> Option<&TreePath> {
        if self.peek.is_none() {
            self.peek = self.iter.next();
        }
        self.peek.as_ref().map(|(path, _)| path)
    }

    pub fn pop(&mut self) -> Option<(TreePath, L)> {
        if let Some(item) = self.peek.take() {
            return Some(item);
        }
        self.iter.next()
    }

    pub fn is_empty(&mut self) -> bool {
        if self.peek.is_some() {
            return false;
        }
        match self.iter.next() {
            None => true,
            Some(item) => {
                self.peek = Some(item);
                false
            }
        }
    }

    pub fn assert_consumed(&mut self, context: &str) {
        let Some((path, _)) = self.pop() else {
            return;
        };
        let mut extra = vec![format_param_path(&path)];
        while let Some((path, _)) = self.pop() {
            extra.push(format_param_path(&path));
        }
        panic!(
            "Tree::unflatten for {context}: {} unmatched leaves: {extra:?}",
            extra.len()
        );
    }
}

/// Pop the next leaf and assert its path is exactly `[name]`.
pub fn take_next_leaf<L, I>(leaves: &mut TreeLeaves<L, I>, name: &str) -> L
where
    I: Iterator<Item = (TreePath, L)>,
{
    let want = path_name(name);
    let (path, leaf) = leaves
        .pop()
        .unwrap_or_else(|| panic!("Tree::unflatten: missing leaf at `{name}`"));
    assert_eq!(
        path,
        want,
        "Tree::unflatten: expected path `{name}`, got `{}`",
        format_param_path(&path)
    );
    leaf
}

/// Pop the next leaf at `[name]` if present (optional fields in visit order).
pub fn take_optional_leaf<L, I>(leaves: &mut TreeLeaves<L, I>, name: &str) -> Option<L>
where
    I: Iterator<Item = (TreePath, L)>,
{
    let want = path_name(name);
    if leaves.peek_path() == Some(&want) {
        Some(leaves.pop().expect("peek implied leaf").1)
    } else {
        None
    }
}

/// Drain consecutive head leaves whose first segment is `name`, stripping that prefix.
pub fn take_named_children<L, I>(
    leaves: &mut TreeLeaves<L, I>,
    name: &str,
) -> TreeLeaves<L, std::vec::IntoIter<(TreePath, L)>>
where
    I: Iterator<Item = (TreePath, L)>,
{
    let want = TreePathElement::name(name);
    let mut taken = Vec::new();
    while leaves
        .peek_path()
        .is_some_and(|path| path_starts_with(path, &want))
    {
        let (path, leaf) = leaves.pop().expect("peek implied leaf");
        taken.push((path_tail(path), leaf));
    }
    TreeLeaves::new(taken)
}

/// Drain `name` / `index` / … leaves (visit order), grouped by index in order `0..=max`.
pub fn take_named_indexed_children<L, I>(
    leaves: &mut TreeLeaves<L, I>,
    name: &str,
) -> Vec<TreeLeaves<L, std::vec::IntoIter<(TreePath, L)>>>
where
    I: Iterator<Item = (TreePath, L)>,
{
    let mut children = take_named_children(leaves, name);
    let mut groups: BTreeMap<usize, Vec<(TreePath, L)>> = BTreeMap::new();
    while let Some((path, leaf)) = children.pop() {
        match path.front() {
            Some(TreePathElement::Index(i)) => {
                groups.entry(*i).or_default().push((path_tail(path), leaf));
            }
            _ => panic!(
                "expected index segment under field {name:?}, got {}",
                format_param_path(&path)
            ),
        }
    }
    match groups.keys().max().copied() {
        None => Vec::new(),
        Some(max) => (0..=max)
            .map(|i| TreeLeaves::new(groups.remove(&i).unwrap_or_default()))
            .collect(),
    }
}

pub(crate) fn take_indexed_children<L, I>(
    leaves: &mut TreeLeaves<L, I>,
) -> Vec<TreeLeaves<L, std::vec::IntoIter<(TreePath, L)>>>
where
    I: Iterator<Item = (TreePath, L)>,
{
    let mut groups: BTreeMap<usize, Vec<(TreePath, L)>> = BTreeMap::new();
    while let Some((path, leaf)) = leaves.pop() {
        match path.front() {
            Some(TreePathElement::Index(i)) => {
                groups.entry(*i).or_default().push((path_tail(path), leaf));
            }
            _ => panic!(
                "Tree::unflatten for Vec: expected index path prefix, got {}",
                format_param_path(&path)
            ),
        }
    }
    match groups.keys().max().copied() {
        None => Vec::new(),
        Some(max) => (0..=max)
            .map(|i| TreeLeaves::new(groups.remove(&i).unwrap_or_default()))
            .collect(),
    }
}
