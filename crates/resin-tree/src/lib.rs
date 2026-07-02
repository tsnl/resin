//! Host module / param trees (Python `PyTree` analogue).
//!
//! Paths are [`TreePath`] (`im::Vector` with `push_front`). [`Tree::unflatten`]
//! consumes a [`TreeLeaves`] stream in [`Tree::flatten`] visit order.

extern crate self as resin_tree;

mod leaf_stream;

use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

pub use leaf_stream::{
    take_named_children, take_named_indexed_children, take_next_leaf, take_optional_leaf,
    TreeLeaves,
};
pub use resin_tree_derive::Tree;

/// One segment of a structured param path (`layers` / `0` / `weight`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TreePathElement {
    Name(Arc<str>),
    Index(usize),
}

impl TreePathElement {
    pub fn name(s: impl AsRef<str>) -> Self {
        Self::Name(Arc::from(s.as_ref()))
    }
}

impl fmt::Display for TreePathElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Name(s) => f.write_str(s),
            Self::Index(i) => write!(f, "{i}"),
        }
    }
}

/// Structured path to a leaf (e.g. `layers` / `0` / `weight`).
pub type TreePath = im::Vector<TreePathElement>;

/// Empty path (single-leaf root).
pub fn path_empty() -> TreePath {
    TreePath::new()
}

/// Path with one segment.
pub fn path_single(el: TreePathElement) -> TreePath {
    let mut path = TreePath::new();
    path.push_front(el);
    path
}

/// Path `[name]`.
pub fn path_name(name: &str) -> TreePath {
    path_single(TreePathElement::name(name))
}

/// Format `path` as a dotted buffer name (`layers.0.weight`).
pub fn format_param_path(path: &TreePath) -> String {
    let mut s = String::new();
    for (i, el) in path.iter().enumerate() {
        if i > 0 {
            s.push('.');
        }
        use std::fmt::Write;
        let _ = write!(s, "{el}");
    }
    s
}

/// `prefix` (host register name) + structured path → dotted buffer name.
pub fn join_param_path(prefix: &str, path: &TreePath) -> String {
    let rest = format_param_path(path);
    if prefix.is_empty() {
        rest
    } else if rest.is_empty() {
        prefix.to_string()
    } else {
        format!("{prefix}.{rest}")
    }
}

fn prepend(el: TreePathElement, path: &TreePath) -> TreePath {
    let mut out = path.clone();
    out.push_front(el);
    out
}

/// Prepend a field-name segment (for `#[derive(Tree)]` and hand-written impls).
pub fn prepend_name(name: &str, path: &TreePath) -> TreePath {
    prepend(TreePathElement::name(name), path)
}

/// Prepend `name` / `index` before nested path segments.
pub fn prepend_name_index(name: &str, index: usize, path: &TreePath) -> TreePath {
    let mut out = path.clone();
    out.push_front(TreePathElement::Index(index));
    out.push_front(TreePathElement::name(name));
    out
}

pub(crate) fn path_tail(mut path: TreePath) -> TreePath {
    path.pop_front();
    path
}

pub(crate) fn path_starts_with(path: &TreePath, head: &TreePathElement) -> bool {
    path.front() == Some(head)
}

/// Drop the first path segment.
pub fn path_drop_first(path: &TreePath) -> TreePath {
    let mut out = path.clone();
    out.pop_front();
    out
}

/// Total order on paths: `Name` before `Index`; lexicographic / numeric within kind.
pub fn cmp_path(a: &TreePath, b: &TreePath) -> Ordering {
    for (ae, be) in a.iter().zip(b.iter()) {
        let ord = match (ae, be) {
            (TreePathElement::Name(a), TreePathElement::Name(b)) => a.as_ref().cmp(b.as_ref()),
            (TreePathElement::Name(_), TreePathElement::Index(_)) => Ordering::Less,
            (TreePathElement::Index(_), TreePathElement::Name(_)) => Ordering::Greater,
            (TreePathElement::Index(a), TreePathElement::Index(b)) => a.cmp(b),
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    a.len().cmp(&b.len())
}

/// Structured host tree of leaves (e.g. `View` params).
///
/// Shape lives in the concrete type (`Linear`, `Mlp`, `Vec<…>`); there is no live
/// Object/Array/Leaf enum.
pub trait Tree: Sized {
    type Leaf;
    type Map<U>: Tree<Leaf = U>;

    /// Lazy depth-first leaf stream `(path, leaf)` in field-declaration order.
    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_;

    /// Reconstruct `Self` from `(path, leaf)` pairs in [`flatten`] visit order.
    fn unflatten(leaves: impl IntoIterator<Item = (TreePath, Self::Leaf)>) -> Self {
        let mut stream = TreeLeaves::new(leaves);
        let out = Self::consume_unflatten(&mut stream);
        stream.assert_consumed(std::any::type_name::<Self>());
        out
    }

    /// Visit-order rebuild from a shared leaf stream (used by nested types and tuples).
    #[doc(hidden)]
    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>;

    /// Map each `(path, leaf)` via `f`, then [`unflatten`].
    fn map<U>(&self, mut f: impl FnMut(&TreePath, &Self::Leaf) -> U) -> Self::Map<U> {
        <Self::Map<U> as Tree>::unflatten(self.flatten().map(|(path, leaf)| {
            let mapped = f(&path, leaf);
            (path, mapped)
        }))
    }

    /// Zip two same-shape trees leaf-wise (paths must match), then [`unflatten`].
    fn zip<U, O>(
        &self,
        other: &Self::Map<U>,
        mut f: impl FnMut(&TreePath, &Self::Leaf, &U) -> O,
    ) -> Self::Map<O> {
        <Self::Map<O> as Tree>::unflatten(self.flatten().zip(other.flatten()).map(
            |((path, a), (path_b, b))| {
                debug_assert_eq!(path, path_b, "Tree::zip: path mismatch");
                let zipped = f(&path, a, b);
                (path, zipped)
            },
        ))
    }
}

impl<T: Tree> Tree for Vec<T> {
    type Leaf = T::Leaf;
    type Map<U> = Vec<T::Map<U>>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.iter().enumerate().flat_map(|(i, x)| {
            x.flatten()
                .map(move |(path, leaf)| (prepend(TreePathElement::Index(i), &path), leaf))
        })
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        leaf_stream::take_indexed_children(stream)
            .into_iter()
            .map(|mut group| {
                let item = T::consume_unflatten(&mut group);
                group.assert_consumed("Vec element");
                item
            })
            .collect()
    }
}

impl<T: Tree> Tree for Option<T> {
    type Leaf = T::Leaf;
    type Map<U> = Option<T::Map<U>>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.iter().flat_map(Tree::flatten)
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        if stream.is_empty() {
            None
        } else {
            Some(T::consume_unflatten(stream))
        }
    }
}

/// Single-leaf tree. Use for bare values at a tree root (e.g. one sink `View`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Leaf<T>(pub T);

impl<T> From<T> for Leaf<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<T> Tree for Leaf<T> {
    type Leaf = T;
    type Map<U> = Leaf<U>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &T)> + '_ {
        std::iter::once((path_empty(), &self.0))
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<T, I>) -> Self
    where
        I: Iterator<Item = (TreePath, T)>,
    {
        match stream.pop() {
            None => panic!("Tree::unflatten for Leaf: missing leaf"),
            Some((path, value)) => {
                assert!(
                    path.is_empty(),
                    "Tree::unflatten for Leaf: unexpected path {path:?}"
                );
                Leaf(value)
            }
        }
    }
}

/// Name-prefix wrapper: flattens as `name` / nested path.
#[derive(Debug, Clone)]
pub struct Named<T> {
    pub name: Arc<str>,
    pub value: T,
}

impl<T> Named<T> {
    pub fn new(name: impl AsRef<str>, value: T) -> Self {
        Self {
            name: Arc::from(name.as_ref()),
            value,
        }
    }
}

/// Convenience constructor for [`Named`].
pub fn named<T>(name: impl AsRef<str>, value: T) -> Named<T> {
    Named::new(name, value)
}

impl<T: Tree> Tree for Named<T> {
    type Leaf = T::Leaf;
    type Map<U> = Named<T::Map<U>>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        let name = Arc::clone(&self.name);
        self.value.flatten().map(move |(path, leaf)| {
            (
                prepend(TreePathElement::Name(Arc::clone(&name)), &path),
                leaf,
            )
        })
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let name = stream
            .peek_path()
            .and_then(|path| match path.front() {
                Some(TreePathElement::Name(s)) => Some(Arc::clone(s)),
                _ => None,
            })
            .unwrap_or_else(|| panic!("Tree::unflatten for Named: empty or missing name prefix"));
        let mut child = take_named_children(stream, name.as_ref());
        let value = T::consume_unflatten(&mut child);
        child.assert_consumed("Named child");
        Named { name, value }
    }
}

/// Empty tree (no leaves). Useful for const-only `compile` inputs.
#[derive(Debug, Clone, Copy, Default)]
pub struct Empty<L = ()>(PhantomData<fn() -> L>);

impl<L> Empty<L> {
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<L> Tree for Empty<L> {
    type Leaf = L;
    type Map<U> = Empty<U>;

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &L)> + '_ {
        std::iter::empty()
    }

    fn consume_unflatten<I>(_stream: &mut TreeLeaves<L, I>) -> Self
    where
        I: Iterator<Item = (TreePath, L)>,
    {
        Empty(PhantomData)
    }
}

// Tuples concatenate children. Each child should carry its own `Named` prefix
// (no automatic index segments — use `Vec` when you want indices).

impl<A, B> Tree for (A, B)
where
    A: Tree,
    B: Tree<Leaf = A::Leaf>,
{
    type Leaf = A::Leaf;
    type Map<U> = (A::Map<U>, B::Map<U>);

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.0.flatten().chain(self.1.flatten())
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let a = A::consume_unflatten(stream);
        let b = B::consume_unflatten(stream);
        (a, b)
    }
}

impl<A, B, C> Tree for (A, B, C)
where
    A: Tree,
    B: Tree<Leaf = A::Leaf>,
    C: Tree<Leaf = A::Leaf>,
{
    type Leaf = A::Leaf;
    type Map<U> = (A::Map<U>, B::Map<U>, C::Map<U>);

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.0
            .flatten()
            .chain(self.1.flatten())
            .chain(self.2.flatten())
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let a = A::consume_unflatten(stream);
        let b = B::consume_unflatten(stream);
        let c = C::consume_unflatten(stream);
        (a, b, c)
    }
}

impl<A, B, C, D> Tree for (A, B, C, D)
where
    A: Tree,
    B: Tree<Leaf = A::Leaf>,
    C: Tree<Leaf = A::Leaf>,
    D: Tree<Leaf = A::Leaf>,
{
    type Leaf = A::Leaf;
    type Map<U> = (A::Map<U>, B::Map<U>, C::Map<U>, D::Map<U>);

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.0
            .flatten()
            .chain(self.1.flatten())
            .chain(self.2.flatten())
            .chain(self.3.flatten())
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let a = A::consume_unflatten(stream);
        let b = B::consume_unflatten(stream);
        let c = C::consume_unflatten(stream);
        let d = D::consume_unflatten(stream);
        (a, b, c, d)
    }
}

impl<A, B, C, D, E> Tree for (A, B, C, D, E)
where
    A: Tree,
    B: Tree<Leaf = A::Leaf>,
    C: Tree<Leaf = A::Leaf>,
    D: Tree<Leaf = A::Leaf>,
    E: Tree<Leaf = A::Leaf>,
{
    type Leaf = A::Leaf;
    type Map<U> = (A::Map<U>, B::Map<U>, C::Map<U>, D::Map<U>, E::Map<U>);

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.0
            .flatten()
            .chain(self.1.flatten())
            .chain(self.2.flatten())
            .chain(self.3.flatten())
            .chain(self.4.flatten())
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let a = A::consume_unflatten(stream);
        let b = B::consume_unflatten(stream);
        let c = C::consume_unflatten(stream);
        let d = D::consume_unflatten(stream);
        let e = E::consume_unflatten(stream);
        (a, b, c, d, e)
    }
}

impl<A, B, C, D, E, F> Tree for (A, B, C, D, E, F)
where
    A: Tree,
    B: Tree<Leaf = A::Leaf>,
    C: Tree<Leaf = A::Leaf>,
    D: Tree<Leaf = A::Leaf>,
    E: Tree<Leaf = A::Leaf>,
    F: Tree<Leaf = A::Leaf>,
{
    type Leaf = A::Leaf;
    type Map<U> = (
        A::Map<U>,
        B::Map<U>,
        C::Map<U>,
        D::Map<U>,
        E::Map<U>,
        F::Map<U>,
    );

    fn flatten(&self) -> impl Iterator<Item = (TreePath, &Self::Leaf)> + '_ {
        self.0
            .flatten()
            .chain(self.1.flatten())
            .chain(self.2.flatten())
            .chain(self.3.flatten())
            .chain(self.4.flatten())
            .chain(self.5.flatten())
    }

    fn consume_unflatten<I>(stream: &mut TreeLeaves<Self::Leaf, I>) -> Self
    where
        I: Iterator<Item = (TreePath, Self::Leaf)>,
    {
        let a = A::consume_unflatten(stream);
        let b = B::consume_unflatten(stream);
        let c = C::consume_unflatten(stream);
        let d = D::consume_unflatten(stream);
        let e = E::consume_unflatten(stream);
        let f = F::consume_unflatten(stream);
        (a, b, c, d, e, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Linear<T> {
        weight: T,
        bias: Option<T>,
    }

    impl<T> Tree for Linear<T> {
        type Leaf = T;
        type Map<U> = Linear<U>;

        fn flatten(&self) -> impl Iterator<Item = (TreePath, &T)> + '_ {
            std::iter::once((path_name("weight"), &self.weight)).chain(
                self.bias
                    .as_ref()
                    .into_iter()
                    .map(|b| (path_name("bias"), b)),
            )
        }

        fn consume_unflatten<I>(stream: &mut TreeLeaves<T, I>) -> Self
        where
            I: Iterator<Item = (TreePath, T)>,
        {
            Linear {
                weight: take_next_leaf(stream, "weight"),
                bias: take_optional_leaf(stream, "bias"),
            }
        }
    }

    #[test]
    fn flatten_linear_and_vec() {
        let mlp = vec![
            Linear {
                weight: 1i32,
                bias: Some(2),
            },
            Linear {
                weight: 3,
                bias: None,
            },
        ];
        let paths: Vec<_> = mlp
            .flatten()
            .map(|(p, v)| (format_param_path(&p), *v))
            .collect();
        assert_eq!(
            paths,
            vec![
                ("0.weight".into(), 1),
                ("0.bias".into(), 2),
                ("1.weight".into(), 3),
            ]
        );
    }

    #[test]
    fn map_and_zip() {
        let layer = Linear {
            weight: 2i32,
            bias: Some(3),
        };
        let doubled = layer.map(|_path, &v| v * 2);
        assert_eq!(doubled.weight, 4);
        assert_eq!(doubled.bias, Some(6));

        let zipped = layer.zip(&doubled, |_path, &a, &b| a + b);
        assert_eq!(zipped.weight, 6);
        assert_eq!(zipped.bias, Some(9));
    }

    #[test]
    fn named_leaf_and_tuple_paths() {
        let tree = (
            named("xs", Leaf(10i32)),
            named(
                "model",
                Linear {
                    weight: 1,
                    bias: Some(2),
                },
            ),
            named("loss", Leaf(99)),
        );
        let paths: Vec<_> = tree
            .flatten()
            .map(|(p, v)| (format_param_path(&p), *v))
            .collect();
        assert_eq!(
            paths,
            vec![
                ("xs".into(), 10),
                ("model.weight".into(), 1),
                ("model.bias".into(), 2),
                ("loss".into(), 99),
            ]
        );
    }

    #[test]
    fn empty_tree() {
        let e = Empty::<i32>::new();
        assert_eq!(e.flatten().count(), 0);
    }

    #[test]
    fn unflatten_round_trip() {
        let layer = Linear {
            weight: 2i32,
            bias: Some(3),
        };
        let leaves: Vec<_> = layer.flatten().map(|(path, &v)| (path, v)).collect();
        assert_eq!(Linear::unflatten(leaves), layer);
    }

    #[test]
    fn tuple_unflatten_round_trip() {
        let tree = (
            named("xs", Leaf(10i32)),
            named(
                "model",
                Linear {
                    weight: 1,
                    bias: Some(2),
                },
            ),
            named("loss", Leaf(99)),
        );
        let leaves: Vec<_> = tree.flatten().map(|(path, &v)| (path, v)).collect();
        let back = <(Named<Leaf<i32>>, Named<Linear<i32>>, Named<Leaf<i32>>)>::unflatten(leaves);
        assert_eq!(back.0.value.0, 10);
        assert_eq!(back.1.value.weight, 1);
        assert_eq!(back.1.value.bias, Some(2));
        assert_eq!(back.2.value.0, 99);
    }

    #[test]
    #[should_panic(expected = "expected path `weight`")]
    fn unflatten_rejects_wrong_visit_order() {
        let leaves = vec![(path_name("bias"), 3i32), (path_name("weight"), 2i32)];
        let _ = Linear::<i32>::unflatten(leaves);
    }

    #[test]
    fn tree_leaves_does_not_eagerly_pull() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct CountingIter {
            inner: std::vec::IntoIter<(TreePath, i32)>,
            pulls: Rc<Cell<usize>>,
        }

        impl Iterator for CountingIter {
            type Item = (TreePath, i32);

            fn next(&mut self) -> Option<Self::Item> {
                self.pulls.set(self.pulls.get() + 1);
                self.inner.next()
            }
        }

        let pulls = Rc::new(Cell::new(0));
        let leaves = vec![(path_name("weight"), 1), (path_name("bias"), 2)];
        let mut stream = TreeLeaves::new(CountingIter {
            inner: leaves.into_iter(),
            pulls: Rc::clone(&pulls),
        });
        assert_eq!(pulls.get(), 0);
        assert_eq!(stream.peek_path(), Some(&path_name("weight")));
        assert_eq!(pulls.get(), 1);
        let layer = Linear {
            weight: take_next_leaf(&mut stream, "weight"),
            bias: take_optional_leaf(&mut stream, "bias"),
        };
        assert_eq!(layer.weight, 1);
        assert_eq!(layer.bias, Some(2));
        assert_eq!(pulls.get(), 2);
    }

    #[test]
    fn map_with_path() {
        let layer = Linear {
            weight: 2i32,
            bias: Some(3),
        };
        let paths = layer.map(|path, &v| (format_param_path(path), v * 10));
        assert_eq!(paths.weight, ("weight".into(), 20));
        assert_eq!(paths.bias, Some(("bias".into(), 30)));
    }
}
