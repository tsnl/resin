pub trait Tree<T: Clone> {
    type Mapped<U: Clone>: Tree<U>; // = Self<U>

    fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> Self::Mapped<U>;

    fn for_each_leaf(&self, f: impl FnMut(&T));
}

/// A bare leaf is a single-node tree.
impl<T: Clone> Tree<T> for T {
    type Mapped<U: Clone> = U;

    fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> U {
        f(self)
    }

    fn for_each_leaf(&self, mut f: impl FnMut(&T)) {
        f(self);
    }
}

impl<T: Clone> Tree<T> for Vec<T> {
    type Mapped<U: Clone> = Vec<U>;

    fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> Vec<U> {
        self.iter().map(|leaf| f(leaf)).collect()
    }

    fn for_each_leaf(&self, mut f: impl FnMut(&T)) {
        for leaf in self {
            f(leaf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate as resin_core;
    use resin_macros::*;

    #[derive(Tree)]
    struct TestNode<T> {
        x: T,
        y: Vec<T>,
    }

    #[derive(Tree)]
    enum TestEnum<T> {
        One(T),
        Pair { a: T, b: T },
    }

    #[test]
    fn map_doubles_leaves() {
        let tree = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        let mapped = tree.map(|&v| v * 2);
        assert_eq!(mapped.x, 2);
        assert_eq!(mapped.y, vec![4, 6]);
    }

    #[test]
    fn map_enum_variants() {
        let tree = TestEnum::Pair { a: 1, b: 2 };
        let mapped = tree.map(|&v| v * 2);
        assert!(matches!(mapped, TestEnum::Pair { a: 2, b: 4 }));
    }

    #[test]
    fn for_each_leaf_visits_all_leaves() {
        let tree = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        let mut leaves = Vec::new();
        tree.for_each_leaf(|v| leaves.push(*v));
        leaves.sort_unstable();
        assert_eq!(leaves, vec![1, 2, 3]);
    }
}
