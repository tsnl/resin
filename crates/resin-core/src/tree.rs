pub trait Tree<T: Clone> {
    type Mapped<U: Clone>: Tree<U>; // = Self<U>

    fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> Self::Mapped<U>;

    fn zip<U: Clone, V: Clone>(
        &self,
        other: &Self::Mapped<U>,
        f: impl Fn(&T, &U) -> V,
    ) -> Self::Mapped<V>;
}

impl<T: Clone> Tree<T> for Vec<T> {
    type Mapped<U: Clone> = Vec<U>;

    fn map<U: Clone>(&self, f: impl Fn(&T) -> U) -> Vec<U> {
        self.iter().map(|leaf| f(leaf)).collect()
    }

    fn zip<U: Clone, V: Clone>(&self, other: &Vec<U>, f: impl Fn(&T, &U) -> V) -> Vec<V> {
        assert_eq!(self.len(), other.len(), "Tree::zip: Vec length mismatch");
        self.iter().zip(other).map(|(a, b)| f(a, b)).collect()
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
    fn zip_adds_leaves() {
        let left = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        let right = TestNode {
            x: 10,
            y: vec![20, 30],
        };
        let zipped = left.zip(&right, |&a, &b| a + b);
        assert_eq!(zipped.x, 11);
        assert_eq!(zipped.y, vec![22, 33]);
    }

    #[test]
    fn zip_enum_requires_matching_variant() {
        let left = TestEnum::Pair { a: 1, b: 2 };
        let right = TestEnum::Pair { a: 10, b: 20 };
        let zipped = left.zip(&right, |&a, &b| a + b);
        assert!(matches!(zipped, TestEnum::Pair { a: 11, b: 22 }));
    }
}
