//! Pytrees: structured containers of leaves, mapped and walked generically.
//!
//! A [`Tree<T>`] is anything holding `T` leaves — a bare leaf, a `Vec`, or a
//! user struct/enum with `#[derive(Tree)]`. All methods visit leaves in a
//! stable order (field declaration order), which the JIT relies on to match
//! parameter and output leaves to compiled buffer slots.

use std::convert::Infallible;

pub trait Tree<T> {
    /// The same container shape with `U` leaves.
    type Mapped<U: Clone>: Tree<U>;

    fn try_map<U: Clone, E>(
        &self,
        f: &mut dyn FnMut(&T) -> Result<U, E>,
    ) -> Result<Self::Mapped<U>, E>;

    fn for_each<'a>(&'a self, f: &mut dyn FnMut(&'a T));

    fn for_each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut T));

    fn map<U: Clone>(&self, mut f: impl FnMut(&T) -> U) -> Self::Mapped<U> {
        match self.try_map(&mut |leaf| Ok::<U, Infallible>(f(leaf))) {
            Ok(mapped) => mapped,
            Err(never) => match never {},
        }
    }

    fn leaves(&self) -> Vec<&T> {
        let mut leaves = Vec::new();
        self.for_each(&mut |leaf| leaves.push(leaf));
        leaves
    }

    fn leaves_mut(&mut self) -> Vec<&mut T> {
        let mut leaves = Vec::new();
        self.for_each_mut(&mut |leaf| leaves.push(leaf));
        leaves
    }
}

/// A bare leaf is a single-node tree.
///
/// The `Clone` bound is not used; it keeps this impl out of method resolution
/// for containers (which don't derive `Clone` just to be trees), so calls like
/// `tree.map(…)` stay unambiguous.
impl<T: Clone> Tree<T> for T {
    type Mapped<U: Clone> = U;

    fn try_map<U: Clone, E>(&self, f: &mut dyn FnMut(&T) -> Result<U, E>) -> Result<U, E> {
        f(self)
    }

    fn for_each<'a>(&'a self, f: &mut dyn FnMut(&'a T)) {
        f(self);
    }

    fn for_each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut T)) {
        f(self);
    }
}

impl<T> Tree<T> for Vec<T> {
    type Mapped<U: Clone> = Vec<U>;

    fn try_map<U: Clone, E>(&self, f: &mut dyn FnMut(&T) -> Result<U, E>) -> Result<Vec<U>, E> {
        self.iter().map(f).collect()
    }

    fn for_each<'a>(&'a self, f: &mut dyn FnMut(&'a T)) {
        for leaf in self {
            f(leaf);
        }
    }

    fn for_each_mut<'a>(&'a mut self, f: &mut dyn FnMut(&'a mut T)) {
        for leaf in self {
            f(leaf);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(resin_macros::Tree)]
    struct TestNode<T> {
        x: T,
        y: Vec<T>,
    }

    #[derive(resin_macros::Tree)]
    enum TestEnum<T> {
        One(T),
        Pair { a: T, b: T },
    }

    #[derive(resin_macros::Tree)]
    struct TupleNode<T>(T, Vec<T>);

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
    fn try_map_propagates_errors() {
        let tree = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        let result: Result<TestNode<i32>, &str> =
            tree.try_map(&mut |&v| if v == 3 { Err("three") } else { Ok(v) });
        assert_eq!(result.err(), Some("three"));
    }

    #[test]
    fn map_enum_variants() {
        let tree = TestEnum::Pair { a: 1, b: 2 };
        let mapped = tree.map(|&v| v * 2);
        assert!(matches!(mapped, TestEnum::Pair { a: 2, b: 4 }));
    }

    #[test]
    fn leaves_visit_in_declaration_order() {
        let tree = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        assert_eq!(tree.leaves(), vec![&1, &2, &3]);
    }

    #[test]
    fn leaves_mut_can_write() {
        let mut tree = TestNode {
            x: 1,
            y: vec![2, 3],
        };
        for leaf in tree.leaves_mut() {
            *leaf *= 10;
        }
        assert_eq!(tree.x, 10);
        assert_eq!(tree.y, vec![20, 30]);
    }

    #[test]
    fn tuple_struct_maps_and_visits() {
        let tree = TupleNode(1, vec![2, 3]);
        let mapped = tree.map(|&v| v * 2);
        assert_eq!(mapped.0, 2);
        assert_eq!(mapped.1, vec![4, 6]);
        assert_eq!(tree.leaves(), vec![&1, &2, &3]);
    }
}
