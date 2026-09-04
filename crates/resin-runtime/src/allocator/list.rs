//! Cons list used by [`super::range::RangeAllocator`].

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

pub type List<T> = Option<NonEmptyList<T>>;
pub type NonEmptyList<T> = Rc<RefCell<Cons<T>>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cons<T: Debug + Clone> {
    pub data: T,
    pub tail: List<T>,
}

#[cfg_attr(not(test), allow(dead_code))]
pub fn list<T: Debug + Clone, X: IntoIterator<Item = T>>(x: X) -> List<T> {
    let mut xs = x.into_iter();
    xs.next().map(|it| cons(it, list(xs)))
}

pub fn cons<T: Debug + Clone>(data: T, tail: List<T>) -> NonEmptyList<T> {
    Rc::new(RefCell::new(Cons { data, tail }))
}

pub fn iter<T: Debug + Clone>(list: List<T>) -> Iter<T> {
    Iter { curr: list }
}

pub fn iter_with_prev<T: Debug + Clone>(list: List<T>) -> IterWithPrev<T> {
    IterWithPrev {
        prev: None,
        curr: list,
    }
}

pub struct Iter<T: Debug + Clone> {
    curr: List<T>,
}

impl<T: Debug + Clone> Iterator for Iter<T> {
    type Item = NonEmptyList<T>;

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr.take()?;
        self.curr = curr.borrow().tail.clone();
        Some(curr)
    }
}

#[derive(Clone)]
pub struct IterWithPrev<T: Debug + Clone> {
    prev: List<T>,
    curr: List<T>,
}

impl<T: Debug + Clone> Iterator for IterWithPrev<T> {
    type Item = (List<T>, NonEmptyList<T>);

    fn next(&mut self) -> Option<Self::Item> {
        let curr = self.curr.clone()?;
        let prev = self.prev.clone();

        self.prev = Some(curr.clone());
        self.curr = curr.clone().borrow().tail.clone();

        Some((prev, curr))
    }
}
