//! Best-fit free-list over a single interval.

use std::cell::RefMut;
use std::ops::Range;

use super::list::{self, Cons, List, NonEmptyList, cons};

/// Suballocates ranges from a larger interval. Does not own GPU resources.
#[derive(Debug)]
pub struct RangeAllocator {
    root: Range<u64>,
    free_list: FreeList,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeAllocationError {
    HeapExhausted,
}

impl RangeAllocator {
    pub fn new(root_range: Range<u64>) -> Self {
        Self {
            root: root_range.clone(),
            free_list: FreeList::new(root_range),
        }
    }

    pub fn is_fully_free(&self) -> bool {
        let Some(head) = &self.free_list.list else {
            return false;
        };
        head.borrow().tail.is_none() && head.borrow().data == self.root
    }

    pub fn allocate(&mut self, n: u64, align: u64) -> Result<Range<u64>, RangeAllocationError> {
        if n == 0 || align == 0 || !align.is_power_of_two() {
            return Err(RangeAllocationError::HeapExhausted);
        }
        let Some(best_fit) = self.free_list.find_best_fit(n, align) else {
            return Err(RangeAllocationError::HeapExhausted);
        };
        let span = best_fit.curr.borrow().data.clone();
        let start = align_up(span.start, align);
        let end = start
            .checked_add(n)
            .ok_or(RangeAllocationError::HeapExhausted)?;
        self.free_list.remove(best_fit.prev);
        if span.start < start {
            self.free_list.insert(span.start..start);
        }
        if end < span.end {
            self.free_list.insert(end..span.end);
        }
        Ok(start..end)
    }

    pub fn free(&mut self, range: Range<u64>) {
        if range.start < range.end {
            self.free_list.insert(range);
        }
    }
}

#[derive(Debug)]
struct FreeList {
    list: List<Range<u64>>,
}

struct FreeListBestFit {
    prev: List<Range<u64>>,
    curr: NonEmptyList<Range<u64>>,
}

impl FreeList {
    fn new(root_range: Range<u64>) -> Self {
        Self {
            list: Some(cons(root_range, None)),
        }
    }

    fn find_best_fit(&self, n: u64, align: u64) -> Option<FreeListBestFit> {
        let mut best_fit: Option<FreeListBestFit> = None;
        let mut best_len = u64::MAX;

        for (prev, curr) in list::iter_with_prev(self.list.clone()) {
            let span = curr.borrow().data.clone();
            let Some(start) = aligned_start_in(&span, n, align) else {
                continue;
            };
            let span_len = span.end - span.start;
            if span_len < best_len {
                best_len = span_len;
                best_fit = Some(FreeListBestFit { prev, curr });
            }
            if start == span.start && start + n == span.end {
                break;
            }
        }

        best_fit
    }

    fn remove(&mut self, prev: List<Range<u64>>) -> Range<u64> {
        let removed = match prev {
            None => {
                let curr = self.list.clone().expect("cannot remove from an empty list");
                self.list = curr.borrow().tail.clone();
                curr
            }
            Some(prev) => {
                let curr = prev
                    .borrow()
                    .tail
                    .clone()
                    .expect("non-None prev should have a tail");
                prev.borrow_mut().tail = curr.borrow().tail.clone();
                curr
            }
        };
        removed.borrow().data.clone()
    }

    fn insert(&mut self, range: Range<u64>) {
        if try_insert_before_head(&mut self.list, &range) {
            return;
        }

        for node in list::iter(self.list.clone()) {
            if try_insert_immediately_after_node(node, &range) {
                return;
            }
        }

        // Overlap, double-free, or a range outside the root: ignore.

        fn try_insert_before_head(list: &mut List<Range<u64>>, range: &Range<u64>) -> bool {
            match list {
                None => {
                    *list = Some(cons(range.clone(), None));
                    true
                }
                Some(head) => {
                    if range.end == head.borrow().data.start {
                        head.borrow_mut().data.start = range.start;
                        return true;
                    }
                    if range.end < head.borrow().data.start {
                        *list = Some(cons(range.clone(), list.clone()));
                        return true;
                    }
                    false
                }
            }
        }

        fn try_insert_immediately_after_node(
            curr: NonEmptyList<Range<u64>>,
            range: &Range<u64>,
        ) -> bool {
            let mut curr = curr.borrow_mut();

            if curr.data.end == range.start {
                curr.data.end = range.end;
                right_extend_node_and_delete_successor_if_needed(curr);
                return true;
            }

            if let Some(next) = curr.tail.as_ref()
                && range.end == next.borrow().data.start
            {
                next.borrow_mut().data.start = range.start;
                return true;
            }

            if curr.data.end < range.start {
                let next = curr.tail.clone();
                let should_insert = match &next {
                    None => true,
                    Some(next) => range.end < next.borrow().data.start,
                };
                if should_insert {
                    curr.tail = Some(cons(range.clone(), next));
                    return true;
                }
            }

            false
        }

        fn right_extend_node_and_delete_successor_if_needed(mut curr: RefMut<Cons<Range<u64>>>) {
            let Some(next) = curr.tail.clone() else {
                return;
            };
            if curr.data.end == next.borrow().data.start {
                let next = next.borrow();
                curr.tail = next.tail.clone();
                curr.data.end = next.data.end;
            }
        }
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two() && align > 0);
    let mask = align - 1;
    value.wrapping_add(mask) & !mask
}

fn aligned_start_in(span: &Range<u64>, n: u64, align: u64) -> Option<u64> {
    let start = align_up(span.start, align);
    if start < span.start {
        return None;
    }
    let end = start.checked_add(n)?;
    (end <= span.end).then_some(start)
}

#[cfg(test)]
mod tests {
    use super::list::{self, cons};
    use super::*;

    #[test]
    fn test_free_list() {
        let mut allocator = RangeAllocator::new(0..8);

        assert_eq!(allocator.allocate(4, 1), Ok(0..4));
        assert_eq!(allocator.allocate(4, 1), Ok(4..8));
        assert_eq!(
            allocator.allocate(1, 1),
            Err(RangeAllocationError::HeapExhausted)
        );
        allocator.free(0..4);
        allocator.free(4..8);
        assert_eq!(allocator.free_list.list, Some(cons(0..8, None)));

        assert_eq!(allocator.allocate(2, 1), Ok(0..2));
        assert_eq!(allocator.allocate(2, 1), Ok(2..4));
        assert_eq!(allocator.allocate(2, 1), Ok(4..6));
        assert_eq!(allocator.allocate(2, 1), Ok(6..8));
        allocator.free(6..8);
        assert_eq!(allocator.free_list.list, Some(cons(6..8, None)));
        allocator.free(2..4);
        assert_eq!(allocator.free_list.list, list::list([2..4, 6..8]));
        allocator.free(0..2);
        assert_eq!(allocator.free_list.list, list::list([0..4, 6..8]));
        allocator.free(4..6);
        assert_eq!(allocator.free_list.list, Some(cons(0..8, None)));
    }

    #[test]
    fn aligned_allocate_skips_and_returns_prefix() {
        let mut allocator = RangeAllocator::new(1..32);
        assert_eq!(allocator.allocate(4, 8), Ok(8..12));
        allocator.free(8..12);
        assert_eq!(allocator.allocate(8, 8), Ok(8..16));
    }

    #[test]
    fn overlapping_free_does_not_panic() {
        let mut allocator = RangeAllocator::new(0..8);
        assert_eq!(allocator.allocate(4, 1), Ok(0..4));
        allocator.free(0..4);
        allocator.free(0..4);
        assert!(allocator.is_fully_free());
        assert_eq!(allocator.allocate(8, 1), Ok(0..8));
    }
}
