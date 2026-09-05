//! Best-fit suballocation over a sorted vector of disjoint free ranges.

use std::ops::Range;

/// Suballocates ranges from a larger interval. Does not own GPU resources.
#[derive(Debug)]
pub struct RangeAllocator {
    root: Range<u64>,
    free: Vec<Range<u64>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeAllocationError {
    HeapExhausted,
}

impl RangeAllocator {
    pub fn new(root: Range<u64>) -> Self {
        let free = if root.is_empty() {
            Vec::new()
        } else {
            vec![root.clone()]
        };
        Self { root, free }
    }

    pub fn is_fully_free(&self) -> bool {
        self.root.is_empty() || self.free.as_slice() == [self.root.clone()]
    }

    pub fn allocate(&mut self, n: u64, align: u64) -> Result<Range<u64>, RangeAllocationError> {
        if n == 0 || !align.is_power_of_two() {
            return Err(RangeAllocationError::HeapExhausted);
        }
        let (index, start) = self
            .free
            .iter()
            .enumerate()
            .filter_map(|(index, span)| {
                let start = span.start.checked_add(align - 1)? & !(align - 1);
                (start.checked_add(n)? <= span.end).then_some((index, start))
            })
            .min_by_key(|&(index, _)| self.free[index].end - self.free[index].start)
            .ok_or(RangeAllocationError::HeapExhausted)?;
        let end = start + n;
        let span = self.free[index].clone();
        match (span.start < start, end < span.end) {
            (true, true) => {
                self.free[index].end = start;
                self.free.insert(index + 1, end..span.end);
            }
            (true, false) => self.free[index].end = start,
            (false, true) => self.free[index].start = end,
            (false, false) => {
                self.free.remove(index);
            }
        }
        Ok(start..end)
    }

    /// Return a previously allocated range.
    /// Empty, out-of-bounds, or already-free ranges are ignored.
    pub fn free(&mut self, range: Range<u64>) {
        if range.is_empty() || range.start < self.root.start || range.end > self.root.end {
            return;
        }
        let index = self.free.partition_point(|span| span.start < range.start);
        if index > 0 && self.free[index - 1].end > range.start
            || index < self.free.len() && self.free[index].start < range.end
        {
            return;
        }
        let left = index > 0 && self.free[index - 1].end == range.start;
        let right = index < self.free.len() && self.free[index].start == range.end;
        match (left, right) {
            (true, true) => {
                self.free[index - 1].end = self.free[index].end;
                self.free.remove(index);
            }
            (true, false) => self.free[index - 1].end = range.end,
            (false, true) => self.free[index].start = range.start,
            (false, false) => self.free.insert(index, range),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_in_every_free_order() {
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        let order = [a, b, c, d];
                        if (0..4).any(|i| order[..i].contains(&order[i])) {
                            continue;
                        }
                        let mut allocator = RangeAllocator::new(0..8);
                        let ranges: Vec<_> =
                            (0..4).map(|_| allocator.allocate(2, 1).unwrap()).collect();
                        assert_eq!(
                            allocator.allocate(1, 1),
                            Err(RangeAllocationError::HeapExhausted)
                        );
                        for index in order {
                            allocator.free(ranges[index].clone());
                        }
                        assert!(allocator.is_fully_free(), "free order {order:?}");
                        assert_eq!(allocator.allocate(8, 1), Ok(0..8));
                    }
                }
            }
        }
    }

    #[test]
    fn best_fit_and_alignment_preserve_unused_space() {
        let mut allocator = RangeAllocator::new(1..32);
        assert_eq!(allocator.allocate(4, 8), Ok(8..12));
        assert_eq!(allocator.allocate(6, 1), Ok(1..7));
        assert_eq!(allocator.allocate(1, 1), Ok(7..8));
        allocator.free(8..12);
        allocator.free(1..7);
        allocator.free(7..8);
        assert!(allocator.is_fully_free());
        assert_eq!(allocator.allocate(8, 8), Ok(8..16));
    }

    #[test]
    fn invalid_frees_cannot_corrupt_the_free_ranges() {
        let mut allocator = RangeAllocator::new(10..30);
        assert_eq!(allocator.allocate(10, 1), Ok(10..20));
        for invalid in [0..10, 25..35, 15..25, 19..30, 20..30, 12..12] {
            allocator.free(invalid);
            assert_eq!(allocator.free, vec![20..30]);
        }
        allocator.free(10..20);
        allocator.free(10..20);
        assert!(allocator.is_fully_free());
    }

    #[test]
    fn rejects_bad_sizes_and_alignment_overflow() {
        let mut empty = RangeAllocator::new(0..0);
        assert!(empty.is_fully_free());
        assert_eq!(
            empty.allocate(1, 1),
            Err(RangeAllocationError::HeapExhausted)
        );
        let mut allocator = RangeAllocator::new(u64::MAX - 7..u64::MAX);
        for (n, align) in [(0, 1), (1, 0), (1, 3), (8, 1), (1, 16), (u64::MAX, 8)] {
            assert_eq!(
                allocator.allocate(n, align),
                Err(RangeAllocationError::HeapExhausted)
            );
        }
        assert_eq!(allocator.allocate(1, 8), Ok(u64::MAX - 7..u64::MAX - 6));
    }

    #[test]
    fn fragmented_allocations_match_a_byte_occupancy_model() {
        let mut allocator = RangeAllocator::new(0..128);
        let mut used = [false; 128];
        let mut live = Vec::<Range<u64>>::new();
        let mut seed = 7u64;
        for _ in 0..4000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            if seed & 3 == 0 && !live.is_empty() {
                let range = live.swap_remove((seed >> 8) as usize % live.len());
                used[range.start as usize..range.end as usize].fill(false);
                allocator.free(range);
            } else {
                let n = (seed >> 8) % 13 + 1;
                let align = 1 << ((seed >> 16) % 5);
                let possible = (0..=128 - n).any(|start| {
                    start % align == 0
                        && used[start as usize..(start + n) as usize]
                            .iter()
                            .all(|&b| !b)
                });
                let result = allocator.allocate(n, align);
                assert_eq!(result.is_ok(), possible);
                if let Ok(range) = result {
                    assert_eq!(range.start % align, 0);
                    assert!(
                        used[range.start as usize..range.end as usize]
                            .iter()
                            .all(|&b| !b)
                    );
                    used[range.start as usize..range.end as usize].fill(true);
                    live.push(range);
                }
            }
        }
        for range in live {
            allocator.free(range);
        }
        assert!(allocator.is_fully_free());
    }
}
