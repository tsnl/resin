use std::{collections::VecDeque, ops::Range};

/// RangeAllocator implements a best-fit allocation strategy over a linear address space.
pub struct RangeAllocator {
    name: &'static str,
    sorted_free_ranges: VecDeque<Range<usize>>,
}
impl RangeAllocator {
    pub fn new(name: &'static str, range: Range<usize>) -> Self {
        let sorted_free_ranges = VecDeque::from([range]);
        Self {
            name,
            sorted_free_ranges,
        }
    }

    pub fn allocate(&mut self, size: usize) -> Result<Range<usize>, RangeAllocationError> {
        let best_fit_index = self.find_best_fit_free_range(size)?;
        Ok(self.satisfy_allocation(best_fit_index, size))
    }
    fn find_best_fit_free_range(&self, size: usize) -> Result<usize, RangeAllocationError> {
        let mut best_fit_index: Option<usize> = None;
        let mut best_fit_size: usize = usize::MAX;
        for (i, free_range) in self.sorted_free_ranges.iter().enumerate() {
            if free_range.len() >= size && free_range.len() < best_fit_size {
                best_fit_size = free_range.len();
                best_fit_index = Some(i);
            }
            if free_range.len() == size {
                break;
            }
        }
        match best_fit_index {
            Some(index) => Ok(index),
            None => Err(RangeAllocationError::NoFreeRange(self.name, size)),
        }
    }
    fn satisfy_allocation(&mut self, free_range_index: usize, size: usize) -> Range<usize> {
        if self.sorted_free_ranges[free_range_index].len() == size {
            self.sorted_free_ranges.remove(free_range_index).unwrap()
        } else {
            let oversize_range = self.sorted_free_ranges[free_range_index].clone();
            let alloc_range = oversize_range.start..(oversize_range.start + size);
            let rem_range = (oversize_range.start + size)..oversize_range.end;
            self.sorted_free_ranges[free_range_index] = rem_range;
            alloc_range
        }
    }

    pub fn deallocate(&mut self, dealloc_range: Range<usize>) {
        // Find the position to insert the dealloc_range
        let num_free_ranges_before_dealloc_range = self
            .sorted_free_ranges
            .partition_point(|r| r.end <= dealloc_range.start);

        // Handle edge cases where dealloc_range is at the beginning or end
        {
            if num_free_ranges_before_dealloc_range == 0 {
                if let Some(front) = self.sorted_free_ranges.front()
                    && front.start == dealloc_range.end
                {
                    // The dealloc_range extends the next free range to the front
                    self.sorted_free_ranges[0].start = dealloc_range.start;
                } else {
                    self.sorted_free_ranges.push_front(dealloc_range);
                }
                return;
            }
            if num_free_ranges_before_dealloc_range == self.sorted_free_ranges.len() {
                if let Some(back) = self.sorted_free_ranges.back()
                    && back.end == dealloc_range.start
                {
                    // The dealloc_range extends the previous free range to the back
                    self.sorted_free_ranges.back_mut().unwrap().end = dealloc_range.end;
                } else {
                    self.sorted_free_ranges.push_back(dealloc_range);
                }
                return;
            }
        }

        // Check for merging with adjacent free ranges
        {
            let prev_range_index = num_free_ranges_before_dealloc_range - 1;
            let next_range_index = num_free_ranges_before_dealloc_range;
            let prev_range = self.sorted_free_ranges[prev_range_index].clone();
            let next_range = self.sorted_free_ranges[next_range_index].clone();
            if prev_range.end == dealloc_range.start && next_range.start == dealloc_range.end {
                // The dealloc_range bridges the previous and next free ranges
                self.sorted_free_ranges.remove(next_range_index).unwrap();
                self.sorted_free_ranges[prev_range_index] = prev_range.start..next_range.end;
                return;
            }
            if prev_range.end == dealloc_range.start {
                // The dealloc_range extends the previous free range to the back
                self.sorted_free_ranges[prev_range_index].end = dealloc_range.end;
                return;
            }
            if next_range.start == dealloc_range.end {
                // The dealloc_range extends the next free range to the front
                self.sorted_free_ranges[next_range_index].start = dealloc_range.start;
                return;
            }
        }

        // The dealloc_range is isolated, just insert it into the free list.
        {
            self.sorted_free_ranges
                .insert(num_free_ranges_before_dealloc_range, dealloc_range);
        }
    }
}

/// AllocationError indicates a failure to allocate a range.
#[derive(thiserror::Error, Debug)]
pub enum RangeAllocationError {
    #[error("Heap {0} too full or fragmented to satisfy {1}-element allocation.")]
    NoFreeRange(&'static str, usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_deallocate() {
        let mut allocator = RangeAllocator::new("test_heap", 0..1024);
        let range1 = allocator.allocate(100).unwrap();
        let range2 = allocator.allocate(200).unwrap();
        assert!(range1.end <= range2.start || range2.end <= range1.start);
        allocator.deallocate(range1.clone());
        let range3 = allocator.allocate(50).unwrap();
        allocator.deallocate(range2);
        allocator.deallocate(range3);
        eprintln!("{:?}", allocator.sorted_free_ranges);
        assert_eq!(allocator.sorted_free_ranges.len(), 1);
        assert_eq!(allocator.sorted_free_ranges[0], 0..1024);
    }
}
