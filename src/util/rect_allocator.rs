use simd_math::{SimdRect2, SimdVec2};

/// RectAllocator allocates rectangles in an array of fixed-size pages.
/// Intended for use in texture heaps: each page is a 4096x4096px texture.
/// TODO: Support deallocation and a free-list.
/// TODO: Support compaction.
pub struct RectAllocator {
    name: &'static str,

    page_count: usize,
    page_height: usize,
    page_width: usize,

    cursor_page_index: usize,
    cursor_row_offset: usize,
    cursor_row_height: usize,
    cursor_col_offset: usize,
}

#[derive(thiserror::Error, Debug)]
pub enum RectAllocationError {
    #[error("Allocation too large for configured page size: {1}x{2}px")]
    AllocationTooLarge(&'static str, usize, usize),

    #[error("Heap {0} too full or fragmented to satisfy allocation: {1}x{2}px")]
    NoFreeSpace(&'static str, usize, usize),
}
impl RectAllocationError {
    /// `resolvable_via_compaction` indicates whether this error _may_ be resolved by compacting the
    /// allocator and retrying the allocation. Note that the allocation may still fail after
    /// retrying.
    pub fn resolvable_via_compaction(&self) -> bool {
        match self {
            RectAllocationError::NoFreeSpace(_, _, _) => true,
            _ => false,
        }
    }
}

impl RectAllocator {
    pub fn new(
        name: &'static str,
        page_count: usize,
        page_width: usize,
        page_height: usize,
    ) -> Self {
        Self {
            name,
            page_count,
            page_width,
            page_height,
            cursor_page_index: 0,
            cursor_row_height: 0,
            cursor_row_offset: 0,
            cursor_col_offset: 0,
        }
    }

    pub fn allocate(&mut self, w: usize, h: usize) -> Result<SimdRect2, RectAllocationError> {
        // Ensure the allocation is not too large for this allocator:
        if w > self.page_width || h > self.page_height {
            return Err(RectAllocationError::AllocationTooLarge(self.name, w, h));
        }

        // Backup the cursor state in case we need to roll back later.
        let rollback_page_index = self.cursor_page_index;
        let rollback_row_offset = self.cursor_row_offset;
        let rollback_row_height = self.cursor_row_height;
        let rollback_col_offset = self.cursor_col_offset;

        // Bump the cursor state until this allocation is satisfiable. If we overflow all pages,
        // restore the backed up cursor state and error out.
        if w + self.cursor_col_offset > self.page_width {
            self.cursor_row_offset += self.cursor_row_height;
            self.cursor_row_height = 0;
            self.cursor_col_offset = 0;
        }
        if h + self.cursor_row_offset > self.page_height {
            self.cursor_page_index += 1;
            self.cursor_row_offset = 0;
            self.cursor_row_height = 0;
            self.cursor_col_offset = 0;
        }
        if self.cursor_page_index >= self.page_count {
            // Overflow: rollback and return error.
            self.cursor_page_index = rollback_page_index;
            self.cursor_row_offset = rollback_row_offset;
            self.cursor_row_height = rollback_row_height;
            self.cursor_col_offset = rollback_col_offset;
            return Err(RectAllocationError::NoFreeSpace(self.name, w, h));
        }
        debug_assert!(self.cursor_col_offset + w <= self.page_width);
        debug_assert!(self.cursor_row_offset + h <= self.page_height);

        // Finalize this allocation in pixel coordinates:
        let alloc_page_index = self.cursor_page_index;
        let alloc_y_offset = self.cursor_row_offset;
        let alloc_x_offset = self.cursor_col_offset;
        self.cursor_row_height = h.max(self.cursor_row_height);
        self.cursor_col_offset += w;

        // Convert the pixel rectangles to UVs
        // See: https://asawicki.info/news_1516_half-pixel_offset_in_directx_11.html
        let alloc_xy_px_min = SimdVec2::from([alloc_x_offset as f32, alloc_y_offset as f32]);
        let alloc_wh_px = SimdVec2::from([w as f32, h as f32]);
        let page_wh_px = SimdVec2::from([self.page_width as f32, self.page_height as f32]);
        let alloc_xy_uv_min = (alloc_xy_px_min + SimdVec2::splat(0.5)) / page_wh_px;
        let alloc_xy_uv_max = (alloc_xy_uv_min + alloc_wh_px - SimdVec2::splat(1.0)) / page_wh_px;

        // Add an offset to the Y coordinate for the page index:
        let page_uv_offset = SimdVec2::from([0.0, alloc_page_index as f32]);
        let alloc_xy_uv_min = alloc_xy_uv_min + page_uv_offset;
        let alloc_xy_uv_max = alloc_xy_uv_max + page_uv_offset;

        // Done:
        Ok(SimdRect2::new(alloc_xy_uv_min, alloc_xy_uv_max))
    }

    pub fn deallocate(&mut self) {
        todo!()
    }

    pub fn compact(&mut self) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut allocator = RectAllocator::new("test_rect_allocator", 1, 1024, 1024);
        allocator.allocate(1024, 1024).unwrap();
        allocator
            .allocate(1, 1)
            .expect_err("Expected no remaining space");
    }

    #[test]
    fn test_correct_fragmentation_behavior() {
        let mut allocator = RectAllocator::new("test_rect_allocator", 1, 1024, 1024);
        allocator.allocate(512, 512).unwrap(); // at (0, 0)
        allocator.allocate(512, 256).unwrap(); // at (512, 0)
        allocator.allocate(512, 513).unwrap_err();
        allocator.allocate(512, 512).unwrap(); // at (0, 512)
        allocator.allocate(512, 512).unwrap(); // at (512, 512)
        allocator
            .allocate(1, 1)
            .expect_err("Expected no remaining space");
    }
}
