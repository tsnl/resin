//! Strided views over flat buffers (offset / shape / stride).
//!
//! Stride tricks express broadcast (stride 0), permute (reordered strides),
//! unfold (window stride + step), and squeeze without touching storage.
//! Equality is by the affine map — two accessors that address the same way
//! compare equal regardless of how they were built.

use super::program::Error;

/// Maps N-d coordinates to a linear element offset: `offset + Σ coordᵢ · strideᵢ`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Accessor {
    pub offset: usize,
    pub shape: Box<[usize]>,
    pub stride: Box<[usize]>,
}

impl Accessor {
    /// C-contiguous accessor addressing `shape` from `offset`.
    pub fn dense(shape: impl Into<Box<[usize]>>, offset: usize) -> Self {
        let shape = shape.into();
        let stride = dense_stride(&shape);
        Self {
            offset,
            shape,
            stride,
        }
    }

    /// Whether this is C-contiguous (kernel-output law).
    pub fn is_dense(&self) -> bool {
        self.stride.as_ref() == dense_stride(&self.shape).as_ref()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Linear element offset of `coords`.
    pub fn index(&self, coords: &[usize]) -> usize {
        debug_assert_eq!(coords.len(), self.rank());
        self.offset
            + coords
                .iter()
                .zip(&self.stride)
                .map(|(c, s)| c * s)
                .sum::<usize>()
    }

    /// Highest linear element index this accessor can touch (empty shape → offset).
    pub fn max_index(&self) -> usize {
        if self.shape.iter().any(|&d| d == 0) {
            return self.offset;
        }
        self.offset
            + self
                .shape
                .iter()
                .zip(&self.stride)
                .map(|(&d, &s)| (d - 1) * s)
                .sum::<usize>()
    }

    /// Reorder axes: `out.shape[i] = self.shape[axes[i]]` (NumPy / PyTorch axes).
    pub fn permute(&self, axes: &[usize]) -> Result<Self, Error> {
        let n = self.rank();
        if axes.len() != n {
            return Err(Error(format!(
                "permute axes {axes:?} do not match rank {n}"
            )));
        }
        let mut seen = vec![false; n];
        for &a in axes {
            if a >= n {
                return Err(Error(format!(
                    "permute axis {a} out of range for rank {n}"
                )));
            }
            if seen[a] {
                return Err(Error(format!(
                    "permute axes must be a permutation, got {axes:?}"
                )));
            }
            seen[a] = true;
        }
        Ok(Self {
            offset: self.offset,
            shape: axes.iter().map(|&a| self.shape[a]).collect(),
            stride: axes.iter().map(|&a| self.stride[a]).collect(),
        })
    }

    /// Swap the last two axes.
    pub fn transpose(&self) -> Result<Self, Error> {
        let n = self.rank();
        if n < 2 {
            return Err(Error(format!("transpose requires rank >= 2, got {n}")));
        }
        let mut axes: Vec<usize> = (0..n).collect();
        axes.swap(n - 2, n - 1);
        self.permute(&axes)
    }

    /// Sliding windows along `axis`: shape gains a trailing window axis of
    /// length `size`, and `axis` shrinks to the number of windows.
    ///
    /// `out[..., i, ..., j] = self[..., i * step + j, ...]` for
    /// `j ∈ 0..size` and `i ∈ 0..n_win` with `n_win = (dim - size) / step + 1`
    /// (remainder dropped, PyTorch-style).
    pub fn unfold(&self, axis: usize, size: usize, step: usize) -> Result<Self, Error> {
        let n = self.rank();
        if axis >= n {
            return Err(Error(format!(
                "unfold axis {axis} out of range for rank {n}"
            )));
        }
        if size == 0 {
            return Err(Error("unfold size must be >= 1".into()));
        }
        if step == 0 {
            return Err(Error("unfold step must be >= 1".into()));
        }
        let dim = self.shape[axis];
        if size > dim {
            return Err(Error(format!(
                "unfold size {size} exceeds axis {axis} dim {dim}"
            )));
        }
        let n_win = (dim - size) / step + 1;
        let base = self.stride[axis];
        let mut shape = self.shape.to_vec();
        shape[axis] = n_win;
        shape.push(size);
        let mut stride = self.stride.to_vec();
        stride[axis] = base * step;
        stride.push(base);
        Ok(Self {
            offset: self.offset,
            shape: shape.into(),
            stride: stride.into(),
        })
    }

    /// Remove the given size-1 axes.
    pub fn squeeze(&self, axes: &[usize]) -> Result<Self, Error> {
        for &axis in axes {
            match self.shape.get(axis) {
                None => {
                    return Err(Error(format!(
                        "squeeze axis {axis} out of range for rank {}",
                        self.rank()
                    )));
                }
                Some(&size) if size != 1 => {
                    return Err(Error(format!(
                        "cannot squeeze axis {axis} with size {size}"
                    )));
                }
                Some(_) => {}
            }
        }
        let keep = |i: &usize| !axes.contains(i);
        Ok(Self {
            offset: self.offset,
            shape: (0..self.rank())
                .filter(keep)
                .map(|i| self.shape[i])
                .collect(),
            stride: (0..self.rank())
                .filter(keep)
                .map(|i| self.stride[i])
                .collect(),
        })
    }

    /// Broadcast to `target` with NumPy trailing alignment: missing or size-1
    /// axes get stride 0.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self, Error> {
        if self.shape.as_ref() == target {
            return Ok(self.clone());
        }
        let lead = target.len().checked_sub(self.rank()).ok_or_else(|| {
            Error(format!(
                "cannot broadcast {:?} down to {target:?}",
                self.shape
            ))
        })?;
        let mut stride = vec![0; target.len()];
        for (i, (&dim, &s)) in self.shape.iter().zip(&self.stride).enumerate() {
            let want = target[lead + i];
            if dim == want {
                stride[lead + i] = s;
            } else if dim != 1 {
                return Err(Error(format!(
                    "cannot broadcast {:?} to {target:?}",
                    self.shape
                )));
            }
        }
        Ok(Self {
            offset: self.offset,
            shape: target.into(),
            stride: stride.into(),
        })
    }

    /// Explicit-axis broadcast: `axes[i]` is the output axis for input axis `i`.
    /// Unmapped axes are broadcast (stride 0).
    pub fn map_axes(&self, target: &[usize], axes: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(Error(format!(
                "broadcast axes {axes:?} do not match rank {rank}"
            )));
        }
        let mut stride = vec![0; target.len()];
        for (input_axis, &out_axis) in axes.iter().enumerate() {
            if out_axis >= target.len() {
                return Err(Error(format!(
                    "broadcast axis {out_axis} out of range for target {target:?}"
                )));
            }
            // Size-1 dims that expand keep stride 0.
            if self.shape[input_axis] != 1 || target[out_axis] == 1 {
                stride[out_axis] = self.stride[input_axis];
            }
        }
        Ok(Self {
            offset: self.offset,
            shape: target.into(),
            stride: stride.into(),
        })
    }

    /// Whether `self` is `source` expanded by trailing broadcast (or equal).
    pub fn is_broadcast_of(&self, source: &Accessor) -> bool {
        if self == source {
            return true;
        }
        if self.offset != source.offset || self.rank() < source.rank() {
            return false;
        }
        let lead = self.rank() - source.rank();
        if self.stride[..lead].iter().any(|&s| s != 0) {
            return false;
        }
        source
            .shape
            .iter()
            .zip(&source.stride)
            .enumerate()
            .all(|(i, (&dim, &src_stride))| {
                let (out_dim, out_stride) = (self.shape[lead + i], self.stride[lead + i]);
                (out_dim == dim && out_stride == src_stride)
                    || (dim == 1 && out_stride == 0)
            })
    }

    /// Re-address this accessor through a consumer view of a producer write
    /// (`source`): identity when `expansion == source`, otherwise stride-compose
    /// through `expansion`'s shape (broadcast axes stay stride 0).
    pub fn compose_through(&self, source: &Accessor, expansion: &Accessor) -> Accessor {
        if expansion == source {
            return self.clone();
        }
        debug_assert!(expansion.rank() >= self.rank());
        let lead = expansion.rank() - self.rank();
        let mut stride = vec![0; expansion.rank()];
        for (i, &s) in self.stride.iter().enumerate() {
            if expansion.stride[lead + i] != 0 {
                stride[lead + i] = s;
            }
        }
        Accessor {
            offset: self.offset,
            shape: expansion.shape.clone(),
            stride: stride.into(),
        }
    }
}

/// C-contiguous (row-major) strides for a shape.
pub fn dense_stride(shape: &[usize]) -> Box<[usize]> {
    let mut stride = vec![0; shape.len()];
    let mut run = 1;
    for i in (0..shape.len()).rev() {
        stride[i] = run;
        run *= shape[i];
    }
    stride.into()
}

/// Number of elements in a shape (scalars have one).
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_stride_2d() {
        let a = Accessor::dense([2, 3], 0);
        assert_eq!(&*a.stride, &[3, 1]);
        assert_eq!(a.index(&[1, 2]), 5);
        assert!(a.is_dense());
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let a = Accessor::dense([], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.stride, &[0, 0]);
        assert!(!b.is_dense());
    }

    #[test]
    fn broadcast_trailing_alignment() {
        let a = Accessor::dense([3], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.stride, &[0, 1]);
    }

    #[test]
    fn broadcast_rejects_mismatch() {
        assert!(Accessor::dense([3], 0).broadcast_to(&[2, 4]).is_err());
    }

    #[test]
    fn transpose_swaps_last_two() {
        let a = Accessor::dense([2, 3], 0).transpose().unwrap();
        assert_eq!(&*a.shape, &[3, 2]);
        assert_eq!(&*a.stride, &[1, 3]);
        assert!(!a.is_dense());
    }

    #[test]
    fn permute_reorders_axes() {
        let a = Accessor::dense([2, 3, 4], 0).permute(&[2, 0, 1]).unwrap();
        assert_eq!(&*a.shape, &[4, 2, 3]);
        assert_eq!(&*a.stride, &[1, 12, 4]);
        assert_eq!(a.index(&[1, 1, 2]), 1 + 12 + 8);
    }

    #[test]
    fn permute_rejects_non_permutation() {
        assert!(Accessor::dense([2, 3], 0).permute(&[0, 0]).is_err());
        assert!(Accessor::dense([2, 3], 0).permute(&[0]).is_err());
    }

    #[test]
    fn unfold_1d_overlapping() {
        // [0,1,2,3,4,5,6] → windows of 3 step 1
        let a = Accessor::dense([7], 0).unfold(0, 3, 1).unwrap();
        assert_eq!(&*a.shape, &[5, 3]);
        assert_eq!(&*a.stride, &[1, 1]);
        assert_eq!(a.index(&[0, 0]), 0);
        assert_eq!(a.index(&[0, 2]), 2);
        assert_eq!(a.index(&[1, 0]), 1);
        assert_eq!(a.index(&[4, 2]), 6);
    }

    #[test]
    fn unfold_2d_spatial_axis() {
        // Row-major [2, 5], unfold width with size 3 step 2 → [2, 2, 3]
        let a = Accessor::dense([2, 5], 0).unfold(1, 3, 2).unwrap();
        assert_eq!(&*a.shape, &[2, 2, 3]);
        assert_eq!(&*a.stride, &[5, 2, 1]);
        assert_eq!(a.index(&[0, 0, 0]), 0);
        assert_eq!(a.index(&[0, 0, 2]), 2);
        assert_eq!(a.index(&[0, 1, 0]), 2);
        assert_eq!(a.index(&[1, 0, 1]), 6);
    }

    #[test]
    fn squeeze_drops_unit_axes() {
        let a = Accessor::dense([1, 4], 0).squeeze(&[0]).unwrap();
        assert_eq!(&*a.shape, &[4]);
        assert!(a.is_dense());
    }

    #[test]
    fn is_broadcast_of_trailing() {
        let dense = Accessor::dense([3], 0);
        let b = dense.broadcast_to(&[2, 3]).unwrap();
        assert!(b.is_broadcast_of(&dense));
        assert!(dense.is_broadcast_of(&dense));
    }

    #[test]
    fn compose_through_broadcast() {
        let write = Accessor::dense([3], 0);
        let read = write.broadcast_to(&[2, 3]).unwrap();
        let arg = Accessor::dense([3], 0);
        let composed = arg.compose_through(&write, &read);
        assert_eq!(&*composed.shape, &[2, 3]);
        assert_eq!(&*composed.stride, &[0, 1]);
    }

    #[test]
    fn map_axes_places_stride() {
        let a = Accessor::dense([3], 0);
        let b = a.map_axes(&[2, 3], &[1]).unwrap();
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.stride, &[0, 1]);
    }
}
