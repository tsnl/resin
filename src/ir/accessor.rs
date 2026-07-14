//! Strided views over flat buffers (offset / shape / pitch).
//!
//! Pitch tricks express broadcast (pitch 0), transpose (swapped pitches), and
//! squeeze without touching storage. Equality is by the affine map — two
//! accessors that address the same way compare equal regardless of how they
//! were built.

use super::program::Error;

/// Maps N-d coordinates to a linear element offset: `offset + Σ coordᵢ · pitchᵢ`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Accessor {
    pub offset: usize,
    pub shape: Box<[usize]>,
    pub pitch: Box<[usize]>,
}

impl Accessor {
    /// C-contiguous accessor addressing `shape` from `offset`.
    pub fn dense(shape: impl Into<Box<[usize]>>, offset: usize) -> Self {
        let shape = shape.into();
        let pitch = dense_pitch(&shape);
        Self {
            offset,
            shape,
            pitch,
        }
    }

    /// Whether this is C-contiguous (kernel-output law).
    pub fn is_dense(&self) -> bool {
        self.pitch.as_ref() == dense_pitch(&self.shape).as_ref()
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
                .zip(&self.pitch)
                .map(|(c, p)| c * p)
                .sum::<usize>()
    }

    /// Swap the last two axes.
    pub fn transpose(&self) -> Result<Self, Error> {
        let n = self.rank();
        if n < 2 {
            return Err(Error(format!("transpose requires rank >= 2, got {n}")));
        }
        let mut out = self.clone();
        out.shape.swap(n - 2, n - 1);
        out.pitch.swap(n - 2, n - 1);
        Ok(out)
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
            pitch: (0..self.rank())
                .filter(keep)
                .map(|i| self.pitch[i])
                .collect(),
        })
    }

    /// Broadcast to `target` with NumPy trailing alignment: missing or size-1
    /// axes get pitch 0.
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
        let mut pitch = vec![0; target.len()];
        for (i, (&dim, &p)) in self.shape.iter().zip(&self.pitch).enumerate() {
            let want = target[lead + i];
            if dim == want {
                pitch[lead + i] = p;
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
            pitch: pitch.into(),
        })
    }

    /// Explicit-axis broadcast: `axes[i]` is the output axis for input axis `i`.
    /// Unmapped axes are broadcast (pitch 0).
    pub fn map_axes(&self, target: &[usize], axes: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(Error(format!(
                "broadcast axes {axes:?} do not match rank {rank}"
            )));
        }
        let mut pitch = vec![0; target.len()];
        for (input_axis, &out_axis) in axes.iter().enumerate() {
            if out_axis >= target.len() {
                return Err(Error(format!(
                    "broadcast axis {out_axis} out of range for target {target:?}"
                )));
            }
            // Size-1 dims that expand keep pitch 0.
            if self.shape[input_axis] != 1 || target[out_axis] == 1 {
                pitch[out_axis] = self.pitch[input_axis];
            }
        }
        Ok(Self {
            offset: self.offset,
            shape: target.into(),
            pitch: pitch.into(),
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
        if self.pitch[..lead].iter().any(|&p| p != 0) {
            return false;
        }
        source
            .shape
            .iter()
            .zip(&source.pitch)
            .enumerate()
            .all(|(i, (&dim, &pitch))| {
                let (out_dim, out_pitch) = (self.shape[lead + i], self.pitch[lead + i]);
                (out_dim == dim && out_pitch == pitch) || (dim == 1 && out_pitch == 0)
            })
    }

    /// Re-address this accessor through a consumer view of a producer write
    /// (`source`): identity when `expansion == source`, otherwise pitch compose
    /// through `expansion`'s shape (broadcast axes stay pitch 0).
    pub fn compose_through(&self, source: &Accessor, expansion: &Accessor) -> Accessor {
        if expansion == source {
            return self.clone();
        }
        debug_assert!(expansion.rank() >= self.rank());
        let lead = expansion.rank() - self.rank();
        let mut pitch = vec![0; expansion.rank()];
        for (i, &p) in self.pitch.iter().enumerate() {
            if expansion.pitch[lead + i] != 0 {
                pitch[lead + i] = p;
            }
        }
        Accessor {
            offset: self.offset,
            shape: expansion.shape.clone(),
            pitch: pitch.into(),
        }
    }
}

/// C-contiguous (row-major) pitch for a shape.
pub fn dense_pitch(shape: &[usize]) -> Box<[usize]> {
    let mut pitch = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        pitch[i] = stride;
        stride *= shape[i];
    }
    pitch.into()
}

/// Number of elements in a shape (scalars have one).
pub fn element_count(shape: &[usize]) -> usize {
    shape.iter().product()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_pitch_2d() {
        let a = Accessor::dense([2, 3], 0);
        assert_eq!(&*a.pitch, &[3, 1]);
        assert_eq!(a.index(&[1, 2]), 5);
        assert!(a.is_dense());
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let a = Accessor::dense([], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.pitch, &[0, 0]);
        assert!(!b.is_dense());
    }

    #[test]
    fn broadcast_trailing_alignment() {
        let a = Accessor::dense([3], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.pitch, &[0, 1]);
    }

    #[test]
    fn broadcast_rejects_mismatch() {
        assert!(Accessor::dense([3], 0).broadcast_to(&[2, 4]).is_err());
    }

    #[test]
    fn transpose_swaps_last_two() {
        let a = Accessor::dense([2, 3], 0).transpose().unwrap();
        assert_eq!(&*a.shape, &[3, 2]);
        assert_eq!(&*a.pitch, &[1, 3]);
        assert!(!a.is_dense());
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
        assert_eq!(&*composed.pitch, &[0, 1]);
    }

    #[test]
    fn map_axes_places_pitch() {
        let a = Accessor::dense([3], 0);
        let b = a.map_axes(&[2, 3], &[1]).unwrap();
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.pitch, &[0, 1]);
    }
}
