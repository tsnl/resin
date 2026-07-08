//! Strided views over flat buffers (offset / shape / pitch).

use super::Error;

/// Maps N-d coordinates to a linear element offset: `offset + Σ coordᵢ · pitchᵢ`.
///
/// Pitch tricks express broadcast (pitch 0), transpose (swapped pitches), and
/// squeeze without touching storage.
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
        Self { offset, shape, pitch }
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
                    return Err(Error(format!("cannot squeeze axis {axis} with size {size}")));
                }
                Some(_) => {}
            }
        }
        let keep = |i: &usize| !axes.contains(i);
        Ok(Self {
            offset: self.offset,
            shape: (0..self.rank()).filter(keep).map(|i| self.shape[i]).collect(),
            pitch: (0..self.rank()).filter(keep).map(|i| self.pitch[i]).collect(),
        })
    }

    /// Broadcast to `target` with NumPy trailing alignment: missing or size-1
    /// axes get pitch 0.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self, Error> {
        if self.shape.as_ref() == target {
            return Ok(self.clone());
        }
        let lead = target.len().checked_sub(self.rank()).ok_or_else(|| {
            Error(format!("cannot broadcast {:?} down to {target:?}", self.shape))
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
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let a = Accessor::dense([], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.pitch, &[0, 0]);
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
    }

    #[test]
    fn squeeze_drops_unit_axes() {
        let a = Accessor::dense([1, 4], 0).squeeze(&[0]).unwrap();
        assert_eq!(&*a.shape, &[4]);
    }
}
