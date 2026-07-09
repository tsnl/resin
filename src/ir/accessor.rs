//! Accessors: deferred view programs over flat buffers.
//!
//! An [`Accessor`] is a composition of layout ops (dense base, transpose,
//! squeeze, broadcast, axis remap). Offset/shape/pitch are derived on demand
//! via [`Accessor::strided`] — backends and bounds checks use that affine
//! form; fusion and lowering keep the structured form when they can.

use super::program::Error;

/// How coordinates map into a buffer: a small program of view ops.
///
/// Evaluated by [`Accessor::strided`] / [`Accessor::index`]. Structural
/// equality is by op tree (not by materialised pitches).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Accessor {
    /// C-contiguous layout of `shape` starting at `offset`.
    Dense {
        shape: Box<[usize]>,
        offset: usize,
    },
    /// Swap the last two axes of `base`.
    Transpose(Box<Accessor>),
    /// Drop the given size-1 axes of `base`.
    Squeeze {
        base: Box<Accessor>,
        axes: Box<[usize]>,
    },
    /// NumPy trailing broadcast of `base` to `shape`.
    Broadcast {
        base: Box<Accessor>,
        shape: Box<[usize]>,
    },
    /// Place `base`'s axes onto `axes[i]` of `shape` (DSL explicit-axis
    /// `broadcast_to`). Unmapped output axes are broadcast (pitch 0).
    Map {
        base: Box<Accessor>,
        shape: Box<[usize]>,
        axes: Box<[usize]>,
    },
    /// Fully resolved affine map — escape hatch when a transform is not one
    /// of the structured ops (e.g. general fusion compose).
    Affine {
        offset: usize,
        shape: Box<[usize]>,
        pitch: Box<[usize]>,
    },
}

/// Flattened affine map: `offset + Σ coordᵢ · pitchᵢ`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Strided {
    pub offset: usize,
    pub shape: Box<[usize]>,
    pub pitch: Box<[usize]>,
}

impl Strided {
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn index(&self, coords: &[usize]) -> usize {
        debug_assert_eq!(coords.len(), self.rank());
        self.offset
            + coords
                .iter()
                .zip(&self.pitch)
                .map(|(c, p)| c * p)
                .sum::<usize>()
    }
}

impl Accessor {
    /// C-contiguous accessor addressing `shape` from `offset`.
    pub fn dense(shape: impl Into<Box<[usize]>>, offset: usize) -> Self {
        Accessor::Dense { shape: shape.into(), offset }
    }

    /// Whether this is a structural [`Accessor::Dense`] (kernel-output law).
    pub fn is_dense(&self) -> bool {
        matches!(self, Accessor::Dense { .. })
    }

    /// Iteration / result shape of this accessor.
    pub fn shape(&self) -> Box<[usize]> {
        self.strided().shape
    }

    pub fn rank(&self) -> usize {
        match self {
            Accessor::Dense { shape, .. }
            | Accessor::Broadcast { shape, .. }
            | Accessor::Map { shape, .. }
            | Accessor::Affine { shape, .. } => shape.len(),
            Accessor::Transpose(base) => base.rank(),
            Accessor::Squeeze { base, axes } => base.rank() - axes.len(),
        }
    }

    /// Materialise the affine map (for backends, bounds checks, pitch math).
    pub fn strided(&self) -> Strided {
        match self {
            Accessor::Dense { shape, offset } => Strided {
                offset: *offset,
                pitch: dense_pitch(shape),
                shape: shape.clone(),
            },
            Accessor::Affine { offset, shape, pitch } => Strided {
                offset: *offset,
                shape: shape.clone(),
                pitch: pitch.clone(),
            },
            Accessor::Transpose(base) => {
                let mut s = base.strided();
                let n = s.rank();
                debug_assert!(n >= 2);
                s.shape.swap(n - 2, n - 1);
                s.pitch.swap(n - 2, n - 1);
                s
            }
            Accessor::Squeeze { base, axes } => {
                let s = base.strided();
                let keep = |i: &usize| !axes.contains(i);
                Strided {
                    offset: s.offset,
                    shape: (0..s.rank()).filter(keep).map(|i| s.shape[i]).collect(),
                    pitch: (0..s.rank()).filter(keep).map(|i| s.pitch[i]).collect(),
                }
            }
            Accessor::Broadcast { base, shape } => {
                let inner = base.strided();
                let target = shape.as_ref();
                if inner.shape.as_ref() == target {
                    return inner;
                }
                let lead = target.len() - inner.rank();
                let mut pitch = vec![0; target.len()];
                for (i, (&dim, &p)) in inner.shape.iter().zip(&inner.pitch).enumerate() {
                    let want = target[lead + i];
                    if dim == want {
                        pitch[lead + i] = p;
                    } else {
                        debug_assert_eq!(dim, 1);
                    }
                }
                Strided {
                    offset: inner.offset,
                    shape: shape.clone(),
                    pitch: pitch.into(),
                }
            }
            Accessor::Map { base, shape, axes } => {
                let inner = base.strided();
                let mut pitch = vec![0; shape.len()];
                for (input_axis, &out_axis) in axes.iter().enumerate() {
                    if inner.shape[input_axis] != 1 || shape[out_axis] == 1 {
                        pitch[out_axis] = inner.pitch[input_axis];
                    }
                }
                Strided {
                    offset: inner.offset,
                    shape: shape.clone(),
                    pitch: pitch.into(),
                }
            }
        }
    }

    /// Linear element offset of `coords`.
    pub fn index(&self, coords: &[usize]) -> usize {
        self.strided().index(coords)
    }

    /// Swap the last two axes.
    pub fn transpose(&self) -> Result<Self, Error> {
        let n = self.rank();
        if n < 2 {
            return Err(Error(format!("transpose requires rank >= 2, got {n}")));
        }
        Ok(Accessor::Transpose(Box::new(self.clone())))
    }

    /// Remove the given size-1 axes.
    pub fn squeeze(&self, axes: &[usize]) -> Result<Self, Error> {
        let shape = self.shape();
        for &axis in axes {
            match shape.get(axis) {
                None => {
                    return Err(Error(format!(
                        "squeeze axis {axis} out of range for rank {}",
                        shape.len()
                    )));
                }
                Some(&size) if size != 1 => {
                    return Err(Error(format!("cannot squeeze axis {axis} with size {size}")));
                }
                Some(_) => {}
            }
        }
        Ok(Accessor::Squeeze {
            base: Box::new(self.clone()),
            axes: axes.into(),
        })
    }

    /// Broadcast to `target` with NumPy trailing alignment.
    pub fn broadcast_to(&self, target: &[usize]) -> Result<Self, Error> {
        let shape = self.shape();
        if shape.as_ref() == target {
            return Ok(self.clone());
        }
        let lead = target.len().checked_sub(shape.len()).ok_or_else(|| {
            Error(format!("cannot broadcast {:?} down to {target:?}", shape))
        })?;
        for (i, &dim) in shape.iter().enumerate() {
            let want = target[lead + i];
            if dim != want && dim != 1 {
                return Err(Error(format!("cannot broadcast {:?} to {target:?}", shape)));
            }
        }
        Ok(Accessor::Broadcast {
            base: Box::new(self.clone()),
            shape: target.into(),
        })
    }

    /// Explicit-axis broadcast: `axes[i]` is the output axis for input axis `i`.
    pub fn map_axes(&self, target: &[usize], axes: &[usize]) -> Result<Self, Error> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(Error(format!(
                "broadcast axes {axes:?} do not match rank {rank}"
            )));
        }
        for &out_axis in axes {
            if out_axis >= target.len() {
                return Err(Error(format!(
                    "broadcast axis {out_axis} out of range for target {target:?}"
                )));
            }
        }
        Ok(Accessor::Map {
            base: Box::new(self.clone()),
            shape: target.into(),
            axes: axes.into(),
        })
    }

    /// Whether `self` is `source` or a (possibly nested) trailing broadcast of it.
    pub fn is_broadcast_of(&self, source: &Accessor) -> bool {
        if self == source {
            return true;
        }
        match self {
            Accessor::Broadcast { base, .. } => {
                base.as_ref() == source || base.is_broadcast_of(source)
            }
            _ => is_broadcast_strided(&self.strided(), &source.strided()),
        }
    }

    /// Re-address this accessor through a consumer view of a producer write
    /// (`source`): identity when `expansion == source`, structured broadcast
    /// when possible, otherwise pitch compose → [`Accessor::Affine`].
    pub fn compose_through(&self, source: &Accessor, expansion: &Accessor) -> Accessor {
        if expansion == source {
            return self.clone();
        }
        if let Accessor::Broadcast { base, shape } = expansion
            && (base.as_ref() == source || base.is_broadcast_of(source))
            && let Ok(composed) = self.broadcast_to(shape)
        {
            return composed;
        }
        compose_strided(self, expansion)
    }
}

/// Whether `accessor` is `source` expanded by trailing broadcast (pitch form).
fn is_broadcast_strided(accessor: &Strided, source: &Strided) -> bool {
    if accessor.offset != source.offset || accessor.rank() < source.rank() {
        return false;
    }
    let lead = accessor.rank() - source.rank();
    if accessor.pitch[..lead].iter().any(|&p| p != 0) {
        return false;
    }
    source.shape.iter().zip(&source.pitch).enumerate().all(|(i, (&dim, &pitch))| {
        let (out_dim, out_pitch) = (accessor.shape[lead + i], accessor.pitch[lead + i]);
        (out_dim == dim && out_pitch == pitch) || (dim == 1 && out_pitch == 0)
    })
}

/// Pitch-level compose: re-read `accessor` through `expansion`'s shape.
fn compose_strided(accessor: &Accessor, expansion: &Accessor) -> Accessor {
    let a = accessor.strided();
    let e = expansion.strided();
    debug_assert!(e.rank() >= a.rank());
    let lead = e.rank() - a.rank();
    let mut pitch = vec![0; e.rank()];
    for (i, &p) in a.pitch.iter().enumerate() {
        if e.pitch[lead + i] != 0 {
            pitch[lead + i] = p;
        }
    }
    Accessor::Affine {
        offset: a.offset,
        shape: e.shape,
        pitch: pitch.into(),
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
        let s = a.strided();
        assert_eq!(&*s.pitch, &[3, 1]);
        assert_eq!(a.index(&[1, 2]), 5);
    }

    #[test]
    fn broadcast_scalar_to_matrix() {
        let a = Accessor::dense([], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert!(matches!(b, Accessor::Broadcast { .. }));
        let s = b.strided();
        assert_eq!(&*s.shape, &[2, 3]);
        assert_eq!(&*s.pitch, &[0, 0]);
    }

    #[test]
    fn broadcast_trailing_alignment() {
        let a = Accessor::dense([3], 0);
        let b = a.broadcast_to(&[2, 3]).unwrap();
        assert_eq!(&*b.strided().pitch, &[0, 1]);
    }

    #[test]
    fn broadcast_rejects_mismatch() {
        assert!(Accessor::dense([3], 0).broadcast_to(&[2, 4]).is_err());
    }

    #[test]
    fn transpose_swaps_last_two() {
        let a = Accessor::dense([2, 3], 0).transpose().unwrap();
        assert!(matches!(a, Accessor::Transpose(_)));
        let s = a.strided();
        assert_eq!(&*s.shape, &[3, 2]);
        assert_eq!(&*s.pitch, &[1, 3]);
    }

    #[test]
    fn squeeze_drops_unit_axes() {
        let a = Accessor::dense([1, 4], 0).squeeze(&[0]).unwrap();
        assert_eq!(&*a.shape(), &[4]);
    }

    #[test]
    fn is_broadcast_of_structural() {
        let dense = Accessor::dense([3], 0);
        let b = dense.broadcast_to(&[2, 3]).unwrap();
        assert!(b.is_broadcast_of(&dense));
        assert!(dense.is_broadcast_of(&dense));
    }

    #[test]
    fn compose_through_broadcast_keeps_structure() {
        let write = Accessor::dense([3], 0);
        let read = write.broadcast_to(&[2, 3]).unwrap();
        let arg = Accessor::dense([3], 0);
        let composed = arg.compose_through(&write, &read);
        assert!(matches!(composed, Accessor::Broadcast { .. }));
        assert_eq!(&*composed.shape(), &[2, 3]);
    }
}
