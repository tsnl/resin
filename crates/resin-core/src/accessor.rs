//! Strided buffer views (offset / shape / pitch).

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Accessor {
    pub offset: u32,
    pub shape: Box<[u32]>,
    pub pitch: Box<[u32]>,
}

impl Accessor {
    pub fn new(offset: u32, shape: impl Into<Box<[u32]>>, pitch: impl Into<Box<[u32]>>) -> Self {
        let shape = shape.into();
        let pitch = pitch.into();
        assert_eq!(shape.len(), pitch.len(), "inconsistent rank");
        Self {
            offset,
            shape,
            pitch,
        }
    }

    /// C-contiguous accessor addressing `shape` from `offset`.
    pub fn dense(shape: impl Into<Box<[u32]>>, offset: u32) -> Self {
        let shape = shape.into();
        let pitch = c_contiguous_pitch_for_shape(&shape);
        Self {
            offset,
            shape,
            pitch,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_dense_c_contiguous(&self, node_shape: &[u32]) -> bool {
        if self.offset != 0 || self.shape.as_ref() != node_shape {
            return false;
        }
        self.pitch.as_ref() == c_contiguous_pitch_for_shape(node_shape).as_ref()
    }

    pub fn broadcast(&self, leading_shape: &[u32]) -> Self {
        let mut shape = Vec::with_capacity(leading_shape.len() + self.shape.len());
        shape.extend_from_slice(leading_shape);
        shape.extend_from_slice(&self.shape);
        let mut pitch = Vec::with_capacity(leading_shape.len() + self.pitch.len());
        pitch.extend(std::iter::repeat_n(0u32, leading_shape.len()));
        pitch.extend_from_slice(&self.pitch);
        Self::new(self.offset, shape, pitch)
    }

    pub fn permute(&self, permutation: &[usize]) -> Result<Self, String> {
        let mut sorted = permutation.to_vec();
        sorted.sort_unstable();
        if sorted != (0..self.rank()).collect::<Vec<_>>() {
            return Err(format!(
                "bad permutation {permutation:?} for shape {:?}",
                self.shape
            ));
        }
        let shape: Box<[u32]> = permutation.iter().map(|&i| self.shape[i]).collect();
        let pitch: Box<[u32]> = permutation.iter().map(|&i| self.pitch[i]).collect();
        Ok(Self {
            offset: self.offset,
            shape,
            pitch,
        })
    }

    pub fn transpose(&self) -> Result<Self, String> {
        if self.rank() < 2 {
            return Err("transpose requires rank >= 2".into());
        }
        let mut perm: Vec<usize> = (0..self.rank()).collect();
        let n = perm.len();
        perm.swap(n - 2, n - 1);
        self.permute(&perm)
    }

    pub fn squeeze(&self, axes: &[usize]) -> Result<Self, String> {
        for &axis in axes {
            if axis >= self.rank() {
                return Err(format!("squeeze axis {axis} out of range for rank {}", self.rank()));
            }
            if self.shape[axis] != 1 {
                return Err(format!(
                    "cannot squeeze axis {axis} with size {}",
                    self.shape[axis]
                ));
            }
        }
        let mut shape = Vec::new();
        let mut pitch = Vec::new();
        for i in 0..self.rank() {
            if axes.contains(&i) {
                continue;
            }
            shape.push(self.shape[i]);
            pitch.push(self.pitch[i]);
        }
        Ok(Self::new(self.offset, shape, pitch))
    }
}

pub fn c_contiguous_pitch_for_shape(shape: &[u32]) -> Box<[u32]> {
    let mut pitch = vec![0u32; shape.len()];
    let mut stride = 1u32;
    for i in (0..shape.len()).rev() {
        pitch[i] = stride;
        stride = stride.saturating_mul(shape[i]);
    }
    pitch.into_boxed_slice()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShapeJoin {
    pub shape: Box<[u32]>,
    pub pitch1: Box<[u32]>,
    pub pitch2: Box<[u32]>,
}

/// Broadcast-join two shapes/pitches for elementwise ops (NumPy-style trailing align).
pub fn shape_join(
    shape1: &[u32],
    pitch1: &[u32],
    shape2: &[u32],
    pitch2: &[u32],
) -> Result<ShapeJoin, String> {
    assert_eq!(shape1.len(), pitch1.len());
    assert_eq!(shape2.len(), pitch2.len());
    let rank = shape1.len().max(shape2.len());
    let mut out_shape = vec![1u32; rank];
    let mut out_pitch1 = vec![0u32; rank];
    let mut out_pitch2 = vec![0u32; rank];

    for i in 0..rank {
        let i1 = shape1.len().checked_sub(rank - i);
        let i2 = shape2.len().checked_sub(rank - i);
        let d1 = i1.map(|j| shape1[j]).unwrap_or(1);
        let d2 = i2.map(|j| shape2[j]).unwrap_or(1);
        let p1 = i1.map(|j| pitch1[j]).unwrap_or(0);
        let p2 = i2.map(|j| pitch2[j]).unwrap_or(0);

        let d = if d1 == d2 {
            d1
        } else if d1 == 1 {
            d2
        } else if d2 == 1 {
            d1
        } else {
            return Err(format!(
                "incompatible shapes for elementwise: {shape1:?} vs {shape2:?}"
            ));
        };
        out_shape[i] = d;
        out_pitch1[i] = if d1 == 1 { 0 } else { p1 };
        out_pitch2[i] = if d2 == 1 { 0 } else { p2 };
    }

    Ok(ShapeJoin {
        shape: out_shape.into_boxed_slice(),
        pitch1: out_pitch1.into_boxed_slice(),
        pitch2: out_pitch2.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_pitch_2d() {
        let a = Accessor::dense([2, 3], 0);
        assert_eq!(&*a.pitch, &[3, 1]);
    }

    #[test]
    fn broadcast_leading() {
        let a = Accessor::dense([3], 0);
        let b = a.broadcast(&[2]);
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.pitch, &[0, 1]);
    }
}
