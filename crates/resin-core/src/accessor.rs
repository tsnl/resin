//! Strided buffer views (offset / shape / pitch).

use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

use serde::{Deserialize, Serialize};

/// One axis in a `narrow` / index key (Python `int | slice` per dimension).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AxisIndex {
    /// Integer index; removes the axis. Supports negative indices (`-1` = last).
    At(i32),
    /// Slice `[start:end:step]`. `step` must be `> 0` (pitch is unsigned in IR/WGSL).
    Slice {
        start: Option<i32>,
        end: Option<i32>,
        step: i32,
    },
}

impl AxisIndex {
    pub fn all() -> Self {
        Self::Slice {
            start: None,
            end: None,
            step: 1,
        }
    }

    pub fn at(i: i32) -> Self {
        Self::At(i)
    }

    pub fn slice(start: Option<i32>, end: Option<i32>, step: i32) -> Self {
        Self::Slice { start, end, step }
    }
}

/// Convert a value into one axis of an index key (`0`, `1..3`, `..`, …).
pub trait IntoAxisIndex {
    fn into_axis_index(self) -> AxisIndex;
}

impl IntoAxisIndex for AxisIndex {
    fn into_axis_index(self) -> AxisIndex {
        self
    }
}

impl IntoAxisIndex for i32 {
    fn into_axis_index(self) -> AxisIndex {
        AxisIndex::At(self)
    }
}

impl IntoAxisIndex for i64 {
    fn into_axis_index(self) -> AxisIndex {
        AxisIndex::At(i32::try_from(self).expect("axis index fits i32"))
    }
}

impl IntoAxisIndex for usize {
    fn into_axis_index(self) -> AxisIndex {
        AxisIndex::At(i32::try_from(self).expect("axis index fits i32"))
    }
}

impl IntoAxisIndex for u32 {
    fn into_axis_index(self) -> AxisIndex {
        AxisIndex::At(i32::try_from(self).expect("axis index fits i32"))
    }
}

impl IntoAxisIndex for RangeFull {
    fn into_axis_index(self) -> AxisIndex {
        AxisIndex::all()
    }
}

macro_rules! impl_range_into_axis {
    ($($t:ty),+ $(,)?) => {$(
        impl IntoAxisIndex for Range<$t> {
            fn into_axis_index(self) -> AxisIndex {
                AxisIndex::Slice {
                    start: Some(self.start as i32),
                    end: Some(self.end as i32),
                    step: 1,
                }
            }
        }
        impl IntoAxisIndex for RangeFrom<$t> {
            fn into_axis_index(self) -> AxisIndex {
                AxisIndex::Slice {
                    start: Some(self.start as i32),
                    end: None,
                    step: 1,
                }
            }
        }
        impl IntoAxisIndex for RangeTo<$t> {
            fn into_axis_index(self) -> AxisIndex {
                AxisIndex::Slice {
                    start: None,
                    end: Some(self.end as i32),
                    step: 1,
                }
            }
        }
        impl IntoAxisIndex for RangeInclusive<$t> {
            fn into_axis_index(self) -> AxisIndex {
                let start = *self.start() as i32;
                let end = *self.end() as i32;
                AxisIndex::Slice {
                    start: Some(start),
                    end: Some(end.saturating_add(1)),
                    step: 1,
                }
            }
        }
        impl IntoAxisIndex for RangeToInclusive<$t> {
            fn into_axis_index(self) -> AxisIndex {
                AxisIndex::Slice {
                    start: None,
                    end: Some((self.end as i32).saturating_add(1)),
                    step: 1,
                }
            }
        }
    )+};
}

impl_range_into_axis!(i32, i64, u32, usize);

/// Multi-axis index key for [`Accessor::narrow`] / `View::index`.
///
/// Accepts a single axis (`0`, `1..3`, `..`) or a tuple of up to **5** axes.
pub trait IntoIndexKey {
    fn into_index_key(self) -> Box<[AxisIndex]>;
}

impl<T: IntoAxisIndex> IntoIndexKey for T {
    fn into_index_key(self) -> Box<[AxisIndex]> {
        Box::from([self.into_axis_index()])
    }
}

macro_rules! impl_index_key_tuple {
    ($($T:ident),+) => {
        impl<$($T: IntoAxisIndex),+> IntoIndexKey for ($($T,)+) {
            fn into_index_key(self) -> Box<[AxisIndex]> {
                #[allow(non_snake_case)]
                let ($($T,)+) = self;
                Box::from([$($T.into_axis_index()),+])
            }
        }
    };
}

impl_index_key_tuple!(A);
impl_index_key_tuple!(A, B);
impl_index_key_tuple!(A, B, C);
impl_index_key_tuple!(A, B, C, D);
impl_index_key_tuple!(A, B, C, D, E);

impl IntoIndexKey for &[AxisIndex] {
    fn into_index_key(self) -> Box<[AxisIndex]> {
        self.to_vec().into_boxed_slice()
    }
}

impl<const N: usize> IntoIndexKey for [AxisIndex; N] {
    fn into_index_key(self) -> Box<[AxisIndex]> {
        Box::from(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
    pub fn new_c_contiguous(shape: impl Into<Box<[u32]>>, offset: u32) -> Self {
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
                return Err(format!(
                    "squeeze axis {axis} out of range for rank {}",
                    self.rank()
                ));
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

    /// Python `Accessor.narrow`: apply per-axis int / slice keys (trailing axes kept).
    ///
    /// Integer axes are removed (rank shrinks). Slice axes keep a strided subrange.
    /// Negative slice steps are not supported while pitch is `u32`.
    pub fn narrow(&self, key: &[AxisIndex]) -> Result<Self, String> {
        if key.len() > self.rank() {
            return Err(format!(
                "index rank {} exceeds accessor rank {}",
                key.len(),
                self.rank()
            ));
        }

        let mut new_offset = self.offset as i64;
        let mut new_pitch = Vec::new();
        let mut new_shape = Vec::new();

        for (dim, k) in key.iter().enumerate() {
            let d = self.shape[dim] as i64;
            let p = self.pitch[dim] as i64;
            match *k {
                AxisIndex::At(idx) => {
                    let i = bounded_index(idx, d, dim)?;
                    new_offset += i * p;
                }
                AxisIndex::Slice { start, end, step } => {
                    if step == 0 {
                        return Err("slice step must not be 0".into());
                    }
                    if step < 0 {
                        return Err(
                            "negative slice step not supported (unsigned pitch in IR/WGSL)".into(),
                        );
                    }
                    let b = match start {
                        Some(s) => bounded_index(s, d, dim)?,
                        None => 0,
                    };
                    let e = match end {
                        Some(stop) => bounded_end(stop, d, dim)?,
                        None => d,
                    };
                    let a = step as i64;
                    new_offset += b * p;
                    new_pitch.push((p * a) as u32);
                    new_shape.push(std::cmp::max(0, (e - b + (a - 1)) / a) as u32);
                }
            }
        }

        for dim in key.len()..self.rank() {
            new_pitch.push(self.pitch[dim]);
            new_shape.push(self.shape[dim]);
        }

        if new_offset < 0 {
            return Err(format!("narrow produced negative offset {new_offset}"));
        }
        Ok(Self::new(new_offset as u32, new_shape, new_pitch))
    }
}

fn bounded_index(k: i32, dim_size: i64, dim: usize) -> Result<i64, String> {
    let k = i64::from(k);
    if !(-dim_size..dim_size).contains(&k) {
        return Err(format!(
            "index {k} out of bounds for dimension {dim} of size {dim_size}"
        ));
    }
    Ok(k.rem_euclid(dim_size))
}

fn bounded_end(k: i32, dim_size: i64, dim: usize) -> Result<i64, String> {
    let k = i64::from(k);
    if !(0..=dim_size).contains(&k) {
        return Err(format!(
            "slice end {k} out of bounds for dimension {dim} of size {dim_size}"
        ));
    }
    Ok(k)
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
        let a = Accessor::new_c_contiguous([2, 3], 0);
        assert_eq!(&*a.pitch, &[3, 1]);
    }

    #[test]
    fn broadcast_leading() {
        let a = Accessor::new_c_contiguous([3], 0);
        let b = a.broadcast(&[2]);
        assert_eq!(&*b.shape, &[2, 3]);
        assert_eq!(&*b.pitch, &[0, 1]);
    }

    #[test]
    fn narrow_row_and_slice() {
        // [[1,2,3],[4,5,6]] row-major, pitch [3,1]
        let a = Accessor::new_c_contiguous([2, 3], 0);
        // t[(0, :)] → shape [3], offset 0, pitch [1]
        let key = (0i32, ..).into_index_key();
        let row0 = a.narrow(&key).unwrap();
        assert_eq!(row0.offset, 0);
        assert_eq!(&*row0.shape, &[3]);
        assert_eq!(&*row0.pitch, &[1]);

        // t[(1, 1:3)] → elements 5,6 → offset 3+1=4, shape [2], pitch [1]
        let key = (1usize, 1..3).into_index_key();
        let sub = a.narrow(&key).unwrap();
        assert_eq!(sub.offset, 4);
        assert_eq!(&*sub.shape, &[2]);
        assert_eq!(&*sub.pitch, &[1]);
    }

    #[test]
    fn narrow_step_2() {
        let a = Accessor::new_c_contiguous([6], 0);
        let s = a.narrow(&[AxisIndex::slice(None, None, 2)]).unwrap();
        assert_eq!(s.offset, 0);
        assert_eq!(&*s.shape, &[3]);
        assert_eq!(&*s.pitch, &[2]);
    }

    #[test]
    fn narrow_negative_index() {
        let a = Accessor::new_c_contiguous([2, 3], 0);
        let last_row = a.narrow(&[AxisIndex::at(-1), AxisIndex::all()]).unwrap();
        assert_eq!(last_row.offset, 3);
        assert_eq!(&*last_row.shape, &[3]);
    }
}
