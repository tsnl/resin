//! Higher-order scan: a combinator, not a kernel.
//!
//! [`scan`] takes an ordinary Rust closure for its binary operator, traces it
//! once per doubling step, and unrolls a Hillis–Steele inclusive scan into
//! `⌈log₂ n⌉` passes of existing ops (a shift + the user's combine). Because
//! the result is a plain feed-forward graph of differentiable nodes, autodiff
//! and lowering need no new rules — and dead branches never lower at all.
//!
//! Cost model: `O(n log n)` work over `2⌈log₂ n⌉` dispatches — the right
//! order of magnitude without a dedicated kernel; a fused/work-efficient scan
//! can later slot in *underneath* this API without changing user code.

use super::{ConstantData, IndexKeyElement, Tensor, TensorKind};
use crate::ops::ElementType;

/// `out[i] = x[i - offset]` along `axis`, with the leading `offset` slots
/// taking `fill` (a scalar tensor of the same element type).
pub fn shift_axis(x: &Tensor, axis: usize, offset: usize, fill: &Tensor) -> Tensor {
    let shape = x.shape();
    assert!(axis < shape.len(), "shift_axis: axis {axis} out of range");
    assert!(fill.shape().is_empty(), "shift_axis: fill must be a scalar tensor");
    assert_eq!(
        fill.element_type(),
        x.element_type(),
        "shift_axis: fill element type must match"
    );
    let n = shape[axis];
    if offset == 0 {
        return x.clone();
    }
    if offset >= n {
        // Fully shifted out: a constant-filled tensor.
        return fill.broadcast_to(shape, &[]);
    }

    let full_key = |range_for_axis: std::ops::Range<usize>| -> Vec<IndexKeyElement> {
        shape
            .iter()
            .enumerate()
            .map(|(a, &dim)| {
                if a == axis {
                    IndexKeyElement::Slice(range_for_axis.clone())
                } else {
                    IndexKeyElement::Slice(0..dim)
                }
            })
            .collect()
    };

    // Body: x[..n-offset] embedded at [offset..n] (zeros elsewhere).
    let head = x.index(&full_key(0..n - offset));
    let body = head.scatter_index(shape, &full_key(offset..n));

    if is_zero_constant(fill) {
        return body;
    }

    // Non-zero fill: add a disjoint front region holding the fill value.
    let mut front_shape = shape.to_vec();
    front_shape[axis] = offset;
    let front_region = fill.broadcast_to(&front_shape, &[]);
    let front = front_region.scatter_index(shape, &full_key(0..offset));
    body + front
}

/// Inclusive scan of `x` along `axis` with a user-traced binary operator.
///
/// `op` must be associative with identity element `identity` (a scalar
/// tensor) — a monoid. Commutativity and inverses are not required. The
/// closure is called once per doubling step at trace time to build the
/// unrolled graph; gradients flow through whatever graph `op` builds.
///
/// **Implementation:** Hillis–Steele (shift-by-stride + combine), unrolled into
/// existing ops. Work is O(n log n).
///
/// TODO: swap the unroll under the hood for work-efficient Blelloch
/// (upsweep + downsweep) under the same monoid contract — same API, no inverse
/// operator. Real O(n) work needs shrinking/expanding intermediate shapes, not
/// n-wide masked kernels at every level.
pub fn scan(
    x: &Tensor,
    axis: usize,
    identity: &Tensor,
    mut op: impl FnMut(&Tensor, &Tensor) -> Tensor,
) -> Tensor {
    let shape = x.shape();
    assert!(axis < shape.len(), "scan: axis {axis} out of range");
    let n = shape[axis];
    let mut acc = x.clone();
    let mut stride = 1;
    while stride < n {
        let shifted = shift_axis(&acc, axis, stride, identity);
        acc = op(&acc, &shifted);
        assert_eq!(acc.shape(), shape, "scan: op must preserve the operand shape");
        stride *= 2;
    }
    acc
}

/// Exclusive scan: like [`scan`] but `out[i]` combines strictly-preceding
/// elements; `out[0] = identity`.
pub fn scan_exclusive(
    x: &Tensor,
    axis: usize,
    identity: &Tensor,
    op: impl FnMut(&Tensor, &Tensor) -> Tensor,
) -> Tensor {
    let inclusive = scan(x, axis, identity, op);
    shift_axis(&inclusive, axis, 1, identity)
}

/// Inclusive cumulative sum along `axis`.
pub fn cumsum(x: &Tensor, axis: usize) -> Tensor {
    let zero = zero_scalar(x);
    scan(x, axis, &zero, |a, b| a.clone() + b.clone())
}

/// Exclusive cumulative sum along `axis` (`out[0] = 0`).
pub fn cumsum_exclusive(x: &Tensor, axis: usize) -> Tensor {
    let zero = zero_scalar(x);
    scan_exclusive(x, axis, &zero, |a, b| a.clone() + b.clone())
}

/// Inclusive cumulative product along `axis`.
pub fn cumprod(x: &Tensor, axis: usize) -> Tensor {
    let one = one_scalar(x);
    scan(x, axis, &one, |a, b| a.clone() * b.clone())
}

/// Exclusive cumulative product along `axis` (`out[0] = 1`). This is the
/// transmittance pattern: `T[i] = Π_{j<i} x[j]`.
pub fn cumprod_exclusive(x: &Tensor, axis: usize) -> Tensor {
    let one = one_scalar(x);
    scan_exclusive(x, axis, &one, |a, b| a.clone() * b.clone())
}

fn zero_scalar(like: &Tensor) -> Tensor {
    match like.element_type() {
        ElementType::U32 => Tensor::full_u32(&[], 0),
        ElementType::F32 => Tensor::full(&[], 0.0),
    }
}

fn one_scalar(like: &Tensor) -> Tensor {
    match like.element_type() {
        ElementType::U32 => Tensor::full_u32(&[], 1),
        ElementType::F32 => Tensor::full(&[], 1.0),
    }
}

fn is_zero_constant(t: &Tensor) -> bool {
    match t.kind() {
        TensorKind::Constant { values: ConstantData::F32(v) } => v.iter().all(|&x| x == 0.0),
        TensorKind::Constant { values: ConstantData::U32(v) } => v.iter().all(|&x| x == 0),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_preserves_shape_and_type() {
        let x = Tensor::parameter(&[7]);
        let out = cumsum(&x, 0);
        assert_eq!(out.shape(), &[7]);
        assert_eq!(out.element_type(), ElementType::F32);
    }

    #[test]
    fn scan_on_u32() {
        let x = Tensor::parameter_typed(&[8], ElementType::U32);
        let out = cumsum(&x, 0);
        assert_eq!(out.element_type(), ElementType::U32);
    }

    #[test]
    fn scan_along_leading_axis_of_matrix() {
        let x = Tensor::parameter(&[5, 3]);
        let out = cumprod(&x, 0);
        assert_eq!(out.shape(), &[5, 3]);
    }

    #[test]
    fn shift_by_zero_is_identity() {
        let x = Tensor::parameter(&[4]);
        let fill = Tensor::full(&[], 0.0);
        assert!(shift_axis(&x, 0, 0, &fill) == x);
    }

    #[test]
    fn shift_past_length_is_fill() {
        let x = Tensor::parameter(&[4]);
        let fill = Tensor::full(&[], 1.0);
        let out = shift_axis(&x, 0, 4, &fill);
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    #[should_panic(expected = "fill must be a scalar")]
    fn shift_rejects_non_scalar_fill() {
        let x = Tensor::parameter(&[4]);
        let fill = Tensor::full(&[4], 0.0);
        shift_axis(&x, 0, 1, &fill);
    }

    #[test]
    fn scan_length_one_traces_no_steps() {
        let x = Tensor::parameter(&[1]);
        let zero = Tensor::full(&[], 0.0);
        let mut calls = 0;
        let out = scan(&x, 0, &zero, |a, b| {
            calls += 1;
            a.clone() + b.clone()
        });
        assert_eq!(calls, 0);
        assert!(out == x, "scan over length-1 axis is the input");
    }
}
