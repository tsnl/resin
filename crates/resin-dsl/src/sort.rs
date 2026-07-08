//! Sorting as a composition: LSD radix sort built from bit ops, [`scan`],
//! and row scatter — no dedicated sort kernel.
//!
//! Each pass splits on one key bit with a stable counting placement (two
//! exclusive prefix sums), then permutes keys and the running order with a
//! `scatter_rows` write. `bits` passes over `n` elements cost
//! `O(bits · n log n)` work; callers sort only the bits they need (e.g. 16-bit
//! quantized depths, or full 32 for exact float order).

use crate::scan::cumsum;
use crate::tensor::{ElementType, IndexKeyElement, ScatterOp, Tensor};

/// Stable ascending argsort of U32 `keys` over their low `bits` bits.
///
/// Returns a U32 permutation `order` with **gather semantics**:
/// `keys.gather_rows(&order)` is sorted, and `order[i]` is the original index
/// of the i-th smallest key. Keys must be rank 1.
pub fn argsort_u32(keys: &Tensor, bits: u32) -> Tensor {
    assert_eq!(
        keys.element_type(),
        ElementType::U32,
        "argsort_u32: keys must be U32"
    );
    assert_eq!(keys.shape().len(), 1, "argsort_u32: keys must be rank 1");
    assert!(
        (1..=32).contains(&bits),
        "argsort_u32: bits must be in 1..=32"
    );
    let n = keys.shape()[0];
    let mut order = Tensor::iota(n);
    if n <= 1 {
        return order;
    }
    let mut keys_cur = keys.clone();
    let one = Tensor::full_u32(&[n], 1);

    for b in 0..bits {
        let shift = Tensor::full_u32(&[n], b);
        let bit = (keys_cur.clone() >> shift) & one.clone();
        let zeros_flag = bit.clone() ^ one.clone();

        // Stable split positions from two prefix sums (exclusive = incl - x).
        let zeros_incl = cumsum(&zeros_flag, 0);
        let zeros_before = zeros_incl.clone() - zeros_flag.clone();
        let ones_incl = cumsum(&bit, 0);
        let ones_before = ones_incl - bit.clone();
        let total_zeros = zeros_incl
            .index(&[IndexKeyElement::Slice(n - 1..n)])
            .broadcast_to(&[n], &[0]);

        let pos = bit.select(&(total_zeros + ones_before), &zeros_before);

        // pos is a permutation, so a plain scatter write is race-free.
        keys_cur = keys_cur.scatter_rows(&pos, n, ScatterOp::Write);
        order = order.scatter_rows(&pos, n, ScatterOp::Write);
    }
    order
}

/// Order-preserving bijection f32 → u32: `a < b` (totally ordered floats)
/// iff `float_sort_key(a) < float_sort_key(b)` as unsigned integers.
///
/// The usual trick: flip all bits of negative floats, flip only the sign bit
/// of non-negative ones. Sort the result with [`argsort_u32`] (32 bits).
pub fn float_sort_key(x: &Tensor) -> Tensor {
    assert_eq!(
        x.element_type(),
        ElementType::F32,
        "float_sort_key: input must be F32"
    );
    let bits = x.bitcast(ElementType::U32);
    let sign = bits.clone() >> Tensor::full_u32(x.shape(), 31);
    let mask = sign.select(
        &Tensor::full_u32(x.shape(), 0xFFFF_FFFF),
        &Tensor::full_u32(x.shape(), 0x8000_0000),
    );
    bits ^ mask
}

/// Stable ascending argsort of F32 `keys` (gather semantics, like
/// [`argsort_u32`]).
pub fn argsort_f32(keys: &Tensor) -> Tensor {
    argsort_u32(&float_sort_key(keys), 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argsort_shapes() {
        let keys = Tensor::parameter(&[10], ElementType::U32);
        let order = argsort_u32(&keys, 4);
        assert_eq!(order.shape(), &[10]);
        assert_eq!(order.element_type(), ElementType::U32);
    }

    #[test]
    fn argsort_length_one_is_iota() {
        let keys = Tensor::parameter(&[1], ElementType::U32);
        let order = argsort_u32(&keys, 32);
        assert!(matches!(
            order.kind(),
            crate::tensor::TensorKind::Constant { .. }
        ));
    }

    #[test]
    fn float_sort_key_type() {
        let x = Tensor::parameter(&[4], ElementType::F32);
        let key = float_sort_key(&x);
        assert_eq!(key.element_type(), ElementType::U32);
        assert_eq!(key.shape(), &[4]);
    }

    #[test]
    #[should_panic(expected = "keys must be U32")]
    fn argsort_rejects_f32_keys() {
        let keys = Tensor::parameter(&[4], ElementType::F32);
        argsort_u32(&keys, 32);
    }
}
