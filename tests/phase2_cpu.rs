//! Phase 2 end-to-end tests on the CPU interpreter: scan/cumsum/cumprod
//! combinators, radix argsort, and gradients through composed graphs.
//!
//! Elementwise fusion already lives on main (#57); these tests exercise the
//! combinators under the default optimize pipeline.

use resin::Tree;
use resin::dsl::{
    Tensor, argsort_f32, argsort_u32, cumprod_exclusive, cumsum, cumsum_exclusive, grad_wrt, scan,
};
use resin::jit::{CpuJit, HostArray, Jit};

#[derive(Tree)]
struct In<T> {
    x: T,
}

#[test]
fn cumsum_matches_host() {
    let f = CpuJit.jit(|input: &In<Tensor>| cumsum(&input.x, 0));
    let values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]; // non-power-of-two length
    let x = HostArray::from_f32(&[7], &values);
    let out = f.call(&In { x }).unwrap();
    let mut expected = Vec::new();
    let mut acc = 0.0;
    for v in values {
        acc += v;
        expected.push(acc);
    }
    assert_eq!(out.to_f32(), expected);
}

#[test]
fn cumsum_exclusive_starts_at_zero() {
    let f = CpuJit.jit(|input: &In<Tensor>| cumsum_exclusive(&input.x, 0));
    let x = HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 1.0, 3.0, 6.0]);
}

#[test]
fn cumsum_on_u32() {
    let f = CpuJit.jit(|input: &In<Tensor>| cumsum(&input.x, 0));
    let x = HostArray::from_u32(&[5], &[1, 0, 1, 1, 0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![1, 1, 2, 3, 3]);
}

#[test]
fn cumprod_exclusive_is_transmittance() {
    // T[i] = prod_{j<i} x[j], the alpha-compositing transmittance pattern.
    let f = CpuJit.jit(|input: &In<Tensor>| cumprod_exclusive(&input.x, 0));
    let x = HostArray::from_f32(&[4], &[0.5, 0.5, 0.25, 1.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 0.5, 0.25, 0.0625]);
}

#[test]
fn scan_along_axis0_of_matrix() {
    let f = CpuJit.jit(|input: &In<Tensor>| cumsum(&input.x, 0));
    let x = HostArray::from_f32(&[3, 2], &[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 10.0, 3.0, 30.0, 6.0, 60.0]);
}

#[test]
fn scan_with_max_operator() {
    // Running maximum: scan with the user's own traced closure.
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let identity = Tensor::full(&[], f32::NEG_INFINITY);
        scan(&input.x, 0, &identity, |a, b| a.maximum(b))
    });
    let x = HostArray::from_f32(&[6], &[1.0, 3.0, 2.0, 5.0, 4.0, 0.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 3.0, 3.0, 5.0, 5.0, 5.0]);
}

#[test]
fn scan_folds_left_to_right_for_noncommutative_op() {
    // `op(l, r) = r` ("keep the later element") is associative and
    // non-commutative, and 0.0 is a left identity — all a left-to-right scan
    // relies on, since the shift only ever feeds the identity as the earlier
    // (left) operand. The inclusive scan of "keep later" is therefore the
    // input unchanged: x[0] ⊕ … ⊕ x[i] = x[i]. Folding the later element on
    // the left instead collapses the scan to a single shifted element.
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let identity = Tensor::full(&[], 0.0);
        scan(&input.x, 0, &identity, |_earlier, later| later.clone())
    });
    let values = [3.0, 1.0, 4.0, 1.0, 5.0]; // non-power-of-two length
    let x = HostArray::from_f32(&[5], &values);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), values);
}

#[test]
fn grad_of_cumsum_is_reverse_cumsum() {
    // loss = sum(w * cumsum(x)) → dloss/dx[i] = sum_{j>=i} w[j].
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let w = Tensor::constant_f32(&[4], &[1.0, 10.0, 100.0, 1000.0]);
        let loss = (cumsum(&input.x, 0) * w).sum_axes(&[0]).squeeze_all();
        grad_wrt(&loss, &input.x).unwrap()
    });
    let x = HostArray::from_f32(&[4], &[0.5, -1.0, 2.0, 3.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![1111.0, 1110.0, 1100.0, 1000.0]);
}

#[test]
fn argsort_u32_sorts_with_duplicates_stably() {
    let f = CpuJit.jit(|input: &In<Tensor>| argsort_u32(&input.x, 4));
    let x = HostArray::from_u32(&[8], &[3, 1, 3, 0, 2, 1, 0, 2]);
    let out = f.call(&In { x }).unwrap();
    // Stable: equal keys keep original relative order.
    assert_eq!(out.to_u32(), vec![3, 6, 1, 5, 4, 7, 0, 2]);
}

#[test]
fn argsort_u32_gathers_sorted_values() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let order = argsort_u32(&input.x, 8);
        input.x.gather_rows(&order)
    });
    let x = HostArray::from_u32(&[7], &[42, 7, 255, 0, 13, 128, 1]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![0, 1, 7, 13, 42, 128, 255]);
}

#[test]
fn argsort_f32_handles_negatives() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let order = argsort_f32(&input.x);
        input.x.gather_rows(&order)
    });
    let x = HostArray::from_f32(&[6], &[0.5, -1.5, 3.0, -0.25, 0.0, 2.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![-1.5, -0.25, 0.0, 0.5, 2.0, 3.0]);
}

#[test]
fn sorted_gather_is_differentiable() {
    // loss = sum(w * sort(x)): gradient routes each weight to the source
    // element that landed in that sorted slot.
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let order = argsort_f32(&input.x);
        let sorted = input.x.gather_rows(&order);
        let w = Tensor::constant_f32(&[4], &[1.0, 10.0, 100.0, 1000.0]);
        let loss = (sorted * w).sum_axes(&[0]).squeeze_all();
        grad_wrt(&loss, &input.x).unwrap()
    });
    let x = HostArray::from_f32(&[4], &[2.0, 0.5, 3.0, 1.0]);
    let out = f.call(&In { x }).unwrap();
    // Ascending order is [0.5, 1.0, 2.0, 3.0] = indices [1, 3, 0, 2].
    assert_eq!(out.to_f32(), vec![100.0, 1.0, 1000.0, 10.0]);
}
