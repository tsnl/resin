//! Phase 1 end-to-end tests on the CPU interpreter: integer dtype, bitwise /
//! compare / select ops, casts, slicing views, and row gather/scatter.

use resin::dsl::{grad_wrt, IndexKeyElement, ScatterOp, Tensor};
use resin::jit::{HostArray, CpuJit, Jit};
use resin::ops::ElementType;
use resin::Tree;

#[derive(Tree, Clone)]
struct In<T> {
    x: T,
}

#[derive(Tree, Clone)]
struct Pair<T> {
    a: T,
    b: T,
}

#[test]
fn u32_bitwise_and_shift() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let one = Tensor::full_u32(input.x.shape(), 1);
        // Low bit of (x >> 1), then flipped.
        ((input.x.clone() >> one.clone()) & one.clone()) ^ one
    });
    let x = HostArray::from_u32(&[6], &[0, 1, 2, 3, 6, 7]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![1, 1, 0, 0, 0, 0]);
}

#[test]
fn u32_arithmetic_and_sum() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let two = Tensor::full_u32(input.x.shape(), 2);
        (input.x.clone() * two).sum_axes(&[0]).squeeze_all()
    });
    let x = HostArray::from_u32(&[4], &[1, 2, 3, 4]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![20]);
}

#[test]
fn compare_produces_mask() {
    let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.cmp_lt(&p.b));
    let p = Pair {
        a: HostArray::from_f32(&[4], &[1.0, 5.0, 3.0, 0.0]),
        b: HostArray::from_f32(&[4], &[2.0, 4.0, 3.0, 1.0]),
    };
    let out = f.call(&p).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn select_is_elementwise_choice() {
    let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.cmp_le(&p.b).select(&p.a, &p.b));
    let p = Pair {
        a: HostArray::from_f32(&[4], &[1.0, 5.0, 3.0, 0.5]),
        b: HostArray::from_f32(&[4], &[2.0, 4.0, 3.0, 0.25]),
    };
    let out = f.call(&p).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 4.0, 3.0, 0.25]);
}

#[test]
fn cast_f32_to_u32_truncates() {
    let f = CpuJit.jit(|input: &In<Tensor>| input.x.cast(ElementType::U32));
    let x = HostArray::from_f32(&[4], &[0.5, 1.0, 2.9, 7.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![0, 1, 2, 7]);
}

#[test]
fn cast_u32_to_f32() {
    let f = CpuJit.jit(|input: &In<Tensor>| input.x.cast(ElementType::F32));
    let x = HostArray::from_u32(&[3], &[0, 3, 100]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 3.0, 100.0]);
}

#[test]
fn bitcast_round_trips() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        input
            .x
            .bitcast(ElementType::U32)
            .bitcast(ElementType::F32)
    });
    let x = HostArray::from_f32(&[3], &[-1.5, 0.0, 42.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![-1.5, 0.0, 42.0]);
}

#[test]
fn bitcast_matches_host_bits() {
    let f = CpuJit.jit(|input: &In<Tensor>| input.x.bitcast(ElementType::U32));
    let x = HostArray::from_f32(&[2], &[1.0, -2.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![1.0f32.to_bits(), (-2.0f32).to_bits()]);
}

#[test]
fn floor_sqrt_min_max() {
    let f = CpuJit.jit(|p: &Pair<Tensor>| {
        p.a.floor().maximum(&p.b) + p.a.sqrt().minimum(&p.b)
    });
    let p = Pair {
        a: HostArray::from_f32(&[2], &[4.9, 9.1]),
        b: HostArray::from_f32(&[2], &[3.0, 10.0]),
    };
    let out = f.call(&p).unwrap();
    // max(floor(4.9), 3) + min(sqrt(4.9), 3) ≈ 4 + 2.2136
    assert!((out.to_f32()[0] - (4.0 + 4.9f32.sqrt())).abs() < 1e-5);
    // max(floor(9.1), 10) + min(sqrt(9.1), 10) ≈ 10 + 3.0166
    assert!((out.to_f32()[1] - (10.0 + 9.1f32.sqrt())).abs() < 1e-5);
}

#[test]
fn iota_is_a_u32_range() {
    let f = CpuJit.jit(|input: &In<Tensor>| Tensor::iota(5) + input.x.clone());
    let x = HostArray::from_u32(&[5], &[0, 0, 0, 0, 0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![0, 1, 2, 3, 4]);
}

#[test]
fn index_slice_is_a_view() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        input.x.index(&[IndexKeyElement::Slice(2..5)])
    });
    let x = HostArray::from_f32(&[6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(out.to_f32(), vec![2.0, 3.0, 4.0]);
}

#[test]
fn index_single_keeps_unit_axis() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        input
            .x
            .index(&[IndexKeyElement::Single(1), IndexKeyElement::Slice(0..3)])
    });
    let x = HostArray::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.shape(), &[1, 3]);
    assert_eq!(out.to_f32(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn scatter_index_pads_with_zeros() {
    // Shift x right by 2 inside a length-6 output.
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let head = input.x.index(&[IndexKeyElement::Slice(0..4)]);
        head.scatter_index(&[6], &[IndexKeyElement::Slice(2..6)])
    });
    let x = HostArray::from_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn gather_rows_2d() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[2, 0, 3]);
        input.x.gather_rows(&indices)
    });
    let x = HostArray::from_f32(&[4, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.to_f32(), vec![20.0, 21.0, 0.0, 1.0, 30.0, 31.0]);
}

#[test]
fn scatter_rows_write_permutes() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[2, 0, 1]);
        input.x.scatter_rows(&indices, 3, ScatterOp::Write)
    });
    let x = HostArray::from_f32(&[3], &[10.0, 20.0, 30.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![20.0, 30.0, 10.0]);
}

#[test]
fn scatter_rows_add_accumulates_duplicates() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[4], &[0, 0, 1, 2]);
        input.x.scatter_rows(&indices, 3, ScatterOp::Add)
    });
    let x = HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![3.0, 3.0, 4.0]);
}

#[test]
fn scatter_rows_drops_out_of_range() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[0, 9, 1]);
        input.x.scatter_rows(&indices, 2, ScatterOp::Add)
    });
    let x = HostArray::from_f32(&[3], &[1.0, 2.0, 3.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 3.0]);
}

#[test]
fn u32_gather_and_scatter() {
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let perm = Tensor::constant_u32(&[4], &[3, 2, 1, 0]);
        input.x.gather_rows(&perm)
    });
    let x = HostArray::from_u32(&[4], &[5, 6, 7, 8]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_u32(), vec![8, 7, 6, 5]);
}

#[test]
fn grad_flows_through_gather_as_scatter_add() {
    // loss = sum(x[[1, 1, 3]]) → dloss/dx = [0, 2, 0, 1].
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[1, 1, 3]);
        let loss = input
            .x
            .gather_rows(&indices)
            .sum_axes(&[0])
            .squeeze_all();
        grad_wrt(&loss, &input.x).unwrap()
    });
    let x = HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 2.0, 0.0, 1.0]);
}

#[test]
fn grad_flows_through_scatter_as_gather() {
    // y = scatter_add(x, [1, 1], len 3); loss = sum(y * w) with w = [1, 10, 100]
    // → dloss/dx = [10, 10].
    let f = CpuJit.jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[2], &[1, 1]);
        let w = Tensor::constant_f32(&[3], &[1.0, 10.0, 100.0]);
        let y = input.x.scatter_rows(&indices, 3, ScatterOp::Add);
        let loss = (y * w).sum_axes(&[0]).squeeze_all();
        grad_wrt(&loss, &input.x).unwrap()
    });
    let x = HostArray::from_f32(&[2], &[5.0, 7.0]);
    let out = f.call(&In { x }).unwrap();
    assert_eq!(out.to_f32(), vec![10.0, 10.0]);
}

#[test]
fn grad_of_minimum_routes_to_smaller_operand() {
    let f = CpuJit.jit(|p: &Pair<Tensor>| {
        let loss = p.a.minimum(&p.b).sum_axes(&[0]).squeeze_all();
        grad_wrt(&loss, &p.a).unwrap()
    });
    let p = Pair {
        a: HostArray::from_f32(&[3], &[1.0, 5.0, 2.0]),
        b: HostArray::from_f32(&[3], &[2.0, 4.0, 2.0]),
    };
    let out = f.call(&p).unwrap();
    // a wins (<=) at 0 and 2 (tie), b wins at 1.
    assert_eq!(out.to_f32(), vec![1.0, 0.0, 1.0]);
}

#[test]
fn grad_through_select_ignores_mask() {
    // loss = sum(select(a < b, a * 2, b)); only the taken branch gets gradient.
    let f = CpuJit.jit(|p: &Pair<Tensor>| {
        let two = p.a.ones_like() + p.a.ones_like();
        let mask = p.a.cmp_lt(&p.b);
        let loss = mask
            .select(&(p.a.clone() * two), &p.b)
            .sum_axes(&[0])
            .squeeze_all();
        grad_wrt(&loss, &p.a).unwrap()
    });
    let p = Pair {
        a: HostArray::from_f32(&[2], &[1.0, 9.0]),
        b: HostArray::from_f32(&[2], &[5.0, 5.0]),
    };
    let out = f.call(&p).unwrap();
    assert_eq!(out.to_f32(), vec![2.0, 0.0]);
}
