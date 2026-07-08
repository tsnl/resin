//! Reshape view op: pure accessor re-striding, gradients, and rejection of
//! non-contiguous operands.

#![cfg(feature = "cpu")]

use resin_dsl::{grad_wrt, Tensor};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::{ConcreteTensor, Jit};

#[test]
fn reshape_is_a_view() {
    let jit = CpuJit;
    let f = jit.jit(|x: &Tensor| {
        // [6] -> [2, 3], reduce rows -> [1, 3] -> squeeze -> [3]
        x.reshape(&[2, 3]).sum_axes(&[0]).squeeze(&[0])
    });
    let x = CpuTensor::from_f32(&[6], &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
    let out = f.call(&x).unwrap();
    assert_eq!(out.to_f32(), vec![11.0, 22.0, 33.0]);
}

#[test]
fn reshape_of_contiguous_slice_offsets_correctly() {
    let jit = CpuJit;
    let f = jit.jit(|x: &Tensor| {
        x.index(&[resin_dsl::IndexKeyElement::Slice(2..6)])
            .reshape(&[2, 2])
            .sum_axes(&[1])
            .squeeze(&[1])
    });
    let x = CpuTensor::from_f32(&[6], &[0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
    let out = f.call(&x).unwrap();
    assert_eq!(out.to_f32(), vec![3.0, 7.0]);
}

#[test]
fn grad_flows_through_reshape() {
    let jit = CpuJit;
    let f = jit.jit(|x: &Tensor| {
        let w = Tensor::constant_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let loss = (x.reshape(&[2, 3]) * w).sum_axes(&[0, 1]).squeeze_all();
        grad_wrt(&loss, x).unwrap()
    });
    let x = CpuTensor::from_f32(&[6], &[0.0; 6]);
    let out = f.call(&x).unwrap();
    assert_eq!(out.shape(), &[6]);
    assert_eq!(out.to_f32(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_of_broadcast_view_is_rejected() {
    let jit = CpuJit;
    let f = jit.jit(|x: &Tensor| x.broadcast_to(&[2, 4], &[1]).reshape(&[8]));
    let x = CpuTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let err = f.call(&x).unwrap_err();
    assert!(
        err.to_string().contains("non-contiguous"),
        "unexpected error: {err}"
    );
}
