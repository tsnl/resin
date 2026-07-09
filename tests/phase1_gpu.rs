//! Phase 1 end-to-end tests on the GPU backend.
//!
//! Exercises the same surface as `phase1_cpu` where it matters for codegen:
//! u32 elementwise, cast/bitcast, remaps (gather / scatter write / scatter-add
//! / scatter-view), and a small graph that mixes them. Skips cleanly when no
//! adapter is present (CI without Metal/Vulkan).

#![cfg(feature = "wgpu")]

use resin::dsl::{grad_wrt, IndexKeyElement, ScatterOp, Tensor};
use resin::jit::wgpu::{gpu_available, WgpuJit};
use resin::jit::{DeviceValue, HostArray, Jit};
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

fn no_gpu() -> bool {
    if gpu_available() {
        false
    } else {
        eprintln!("skip: no GPU adapter");
        true
    }
}

fn up(host: HostArray) -> resin::jit::WgpuArray {
    WgpuJit::default().upload(&host).unwrap()
}

#[test]
fn u32_bitwise_and_shift() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let one = Tensor::full_u32(input.x.shape(), 1);
        ((input.x.clone() >> one.clone()) & one.clone()) ^ one
    });
    let x = up(HostArray::from_u32(&[6], &[0, 1, 2, 3, 6, 7]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_u32(), vec![1, 1, 0, 0, 0, 0]);
}

#[test]
fn u32_add_and_sum() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let two = Tensor::full_u32(input.x.shape(), 2);
        (input.x.clone() * two).sum_axes(&[0]).squeeze_all()
    });
    let x = up(HostArray::from_u32(&[4], &[1, 2, 3, 4]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_u32(), vec![20]);
}

#[test]
fn cast_round_trip() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        input.x.cast(ElementType::U32).cast(ElementType::F32)
    });
    let x = up(HostArray::from_f32(&[4], &[0.0, 1.0, 2.0, 7.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 1.0, 2.0, 7.0]);
}

#[test]
fn bitcast_round_trips() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        input.x.bitcast(ElementType::U32).bitcast(ElementType::F32)
    });
    let x = up(HostArray::from_f32(&[3], &[-1.5, 0.0, 42.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![-1.5, 0.0, 42.0]);
}

#[test]
fn compare_select() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|p: &Pair<Tensor>| p.a.cmp_le(&p.b).select(&p.a, &p.b));
    let p = Pair {
        a: up(HostArray::from_f32(&[4], &[1.0, 5.0, 3.0, 0.5])),
        b: up(HostArray::from_f32(&[4], &[2.0, 4.0, 3.0, 0.25])),
    };
    let out = f.call(&p).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![1.0, 4.0, 3.0, 0.25]);
}

#[test]
fn gather_rows_2d() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[2, 0, 3]);
        input.x.gather_rows(&indices)
    });
    let x = up(HostArray::from_f32(&[4, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.shape(), &[3, 2]);
    assert_eq!(out.to_f32(), vec![20.0, 21.0, 0.0, 1.0, 30.0, 31.0]);
}

#[test]
fn scatter_rows_write_permutes() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[2, 0, 1]);
        input.x.scatter_rows(&indices, 3, ScatterOp::Write)
    });
    let x = up(HostArray::from_f32(&[3], &[10.0, 20.0, 30.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![20.0, 30.0, 10.0]);
}

#[test]
fn scatter_rows_add_f32_accumulates_duplicates() {
    if no_gpu() {
        return;
    }
    // f32 scatter-add uses a compare-exchange loop in WGSL.
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[4], &[0, 0, 1, 2]);
        input.x.scatter_rows(&indices, 3, ScatterOp::Add)
    });
    let x = up(HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![3.0, 3.0, 4.0]);
}

#[test]
fn scatter_rows_add_u32_uses_atomic() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[4], &[0, 0, 1, 2]);
        input.x.scatter_rows(&indices, 3, ScatterOp::Add)
    });
    let x = up(HostArray::from_u32(&[4], &[1, 2, 3, 4]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_u32(), vec![3, 3, 4]);
}

#[test]
fn scatter_index_pads_with_zeros() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let head = input.x.index(&[IndexKeyElement::Slice(0..4)]);
        head.scatter_index(&[6], &[IndexKeyElement::Slice(2..6)])
    });
    let x = up(HostArray::from_f32(&[6], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn u32_gather_rows() {
    if no_gpu() {
        return;
    }
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let perm = Tensor::constant_u32(&[4], &[3, 2, 1, 0]);
        input.x.gather_rows(&perm)
    });
    let x = up(HostArray::from_u32(&[4], &[5, 6, 7, 8]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_u32(), vec![8, 7, 6, 5]);
}

#[test]
fn grad_through_gather_runs_on_gpu() {
    if no_gpu() {
        return;
    }
    // loss = sum(x[[1, 1, 3]]) → dloss/dx = [0, 2, 0, 1] via scatter-add.
    let f = WgpuJit::default().jit(|input: &In<Tensor>| {
        let indices = Tensor::constant_u32(&[3], &[1, 1, 3]);
        let loss = input.x.gather_rows(&indices).sum_axes(&[0]).squeeze_all();
        grad_wrt(&loss, &input.x).unwrap()
    });
    let x = up(HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]));
    let out = f.call(&In { x }).unwrap().host().unwrap();
    assert_eq!(out.to_f32(), vec![0.0, 2.0, 0.0, 1.0]);
}

/// One graph that mixes u32 ops, cast, select, gather, scatter-add, and grad.
#[test]
fn phase1_mixed_graph() {
    if no_gpu() {
        return;
    }

    #[derive(Tree, Clone)]
    struct MixedIn<T> {
        x: T,
        k: T,
    }

    #[derive(Tree, Clone)]
    struct MixedOut<T> {
        loss: T,
        grad: T,
    }

    let f = WgpuJit::default().jit(|input: &MixedIn<Tensor>| {
        let one = Tensor::full_u32(input.k.shape(), 1);
        let bit = (input.k.clone() >> one.clone()) & one;
        let mask = bit.cast(ElementType::F32);
        let indices = Tensor::constant_u32(&[4], &[3, 2, 1, 0]);
        let gathered = input.x.gather_rows(&indices);
        let blended = mask.select(&gathered, &input.x);
        let scattered = blended.scatter_rows(&indices, 4, ScatterOp::Add);
        let loss = scattered.sum_axes(&[0]).squeeze_all();
        let grad = grad_wrt(&loss, &input.x).unwrap();
        MixedOut { loss, grad }
    });

    let out = f
        .call(&MixedIn {
            x: up(HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0])),
            k: up(HostArray::from_u32(&[4], &[0, 1, 2, 3])),
        })
        .unwrap();
    let loss = out.loss.host().unwrap();
    let grad = out.grad.host().unwrap();
    // Smoke: shapes and finite values; the graph compiled and ran on GPU.
    assert!(loss.shape().is_empty());
    assert_eq!(grad.shape(), &[4]);
    assert!(loss.scalar().is_finite());
    assert!(grad.to_f32().iter().all(|v| v.is_finite()));
}
