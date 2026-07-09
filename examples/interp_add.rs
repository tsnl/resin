//! Smoke test: add two arrays through the CPU JIT.

use resin::Tree;
use resin::dsl::Tensor;
use resin::jit::{DeviceValue, HostArray, CpuJit, Jit};

#[derive(Tree, Clone)]
struct Pair<T> {
    a: T,
    b: T,
}

fn main() {
    let sum = CpuJit.jit(|pair: &Pair<Tensor>| pair.a.clone() + pair.b.clone());
    let out = sum
        .call(&Pair {
            a: HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
            b: HostArray::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
        })
        .expect("jit add")
        .host()
        .expect("host");
    assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    println!("out = {:?}", out.data());
}
