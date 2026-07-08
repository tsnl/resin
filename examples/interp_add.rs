//! Smoke test: const and param paths via the CPU JIT (Python `demo_interp` / `interp_add`).

use resin::dsl::Tensor;
use resin::jit::backends::cpu::{CpuJit, CpuTensor};
use resin::jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

#[derive(Tree)]
struct Pair<J: Jit> {
    a: J::Tensor,
    b: J::Tensor,
}

fn main() {
    let jit = CpuJit;

    let sum = jit.jit(|pair: &PairMapped<Tensor>| pair.a.clone() + pair.b.clone());
    let params = Pair::<CpuJit> {
        a: CpuTensor::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
        b: CpuTensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
    };
    let out = sum.call(&params).expect("jit add");
    assert_eq!(out.to_f32(), vec![11.0, 22.0, 33.0, 44.0]);
    println!("out = {:?}", out.to_f32());
}