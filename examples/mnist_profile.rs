use std::time::Instant;

use resin::dsl::Tensor;
use resin::jit::backends::cpu::{CpuJit, CpuTensor};
use resin::jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

#[derive(Tree, Clone)]
struct LinearParams<T> {
    weight: T,
    bias: T,
}

#[derive(Tree, Clone)]
struct MlpParams<T> {
    layer0: LinearParams<T>,
    layer1: LinearParams<T>,
    layer2: LinearParams<T>,
}

#[derive(Tree, Clone)]
struct TrainStepIn<T> {
    xs: T,
    ys: T,
    model: MlpParams<T>,
}

fn linear(x: &Tensor, layer: &LinearParams<Tensor>) -> Tensor {
    let out = x.matmul(&layer.weight);
    out.clone() + layer.bias.broadcast_to(out.shape(), &[1])
}

fn forward(model: &MlpParams<Tensor>, x: &Tensor) -> Tensor {
    let h0 = linear(x, &model.layer0).relu();
    let h1 = linear(&h0, &model.layer1).relu();
    linear(&h1, &model.layer2)
}

fn main() {
    let jit = CpuJit;
    let f = jit.jit(|step: &TrainStepIn<Tensor>| forward(&step.model, &step.xs));
    let step = TrainStepIn {
        xs: CpuTensor::from_f32(&[8, 784], &vec![0.1; 8 * 784]),
        ys: CpuTensor::from_f32(&[8, 10], &vec![0.0; 80]),
        model: MlpParams {
            layer0: LinearParams {
                weight: CpuTensor::from_f32(&[784, 32], &vec![0.01; 784 * 32]),
                bias: CpuTensor::from_f32(&[32], &vec![0.0; 32]),
            },
            layer1: LinearParams {
                weight: CpuTensor::from_f32(&[32, 32], &vec![0.01; 32 * 32]),
                bias: CpuTensor::from_f32(&[32], &vec![0.0; 32]),
            },
            layer2: LinearParams {
                weight: CpuTensor::from_f32(&[32, 10], &vec![0.01; 32 * 10]),
                bias: CpuTensor::from_f32(&[10], &vec![0.0; 10]),
            },
        },
    };
    let t0 = Instant::now();
    let _ = f.call(&step).unwrap();
    eprintln!("forward-only call: {:?}", t0.elapsed());
}