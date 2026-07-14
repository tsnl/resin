//! Time a single forward pass of the MNIST MLP on the CPU interpreter.

use std::time::Instant;

use resin::Tree;
use resin::dsl::Tensor;
use resin::jit::{CpuJit, HostArray, Jit};

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
struct ForwardIn<T> {
    xs: T,
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
    let f = CpuJit.jit(|input: &ForwardIn<Tensor>| forward(&input.model, &input.xs));
    let layer = |i: usize, o: usize| LinearParams {
        weight: HostArray::from_f32(&[i, o], &vec![0.01; i * o]),
        bias: HostArray::from_f32(&[o], &vec![0.0; o]),
    };
    let input = ForwardIn {
        xs: HostArray::from_f32(&[8, 784], &vec![0.1; 8 * 784]),
        model: MlpParams {
            layer0: layer(784, 32),
            layer1: layer(32, 32),
            layer2: layer(32, 10),
        },
    };
    let t0 = Instant::now();
    let _ = f.call(&input).unwrap();
    eprintln!("forward-only call: {:?}", t0.elapsed());
}
