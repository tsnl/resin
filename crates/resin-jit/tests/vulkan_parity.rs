//! CPU ↔ Vulkan parity across the existing kernel families (elementwise,
//! matmul, reduction, remap) on one composite graph. Skips without a Vulkan
//! driver.

#![cfg(all(feature = "cpu", feature = "vulkan"))]

use resin_dsl::{ScatterOp, Tensor};
use resin_jit::backends::cpu::{CpuJit, CpuTensor};
use resin_jit::backends::vulkan::{shared_context_available, VulkanJit, VulkanTensor};
use resin_jit::{ConcreteTensor, Jit};
use resin_macros::Tree;

#[derive(Tree)]
struct Params<T> {
    x: T,
    w: T,
    bias: T,
}

/// Linear layer + relu + gather/scatter round-trip + reductions: touches
/// every kernel family the backend lowers.
fn graph(p: &Params<Tensor>) -> Vec<Tensor> {
    let y = p.x.matmul(&p.w);
    let y = y.clone() + p.bias.broadcast_to(y.shape(), &[1]);
    let y = y.relu();

    let indices = Tensor::constant_u32(&[4], &[1, 0, 1, 3]);
    let gathered = y.gather_rows(&indices);
    let scattered = gathered.scatter_rows(&indices, y.shape()[0], ScatterOp::Add);

    let loss = (scattered.clone() * scattered.clone())
        .sum_axes(&[0, 1])
        .squeeze_all();
    let row_max = y.clone() - y.clone(); // exercise fused elementwise chains
    vec![loss, scattered, row_max]
}

fn inputs_f32() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let x: Vec<f32> = (0..4 * 3).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let w: Vec<f32> = (0..3 * 2).map(|i| ((i * 7 % 5) as f32) * 0.5 - 1.0).collect();
    let bias = vec![0.5, -0.25];
    (x, w, bias)
}

#[test]
fn vulkan_matches_cpu_on_composite_graph() {
    if !shared_context_available() {
        eprintln!("skip vulkan_matches_cpu_on_composite_graph: no Vulkan device");
        return;
    }
    let (x, w, bias) = inputs_f32();

    let cpu_out = CpuJit
        .jit(graph)
        .call(&Params {
            x: CpuTensor::from_f32(&[4, 3], &x),
            w: CpuTensor::from_f32(&[3, 2], &w),
            bias: CpuTensor::from_f32(&[2], &bias),
        })
        .expect("cpu run");

    let vk_out = VulkanJit
        .jit(graph)
        .call(&Params {
            x: VulkanTensor::from_f32(&[4, 3], &x),
            w: VulkanTensor::from_f32(&[3, 2], &w),
            bias: VulkanTensor::from_f32(&[2], &bias),
        })
        .expect("vulkan run");

    assert_eq!(cpu_out.len(), vk_out.len());
    for (i, (c, v)) in cpu_out.iter().zip(vk_out.iter()).enumerate() {
        let c = c.to_f32();
        let v = v.to_f32();
        assert_eq!(c.len(), v.len(), "output {i} length");
        for (j, (a, b)) in c.iter().zip(v.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "output {i}[{j}]: cpu {a} vs vulkan {b}"
            );
        }
    }
}
