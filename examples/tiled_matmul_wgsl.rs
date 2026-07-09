//! Print the WGSL for a tiled 64x64 matmul, including the cooperative-matrix
//! kernel (`chromium_experimental_subgroup_matrix`).
//!
//! wgpu's WGSL frontend cannot parse the extension yet, so this is the path
//! for trying the shaders on runtimes that can (Chrome / Dawn with unsafe
//! APIs allowed):
//!
//!   cargo run --example tiled_matmul_wgsl            # 8x8 MMAs (common f32 config)
//!   cargo run --example tiled_matmul_wgsl -- 16      # 16x16 MMAs

use resin::dsl::Tensor;
use resin::ir::optimize::tile_matmuls;
use resin::jit::{Jit, WgpuJit, lower};

fn main() {
    let mma: usize = std::env::args().nth(1).map_or(8, |s| s.parse().expect("mma size"));

    let a = Tensor::parameter(&[64, 64]);
    let b = Tensor::parameter(&[64, 64]);
    let out = a.clone().matmul(&b);
    let program = tile_matmuls(lower(&vec![a, b], &out).unwrap());

    let artifact = WgpuJit::with_subgroup_matrix(mma).lower(&program).unwrap();
    for (i, pipeline) in artifact.pipelines.iter().enumerate() {
        println!("// ---- pipeline {i}: workgroups {:?} ----", pipeline.workgroups);
        println!("{}\n", pipeline.wgsl);
    }
}
