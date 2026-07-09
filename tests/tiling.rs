//! Tiled matmul: rewrite structure, numerical equivalence, WGSL emission.

use resin::dsl::Tensor;
use resin::ir::optimize::tile_matmuls;
use resin::ir::{Element, Kernel, Program};
use resin::jit::{Array, CpuJit, Jit, lower};

fn random(shape: &[usize], seed: &mut u64) -> Array {
    let n: usize = shape.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|_| {
            *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((*seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    Array::from_f32(shape, &values)
}

/// Run `program` on the CPU backend and densify its sinks.
fn run(program: &Program, params: &[&Array]) -> Vec<Vec<f32>> {
    let artifact = CpuJit.lower(program).expect("valid program");
    let mut outputs: Vec<Array> = program
        .sinks
        .iter()
        .map(|&sink| Array::zeros(&program.view(sink).accessor.shape()))
        .collect();
    let mut output_refs: Vec<&mut Array> = outputs.iter_mut().collect();
    CpuJit.invoke(&artifact, params, &mut output_refs).expect("run");
    outputs.into_iter().map(|array| array.data().to_vec()).collect()
}

/// Tiled and naive programs must agree on all sinks.
fn assert_equivalent(program: Program, params: &[&Array]) -> Program {
    let tiled = tile_matmuls(program.clone());
    tiled.validate().expect("tiled program stays valid");
    let expected = run(&program, params);
    let actual = run(&tiled, params);
    for (expected, actual) in expected.iter().zip(&actual) {
        for (e, a) in expected.iter().zip(actual) {
            assert!((e - a).abs() < 1e-3, "tiled output diverged: {e} vs {a}");
        }
    }
    tiled
}

fn matmul_program(m: usize, k: usize, n: usize) -> Program {
    let a = Tensor::parameter(&[m, k]);
    let b = Tensor::parameter(&[k, n]);
    let out = a.clone().matmul(&b);
    lower(&vec![a, b], &out).unwrap()
}

#[test]
fn tiled_matmul_structure() {
    let tiled = tile_matmuls(matmul_program(32, 48, 16));
    tiled.validate().unwrap();

    // pack A, pack B, tile multiply, reduce — and no Matmul kernel left.
    assert_eq!(tiled.queue.len(), 4);
    assert!(!tiled.queue.iter().any(|d| matches!(d.kernel, Kernel::Matmul)));
    let tile_mul = &tiled.queue[2];
    assert!(matches!(
        tile_mul.kernel,
        Kernel::Elementwise { element: Element::F32Tile16, .. }
    ));
    // Iteration space is the [Mt, Nt, Kt] tile grid.
    assert_eq!(&*tiled.view(tile_mul.output).accessor.shape(), &[2, 1, 3]);
    assert!(matches!(tiled.queue[3].kernel, Kernel::Reduction { .. }));
}

#[test]
fn tiled_matmul_matches_naive() {
    let mut seed = 3;
    for (m, k, n) in [(16, 16, 16), (32, 32, 48), (64, 16, 32)] {
        let params = [random(&[m, k], &mut seed), random(&[k, n], &mut seed)];
        let param_refs: Vec<&Array> = params.iter().collect();
        assert_equivalent(matmul_program(m, k, n), &param_refs);
    }
}

#[test]
fn non_divisible_matmul_stays_naive() {
    let tiled = tile_matmuls(matmul_program(10, 20, 30));
    assert_eq!(tiled.queue.len(), 1);
    assert!(matches!(tiled.queue[0].kernel, Kernel::Matmul));
}

#[test]
fn transposed_operand_stays_naive_but_correct() {
    // b is consumed through a transpose view — not an identity read, so the
    // rewrite skips it; the program must still run correctly end to end.
    let a = Tensor::parameter(&[32, 16]);
    let b = Tensor::parameter(&[32, 16]);
    let out = a.clone().matmul(&b.transpose());
    let program = lower(&vec![a, b], &out).unwrap();

    let mut seed = 11;
    let params = [random(&[32, 16], &mut seed), random(&[32, 16], &mut seed)];
    let param_refs: Vec<&Array> = params.iter().collect();
    let tiled = assert_equivalent(program, &param_refs);
    assert!(tiled.queue.iter().any(|d| matches!(d.kernel, Kernel::Matmul)));
}

#[test]
fn linear_layer_with_bias_matches() {
    // matmul + broadcast bias + relu around the tiled rewrite.
    let x = Tensor::parameter(&[32, 64]);
    let w = Tensor::parameter(&[64, 16]);
    let bias = Tensor::parameter(&[16]);
    let z = x.clone().matmul(&w);
    let out = (z.clone() + bias.broadcast_to(z.shape(), &[1])).relu();
    let program = lower(&vec![x, w, bias], &out).unwrap();

    let mut seed = 23;
    let params = [
        random(&[32, 64], &mut seed),
        random(&[64, 16], &mut seed),
        random(&[16], &mut seed),
    ];
    let param_refs: Vec<&Array> = params.iter().collect();
    assert_equivalent(program, &param_refs);
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use resin::jit::WgpuJit;

    #[test]
    fn subgroup_matrix_wgsl_emission() {
        // The cooperative-matrix path only runs on Dawn/Chrome; here we check
        // the emitted WGSL (wgpu's WGSL frontend cannot parse it yet).
        let program = tile_matmuls(matmul_program(32, 32, 32));
        let artifact = WgpuJit::with_subgroup_matrix(8).lower(&program).unwrap();

        let tiled = artifact
            .pipelines
            .iter()
            .find(|p| p.wgsl.contains("subgroupMatrixMultiplyAccumulate"))
            .expect("a cooperative-matrix shader");
        let wgsl = &tiled.wgsl;
        assert!(wgsl.contains("enable chromium_experimental_subgroup_matrix;"), "{wgsl}");
        assert!(wgsl.contains("subgroup_matrix_left<f32, 8, 8>"), "{wgsl}");
        assert!(wgsl.contains("subgroup_matrix_right<f32, 8, 8>"), "{wgsl}");
        assert!(wgsl.contains("subgroup_matrix_result<f32, 8, 8>"), "{wgsl}");
        assert!(wgsl.contains("subgroupMatrixLoad"), "{wgsl}");
        assert!(wgsl.contains("subgroupMatrixStore"), "{wgsl}");
        // 16x16 tiles decompose into 2x2 blocks of 8x8 MMAs.
        assert!(wgsl.contains("bi < 2u"), "{wgsl}");
        // One workgroup per output tile.
        assert_eq!(tiled.workgroups, [2 * 2 * 2, 1, 1]);
    }

    #[test]
    fn tiled_matmul_runs_on_wgpu_fallback() {
        if !resin::jit::wgpu::gpu_available() {
            eprintln!("skip: no GPU adapter");
            return;
        }
        // The default config uses the portable per-lane fallback for tiled
        // kernels; the full jit path (which runs tile_matmuls via optimize)
        // must match the CPU backend.
        let f_gpu = WgpuJit::default().jit(|p: &Vec<Tensor>| p[0].matmul(&p[1]));
        let f_cpu = CpuJit.jit(|p: &Vec<Tensor>| p[0].matmul(&p[1]));

        let mut seed = 5;
        let params = vec![random(&[32, 64], &mut seed), random(&[64, 48], &mut seed)];
        let gpu = f_gpu.call(&params).unwrap();
        let cpu = f_cpu.call(&params).unwrap();
        for (g, c) in gpu.data().iter().zip(cpu.data()) {
            assert!((g - c).abs() < 1e-3, "wgpu tiled diverged: {g} vs {c}");
        }
    }
}
