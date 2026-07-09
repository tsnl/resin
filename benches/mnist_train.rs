//! Criterion benchmarks for the MNIST MLP train step.
//!
//! Matrix: **backend** (cpu / wgpu) × **IR opt** (unoptimized / optimized).
//! Compile once outside the timed loop; each iteration is one full train-step
//! invoke (forward + MSE + grads + SGD).
//!
//! ```text
//! cargo bench --bench mnist_train
//! cargo bench --bench mnist_train -- --save-baseline main
//! cargo bench --bench mnist_train -- --baseline main
//! ```
//!
//! wgpu cases are skipped if no adapter is available (`resin::jit::wgpu::gpu_available`).

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use resin::Tree;
use resin::dataset::{IMG_WH, NUM_CLS};
use resin::dsl::{Tensor, grad_wrt};
use resin::ir::optimize::{OptPasses, optimize_with};
use resin::ir::Program;
use resin::jit::lower::lower;
use resin::jit::{Array, CpuJit, Jit};

#[cfg(feature = "wgpu")]
use resin::jit::wgpu::{WgpuJit, WgpuProgram, gpu_available};

const BATCH: usize = 64;
const HIDDEN: usize = 128;
const LR: f32 = 1e-3;

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

#[derive(Tree, Clone)]
struct TrainStepOut<T> {
    loss: T,
    new_model: MlpParams<T>,
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

fn train_step_graph() -> (TrainStepIn<Tensor>, TrainStepOut<Tensor>) {
    let xs = Tensor::parameter(&[BATCH, IMG_WH]);
    let ys = Tensor::parameter(&[BATCH, NUM_CLS]);
    let layer = |i: usize, o: usize| LinearParams {
        weight: Tensor::parameter(&[i, o]),
        bias: Tensor::parameter(&[o]),
    };
    let model = MlpParams {
        layer0: layer(IMG_WH, HIDDEN),
        layer1: layer(HIDDEN, HIDDEN),
        layer2: layer(HIDDEN, NUM_CLS),
    };
    let pred = forward(&model, &xs);
    let diff = pred - ys.clone();
    let loss = (diff.clone() * diff).mean_all();
    let grads = grad_wrt(&loss, &model).expect("grad");
    let lr = Tensor::scalar(LR);
    let new_model = MlpParams {
        layer0: LinearParams {
            weight: model.layer0.weight.clone() - grads.layer0.weight * lr.clone(),
            bias: model.layer0.bias.clone() - grads.layer0.bias * lr.clone(),
        },
        layer1: LinearParams {
            weight: model.layer1.weight.clone() - grads.layer1.weight * lr.clone(),
            bias: model.layer1.bias.clone() - grads.layer1.bias * lr.clone(),
        },
        layer2: LinearParams {
            weight: model.layer2.weight.clone() - grads.layer2.weight * lr.clone(),
            bias: model.layer2.bias.clone() - grads.layer2.bias * lr.clone(),
        },
    };
    (TrainStepIn { xs, ys, model }, TrainStepOut { loss, new_model })
}

fn ir_program(passes: OptPasses) -> Program {
    let (input, output) = train_step_graph();
    let raw = lower(&input, &output).expect("lower");
    // Opt is optional; backend layout (dead-elim + arena pack) always runs.
    resin::ir::layout::prepare_for_backend(optimize_with(raw, passes))
}

fn filled(shape: &[usize], value: f32) -> Array {
    let n: usize = shape.iter().product();
    Array::from_f32(shape, &vec![value; n])
}

/// Parameter leaves matching `TrainStepIn` declaration order.
fn param_arrays() -> TrainStepIn<Array> {
    let layer = |i: usize, o: usize| LinearParams {
        weight: filled(&[i, o], 0.01),
        bias: filled(&[o], 0.0),
    };
    TrainStepIn {
        xs: filled(&[BATCH, IMG_WH], 0.1),
        ys: filled(&[BATCH, NUM_CLS], 0.0),
        model: MlpParams {
            layer0: layer(IMG_WH, HIDDEN),
            layer1: layer(HIDDEN, HIDDEN),
            layer2: layer(HIDDEN, NUM_CLS),
        },
    }
}

fn output_arrays() -> TrainStepOut<Array> {
    let layer = |i: usize, o: usize| LinearParams {
        weight: filled(&[i, o], 0.0),
        bias: filled(&[o], 0.0),
    };
    TrainStepOut {
        loss: filled(&[], 0.0),
        new_model: MlpParams {
            layer0: layer(IMG_WH, HIDDEN),
            layer1: layer(HIDDEN, HIDDEN),
            layer2: layer(HIDDEN, NUM_CLS),
        },
    }
}

fn mnist_train_step(c: &mut Criterion) {
    let opt_cases = [
        ("unoptimized", OptPasses::None),
        ("optimized", OptPasses::All),
    ];

    let mut group = c.benchmark_group("mnist_train_step");
    group.throughput(Throughput::Elements(BATCH as u64));

    let params = param_arrays();
    let mut outputs = output_arrays();

    // --- CPU interpreter -------------------------------------------------
    // Hundreds of ms per step; keep samples modest.
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(2));
    group.measurement_time(Duration::from_secs(45));

    for (opt_name, passes) in opt_cases {
        let program = ir_program(passes);
        let n = program.queue.len();
        let artifact = CpuJit.lower(&program).expect("cpu lower");
        eprintln!("cpu/{opt_name}: {n} dispatches");

        let param_refs: Vec<&Array> = params.leaves();
        let mut out_refs = outputs.leaves_mut();
        CpuJit
            .invoke(&artifact, &param_refs, &mut out_refs)
            .expect("cpu smoke");
        drop(out_refs);
        black_box(outputs.loss.scalar());

        group.bench_with_input(
            BenchmarkId::new("cpu", opt_name),
            &artifact,
            |b, artifact| {
                b.iter(|| {
                    let param_refs: Vec<&Array> = params.leaves();
                    let mut out_refs = outputs.leaves_mut();
                    CpuJit
                        .invoke(artifact, &param_refs, &mut out_refs)
                        .expect("cpu invoke");
                    drop(out_refs);
                    black_box(outputs.loss.scalar());
                });
            },
        );
    }

    // --- WebGPU ----------------------------------------------------------
    #[cfg(feature = "wgpu")]
    {
        if !gpu_available() {
            eprintln!("wgpu: no adapter — skipping gpu cases");
        } else {
            // GPU steps are much faster; more samples, less wall time per cell.
            group.sample_size(50);
            group.warm_up_time(Duration::from_secs(1));
            group.measurement_time(Duration::from_secs(10));

            let jit = WgpuJit::default();
            for (opt_name, passes) in opt_cases {
                let program = ir_program(passes);
                let n = program.queue.len();
                let artifact: WgpuProgram = jit.lower(&program).expect("wgpu lower");
                eprintln!(
                    "wgpu/{opt_name}: {n} dispatches, {} pipelines",
                    artifact.gpu.pipelines.len()
                );

                let param_refs: Vec<&Array> = params.leaves();
                let mut out_refs = outputs.leaves_mut();
                jit.invoke(&artifact, &param_refs, &mut out_refs)
                    .expect("wgpu smoke");
                drop(out_refs);
                black_box(outputs.loss.scalar());

                group.bench_with_input(
                    BenchmarkId::new("wgpu", opt_name),
                    &artifact,
                    |b, artifact| {
                        b.iter(|| {
                            let param_refs: Vec<&Array> = params.leaves();
                            let mut out_refs = outputs.leaves_mut();
                            jit.invoke(artifact, &param_refs, &mut out_refs)
                                .expect("wgpu invoke");
                            drop(out_refs);
                            black_box(outputs.loss.scalar());
                        });
                    },
                );
            }
        }
    }

    group.finish();

    eprintln!(
        "dispatch counts: none={} fuse={} all={}",
        ir_program(OptPasses::None).queue.len(),
        ir_program(OptPasses::Fuse).queue.len(),
        ir_program(OptPasses::All).queue.len(),
    );
}

criterion_group!(benches, mnist_train_step);
criterion_main!(benches);
