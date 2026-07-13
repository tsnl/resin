//! Elementwise fusion: kernel counts and numerical equivalence.

use resin::Tree;
use resin::dsl::{Tensor, grad_wrt};
use resin::ir::optimize::fuse_elementwise;
use resin::ir::{Kernel, Program};
use resin::ir::lower;
use resin::jit::{HostArray, CpuJit, Jit};

/// Run `program` on the CPU backend.
fn run(program: &Program, params: &[&HostArray]) -> Vec<Vec<f32>> {
    let artifact = CpuJit.lower(program).expect("valid program");
    let mut outputs: Vec<HostArray> = program
        .sinks
        .iter()
        .map(|&sink| HostArray::zeros(&program.view(sink).accessor.shape))
        .collect();
    let mut output_refs: Vec<&mut HostArray> = outputs.iter_mut().collect();
    CpuJit.invoke(&artifact, params, &mut output_refs).expect("run");
    outputs.into_iter().map(|array| array.data().to_vec()).collect()
}

/// Fused and unfused programs must agree on all sinks.
fn assert_equivalent(program: Program, params: &[&HostArray]) -> (usize, usize) {
    let fused = fuse_elementwise(program.clone());
    fused.validate().expect("fused program stays valid");
    let expected = run(&program, params);
    let actual = run(&fused, params);
    assert_eq!(expected.len(), actual.len());
    for (expected, actual) in expected.iter().zip(&actual) {
        for (e, a) in expected.iter().zip(actual) {
            assert!((e - a).abs() < 1e-5, "fused output diverged: {e} vs {a}");
        }
    }
    (program.queue.len(), fused.queue.len())
}

#[test]
fn unary_chain_fuses_to_one_kernel() {
    let x = Tensor::parameter(&[8]);
    let out = x.exp().log().sqrt().relu();
    let program = lower(&x, &out).unwrap();
    let input = HostArray::from_f32(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let (before, after) = assert_equivalent(program.clone(), &[&input]);
    assert_eq!(before, 4);
    assert_eq!(after, 1);

    // The intermediates' buffers are gone: params + one output remain.
    let fused = fuse_elementwise(program);
    assert_eq!(fused.buffers.len(), 2);
}

#[test]
fn binary_tree_fuses_to_one_kernel() {
    let (a, b, c, d) = (
        Tensor::parameter(&[4]),
        Tensor::parameter(&[4]),
        Tensor::parameter(&[4]),
        Tensor::parameter(&[4]),
    );
    let out = ((a.clone() + b.clone()) * (c.clone() - d.clone())).relu();
    let program = lower(&vec![a, b, c, d], &out).unwrap();
    let params = [
        HostArray::from_f32(&[4], &[1.0, -2.0, 3.0, -4.0]),
        HostArray::from_f32(&[4], &[0.5, 0.5, 0.5, 0.5]),
        HostArray::from_f32(&[4], &[2.0, 2.0, 2.0, 2.0]),
        HostArray::from_f32(&[4], &[1.0, 3.0, 1.0, 3.0]),
    ];
    let param_refs: Vec<&HostArray> = params.iter().collect();

    let (before, after) = assert_equivalent(program, &param_refs);
    assert_eq!(before, 4);
    assert_eq!(after, 1);
}

#[test]
fn shared_operand_inlines_into_both_slots() {
    // y = exp(x) * x, then t = x*x feeds two separate consumers that later
    // merge — shared subexpressions fuse by recomputation.
    let x = Tensor::parameter(&[4]);
    let t = x.clone() * x.clone();
    let out = t.exp() + t.clone();
    let program = lower(&x, &out).unwrap();
    let input = HostArray::from_f32(&[4], &[0.1, 0.2, 0.3, 0.4]);

    let (before, after) = assert_equivalent(program, &[&input]);
    assert_eq!(before, 3);
    assert_eq!(after, 1);
}

#[test]
fn sink_visible_intermediate_is_not_fused_away() {
    // Both t and relu(t) are outputs: t's kernel must survive (relu may
    // recompute t, but t must still be materialized for the sink).
    let x = Tensor::parameter(&[4]);
    let t = x.clone() + x.clone();
    let y = t.relu();
    let program = lower(&x, &vec![t, y]).unwrap();
    let input = HostArray::from_f32(&[4], &[-1.0, 2.0, -3.0, 4.0]);

    let (before, after) = assert_equivalent(program, &[&input]);
    assert_eq!(before, 2);
    assert_eq!(after, 2, "sink-visible intermediate must stay materialized");
}

#[test]
fn broadcast_consumed_intermediate_fuses() {
    // t is consumed through a pitch-0 broadcast view; the producer's args are
    // composed through the same broadcast and the intermediate disappears.
    let x = Tensor::parameter(&[3]);
    let c = Tensor::parameter(&[2, 3]);
    let t = x.clone() + x.clone();
    let out = c.clone() * t.broadcast_to(&[2, 3], &[1]);
    let program = lower(&vec![x, c], &out).unwrap();
    let params = [
        HostArray::from_f32(&[3], &[1.0, 2.0, 3.0]),
        HostArray::from_f32(&[2, 3], &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    ];
    let param_refs: Vec<&HostArray> = params.iter().collect();

    let (before, after) = assert_equivalent(program, &param_refs);
    assert_eq!(before, 2);
    assert_eq!(after, 1);
}

#[test]
fn sgd_update_with_scalar_lr_fuses_to_one_kernel() {
    // w - (g * g) * lr: the scalar lr is an implicit broadcast throughout.
    let w = Tensor::parameter(&[4, 3]);
    let g = Tensor::parameter(&[4, 3]);
    let out = w.clone() - (g.clone() * g.clone()) * Tensor::scalar(0.1);
    let program = lower(&vec![w, g], &out).unwrap();
    let params = [
        HostArray::from_f32(&[4, 3], &[1.0; 12]),
        HostArray::from_f32(&[4, 3], &[2.0; 12]),
    ];
    let param_refs: Vec<&HostArray> = params.iter().collect();

    let (before, after) = assert_equivalent(program, &param_refs);
    assert_eq!(before, 3);
    assert_eq!(after, 1);
}

#[test]
fn multi_consumer_intermediate_fuses_by_recomputation() {
    // t feeds two separate elementwise consumers, which then merge.
    let x = Tensor::parameter(&[4]);
    let t = x.clone() + x.clone();
    let out = t.exp() * t.relu();
    let program = lower(&x, &out).unwrap();
    let input = HostArray::from_f32(&[4], &[-1.0, 0.5, 2.0, -0.25]);

    let (before, after) = assert_equivalent(program, &[&input]);
    assert_eq!(before, 4);
    assert_eq!(after, 1);
}

#[test]
fn matmul_is_a_fusion_boundary() {
    let x = Tensor::parameter(&[2, 3]);
    let w = Tensor::parameter(&[3, 2]);
    let out = (x.matmul(&w) + Tensor::full(&[2, 2], 1.0)).relu();
    let program = lower(&vec![x, w], &out).unwrap();
    let params = [
        HostArray::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        HostArray::from_f32(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
    ];
    let param_refs: Vec<&HostArray> = params.iter().collect();

    let (before, after) = assert_equivalent(program, &param_refs);
    assert_eq!(before, 3, "matmul, add, relu");
    assert_eq!(after, 2, "add and relu fuse; matmul stays");

    let fused = fuse_elementwise(lower_again(&param_refs));
    assert!(fused.queue.iter().any(|d| matches!(d.kernel, Kernel::Matmul)));

    fn lower_again(_params: &[&HostArray]) -> Program {
        let x = Tensor::parameter(&[2, 3]);
        let w = Tensor::parameter(&[3, 2]);
        let out = (x.clone().matmul(&w) + Tensor::full(&[2, 2], 1.0)).relu();
        lower(&vec![x, w], &out).unwrap()
    }
}

#[test]
fn mlp_train_step_fuses_substantially_and_matches() {
    // A miniature version of the MNIST train step: forward + MSE loss +
    // gradients + SGD update, all in one graph.
    #[derive(Tree, Clone)]
    struct Model<T> {
        w0: T,
        b0: T,
        w1: T,
        b1: T,
    }

    let xs = Tensor::parameter(&[4, 6]);
    let ys = Tensor::parameter(&[4, 3]);
    let model = Model {
        w0: Tensor::parameter(&[6, 5]),
        b0: Tensor::parameter(&[5]),
        w1: Tensor::parameter(&[5, 3]),
        b1: Tensor::parameter(&[3]),
    };

    let h = {
        let z = xs.clone().matmul(&model.w0);
        (z.clone() + model.b0.broadcast_to(z.shape(), &[1])).relu()
    };
    let pred = {
        let z = h.matmul(&model.w1);
        z.clone() + model.b1.broadcast_to(z.shape(), &[1])
    };
    let diff = pred - ys.clone();
    let loss = (diff.clone() * diff).mean_all();
    let grads = grad_wrt(&loss, &model).unwrap();
    let lr = Tensor::scalar(0.1);
    let new_model = Model {
        w0: model.w0.clone() - grads.w0 * lr.clone(),
        b0: model.b0.clone() - grads.b0 * lr.clone(),
        w1: model.w1.clone() - grads.w1 * lr.clone(),
        b1: model.b1.clone() - grads.b1 * lr,
    };

    #[derive(Tree, Clone)]
    struct StepIn<T> {
        xs: T,
        ys: T,
        model: Model<T>,
    }
    #[derive(Tree, Clone)]
    struct StepOut<T> {
        loss: T,
        new_model: Model<T>,
    }
    let inputs = StepIn { xs, ys, model };
    let outputs = StepOut { loss, new_model };
    let program = lower(&inputs, &outputs).unwrap();

    let mut seed = 7u64;
    let mut random = |shape: &[usize]| {
        let n: usize = shape.iter().product();
        let values: Vec<f32> = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect();
        HostArray::from_f32(shape, &values)
    };
    let params = [
        random(&[4, 6]),
        random(&[4, 3]),
        random(&[6, 5]),
        random(&[5]),
        random(&[5, 3]),
        random(&[3]),
    ];
    let param_refs: Vec<&HostArray> = params.iter().collect();

    let (before, after) = assert_equivalent(program, &param_refs);
    // 31 dispatches collapse to 17: 5 matmuls + 3 reductions (boundaries) and
    // 9 elementwise clusters (from 23). Update if the pass improves.
    assert_eq!(before, 31);
    assert!(after <= 17, "fusion regressed: {before} -> {after}");
}
