//! Stage 1: single-node (or tiny) graphs produce expected GPU output.

#[path = "interp_helpers.rs"]
mod interp_helpers;

use interp_helpers::{approx_eq, f4_const, f4_param, run_graph};
use resin_core::{AxisIndex, ElementOperator, F4};
use resin_dsl::{const_bytes, View};

#[test]
fn const_1d() {
    let out = f4_const(&[1.0, 2.0, 3.0]);
    let got = run_graph(&out, &[]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 2.0, 3.0], 1e-5), "{got:?}");
}

#[test]
fn const_2d() {
    // Python TestConst.test_2d — row-major flatten of [[1,2],[3,4]].
    let out = const_bytes([2, 2], F4, {
        let b: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        b.into_boxed_slice()
    });
    let got = run_graph(&out, &[]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 2.0, 3.0, 4.0], 1e-5), "{got:?}");
}

#[test]
fn elementwise_add() {
    let a = f4_param(&[3], "a");
    let b = f4_param(&[3], "b");
    let out = &a + &b;
    let got = run_graph(&out, &[(&a, &[1.0, 2.0, 3.0]), (&b, &[4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&got, &[5.0, 7.0, 9.0], 1e-5), "{got:?}");
}

#[test]
fn elementwise_mul() {
    let a = f4_param(&[2], "a");
    let b = f4_param(&[2], "b");
    let out = &a * &b;
    let got = run_graph(&out, &[(&a, &[2.0, 3.0]), (&b, &[4.0, 5.0])]).expect("gpu");
    assert!(approx_eq(&got, &[8.0, 15.0], 1e-5), "{got:?}");
}

#[test]
fn elementwise_sub_div() {
    let a = f4_param(&[2], "a");
    let b = f4_param(&[2], "b");
    let sub = &a - &b;
    let got = run_graph(&sub, &[(&a, &[5.0, 3.0]), (&b, &[1.0, 4.0])]).expect("gpu");
    assert!(approx_eq(&got, &[4.0, -1.0], 1e-5), "{got:?}");

    let div = &a / &b;
    let got = run_graph(&div, &[(&a, &[8.0, 9.0]), (&b, &[2.0, 3.0])]).expect("gpu");
    assert!(approx_eq(&got, &[4.0, 3.0], 1e-5), "{got:?}");
}

#[test]
fn elementwise_unary_neg_exp() {
    let x = f4_param(&[2], "x");
    let neg = -&x;
    let got = run_graph(&neg, &[(&x, &[1.5, -2.0])]).expect("gpu");
    assert!(approx_eq(&got, &[-1.5, 2.0], 1e-5), "{got:?}");

    let e = x.exp();
    let got = run_graph(&e, &[(&x, &[0.0, 1.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 1.0f32.exp()], 1e-4), "{got:?}");
}

#[test]
fn elementwise_unary_log_sqrt() {
    let x = f4_param(&[2], "x");
    let got = run_graph(&x.log(), &[(&x, &[1.0, std::f32::consts::E])]).expect("gpu");
    assert!(approx_eq(&got, &[0.0, 1.0], 1e-4), "{got:?}");

    let got = run_graph(&x.sqrt(), &[(&x, &[4.0, 9.0])]).expect("gpu");
    assert!(approx_eq(&got, &[2.0, 3.0], 1e-5), "{got:?}");
}

#[test]
fn elementwise_relu() {
    use resin_nn::relu;
    let x = f4_param(&[4], "x");
    let out = relu(&x).unwrap();
    let got = run_graph(&out, &[(&x, &[-1.0, 0.0, 2.0, -3.0])]).expect("gpu");
    assert!(approx_eq(&got, &[0.0, 0.0, 2.0, 0.0], 1e-5), "{got:?}");
}

#[test]
fn reduce_sum() {
    let x = f4_param(&[4], "x");
    let out = x
        .reduce(&[0], ElementOperator::Add)
        .unwrap()
        .squeeze(&[0])
        .unwrap();
    let got = run_graph(&out, &[(&x, &[1.0, 2.0, 3.0, 4.0])]).expect("gpu");
    assert!(approx_eq(&got, &[10.0], 1e-5), "{got:?}");
}

#[test]
fn reduce_max() {
    let x = f4_param(&[4], "x");
    let out = x
        .reduce(&[0], ElementOperator::Max)
        .unwrap()
        .squeeze(&[0])
        .unwrap();
    let got = run_graph(&out, &[(&x, &[1.0, 5.0, 3.0, 2.0])]).expect("gpu");
    assert!(approx_eq(&got, &[5.0], 1e-5), "{got:?}");
}

#[test]
fn matmul_2x3_times_3x2() {
    let a = f4_param(&[2, 3], "a");
    let b = f4_param(&[3, 2], "b");
    let out = a.matmul(&b).unwrap();
    // Python TestMatmul.test_2d
    let got = run_graph(
        &out,
        &[
            (&a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (&b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ],
    )
    .expect("gpu");
    assert!(
        approx_eq(&got, &[58.0, 64.0, 139.0, 154.0], 1e-3),
        "{got:?}"
    );
}

#[test]
fn reduce_sum_axis_1() {
    let x = f4_param(&[2, 3], "x");
    let out = x.reduce(&[1], ElementOperator::Add).unwrap();
    let got = run_graph(&out, &[(&x, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&got, &[6.0, 15.0], 1e-5), "{got:?}");
}

#[test]
fn reduce_max_axis_0() {
    let x = f4_param(&[2, 2], "x");
    let out = x.reduce(&[0], ElementOperator::Max).unwrap();
    let got = run_graph(&out, &[(&x, &[1.0, 4.0, 3.0, 2.0])]).expect("gpu");
    assert!(approx_eq(&got, &[3.0, 4.0], 1e-5), "{got:?}");
}

#[test]
fn identity_copy() {
    let x = f4_param(&[3], "x");
    let got = run_graph(&x.copy(None), &[(&x, &[1.0, 2.0, 3.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 2.0, 3.0], 1e-5), "{got:?}");
}

#[test]
fn index_row_and_step_slice_via_copy() {
    // Narrow is a view; densify with copy() so GPU writes a contiguous buffer.
    let x = f4_param(&[2, 3], "x");
    let row0 = x.index((0, ..)).unwrap().copy(None);
    let got = run_graph(&row0, &[(&x, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 2.0, 3.0], 1e-5), "{got:?}");

    let sub = x.index((1, 1..3)).unwrap().copy(None);
    let got = run_graph(&sub, &[(&x, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&got, &[5.0, 6.0], 1e-5), "{got:?}");

    let x1 = f4_param(&[6], "x1");
    let stepped = x1
        .index(AxisIndex::slice(None, None, 2))
        .unwrap()
        .copy(None);
    let got = run_graph(&stepped, &[(&x1, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 3.0, 5.0], 1e-5), "{got:?}");
}

#[test]
fn broadcast_scalar_adjoint_accumulates() {
    // Python TestScatterAccumulate: g over broadcast(x) → sum into scalar x.
    use resin_grad::accessor_adjoint;
    let x = f4_param(&[], "x");
    let y = x.broadcast(&[5]);
    let g = f4_param(&[5], "g");
    let out = accessor_adjoint(&y, &g).unwrap();
    let mut scalar = out;
    while !scalar.shape().is_empty() {
        scalar = scalar.squeeze(&[0]).unwrap();
    }
    let got = run_graph(&scalar, &[(&g, &[1.0, 2.0, 3.0, 4.0, 5.0])]).expect("gpu");
    assert!(approx_eq(&got, &[15.0], 1e-5), "{got:?}");
}

#[test]
fn softmax_rows_sum_to_one() {
    use resin_nn::softmax;
    let x = f4_param(&[2, 3], "x");
    let probs = softmax(&x, &[1]).unwrap();
    let row_sums = probs.reduce(&[1], ElementOperator::Add).unwrap();
    let got = run_graph(&row_sums, &[(&x, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.0, 1.0], 1e-4), "{got:?}");
}

#[test]
fn matmul_with_transpose_view() {
    // y = x @ W.T with W [2, 3], x [1, 3] → y [1, 2]
    let x = f4_param(&[1, 3], "x");
    let w = f4_param(&[2, 3], "w");
    let out = x.matmul(&w.transpose().unwrap()).unwrap();
    let got = run_graph(
        &out,
        &[
            (&x, &[1.0, 0.0, 2.0]),
            (&w, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), // rows e0, e1 in R^3
        ],
    )
    .expect("gpu");
    // y[0] = x·w0 = 1, y[1] = x·w1 = 0
    assert!(approx_eq(&got, &[1.0, 0.0], 1e-4), "{got:?}");
}

#[test]
fn broadcast_add_bias() {
    let x = f4_param(&[2, 3], "x");
    let bias = f4_param(&[3], "b");
    let out = &x + &bias;
    let got = run_graph(
        &out,
        &[
            (&x, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (&bias, &[10.0, 20.0, 30.0]),
        ],
    )
    .expect("gpu");
    assert!(
        approx_eq(&got, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0], 1e-5),
        "{got:?}"
    );
}

#[test]
fn gather_copy_via_remap() {
    // Identity gather densify: copy() on a non-identity view.
    let x = f4_param(&[2, 3], "x");
    let v = x.broadcast(&[1]); // leading broadcast — not identity on node
    let out = v.copy(None);
    let got = run_graph(&out, &[(&x, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    // broadcast then densify: shape [1,2,3]
    assert_eq!(got.len(), 6);
    assert!(
        approx_eq(&got, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1e-5),
        "{got:?}"
    );
}

#[test]
fn scatter_add_transpose_adjoint_shape() {
    // Scatter-add used by accessor_adjoint for a transposed view of a param.
    // g (dense [3,2]) scatter into out [2,3] via transpose pitch — one-to-one.
    use resin_core::Accessor;
    use resin_dsl::{RemapInfo, RemapScatterInfo};

    let g = f4_param(&[3, 2], "g");
    // W would be [2,3] pitch [3,1]; W.T shape [3,2] pitch [1,3]
    let out = View::remap(
        &g,
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(Accessor::new(0, [3, 2], [1, 3])),
            operator: Some(ElementOperator::Add),
        }),
        [2, 3],
        F4,
        [],
    );
    // g row-major: [[1,2],[3,4],[5,6]] → W = [[1,3,5],[2,4,6]]
    let got = run_graph(&out, &[(&g, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]).expect("gpu");
    assert!(
        approx_eq(&got, &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 1e-5),
        "{got:?}"
    );
}

#[test]
fn const_bytes_sink() {
    let c = const_bytes([2], F4, {
        let b: Vec<u8> = [1.5f32, -2.5]
            .into_iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        b.into_boxed_slice()
    });
    let got = run_graph(&c, &[]).expect("gpu");
    assert!(approx_eq(&got, &[1.5, -2.5], 1e-5), "{got:?}");
}
