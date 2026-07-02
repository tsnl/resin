//! Python `test_interp_compose` parity: composed graphs on GPU.

#[path = "interp_helpers.rs"]
mod interp_helpers;

use interp_helpers::{approx_eq, f4_const, f4_param, run_graph};
use resin_core::F4;
use resin_dsl::{const_bytes, View};
use resin_grad::grad_view;
use resin_nn::{cross_entropy, mean, relu, Linear};

fn squeeze_all(mut v: View) -> View {
    while !v.shape().is_empty() {
        v = v.squeeze(&[0]).unwrap();
    }
    v
}

#[test]
fn linear_forward() {
    let x = f4_param(&[2, 3], "x");
    let layer = Linear::new(3, 2, true);
    let out = layer.forward(&x).unwrap();
    let got = run_graph(
        &out,
        &[
            (&x, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (&layer.weight, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            (layer.bias.as_ref().unwrap(), &[10.0, 20.0]),
        ],
    )
    .expect("gpu");
    assert!(approx_eq(&got, &[11.0, 20.0, 10.0, 21.0], 1e-4), "{got:?}");
}

#[test]
fn small_mlp_relu_linear_stack() {
    // Distinct param names (Linear::new always uses "weight"/"bias").
    let x = f4_param(&[2, 2], "x");
    let w1 = f4_param(&[2, 2], "w1");
    let w2 = f4_param(&[1, 2], "w2");
    let b2 = f4_param(&[1], "b2");
    let h = relu(&x.matmul(&w1.transpose().unwrap()).unwrap()).unwrap();
    let out = &h.matmul(&w2.transpose().unwrap()).unwrap() + &b2;
    let got = run_graph(
        &out,
        &[
            (&x, &[1.0, -1.0, 2.0, 3.0]),
            (&w1, &[1.0, 0.0, 0.0, 1.0]),
            (&w2, &[1.0, -1.0]),
            (&b2, &[0.5]),
        ],
    )
    .expect("gpu");
    assert!(approx_eq(&got, &[1.5, -0.5], 1e-4), "{got:?}");
}

#[test]
fn cross_entropy_single_sample() {
    let probs = f4_param(&[1, 3], "p");
    let label = f4_param(&[1, 3], "y");
    let loss = squeeze_all(cross_entropy(&probs, &label).unwrap());
    let got = run_graph(
        &loss,
        &[(&probs, &[0.7, 0.2, 0.1]), (&label, &[1.0, 0.0, 0.0])],
    )
    .expect("gpu");
    let expected = -0.7f32.ln();
    assert!(
        (got[0] - expected).abs() < 1e-4,
        "got={} expect={expected}",
        got[0]
    );
}

#[test]
fn mean_cross_entropy_batch() {
    let probs = f4_param(&[2, 2], "p");
    let label = f4_param(&[2, 2], "y");
    let loss = mean(&cross_entropy(&probs, &label).unwrap()).unwrap();
    let got = run_graph(
        &loss,
        &[
            (&probs, &[0.6, 0.4, 0.2, 0.8]),
            (&label, &[1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .expect("gpu");
    let expected = (-0.6f32.ln() + -0.8f32.ln()) / 2.0;
    assert!(
        (got[0] - expected).abs() < 1e-4,
        "got={} expect={expected}",
        got[0]
    );
}

#[test]
fn sum_param_grad_via_update() {
    let p = f4_param(&[4], "p");
    let loss = squeeze_all(p.sum(None).unwrap());
    let g = grad_view(&loss, &p).unwrap();
    let one = const_bytes([], F4, 1f32.to_le_bytes());
    let updated = &p + &(&one * &g);
    let got = run_graph(&updated, &[(&p, &[1.0, 2.0, 3.0, 4.0])]).expect("gpu");
    assert!(approx_eq(&got, &[2.0, 3.0, 4.0, 5.0], 1e-4), "{got:?}");
}

#[test]
fn matmul_grad_via_update() {
    let x = f4_param(&[2, 2], "x");
    let w = f4_param(&[2, 2], "w");
    let loss = squeeze_all(x.matmul(&w).unwrap().sum(None).unwrap());
    let g = grad_view(&loss, &w).unwrap();
    let one = const_bytes([], F4, 1f32.to_le_bytes());
    let updated = &w + &(&one * &g);
    let got = run_graph(
        &updated,
        &[(&x, &[1.0, 2.0, 3.0, 4.0]), (&w, &[0.0, 0.0, 0.0, 0.0])],
    )
    .expect("gpu");
    assert!(approx_eq(&got, &[4.0, 4.0, 6.0, 6.0], 1e-4), "{got:?}");
}

#[test]
fn param_update_adds_scaled_grad() {
    let p = f4_param(&[3], "p");
    let c = f4_const(&[1.0, 2.0, 3.0]);
    let loss = squeeze_all((&p * &c).sum(None).unwrap());
    let g = grad_view(&loss, &p).unwrap();
    let lr = const_bytes([], F4, 0.1f32.to_le_bytes());
    let updated = &p + &(&lr * &g);
    let got = run_graph(&updated, &[(&p, &[1.0, 1.0, 1.0])]).expect("gpu");
    assert!(approx_eq(&got, &[1.1, 1.2, 1.3], 1e-4), "{got:?}");
}

#[test]
fn param_update_subtracts_scaled_grad() {
    let p = f4_param(&[3], "p");
    let c = f4_const(&[1.0, 2.0, 3.0]);
    let loss = squeeze_all((&p * &c).sum(None).unwrap());
    let g = grad_view(&loss, &p).unwrap();
    let lr = const_bytes([], F4, 0.1f32.to_le_bytes());
    let updated = &p - &(&lr * &g);
    let got = run_graph(&updated, &[(&p, &[1.0, 1.0, 1.0])]).expect("gpu");
    assert!(approx_eq(&got, &[0.9, 0.8, 0.7], 1e-4), "{got:?}");
}
