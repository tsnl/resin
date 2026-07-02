//! T3: small multi-node forward chains vs hand / host expectations (no download).

#[path = "interp_helpers.rs"]
mod interp_helpers;

use interp_helpers::{approx_eq, cross_entropy, f4_param, mean, run_graph, softmax, Linear};

/// Linear no bias: `y = x @ W.T`.
#[test]
fn chain_linear_no_bias() {
    let x = f4_param(&[2, 3]);
    let w = f4_param(&[2, 3]);
    let y = x.matmul(&w.transpose().unwrap()).unwrap();
    // x = [[1,0,2],[0,1,0]], W = [[1,0,0],[0,1,0]] → y = [[1,0],[0,1]]
    let got = run_graph(
        &y,
        &[
            (&x, &[1.0, 0.0, 2.0, 0.0, 1.0, 0.0]),
            (&w, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        ],
    )
    .expect("gpu");
    assert!(approx_eq(&got, &[1.0, 0.0, 0.0, 1.0], 1e-5), "{got:?}");
}

/// Linear + bias.
#[test]
fn chain_linear_bias() {
    let x = f4_param(&[2, 2]);
    let layer = Linear::new(2, 2, true);
    let y = layer.forward(&x).unwrap();
    let got = run_graph(
        &y,
        &[
            (&x, &[1.0, 0.0, 0.0, 1.0]),
            (&layer.weight, &[1.0, 0.0, 0.0, 1.0]),
            (layer.bias.as_ref().unwrap(), &[0.5, -0.5]),
        ],
    )
    .expect("gpu");
    // y = [[1,0],[0,1]] + [0.5,-0.5] broadcast → [[1.5,-0.5],[0.5,0.5]]
    assert!(approx_eq(&got, &[1.5, -0.5, 0.5, 0.5], 1e-5), "{got:?}");
}

/// Linear + ReLU.
#[test]
fn chain_linear_relu() {
    let x = f4_param(&[2, 2]);
    let layer = Linear::new(2, 2, true);
    let y = layer.forward(&x).unwrap().relu();
    let got = run_graph(
        &y,
        &[
            (&x, &[1.0, 0.0, 0.0, 1.0]),
            (&layer.weight, &[1.0, 0.0, 0.0, 1.0]),
            (layer.bias.as_ref().unwrap(), &[-0.5, -0.5]),
        ],
    )
    .expect("gpu");
    // pre-relu [[0.5,-0.5],[-0.5,0.5]] → relu [[0.5,0],[0,0.5]]
    assert!(approx_eq(&got, &[0.5, 0.0, 0.0, 0.5], 1e-5), "{got:?}");
}

/// Softmax axis 1: rows sum to 1.
#[test]
fn chain_softmax_rows_sum_to_one() {
    let x = f4_param(&[2, 3]);
    let p = softmax(&x, &[1]).unwrap();
    let got = run_graph(&p, &[(&x, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0])]).expect("gpu");
    assert_eq!(got.len(), 6);
    let s0: f32 = got[0..3].iter().sum();
    let s1: f32 = got[3..6].iter().sum();
    assert!((s0 - 1.0).abs() < 1e-4, "row0 sum={s0} probs={got:?}");
    assert!((s1 - 1.0).abs() < 1e-4, "row1 sum={s1} probs={got:?}");
    // row1 all zeros logits → uniform 1/3
    for (i, &p) in got.iter().enumerate().skip(3).take(3) {
        assert!((p - 1.0 / 3.0).abs() < 1e-4, "got[{i}]={p}");
    }
}

/// Softmax + CE + mean on fixed tensors (host formula).
#[test]
fn chain_softmax_ce_mean_host_formula() {
    let x = f4_param(&[2, 2]);
    let y = f4_param(&[2, 2]);
    let layer = Linear::new(2, 2, true);
    let logits = layer.forward(&x).unwrap();
    let probs = softmax(&logits, &[1]).unwrap();
    let loss = mean(&cross_entropy(&probs, &y).unwrap()).unwrap();

    // Identity W, zero bias, x = I → logits = I
    // softmax([1,0]) = [e/(e+1), 1/(e+1)]
    // y one-hot class 0 and 1: CE = -log(p_true)
    let e = std::f32::consts::E;
    let p0 = e / (e + 1.0);
    let p1 = 1.0 / (e + 1.0);
    // row0: y=[1,0], ce = -log(p0); row1: y=[0,1], logits=[0,1] → softmax [1/(e+1), e/(e+1)], ce=-log(p0)
    // logits row1 = [0,1] → same as softmax([0,1]) = [1/(e+1), e/(e+1)], true class 1 → -log(e/(e+1)) = -log(p0)
    let expect = -p0.ln(); // both rows same CE value = -ln(e/(e+1))

    let got = run_graph(
        &loss,
        &[
            (&x, &[1.0, 0.0, 0.0, 1.0]),
            (&y, &[1.0, 0.0, 0.0, 1.0]),
            (&layer.weight, &[1.0, 0.0, 0.0, 1.0]),
            (layer.bias.as_ref().unwrap(), &[0.0, 0.0]),
        ],
    )
    .expect("gpu");
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - expect).abs() < 1e-4,
        "loss={} expect={} (p0={p0} p1={p1})",
        got[0],
        expect
    );
}
