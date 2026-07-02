//! T4 / G1–G4: gradient checks for single ops and short chains (GPU execute).
//!
//! See `tests/README.md` tier map. Prefer analytic checks; use FD where noted.

#[path = "interp_helpers.rs"]
mod interp_helpers;

use interp_helpers::{approx_eq, f4_param, finite_diff_param, max_abs_diff, run_loss_grads};
use resin_grad::{grad_view, grad_wrt};
use resin_nn::{cross_entropy, mean, relu, softmax, Linear};

/// G1: `L = mean(y)`, `y = x @ W.T` (W is `[out, in]`).
/// Analytic: `∂L/∂y = 1/(B*O)`, `∂L/∂W[o,i] = sum_b x[b,i] / (B*O)`,
/// `∂L/∂b` N/A; `∂L/∂x[b,i] = sum_o W[o,i] / (B*O)`.
#[test]
fn g1_mean_of_matmul_weight_and_input_grad() {
    // B=2, in=3, out=2. W [2,3], x [2,3] — use y = x @ W.T so W.T is [3,2].
    let x = f4_param(&[2, 3], "x");
    let w = f4_param(&[2, 3], "w");
    let y = x.matmul(&w.transpose().unwrap()).unwrap();
    let loss = mean(&y).unwrap();
    // Single Views use grad_view (View is not a ParamTree leaf tree).
    let gw = grad_view(&loss, &w).unwrap();
    let gx = grad_view(&loss, &x).unwrap();

    // x = [[1,0,2],[0,1,0]], W = [[1,0,0],[0,1,0]]
    let x_data = [1.0f32, 0.0, 2.0, 0.0, 1.0, 0.0];
    let w_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let (loss_v, grads) = run_loss_grads(
        &loss,
        &[("gw", &gw), ("gx", &gx)],
        &[(&x, &x_data), (&w, &w_data)],
    )
    .expect("gpu");

    // y[0] = [1,0], y[1] = [0,1], mean over B*O=4 → 2/4 = 0.5
    assert!((loss_v - 0.5).abs() < 1e-4, "loss={loss_v}");

    let gw_v = &grads["gw"];
    let gx_v = &grads["gx"];
    // dL/dW[o,i] = sum_b x[b,i] / 4
    // W row0: x col sums [1,1,2]/4
    // W row1: same [1,1,2]/4
    let expect_w = [
        1.0 / 4.0,
        1.0 / 4.0,
        2.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        2.0 / 4.0,
    ];
    assert!(
        approx_eq(gw_v, &expect_w, 1e-4),
        "gw={gw_v:?} expect={expect_w:?}"
    );

    // dL/dx[b,i] = sum_o W[o,i] / 4 → col sums of W are [1,1,0]/4 for each batch row
    let expect_x = [1.0 / 4.0, 1.0 / 4.0, 0.0, 1.0 / 4.0, 1.0 / 4.0, 0.0];
    assert!(
        approx_eq(gx_v, &expect_x, 1e-4),
        "gx={gx_v:?} expect={expect_x:?}"
    );
}

/// G2: `y = x @ W.T + b`, `L = mean(y)`.
/// `∂L/∂b[o] = B / (B*O) = 1/O`.
#[test]
fn g2_mean_of_linear_bias_and_weight_grad() {
    let x = f4_param(&[2, 4], "x");
    let layer = Linear::new(4, 3, true); // W [3,4]
    let y = layer.forward(&x).unwrap();
    let loss = mean(&y).unwrap();
    let grads = grad_wrt(
        &loss,
        Linear {
            weight: layer.weight.clone(),
            bias: layer.bias.clone(),
        },
    )
    .unwrap();

    let x_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let mut w_data = vec![0.1f32; 12];
    for i in 0..3 {
        for j in 0..4 {
            if i == j {
                w_data[i * 4 + j] = 1.0;
            }
        }
    }
    let b_data = [0.0f32, 0.0, 0.0];
    let (loss_v, gmap) = run_loss_grads(
        &loss,
        &[("gw", &grads.weight), ("gb", grads.bias.as_ref().unwrap())],
        &[
            (&x, &x_data),
            (&layer.weight, &w_data),
            (layer.bias.as_ref().unwrap(), &b_data),
        ],
    )
    .expect("gpu");

    assert!(loss_v.is_finite(), "loss={loss_v}");
    let gb = &gmap["gb"];
    // ∂L/∂b[o] = 1/3 for each o (B=2, O=3, mean over 6 entries)
    assert!(
        approx_eq(gb, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], 1e-4),
        "gb={gb:?}"
    );
    let gw = &gmap["gw"];
    // ∂L/∂W[o,i] = sum_b x[b,i] / 6 → [1,1,0,0]/6 for each of 3 rows
    for o in 0..3 {
        assert!(
            (gw[o * 4] - 1.0 / 6.0).abs() < 1e-4 && (gw[o * 4 + 1] - 1.0 / 6.0).abs() < 1e-4,
            "gw row {o}: {:?}",
            &gw[o * 4..o * 4 + 4]
        );
    }
}

/// G3: `y = relu(x @ W.T + b)`, `L = mean(y)` — FD on one weight and one bias.
#[test]
fn g3_relu_linear_grad_matches_fd() {
    let x = f4_param(&[2, 3], "x");
    let layer = Linear::new(3, 2, true);
    let y = relu(&layer.forward(&x).unwrap()).unwrap();
    let loss = mean(&y).unwrap();
    let grads = grad_wrt(
        &loss,
        Linear {
            weight: layer.weight.clone(),
            bias: layer.bias.clone(),
        },
    )
    .unwrap();

    let x_data = [1.0f32, -1.0, 0.5, 0.2, 0.3, -0.4];
    let w_data = [0.5f32, -0.2, 0.1, -0.3, 0.4, 0.2];
    let b_data = [0.1f32, -0.1];
    let params_ref = [
        (&x, x_data.as_slice()),
        (&layer.weight, w_data.as_slice()),
        (layer.bias.as_ref().unwrap(), b_data.as_slice()),
    ];
    let (_loss_v, gmap) = run_loss_grads(
        &loss,
        &[("gw", &grads.weight), ("gb", grads.bias.as_ref().unwrap())],
        &params_ref,
    )
    .expect("gpu");

    let params_owned = [
        (&x, x_data.to_vec()),
        (&layer.weight, w_data.to_vec()),
        (layer.bias.as_ref().unwrap(), b_data.to_vec()),
    ];
    let eps = 1e-3f32;
    let fd_w0 = finite_diff_param(&loss, &params_owned, "weight", 0, eps).expect("fd w");
    let fd_b0 = finite_diff_param(&loss, &params_owned, "bias", 0, eps).expect("fd b");
    let gw0 = gmap["gw"][0];
    let gb0 = gmap["gb"][0];
    assert!(
        (gw0 - fd_w0).abs() < 5e-3,
        "weight[0] autodiff={gw0} FD={fd_w0}"
    );
    assert!(
        (gb0 - fd_b0).abs() < 5e-3,
        "bias[0] autodiff={gb0} FD={fd_b0}"
    );
}

/// G4: `L = mean(CE(softmax(logits), y))`, logits = `x @ W.T + b`.
/// FD on bias (expect match) and one weight entry.
#[test]
fn g4_softmax_ce_linear_grad_bias_and_weight_fd() {
    let x = f4_param(&[2, 4], "x");
    let y = f4_param(&[2, 3], "y");
    let layer = Linear::new(4, 3, true);
    let logits = layer.forward(&x).unwrap();
    let probs = softmax(&logits, &[1]).unwrap();
    let loss = mean(&cross_entropy(&probs, &y).unwrap()).unwrap();
    let grads = grad_wrt(
        &loss,
        Linear {
            weight: layer.weight.clone(),
            bias: layer.bias.clone(),
        },
    )
    .unwrap();

    let x_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let y_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // one-hot
    let w_data = vec![0.1f32; 12];
    let b_data = [0.0f32, 0.0, 0.0];

    let (_loss_v, gmap) = run_loss_grads(
        &loss,
        &[("gw", &grads.weight), ("gb", grads.bias.as_ref().unwrap())],
        &[
            (&x, &x_data),
            (&y, &y_data),
            (&layer.weight, &w_data),
            (layer.bias.as_ref().unwrap(), &b_data),
        ],
    )
    .expect("gpu");

    let params_owned = [
        (&x, x_data.to_vec()),
        (&y, y_data.to_vec()),
        (&layer.weight, w_data.clone()),
        (layer.bias.as_ref().unwrap(), b_data.to_vec()),
    ];
    let eps = 1e-3f32;
    let fd_b: Vec<f32> = (0..3)
        .map(|i| finite_diff_param(&loss, &params_owned, "bias", i, eps).expect("fd b"))
        .collect();
    let gb = &gmap["gb"];
    let bias_err = max_abs_diff(gb, &fd_b);
    assert!(
        bias_err < 5e-3,
        "bias autodiff={gb:?} FD={fd_b:?} max_err={bias_err}"
    );

    let fd_w0 = finite_diff_param(&loss, &params_owned, "weight", 0, eps).expect("fd w");
    let gw0 = gmap["gw"][0];
    let weight_err = (gw0 - fd_w0).abs();
    // Document known risk: full-MLP FD mismatch was large; this isolates last layer.
    // Expect agreement within loose tol; if this fails, weight/adjoint path is wrong.
    assert!(
        weight_err < 5e-3,
        "weight[0] autodiff={gw0} FD={fd_w0} err={weight_err} — G4 weight path regression"
    );
}

/// G4b: same as G4 but **without** stable-softmax max (exp only). Separates Max-adjoint issues.
#[test]
fn g4b_softmax_ce_no_max_sub_weight_fd() {
    let x = f4_param(&[2, 4], "x");
    let y = f4_param(&[2, 3], "y");
    let layer = Linear::new(4, 3, true);
    let logits = layer.forward(&x).unwrap();
    // Unstable softmax: exp / sum (no max sub)
    let exp_x = logits.exp();
    let sum_exp = exp_x
        .reduce(&[1], resin_core::ElementOperator::Add)
        .unwrap();
    let probs = &exp_x / &sum_exp;
    let loss = mean(&cross_entropy(&probs, &y).unwrap()).unwrap();
    let grads = grad_wrt(
        &loss,
        Linear {
            weight: layer.weight.clone(),
            bias: layer.bias.clone(),
        },
    )
    .unwrap();

    let x_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let y_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let w_data = vec![0.1f32; 12];
    let b_data = [0.0f32, 0.0, 0.0];
    let (_loss_v, gmap) = run_loss_grads(
        &loss,
        &[("gw", &grads.weight)],
        &[
            (&x, &x_data),
            (&y, &y_data),
            (&layer.weight, &w_data),
            (layer.bias.as_ref().unwrap(), &b_data),
        ],
    )
    .expect("gpu");

    let params_owned = [
        (&x, x_data.to_vec()),
        (&y, y_data.to_vec()),
        (&layer.weight, w_data),
        (layer.bias.as_ref().unwrap(), b_data.to_vec()),
    ];
    let fd_w0 = finite_diff_param(&loss, &params_owned, "weight", 0, 1e-3).expect("fd");
    let gw0 = gmap["gw"][0];
    assert!(
        (gw0 - fd_w0).abs() < 5e-3,
        "no-max-sub weight[0] autodiff={gw0} FD={fd_w0}"
    );
}

/// G5: two-layer MLP (`relu` hidden then linear) + softmax+CE — FD on last and first weights.
/// Uses `register_param_tree` so layers get unique buffer names (`model.layers.i.weight`).
#[test]
fn g5_two_layer_mlp_weight_fd() {
    use resin_core::{format_param_path, join_param_path, ParamTree, ParamTreePathElement, F4};
    use resin_dsl::param;
    use resin_ir::IrProgram;
    use resin_jit_wgpu::{build_wgpu_program, create_interp, AdmitProgram, Interp, InterpConfig};
    use resin_nn::Mlp;
    use std::collections::BTreeMap;

    let mlp = Mlp::new(4, 3, 1, 5, true); // in=4, hidden=5, out=3
    let xs = param([2, 4], F4, "xs");
    let ys = param([2, 3], F4, "ys");
    let loss = mean(&cross_entropy(&mlp.forward(xs.clone()).unwrap(), &ys).unwrap()).unwrap();
    let grads = grad_wrt(
        &loss,
        Mlp {
            layers: mlp
                .layers
                .iter()
                .map(|l| Linear {
                    weight: l.weight.clone(),
                    bias: l.bias.clone(),
                })
                .collect(),
        },
    )
    .unwrap();

    let mut ir = IrProgram::new();
    ir.register_param("xs", &xs).unwrap();
    ir.register_param("ys", &ys).unwrap();
    ir.register_param_tree(&mlp, "model").unwrap();
    ir.build_sink("loss", &loss).unwrap();
    ir.build_sink_tree("grad", &grads).unwrap();
    ir.seal_params().unwrap();
    let artifact = build_wgpu_program(&ir, None);
    let mut interp = create_interp(InterpConfig::default()).expect("interp");
    let pid = interp.admit_program(artifact).expect("admit");

    let xs_data = [1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let ys_data = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];
    let mut weights: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    let mut rng = 1u64;
    for (path, view) in mlp.flatten() {
        let n: usize = view.shape().iter().map(|&d| d as usize).product();
        let is_bias = matches!(
            path.last(),
            Some(ParamTreePathElement::Name(s)) if s.as_ref() == "bias"
        );
        let vals: Vec<f32> = if is_bias {
            vec![0.0; n]
        } else {
            (0..n)
                .map(|_| {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u = ((rng >> 11) as f32) * (1.0 / ((1u64 << 53) as f32));
                    (u * 2.0 - 1.0) * 0.3
                })
                .collect()
        };
        let key = format_param_path(&path);
        let name = join_param_path("model", &path);
        interp
            .write_param(pid, &name, &interp_helpers::f32_bytes(&vals))
            .unwrap();
        weights.insert(key, vals);
    }
    interp
        .write_param(pid, "xs", &interp_helpers::f32_bytes(&xs_data))
        .unwrap();
    interp
        .write_param(pid, "ys", &interp_helpers::f32_bytes(&ys_data))
        .unwrap();
    interp.run(pid).unwrap();

    let read_grad = |path: &str| -> Vec<f32> {
        let sink = format!("grad.{path}");
        interp
            .read_sink(pid, &sink)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };

    let fd_one = |path: &str, idx: usize, eps: f32| -> f32 {
        let mut ir2 = IrProgram::new();
        ir2.register_param("xs", &xs).unwrap();
        ir2.register_param("ys", &ys).unwrap();
        ir2.register_param_tree(&mlp, "model").unwrap();
        ir2.build_sink("loss", &loss).unwrap();
        ir2.seal_params().unwrap();
        let art = build_wgpu_program(&ir2, None);
        let mut ip = create_interp(InterpConfig::default()).unwrap();
        let p = ip.admit_program(art).unwrap();
        let write_all = |ip: &mut resin_jit_wgpu::WgpuInterp,
                         wmap: &BTreeMap<String, Vec<f32>>,
                         perturb: Option<(&str, usize, f32)>| {
            for (path, vals) in wmap {
                let mut v = vals.clone();
                if let Some((pp, i, d)) = perturb {
                    if path == pp {
                        v[i] += d;
                    }
                }
                let name = format!("model.{path}");
                ip.write_param(p, &name, &interp_helpers::f32_bytes(&v))
                    .unwrap();
            }
            ip.write_param(p, "xs", &interp_helpers::f32_bytes(&xs_data))
                .unwrap();
            ip.write_param(p, "ys", &interp_helpers::f32_bytes(&ys_data))
                .unwrap();
        };
        write_all(&mut ip, &weights, Some((path, idx, eps)));
        ip.run(p).unwrap();
        let lp = f32::from_le_bytes(ip.read_sink(p, "loss").unwrap()[0..4].try_into().unwrap());
        write_all(&mut ip, &weights, Some((path, idx, -eps)));
        ip.run(p).unwrap();
        let lm = f32::from_le_bytes(ip.read_sink(p, "loss").unwrap()[0..4].try_into().unwrap());
        (lp - lm) / (2.0 * eps)
    };

    let eps = 1e-3f32;
    // Last layer weight (layers.1.weight) and first layer weight (layers.0.weight)
    for path in ["layers.1.weight", "layers.0.weight", "layers.1.bias"] {
        let ad = read_grad(path)[0];
        let fd = fd_one(path, 0, eps);
        let err = (ad - fd).abs();
        assert!(err < 5e-3, "G5 {path}[0] autodiff={ad} FD={fd} err={err}");
    }
}
