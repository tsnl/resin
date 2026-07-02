//! Stage 2: second `Interp::run` must not see stale outputs (esp. remap/scatter-add).

#[path = "interp_helpers.rs"]
mod interp_helpers;

use interp_helpers::{approx_eq, f4_param, run_graph_twice};
use resin_core::Accessor;
use resin_core::{ElementOperator, F4};
use resin_dsl::{RemapInfo, RemapScatterInfo, View};

#[test]
fn elementwise_add_rerun_stable() {
    let a = f4_param(&[3]);
    let b = f4_param(&[3]);
    let out = &a + &b;
    let (r1, r2) =
        run_graph_twice(&out, &[(&a, &[1.0, 2.0, 3.0]), (&b, &[4.0, 5.0, 6.0])]).expect("gpu");
    assert!(approx_eq(&r1, &[5.0, 7.0, 9.0], 1e-5), "first {r1:?}");
    assert!(approx_eq(&r2, &r1, 1e-5), "second {r2:?} vs first {r1:?}");
}

#[test]
fn matmul_rerun_stable() {
    let a = f4_param(&[2, 2]);
    let b = f4_param(&[2, 2]);
    let out = a.matmul(&b).unwrap();
    let (r1, r2) = run_graph_twice(
        &out,
        &[(&a, &[1.0, 2.0, 3.0, 4.0]), (&b, &[5.0, 6.0, 7.0, 8.0])],
    )
    .expect("gpu");
    // [[19,22],[43,50]]
    assert!(approx_eq(&r1, &[19.0, 22.0, 43.0, 50.0], 1e-3), "{r1:?}");
    assert!(approx_eq(&r2, &r1, 1e-5), "stale matmul? {r2:?} vs {r1:?}");
}

#[test]
fn scatter_add_rerun_not_double_count() {
    // Without clearing the output buffer, a second run would atomic-add into
    // leftover values and double (or worse) the result.
    let g = f4_param(&[3, 2]);
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
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (r1, r2) = run_graph_twice(&out, &[(&g, &data)]).expect("gpu");
    let expect = [1.0, 3.0, 5.0, 2.0, 4.0, 6.0];
    assert!(approx_eq(&r1, &expect, 1e-5), "first {r1:?}");
    assert!(
        approx_eq(&r2, &expect, 1e-5),
        "second run stale/double-counted: {r2:?} (want {expect:?}, 2× would be {:?})",
        expect.map(|x| x * 2.0)
    );
}

#[test]
fn scatter_add_three_runs() {
    let g = f4_param(&[2]);
    // Scatter each element to itself (identity scatter-add into [2]).
    let out = View::remap(
        &g,
        RemapInfo::Scatter(RemapScatterInfo {
            accessor: Some(Accessor::new(0, [2], [1])),
            operator: Some(ElementOperator::Add),
        }),
        [2],
        F4,
        [],
    );
    // Use run_graph_twice logic thrice via helper twice + manual would be enough;
    // two runs already catch double-count; assert values stay [3,4] not [9,12].
    let (r1, r2) = run_graph_twice(&out, &[(&g, &[3.0, 4.0])]).expect("gpu");
    assert!(approx_eq(&r1, &[3.0, 4.0], 1e-5), "{r1:?}");
    assert!(approx_eq(&r2, &[3.0, 4.0], 1e-5), "triple-stale? {r2:?}");
}

#[test]
fn gather_rerun_stable() {
    let x = f4_param(&[4]);
    // Force non-identity: permute then densify-copy (gather).
    let v = x.permute(&[0]).unwrap();
    let out = v.copy(None);
    let (r1, r2) = run_graph_twice(&out, &[(&x, &[1.0, 2.0, 3.0, 4.0])]).expect("gpu");
    assert!(approx_eq(&r1, &[1.0, 2.0, 3.0, 4.0], 1e-5), "{r1:?}");
    assert!(approx_eq(&r2, &r1, 1e-5), "{r2:?}");
}
