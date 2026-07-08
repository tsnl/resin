//! Validate emitted WGSL with naga — no GPU required. This catches shader
//! syntax/typing bugs (bindings, atomics, address arithmetic) that pure text
//! assertions miss and that otherwise only surface on a real adapter.

#![cfg(feature = "wgpu")]

use resin_core::{Accessor, BinaryAssocElementOperator, ElementType, F4, U4};
use resin_dsl::{grad_wrt, ScatterOp, Tensor};
use resin_ir::{IrKernel, IrRemapKernel, RemapInfo};
use resin_jit::backends::wgpu::WgpuJit;
use resin_jit::Jit;
use resin_macros::Tree;

fn validate_wgsl(wgsl: &str) {
    let module = naga::front::wgsl::parse_str(wgsl)
        .unwrap_or_else(|e| panic!("WGSL parse error: {e}\n---\n{wgsl}"));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("WGSL validation error: {e:?}\n---\n{wgsl}"));
}

fn validate_kernel(kernel: &IrKernel) {
    let wgsl = resin_jit::backends::wgpu::emit_wgsl_for_kernel(
        kernel,
        &resin_jit::backends::wgpu::WgslKernelConfig::default(),
    )
    .expect("emit");
    validate_wgsl(&wgsl);
}

fn remap(info: RemapInfo, etype: ElementType, clear: bool) -> IrKernel {
    IrKernel::Remap(IrRemapKernel {
        arg_accessors: match &info {
            RemapInfo::ScatterView { .. } => vec![Accessor::dense([4, 2], 0)],
            _ => vec![Accessor::dense([4, 2], 0), Accessor::dense([4], 0)],
        },
        arg_element_types: match &info {
            RemapInfo::ScatterView { .. } => vec![etype],
            _ => vec![etype, U4],
        },
        element_type: etype,
        shape: Box::from([4, 2]),
        info,
        clear_output_before_dispatch: clear,
    })
}

#[test]
fn gather_rows_wgsl_validates() {
    validate_kernel(&remap(RemapInfo::GatherRows, F4, false));
    validate_kernel(&remap(RemapInfo::GatherRows, U4, false));
}

#[test]
fn scatter_rows_write_wgsl_validates() {
    validate_kernel(&remap(RemapInfo::ScatterRows { operator: None }, F4, true));
    validate_kernel(&remap(RemapInfo::ScatterRows { operator: None }, U4, true));
}

#[test]
fn scatter_rows_add_wgsl_validates() {
    validate_kernel(&remap(
        RemapInfo::ScatterRows {
            operator: Some(BinaryAssocElementOperator::Add),
        },
        F4,
        true,
    ));
    validate_kernel(&remap(
        RemapInfo::ScatterRows {
            operator: Some(BinaryAssocElementOperator::Add),
        },
        U4,
        true,
    ));
}

#[test]
fn scatter_view_wgsl_validates() {
    validate_kernel(&remap(
        RemapInfo::ScatterView {
            accessor: Accessor::dense([4, 2], 8),
        },
        F4,
        true,
    ));
}

/// Lower a representative Phase-1 graph (u32 ops, casts, compare/select,
/// gather/scatter, gradients) and validate every emitted pipeline.
#[test]
fn phase1_graph_pipelines_validate() {
    #[derive(Tree)]
    struct In<T> {
        x: T,
        k: T,
    }

    let x = Tensor::parameter(&[8], resin_dsl::ElementType::F32);
    let k = Tensor::parameter(&[8], resin_dsl::ElementType::U32);
    let one = Tensor::full_u32(&[8], 1);
    let bit = (k.clone() >> one.clone()) & one.clone();
    let mask = bit.cast(resin_dsl::ElementType::F32);
    let indices = Tensor::constant_u32(&[8], &[7, 6, 5, 4, 3, 2, 1, 0]);
    let gathered = x.gather_rows(&indices);
    let blended = mask.select(&gathered, &x);
    let scattered = blended.scatter_rows(&indices, 8, ScatterOp::Add);
    let loss = scattered.sum_axes(&[0]).squeeze_all();
    let grad = grad_wrt(&loss, &x).unwrap();

    let params = In { x, k };
    let sinks = vec![loss, grad];
    let program = resin_jit::lower_for_tests(&params, &sinks).expect("lower");
    let artifact = WgpuJit.lower(&program).expect("wgpu lower");
    for pipeline in &artifact.pipelines {
        validate_wgsl(&pipeline.wgsl);
    }
}
