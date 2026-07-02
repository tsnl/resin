//! Python `test_reduction_wgsl` / `test_wgsl_kernel` parity (host-side, no GPU).

use resin_core::{Accessor, ElementOperator, ElementType, F4};
use resin_dsl::const_bytes;
use resin_ir::{
    ElementRpnExpr, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel, IrProgram, IrReductionKernel,
    RpnAtom,
};
use resin_jit_wgpu::{dispatch_size_for_kernel, emit_wgsl_for_kernel, WgslKernelConfig};

fn acc(shape: &[u32]) -> Accessor {
    Accessor::new_c_contiguous(shape.to_vec().into_boxed_slice(), 0)
}

fn reduction_kernel(
    in_shape: &[u32],
    out_shape: &[u32],
    op: ElementOperator,
    axes: &[u32],
) -> IrKernel {
    IrKernel::Reduction(IrReductionKernel {
        arg_accessors: vec![acc(in_shape)],
        etype: F4,
        shape: out_shape.to_vec().into_boxed_slice(),
        operator: op,
        axes: axes.to_vec().into_boxed_slice(),
        clear_output_before_dispatch: false,
    })
}

#[test]
fn emit_sum_axis_1_has_loop_and_acc() {
    let kernel = reduction_kernel(&[2, 3], &[2, 1], ElementOperator::Add, &[1]);
    let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default());
    assert!(
        wgsl.contains("fn input_index") || wgsl.contains("input_index"),
        "{wgsl}"
    );
    assert!(
        wgsl.contains("ri < 3u") || wgsl.contains("ri < 3"),
        "{wgsl}"
    );
    assert!(wgsl.contains("acc +=") || wgsl.contains("acc="), "{wgsl}");
}

#[test]
fn emit_max_multi_axis() {
    let kernel = reduction_kernel(&[2, 3], &[1, 1], ElementOperator::Max, &[0, 1]);
    let wgsl = emit_wgsl_for_kernel(&kernel, &WgslKernelConfig::default());
    assert!(
        wgsl.contains("ri < 6u") || wgsl.contains("ri < 6"),
        "{wgsl}"
    );
    assert!(wgsl.contains("max("), "{wgsl}");
}

#[test]
fn program_builder_lowers_reduction() {
    let bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let t = const_bytes([2, 3], F4, bytes.into_boxed_slice());
    let n = t.reduce(&[1], ElementOperator::Add).unwrap();
    let mut ir = IrProgram::new();
    ir.build_sink("out", &n).unwrap();
    ir.seal_params().unwrap();
    assert_eq!(ir.queue.len(), 1);
    match &ir.queue[0].kernel {
        IrKernel::Reduction(k) => {
            assert_eq!(k.operator, ElementOperator::Add);
            assert_eq!(&*k.axes, &[1]);
            assert_eq!(&*k.shape, &[2, 1]);
        }
        other => panic!("expected reduction, got {other:?}"),
    }
}

#[test]
fn dispatch_exact_fit() {
    let config = WgslKernelConfig {
        lg2_items_per_thread: 3,
        workgroup_size: 8,
    };
    let kernel = IrKernel::Matmul(IrMatmulKernel {
        arg_accessors: vec![acc(&[8, 8]), acc(&[8, 8])],
        etype: F4,
        shape: Box::from([8u32, 8]),
        clear_output_before_dispatch: false,
    });
    assert_eq!(dispatch_size_for_kernel(&kernel, &config), [1, 1, 1]);
}

#[test]
fn dispatch_partial_last_workgroup() {
    let config = WgslKernelConfig {
        lg2_items_per_thread: 3,
        workgroup_size: 8,
    };
    let kernel = IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
        arg_accessors: vec![acc(&[65])],
        arg_etypes: vec![ElementType::F4],
        etype: F4,
        shape: Box::from([65u32]),
        rpn_expr: ElementRpnExpr {
            atoms: vec![RpnAtom::Arg(0)],
        },
        clear_output_before_dispatch: false,
    });
    assert_eq!(dispatch_size_for_kernel(&kernel, &config), [2, 1, 1]);
}

#[test]
fn dispatch_empty_output() {
    let config = WgslKernelConfig::default();
    let kernel = IrKernel::Matmul(IrMatmulKernel {
        arg_accessors: vec![acc(&[0, 8]), acc(&[8, 4])],
        etype: F4,
        shape: Box::from([0u32, 4]),
        clear_output_before_dispatch: false,
    });
    assert_eq!(dispatch_size_for_kernel(&kernel, &config), [0, 1, 1]);
}

#[test]
fn dispatch_workgroup_size_affects() {
    let config = WgslKernelConfig {
        lg2_items_per_thread: 3,
        workgroup_size: 4,
    };
    let kernel = IrKernel::Matmul(IrMatmulKernel {
        arg_accessors: vec![acc(&[8, 8]), acc(&[8, 8])],
        etype: F4,
        shape: Box::from([8u32, 8]),
        clear_output_before_dispatch: false,
    });
    assert_eq!(dispatch_size_for_kernel(&kernel, &config), [2, 1, 1]);
}
