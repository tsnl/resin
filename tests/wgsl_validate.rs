//! Validate emitted WGSL with naga — no GPU required. This catches shader
//! syntax/typing bugs (bindings, atomics, address arithmetic) that pure text
//! assertions miss and that otherwise only surface on a real adapter.

#![cfg(feature = "wgpu")]

use resin::dsl::{grad_wrt, ScatterOp, Tensor};
use resin::ir::{Accessor, Kernel, Program, RemapInfo};
use resin::jit::lower::lower;
use resin::jit::wgpu::{KernelConfig, WgpuJit};
use resin::jit::Jit;
use resin::ops::{AssocOp, ElementType};
use resin::Tree;

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

fn remap_program(info: RemapInfo, etype: ElementType) -> Program {
    use resin::ir::{Buffer, BufferRef, BufferView, BufferViewRef, Dispatch};

    let src_shape: Box<[usize]> = Box::from([4usize, 2]);
    let idx_shape: Box<[usize]> = Box::from([4usize]);
    let out_shape = src_shape.clone();

    let mut buffers = vec![
        Buffer {
            shape: src_shape.clone(),
            element_type: etype,
            init: None,
        },
        Buffer {
            shape: out_shape.clone(),
            element_type: etype,
            init: None,
        },
    ];
    let mut views = vec![
        BufferView {
            buffer: BufferRef(0),
            accessor: Accessor::dense(src_shape.clone(), 0),
        },
        BufferView {
            buffer: BufferRef(1),
            accessor: Accessor::dense(out_shape.clone(), 0),
        },
    ];
    let mut args = vec![BufferViewRef(0)];
    match &info {
        RemapInfo::ScatterView { .. } => {}
        _ => {
            buffers.push(Buffer {
                shape: idx_shape.clone(),
                element_type: ElementType::U32,
                init: None,
            });
            views.push(BufferView {
                buffer: BufferRef(2),
                accessor: Accessor::dense(idx_shape, 0),
            });
            args.push(BufferViewRef(2));
        }
    }
    Program {
        params: vec![],
        sinks: vec![],
        queue: vec![Dispatch {
            kernel: Kernel::Remap { info },
            args,
            output: BufferViewRef(1),
        }],
        buffers,
        views,
    }
}

fn validate_remap(info: RemapInfo, etype: ElementType) {
    let program = remap_program(info, etype);
    program.validate().expect("remap IR validates");
    let wgsl = resin::jit::wgpu::emit_wgsl_for_dispatch(&program, 0, &KernelConfig::default());
    validate_wgsl(&wgsl);
}

#[test]
fn gather_rows_wgsl_validates() {
    validate_remap(RemapInfo::GatherRows, ElementType::F32);
    validate_remap(RemapInfo::GatherRows, ElementType::U32);
}

#[test]
fn scatter_rows_write_wgsl_validates() {
    validate_remap(
        RemapInfo::ScatterRows { operator: None },
        ElementType::F32,
    );
    validate_remap(
        RemapInfo::ScatterRows { operator: None },
        ElementType::U32,
    );
}

#[test]
fn scatter_rows_add_wgsl_validates() {
    validate_remap(
        RemapInfo::ScatterRows {
            operator: Some(AssocOp::Add),
        },
        ElementType::F32,
    );
    validate_remap(
        RemapInfo::ScatterRows {
            operator: Some(AssocOp::Add),
        },
        ElementType::U32,
    );
}

#[test]
fn scatter_view_wgsl_validates() {
    validate_remap(
        RemapInfo::ScatterView {
            accessor: Accessor::dense([4, 2], 8),
        },
        ElementType::F32,
    );
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

    let x = Tensor::parameter(&[8]);
    let k = Tensor::parameter_typed(&[8], ElementType::U32);
    let one = Tensor::full_u32(&[8], 1);
    let bit = (k.clone() >> one.clone()) & one.clone();
    let mask = bit.cast(ElementType::F32);
    let indices = Tensor::constant_u32(&[8], &[7, 6, 5, 4, 3, 2, 1, 0]);
    let gathered = x.gather_rows(&indices);
    let blended = mask.select(&gathered, &x);
    let scattered = blended.scatter_rows(&indices, 8, ScatterOp::Add);
    let loss = scattered.sum_axes(&[0]).squeeze_all();
    let grad = grad_wrt(&loss, &x).unwrap();

    let params = In { x, k };
    let sinks = vec![loss, grad];
    let program = lower(&params, &sinks).expect("lower");
    let artifact = WgpuJit::default().lower(&program).expect("wgpu lower");
    for pipeline in &artifact.pipelines {
        validate_wgsl(&pipeline.wgsl);
    }
}
