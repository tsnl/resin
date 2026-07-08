//! IR→IR optimization passes.
//!
//! Implemented today:
//!
//! - **Elementwise RPN fusion** — single-consumer elementwise dispatches are
//!   spliced into their consumer's RPN expression (the RPN kernel already
//!   evaluates arbitrary expressions over N operands), then dead buffers and
//!   views are compacted away. This is what keeps combinator-built graphs
//!   (scan, sort) at a sane dispatch count without any dedicated kernels.
//!
//! Ideas for future passes (ported from the Python `resin.ir.ir_opt` notes):
//!
//! - **Matmul + RPN epilogue fused kernel**: support an RPN scalar epilogue
//!   after each matmul (possibly also an RPN prologue per argument).
//! - **Tiled ElementTypes**: extend `ElementType` with tile types (e.g.
//!   `mat4x4_f4`) so block matmul can use cooperative matrix ops and fuse
//!   with surrounding kernels; long-term, matmul decomposes into elementwise
//!   tiled ops + reduction + stride tricks instead of a dedicated kernel.
//! - **Constant folding** and inlining into RPN / matmul / remap kernels.
//! - **Big symbolic scalar graph**: treat every element as a scalar symbol
//!   and re-derive grouping/scheduling from scratch (radical; needs thought).
//! - **Emitter-side**: SPIR-V extensions like `SPV_NV_tensor_addressing`.

use resin_core::Tree;

use crate::program::{IrDispatch, IrKernel, IrProgram};
use crate::refs::{BufferRef, BufferViewRef};
use crate::rpn::{ElementRpnExpr, RpnAtom};

/// Storage-buffer bindings available to one fused kernel: WebGPU guarantees
/// only 8 per stage, and one is the output.
const MAX_FUSED_ARGS: usize = 7;

/// Run middle-end optimization passes on `program`.
pub fn optimize<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    mut program: IrProgram<P, S>,
) -> IrProgram<P, S> {
    fuse_elementwise_chains(&mut program);
    compact(&mut program);
    program
}

/// Splice single-consumer elementwise producers into their consumer's RPN.
fn fuse_elementwise_chains<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    program: &mut IrProgram<P, S>,
) {
    loop {
        let Some((producer_idx, consumer_idx)) = find_fusible_pair(program) else {
            return;
        };
        let producer = program.queue[producer_idx].clone();
        let consumer = program.queue[consumer_idx].clone();
        if let Some(fused) = splice(&producer, &consumer, program) {
            program.queue[consumer_idx] = fused;
            program.queue.remove(producer_idx);
        } else {
            // Would exceed the binding budget; leave this pair as-is and stop
            // rather than loop forever on it.
            return;
        }
    }
}

fn find_fusible_pair<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    program: &IrProgram<P, S>,
) -> Option<(usize, usize)> {
    // Buffers that must stay materialized: sink-read or param-bound.
    let mut pinned = vec![false; program.buffers.len()];
    program.sinks.for_each_leaf(|view| {
        pinned[program.buffer_views[view.index()].buffer_index.index()] = true;
    });
    program.params.for_each_leaf(|buffer| {
        pinned[buffer.index()] = true;
    });

    // producer[b] = dispatch index that writes buffer b.
    let mut producer_of = vec![None; program.buffers.len()];
    for (i, dispatch) in program.queue.iter().enumerate() {
        producer_of[dispatch.output_buffer_index.index()] = Some(i);
    }

    // consumers[b] = dispatch indices that read buffer b.
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); program.buffers.len()];
    for (j, dispatch) in program.queue.iter().enumerate() {
        for view_ref in &dispatch.arg_view_indices {
            let b = program.buffer_views[view_ref.index()].buffer_index.index();
            if consumers[b].last() != Some(&j) {
                consumers[b].push(j);
            }
        }
    }

    for (j, dispatch) in program.queue.iter().enumerate() {
        let IrKernel::ElementwiseRpn(consumer_kernel) = &dispatch.kernel else {
            continue;
        };
        for view_ref in &dispatch.arg_view_indices {
            let view = &program.buffer_views[view_ref.index()];
            let b = view.buffer_index.index();
            if pinned[b] || consumers[b] != [j] {
                continue;
            }
            let Some(i) = producer_of[b] else {
                continue;
            };
            let IrKernel::ElementwiseRpn(producer_kernel) = &program.queue[i].kernel else {
                continue;
            };
            if producer_kernel.shape != consumer_kernel.shape {
                continue;
            }
            // Splicing replaces EVERY read of the fused buffer with the
            // producer expression evaluated at the output index, so every
            // one of the consumer's views of it must be a plain dense read
            // (a broadcast/slice view — e.g. a scan total re-read at a fixed
            // offset — would silently change meaning).
            let all_reads_dense = dispatch.arg_view_indices.iter().all(|other_ref| {
                let other = &program.buffer_views[other_ref.index()];
                other.buffer_index.index() != b
                    || other
                        .accessor
                        .is_dense_c_contiguous(&program.buffers[b].shape)
            });
            if !all_reads_dense {
                continue;
            }
            return Some((i, j));
        }
    }
    None
}

/// Inline `producer`'s RPN into `consumer` at every slot that reads the
/// producer's output buffer. Returns `None` if the fused kernel would exceed
/// the binding budget.
fn splice<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(
    producer: &IrDispatch,
    consumer: &IrDispatch,
    program: &IrProgram<P, S>,
) -> Option<IrDispatch> {
    let IrKernel::ElementwiseRpn(pk) = &producer.kernel else {
        unreachable!("find_fusible_pair only selects elementwise kernels");
    };
    let IrKernel::ElementwiseRpn(ck) = &consumer.kernel else {
        unreachable!("find_fusible_pair only selects elementwise kernels");
    };
    let fused_buffer = producer.output_buffer_index;

    // New arg tables, deduplicated by view ref.
    let mut arg_views: Vec<BufferViewRef> = Vec::new();
    let mut arg_accessors = Vec::new();
    let mut arg_element_types = Vec::new();
    let intern =
        |view_ref: BufferViewRef,
         accessor: &resin_core::Accessor,
         etype: resin_core::ElementType,
         arg_views: &mut Vec<BufferViewRef>,
         arg_accessors: &mut Vec<resin_core::Accessor>,
         arg_element_types: &mut Vec<resin_core::ElementType>|
         -> u32 {
            if let Some(pos) = arg_views.iter().position(|v| *v == view_ref) {
                return pos as u32;
            }
            arg_views.push(view_ref);
            arg_accessors.push(accessor.clone());
            arg_element_types.push(etype);
            (arg_views.len() - 1) as u32
        };

    // Consumer args that are NOT the fused buffer keep a (remapped) slot.
    let mut consumer_slot_map: Vec<Option<u32>> = Vec::new();
    for (slot, view_ref) in consumer.arg_view_indices.iter().enumerate() {
        let reads_fused =
            program.buffer_views[view_ref.index()].buffer_index == fused_buffer;
        if reads_fused {
            consumer_slot_map.push(None);
        } else {
            consumer_slot_map.push(Some(intern(
                *view_ref,
                &ck.arg_accessors[slot],
                ck.arg_element_types[slot],
                &mut arg_views,
                &mut arg_accessors,
                &mut arg_element_types,
            )));
        }
    }
    // Producer args all get slots.
    let producer_slot_map: Vec<u32> = producer
        .arg_view_indices
        .iter()
        .enumerate()
        .map(|(slot, view_ref)| {
            intern(
                *view_ref,
                &pk.arg_accessors[slot],
                pk.arg_element_types[slot],
                &mut arg_views,
                &mut arg_accessors,
                &mut arg_element_types,
            )
        })
        .collect();

    if arg_views.len() > MAX_FUSED_ARGS {
        return None;
    }

    // Rewrite the consumer RPN, inlining the producer expression wherever the
    // consumer read the fused buffer.
    let mut atoms = Vec::with_capacity(ck.rpn_expr.atoms.len() + pk.rpn_expr.atoms.len());
    for atom in &ck.rpn_expr.atoms {
        match atom {
            RpnAtom::Arg(a) => match consumer_slot_map[*a as usize] {
                Some(mapped) => atoms.push(RpnAtom::Arg(mapped)),
                None => {
                    for producer_atom in &pk.rpn_expr.atoms {
                        match producer_atom {
                            RpnAtom::Arg(pa) => {
                                atoms.push(RpnAtom::Arg(producer_slot_map[*pa as usize]))
                            }
                            RpnAtom::Op(op) => atoms.push(RpnAtom::Op(*op)),
                        }
                    }
                }
            },
            RpnAtom::Op(op) => atoms.push(RpnAtom::Op(*op)),
        }
    }

    let mut fused_kernel = ck.clone();
    fused_kernel.arg_accessors = arg_accessors;
    fused_kernel.arg_element_types = arg_element_types;
    fused_kernel.rpn_expr = ElementRpnExpr { atoms };
    Some(IrDispatch {
        kernel: IrKernel::ElementwiseRpn(fused_kernel),
        arg_view_indices: arg_views,
        output_buffer_index: consumer.output_buffer_index,
    })
}

/// Drop unreferenced views and buffers, remapping every index in place.
fn compact<P: Tree<BufferRef>, S: Tree<BufferViewRef>>(program: &mut IrProgram<P, S>) {
    // Live views: dispatch args + sinks.
    let mut view_live = vec![false; program.buffer_views.len()];
    for dispatch in &program.queue {
        for view_ref in &dispatch.arg_view_indices {
            view_live[view_ref.index()] = true;
        }
    }
    program.sinks.for_each_leaf(|view| {
        view_live[view.index()] = true;
    });

    let mut view_map: Vec<Option<BufferViewRef>> = vec![None; program.buffer_views.len()];
    let mut new_views = Vec::new();
    for (old, view) in program.buffer_views.iter().enumerate() {
        if view_live[old] {
            view_map[old] = Some(BufferViewRef::new(new_views.len()));
            new_views.push(view.clone());
        }
    }

    // Live buffers: dispatch outputs + live view targets + params.
    let mut buffer_live = vec![false; program.buffers.len()];
    for dispatch in &program.queue {
        buffer_live[dispatch.output_buffer_index.index()] = true;
    }
    for view in &new_views {
        buffer_live[view.buffer_index.index()] = true;
    }
    program.params.for_each_leaf(|buffer| {
        buffer_live[buffer.index()] = true;
    });

    let mut buffer_map: Vec<Option<BufferRef>> = vec![None; program.buffers.len()];
    let mut new_buffers = Vec::new();
    for (old, buffer) in program.buffers.iter().enumerate() {
        if buffer_live[old] {
            buffer_map[old] = Some(BufferRef::new(new_buffers.len()));
            new_buffers.push(buffer.clone());
        }
    }

    for view in &mut new_views {
        view.buffer_index = buffer_map[view.buffer_index.index()].expect("live view buffer");
    }
    for dispatch in &mut program.queue {
        dispatch.output_buffer_index = buffer_map[dispatch.output_buffer_index.index()]
            .expect("live dispatch output");
        for view_ref in &mut dispatch.arg_view_indices {
            *view_ref = view_map[view_ref.index()].expect("live dispatch arg view");
        }
    }
    program.params.for_each_leaf_mut(|buffer| {
        *buffer = buffer_map[buffer.index()].expect("live param buffer");
    });
    program.sinks.for_each_leaf_mut(|view| {
        *view = view_map[view.index()].expect("live sink view");
    });

    program.buffer_views = new_views;
    program.buffers = new_buffers;
}

#[cfg(test)]
mod tests {
    use resin_core::{
        Accessor, BinaryAssocElementOperator, BinaryElementOperator, ElementOperator, F4,
    };

    use super::*;
    use crate::program::{IrBuffer, IrBufferView, IrElementwiseRpnKernel};

    fn dense_buffer(shape: &[u32]) -> IrBuffer {
        IrBuffer {
            shape: shape.into(),
            element_type: F4,
            init: None,
            readonly: false,
        }
    }

    fn binary_rpn(op: BinaryAssocElementOperator) -> ElementRpnExpr {
        ElementRpnExpr {
            atoms: vec![
                RpnAtom::Arg(0),
                RpnAtom::Arg(1),
                RpnAtom::Op(ElementOperator::Binary(BinaryElementOperator::Assoc(op))),
            ],
        }
    }

    fn elementwise_dispatch(
        shape: &[u32],
        args: &[usize],
        out: usize,
        op: BinaryAssocElementOperator,
    ) -> IrDispatch {
        IrDispatch {
            kernel: IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                arg_accessors: vec![Accessor::dense(shape, 0); args.len()],
                arg_element_types: vec![F4; args.len()],
                element_type: F4,
                shape: shape.into(),
                rpn_expr: binary_rpn(op),
                clear_output_before_dispatch: false,
            }),
            arg_view_indices: args.iter().map(|&v| BufferViewRef::new(v)).collect(),
            output_buffer_index: BufferRef::new(out),
        }
    }

    /// (a + b) * c over buffers 0..2, intermediate 3, output 4.
    fn chain_program() -> IrProgram<Vec<BufferRef>, BufferViewRef> {
        let shape: &[u32] = &[4];
        IrProgram {
            params: vec![BufferRef::new(0), BufferRef::new(1), BufferRef::new(2)],
            sinks: BufferViewRef::new(4),
            queue: vec![
                elementwise_dispatch(shape, &[0, 1], 3, BinaryAssocElementOperator::Add),
                elementwise_dispatch(shape, &[3, 2], 4, BinaryAssocElementOperator::Mul),
            ],
            buffers: (0..5).map(|_| dense_buffer(shape)).collect(),
            buffer_views: (0..5)
                .map(|b| IrBufferView {
                    buffer_index: BufferRef::new(b),
                    accessor: Accessor::dense(shape, 0),
                })
                .collect(),
        }
    }

    #[test]
    fn fuses_single_consumer_chain() {
        let program = optimize(chain_program());
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 1, "add should fuse into mul");
        let IrKernel::ElementwiseRpn(kernel) = &program.queue[0].kernel else {
            panic!("expected elementwise kernel");
        };
        assert_eq!(kernel.arg_accessors.len(), 3);
        // RPN is (a b add) c mul.
        assert_eq!(kernel.rpn_expr.atoms.len(), 5);
        // Dead intermediate buffer is compacted away.
        assert_eq!(program.buffers.len(), 4);
    }

    #[test]
    fn does_not_fuse_sink_read_intermediate() {
        let mut program = chain_program();
        // The intermediate (buffer 3, view 3) is also a sink → must stay.
        program.sinks = BufferViewRef::new(3);
        let program = optimize(program);
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 2);
    }

    #[test]
    fn does_not_fuse_multi_consumer_intermediate() {
        let shape: &[u32] = &[4];
        let mut program = chain_program();
        // Add a second consumer of the intermediate buffer 3.
        program.buffers.push(dense_buffer(shape));
        program.buffer_views.push(IrBufferView {
            buffer_index: BufferRef::new(5),
            accessor: Accessor::dense(shape, 0),
        });
        program.queue.push(elementwise_dispatch(
            shape,
            &[3, 3],
            5,
            BinaryAssocElementOperator::Add,
        ));
        program.sinks = BufferViewRef::new(5);
        let program = optimize(program);
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 3, "shared intermediate must stay");
    }

    #[test]
    fn fuses_operand_used_twice_by_same_consumer() {
        let shape: &[u32] = &[4];
        // d = a + b; e = d * d — both reads are in one consumer, so fusing is
        // safe (the producer expression is inlined twice).
        let program = IrProgram::<Vec<BufferRef>, BufferViewRef> {
            params: vec![BufferRef::new(0), BufferRef::new(1)],
            sinks: BufferViewRef::new(3),
            queue: vec![
                elementwise_dispatch(shape, &[0, 1], 2, BinaryAssocElementOperator::Add),
                elementwise_dispatch(shape, &[2, 2], 3, BinaryAssocElementOperator::Mul),
            ],
            buffers: (0..4).map(|_| dense_buffer(shape)).collect(),
            buffer_views: (0..4)
                .map(|b| IrBufferView {
                    buffer_index: BufferRef::new(b),
                    accessor: Accessor::dense(shape, 0),
                })
                .collect(),
        };
        let program = optimize(program);
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 1);
        let IrKernel::ElementwiseRpn(kernel) = &program.queue[0].kernel else {
            panic!("expected elementwise kernel");
        };
        // (a b add) (a b add) mul — 7 atoms, 2 dedup'd args.
        assert_eq!(kernel.rpn_expr.atoms.len(), 7);
        assert_eq!(kernel.arg_accessors.len(), 2);
    }

    #[test]
    fn does_not_fuse_when_any_read_is_a_broadcast_view() {
        // Regression: e = d + d[3](broadcast). One read of d is dense, the
        // other re-reads a fixed offset with pitch 0 (a scan-total pattern).
        // Splicing would rewrite BOTH reads to the running expression, so the
        // pair must be rejected even though one view is dense.
        let shape: &[u32] = &[4];
        let mut program = IrProgram::<Vec<BufferRef>, BufferViewRef> {
            params: vec![BufferRef::new(0), BufferRef::new(1)],
            sinks: BufferViewRef::new(3),
            queue: vec![
                elementwise_dispatch(shape, &[0, 1], 2, BinaryAssocElementOperator::Add),
                elementwise_dispatch(shape, &[2, 4], 3, BinaryAssocElementOperator::Add),
            ],
            buffers: (0..4).map(|_| dense_buffer(shape)).collect(),
            buffer_views: (0..4)
                .map(|b| IrBufferView {
                    buffer_index: BufferRef::new(b),
                    accessor: Accessor::dense(shape, 0),
                })
                .collect(),
        };
        // View 4: broadcast of d[3] across the whole output.
        program.buffer_views.push(IrBufferView {
            buffer_index: BufferRef::new(2),
            accessor: Accessor {
                offset: 3,
                shape: Box::from([4]),
                pitch: Box::from([0]),
            },
        });
        let program = optimize(program);
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 2, "mixed dense+broadcast reads must not fuse");
    }

    #[test]
    fn does_not_fuse_through_broadcast_view() {
        let mut program = chain_program();
        // Consumer reads the intermediate through a pitched (broadcast) view.
        program.buffer_views[3].accessor = Accessor {
            offset: 0,
            shape: Box::from([4]),
            pitch: Box::from([0]),
        };
        let program = optimize(program);
        program.validate().unwrap();
        assert_eq!(program.queue.len(), 2);
    }
}
