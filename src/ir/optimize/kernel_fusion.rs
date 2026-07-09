//! Elementwise kernel fusion: inline producer expressions into consumers.
//!
//! Producer→consumer chains of elementwise dispatches collapse into single
//! kernels, to a fixed point. Fusion is tree substitution
//! ([`Expr::substitute`]).

use crate::ir::{
    Accessor, BufferRef, BufferView, BufferViewRef, Dispatch, Expr, Kernel, Program,
};

/// Fuse adjacent elementwise dispatches to a fixed point.
///
/// A producer fuses into a consumer when both are elementwise and the
/// consumer loads the producer's write view (or a broadcast of it): the
/// producer's expression is substituted for that load ([`Expr::substitute`]),
/// with producer loads composed through the same view relationship.
/// Multi-consumer intermediates are inlined into each reader; the producer is
/// dropped once nothing reads its output. Matmul and reduction are fusion
/// boundaries.
///
/// Ends with [`crate::ir::layout::eliminate_dead`] so dead intermediates from
/// fusion are gone before later passes (or backend layout).
pub fn fuse_elementwise(mut program: Program) -> Program {
    while fuse_one(&mut program) {}
    crate::ir::layout::eliminate_dead(program)
}

/// Soft cap on fused tree size. Multi-consumer inlining can otherwise
/// duplicate a shared subexpression without bound.
const MAX_FUSED_NODES: usize = 256;

/// One fusable producer→consumer edge in the dispatch queue.
struct ProducerConsumerPair {
    consumer: usize,
    producer: usize,
    /// Consumer load of the producer's output (identity or broadcast).
    arg_view: BufferViewRef,
    consumer_expr: Expr,
    producer_expr: Expr,
}

/// Find one producer→consumer pair and fuse it. Returns false at fixed point.
fn fuse_one(program: &mut Program) -> bool {
    let Some(pair) = find_fusible_producer_consumer_pair(program) else {
        return false;
    };
    apply_fusion(program, pair);
    true
}

/// Scan the queue for the next fusable elementwise→elementwise edge.
fn find_fusible_producer_consumer_pair(program: &Program) -> Option<ProducerConsumerPair> {
    for (consumer, dispatch) in program.queue.iter().enumerate() {
        let Kernel::Elementwise { expr: consumer_expr } = &dispatch.kernel else {
            continue;
        };
        for &arg_view in &dispatch.args {
            let Some((producer, producer_expr)) =
                fusable_producer(program, consumer, arg_view)
            else {
                continue;
            };
            // Each use of the intermediate becomes a full copy of `producer_expr`.
            let uses = count_load_occurrences(consumer_expr, arg_view);
            let fused_nodes =
                consumer_expr.node_count() + uses * producer_expr.node_count();
            if fused_nodes > MAX_FUSED_NODES {
                continue;
            }
            return Some(ProducerConsumerPair {
                consumer,
                producer,
                arg_view,
                consumer_expr: consumer_expr.clone(),
                producer_expr,
            });
        }
    }
    None
}

/// Splice the producer into the consumer and drop the producer if unused.
fn apply_fusion(program: &mut Program, pair: ProducerConsumerPair) {
    let ProducerConsumerPair {
        consumer,
        producer,
        arg_view,
        consumer_expr,
        producer_expr,
    } = pair;

    // Re-read each producer load through the consumer's view of the intermediate.
    let write = program.view(program.queue[producer].output).accessor.clone();
    let read = program.view(arg_view).accessor.clone();
    let producer_expr = producer_expr.map_loads(&mut |v| {
        let arg = program.view(v).clone();
        let composed = arg.accessor.compose_through(&write, &read);
        Expr::Load(intern_view(program, arg.buffer, composed))
    });

    let fused = consumer_expr.substitute(arg_view, &producer_expr);
    let args = fused.loads();

    program.queue[consumer] = Dispatch {
        kernel: Kernel::Elementwise { expr: fused },
        args,
        output: program.queue[consumer].output,
    };

    let buffer = program.view(arg_view).buffer;
    if !buffer_is_read(program, buffer, producer) {
        program.queue.remove(producer);
    }
}

/// How many times `target` appears as a load leaf (not deduped).
fn count_load_occurrences(expr: &Expr, target: BufferViewRef) -> usize {
    match expr {
        Expr::Load(v) => usize::from(*v == target),
        Expr::Op { args, .. } => args.iter().map(|a| count_load_occurrences(a, target)).sum(),
    }
}

/// Index and body of the dispatch producing `arg_view`, if it can be inlined
/// into `queue[consumer]` at that load.
///
/// A producer is fusable when:
///
/// 1. **Kernel compatibility** — producer is elementwise (same iteration model
///    as the consumer).
/// 2. **Shape compatibility** — [`fusible_view_relationship`].
fn fusable_producer(
    program: &Program,
    consumer: usize,
    arg_view: BufferViewRef,
) -> Option<(usize, Expr)> {
    let buffer = program.view(arg_view).buffer;
    let producer = program
        .queue
        .iter()
        .position(|d| program.view(d.output).buffer == buffer)?;
    if producer >= consumer {
        return None;
    }

    let Kernel::Elementwise { expr } = &program.queue[producer].kernel else {
        return None;
    };

    // Cast/bitcast change element type mid-tree; keep them as fusion barriers
    // so backends evaluate each typed kernel in isolation.
    if expr_changes_element_type(expr) {
        return None;
    }

    if !fusible_view_relationship(program, producer, arg_view) {
        return None;
    }

    // Only fuse when producer and consumer buffers share an element type.
    let prod_etype = program.buffer(program.view(program.queue[producer].output).buffer).element_type;
    let cons_etype = program.buffer(program.view(program.queue[consumer].output).buffer).element_type;
    if prod_etype != cons_etype {
        return None;
    }

    Some((producer, expr.clone()))
}

fn expr_changes_element_type(expr: &Expr) -> bool {
    match expr {
        Expr::Load(_) => false,
        Expr::Op { op, args } => {
            op.changes_element_type() || args.iter().any(expr_changes_element_type)
        }
    }
}

/// Shape compatibility: consumer reads the producer's write view, or a
/// broadcast of it ([`Accessor::is_broadcast_of`]).
fn fusible_view_relationship(
    program: &Program,
    producer: usize,
    arg_view: BufferViewRef,
) -> bool {
    let write = &program.view(program.queue[producer].output).accessor;
    let read = &program.view(arg_view).accessor;
    read.is_broadcast_of(write)
}

/// Whether any dispatch (other than `except`) or sink still reads `buffer`.
fn buffer_is_read(program: &Program, buffer: BufferRef, except: usize) -> bool {
    let reads = |view: BufferViewRef| program.view(view).buffer == buffer;
    program
        .queue
        .iter()
        .enumerate()
        .any(|(i, d)| i != except && d.args.iter().copied().any(reads))
        || program.sinks.iter().copied().any(reads)
}

fn intern_view(program: &mut Program, buffer: BufferRef, accessor: Accessor) -> BufferViewRef {
    let view = BufferView { buffer, accessor };
    let index = program
        .views
        .iter()
        .position(|v| *v == view)
        .unwrap_or_else(|| {
            program.views.push(view);
            program.views.len() - 1
        });
    BufferViewRef(index)
}


