//! Elementwise kernel fusion: inline producer expressions into consumers.
//!
//! Producer→consumer chains of same-[`Element`] elementwise dispatches
//! collapse into single kernels, to a fixed point. Fusion is tree
//! substitution ([`Expr::substitute`]).

use std::collections::HashMap;

use crate::ir::{
    Accessor, BufferRef, BufferView, BufferViewRef, Dispatch, Element, Expr, Kernel, Program,
};

/// Fuse adjacent elementwise dispatches to a fixed point.
///
/// A producer fuses into a consumer when both are elementwise with the same
/// [`Element`], and the consumer loads the producer's write view (or a
/// broadcast of it): the producer's expression is substituted for that load
/// ([`Expr::substitute`]), with producer loads composed through the same view
/// relationship. Multi-consumer intermediates are inlined into each reader;
/// the producer is dropped once nothing reads its output. Matmul and reduction
/// are fusion boundaries.
pub fn fuse_elementwise(mut program: Program) -> Program {
    while fuse_one(&mut program) {}
    compact(program)
}

/// Soft cap on fused tree size. Multi-consumer inlining can otherwise
/// duplicate a shared subexpression without bound.
const MAX_FUSED_NODES: usize = 256;

/// One fusable producer→consumer edge in the dispatch queue.
struct ProducerConsumerPair {
    consumer: usize,
    producer: usize,
    /// Shared element type of both kernels (must match to fuse).
    element: Element,
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
        let Kernel::Elementwise { expr: consumer_expr, element } = &dispatch.kernel else {
            continue;
        };
        for &arg_view in &dispatch.args {
            let Some((producer, producer_expr)) =
                fusable_producer(program, consumer, arg_view, *element)
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
                element: *element,
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
        element,
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
        kernel: Kernel::Elementwise { expr: fused, element },
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
/// A producer is fusable when three criteria hold:
///
/// 1. **Kernel compatibility** — producer is elementwise (same iteration model
///    as the consumer).
/// 2. **Datatype compatibility** — [`same_element_type`].
/// 3. **Shape compatibility** — [`fusible_view_relationship`].
fn fusable_producer(
    program: &Program,
    consumer: usize,
    arg_view: BufferViewRef,
    consumer_element: Element,
) -> Option<(usize, Expr)> {
    let buffer = program.view(arg_view).buffer;
    let producer = program
        .queue
        .iter()
        .position(|d| program.view(d.output).buffer == buffer)?;
    if producer >= consumer {
        return None;
    }

    // Kernel compatibility: elementwise.
    let Kernel::Elementwise { expr, element } = &program.queue[producer].kernel else {
        return None;
    };

    if !same_element_type(*element, consumer_element) {
        return None;
    }
    if !fusible_view_relationship(program, producer, arg_view) {
        return None;
    }

    Some((producer, expr.clone()))
}

/// Datatype compatibility: producer and consumer share an [`Element`].
fn same_element_type(producer: Element, consumer: Element) -> bool {
    producer == consumer
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

/// Drop unreferenced views and buffers, renumbering all references.
///
/// Elementwise expressions hold [`BufferViewRef`]s; those are rewritten too.
fn compact(program: Program) -> Program {
    let mut live_views = vec![false; program.views.len()];
    for &sink in &program.sinks {
        live_views[sink.0] = true;
    }
    for dispatch in &program.queue {
        for &arg in &dispatch.args {
            live_views[arg.0] = true;
        }
        live_views[dispatch.output.0] = true;
        if let Kernel::Elementwise { expr, .. } = &dispatch.kernel {
            for v in expr.loads() {
                live_views[v.0] = true;
            }
        }
    }
    let mut live_buffers = vec![false; program.buffers.len()];
    for &param in &program.params {
        live_buffers[param.0] = true;
    }
    for (index, view) in program.views.iter().enumerate() {
        if live_views[index] {
            live_buffers[view.buffer.0] = true;
        }
    }

    let buffer_map: HashMap<usize, usize> = live_buffers
        .iter()
        .enumerate()
        .filter(|&(_, &live)| live)
        .enumerate()
        .map(|(new, (old, _))| (old, new))
        .collect();
    let view_map: HashMap<usize, usize> = live_views
        .iter()
        .enumerate()
        .filter(|&(_, &live)| live)
        .enumerate()
        .map(|(new, (old, _))| (old, new))
        .collect();

    let map_view = |r: BufferViewRef| BufferViewRef(view_map[&r.0]);
    let map_expr = |expr: Expr| expr.map_loads(&mut |v| Expr::Load(map_view(v)));

    Program {
        params: program.params.iter().map(|p| BufferRef(buffer_map[&p.0])).collect(),
        sinks: program.sinks.iter().map(|s| map_view(*s)).collect(),
        queue: program
            .queue
            .into_iter()
            .map(|d| Dispatch {
                kernel: match d.kernel {
                    Kernel::Elementwise { expr, element } => {
                        Kernel::Elementwise { expr: map_expr(expr), element }
                    }
                    other => other,
                },
                args: d.args.iter().map(|a| map_view(*a)).collect(),
                output: map_view(d.output),
            })
            .collect(),
        buffers: program
            .buffers
            .into_iter()
            .enumerate()
            .filter(|(index, _)| live_buffers[*index])
            .map(|(_, buffer)| buffer)
            .collect(),
        views: program
            .views
            .into_iter()
            .enumerate()
            .filter(|(index, _)| live_views[*index])
            .map(|(_, view)| BufferView {
                buffer: BufferRef(buffer_map[&view.buffer.0]),
                accessor: view.accessor,
            })
            .collect(),
    }
}
