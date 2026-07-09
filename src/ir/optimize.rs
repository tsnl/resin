//! IR→IR optimization passes.
//!
//! Implemented:
//!
//! - **Elementwise fusion** ([`fuse_elementwise`]) — producer→consumer chains
//!   of elementwise dispatches collapse into single kernels, to a fixed
//!   point. Fusion is tree substitution ([`Expr::substitute`]); this module
//!   walks the dispatch queue and decides *when* to apply it.
//!
//! Planned:
//!
//! - **Tiled element types** — block matmul as an *elementwise* operation on
//!   16×16 tiles plus a reduction, so it schedules and fuses like everything
//!   else (and can lower to cooperative-matrix WGSL). No dedicated matmul
//!   kernel long-term.
//! - **Matmul epilogues** — fuse a trailing expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

use std::collections::HashMap;

use super::{
    Accessor, BufferRef, BufferView, BufferViewRef, Dispatch, Expr, Kernel, Program,
};

/// Run middle-end optimization passes on `program`.
pub fn optimize(program: Program) -> Program {
    fuse_elementwise(program)
}

/// Fuse adjacent elementwise dispatches to a fixed point.
///
/// A producer fuses into a consumer when the consumer loads the producer's
/// output buffer through its identity view or a broadcast of it: the
/// producer's expression is substituted for that load ([`Expr::substitute`]),
/// with the producer's own loads composed through the same broadcast.
/// Multi-consumer intermediates are inlined into each reader; the producer is
/// dropped once nothing reads its output. Matmul and reduction are fusion
/// boundaries.
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
        let Kernel::Elementwise(consumer_expr) = &dispatch.kernel else {
            continue;
        };
        for &arg_view in &dispatch.args {
            let Some((producer, producer_expr)) = fusable_producer(program, consumer, arg_view) else {
                continue;
            };
            // Each use of the intermediate becomes a full copy of `producer_expr`.
            let uses = count_load_occurrences(consumer_expr, arg_view);
            let fused_nodes = consumer_expr.node_count() + uses * producer_expr.node_count();
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

    // Re-read each producer load through the consumer's (possibly broadcast)
    // view of the intermediate.
    let expansion = program.view(arg_view).accessor.clone();
    let producer_expr = producer_expr.map_loads(&mut |v| {
        Expr::Load(compose_arg_view(program, v, &expansion))
    });

    let fused = consumer_expr.substitute(arg_view, &producer_expr);
    let args = fused.loads();

    program.queue[consumer] = Dispatch {
        kernel: Kernel::Elementwise(fused),
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

/// Index and body of the elementwise dispatch producing `arg_view`, if it
/// can be inlined into `queue[consumer]` at that load.
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
    let Kernel::Elementwise(expr) = &program.queue[producer].kernel else {
        return None;
    };
    // Producer must write the whole buffer densely; consumer must read it as
    // written or as a broadcast of that layout, so coordinates line up.
    let identity = Accessor::dense(program.buffer(buffer).shape.clone(), 0);
    if program.view(program.queue[producer].output).accessor != identity {
        return None;
    }
    if !is_broadcast_of(&program.view(arg_view).accessor, &identity) {
        return None;
    }
    Some((producer, expr.clone()))
}

/// Producer load `arg`, re-addressed through the consumer's `expansion` of the
/// intermediate (identity or broadcast).
fn compose_arg_view(
    program: &mut Program,
    arg: BufferViewRef,
    expansion: &Accessor,
) -> BufferViewRef {
    let view = program.view(arg).clone();
    let accessor = compose_broadcast(&view.accessor, expansion);
    intern_view(program, view.buffer, accessor)
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

/// Whether `accessor` is `source` expanded by [`Accessor::broadcast_to`]:
/// trailing-aligned, with source pitches kept and everything else pitch 0.
fn is_broadcast_of(accessor: &Accessor, source: &Accessor) -> bool {
    if accessor.offset != source.offset || accessor.rank() < source.rank() {
        return false;
    }
    let lead = accessor.rank() - source.rank();
    if accessor.pitch[..lead].iter().any(|&p| p != 0) {
        return false;
    }
    source.shape.iter().zip(&source.pitch).enumerate().all(|(i, (&dim, &pitch))| {
        let (out_dim, out_pitch) = (accessor.shape[lead + i], accessor.pitch[lead + i]);
        (out_dim == dim && out_pitch == pitch) || (dim == 1 && out_pitch == 0)
    })
}

/// Re-read `accessor` (over the producer's iteration space) through the
/// consumer's `expansion` of that space: pitches follow expanded axes, and
/// broadcast axes stay pitch 0.
fn compose_broadcast(accessor: &Accessor, expansion: &Accessor) -> Accessor {
    let lead = expansion.rank() - accessor.rank();
    let mut pitch = vec![0; expansion.rank()];
    for (i, &p) in accessor.pitch.iter().enumerate() {
        if expansion.pitch[lead + i] != 0 {
            pitch[lead + i] = p;
        }
    }
    Accessor {
        offset: accessor.offset,
        shape: expansion.shape.clone(),
        pitch: pitch.into(),
    }
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
        if let Kernel::Elementwise(expr) = &dispatch.kernel {
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
    let map_expr = |expr: Expr| {
        expr.map_loads(&mut |v| Expr::Load(map_view(v)))
    };

    Program {
        params: program.params.iter().map(|p| BufferRef(buffer_map[&p.0])).collect(),
        sinks: program.sinks.iter().map(|s| map_view(*s)).collect(),
        queue: program
            .queue
            .into_iter()
            .map(|d| Dispatch {
                kernel: match d.kernel {
                    Kernel::Elementwise(expr) => Kernel::Elementwise(map_expr(expr)),
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
