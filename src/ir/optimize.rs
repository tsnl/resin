//! IR→IR optimization passes.
//!
//! Implemented:
//!
//! - **Elementwise RPN fusion** ([`fuse_elementwise`]) — producer→consumer
//!   chains of elementwise dispatches collapse into single kernels, to a
//!   fixed point. The algebraic splice lives on [`RpnExpr::fuse`]; this
//!   module walks the dispatch queue and decides *when* to apply it.
//!
//! Planned:
//!
//! - **Tiled element types** — block matmul as an *elementwise* operation on
//!   16×16 tiles plus a reduction, so it schedules and fuses like everything
//!   else (and can lower to cooperative-matrix WGSL). No dedicated matmul
//!   kernel long-term.
//! - **Matmul epilogues** — fuse a trailing RPN expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

use std::collections::HashMap;

use super::{
    Accessor, BufferRef, BufferView, BufferViewRef, Dispatch, Kernel, Program, RpnAtom, RpnExpr,
};

/// Run middle-end optimization passes on `program`.
pub fn optimize(program: Program) -> Program {
    fuse_elementwise(program)
}

/// Fuse adjacent elementwise dispatches to a fixed point.
///
/// A producer fuses into a consumer when the consumer reads the producer's
/// output buffer through its identity view or a broadcast of it: the
/// producer's RPN expression is spliced into the consumer's
/// ([`RpnExpr::fuse`]) with the producer's argument views composed through
/// the same broadcast. Multi-consumer intermediates are inlined into each
/// reader; the producer is dropped once nothing reads its output. Matmul
/// and reduction kernels are fusion boundaries.
pub fn fuse_elementwise(mut program: Program) -> Program {
    while fuse_one(&mut program) {}
    compact(program)
}

/// Soft cap on fused expression size. Multi-consumer inlining can otherwise
/// duplicate a shared subexpression without bound.
const MAX_FUSED_ATOMS: usize = 256;

/// One fusable producer→consumer edge in the dispatch queue.
struct OuterInnerPair {
    consumer: usize,
    producer: usize,
    /// Consumer arg view of the producer's output (identity or broadcast).
    arg_view: BufferViewRef,
    outer: RpnExpr,
    inner: RpnExpr,
    /// Producer's argument views, before broadcast composition.
    inner_args: Vec<BufferViewRef>,
}

/// Find one producer→consumer pair and fuse it. Returns false at fixed point.
fn fuse_one(program: &mut Program) -> bool {
    let Some(pair) = find_outer_inner_pair(program) else {
        return false;
    };
    apply_fusion(program, pair);
    true
}

/// Scan the queue for the next fusable elementwise→elementwise edge.
fn find_outer_inner_pair(program: &Program) -> Option<OuterInnerPair> {
    for (consumer, dispatch) in program.queue.iter().enumerate() {
        let Kernel::Elementwise(outer) = &dispatch.kernel else {
            continue;
        };
        for &arg_view in &dispatch.args {
            let Some(producer) = fusable_producer(program, consumer, arg_view) else {
                continue;
            };
            let Kernel::Elementwise(inner) = &program.queue[producer].kernel else {
                unreachable!("fusable_producer only returns elementwise dispatches");
            };
            if fused_atom_count(outer, &dispatch.args, arg_view, inner) > MAX_FUSED_ATOMS {
                continue;
            }
            return Some(OuterInnerPair {
                consumer,
                producer,
                arg_view,
                outer: outer.clone(),
                inner: inner.clone(),
                inner_args: program.queue[producer].args.clone(),
            });
        }
    }
    None
}

/// Splice the producer into the consumer and drop the producer if unused.
fn apply_fusion(program: &mut Program, pair: OuterInnerPair) {
    let OuterInnerPair {
        consumer,
        producer,
        arg_view,
        outer,
        inner,
        inner_args,
    } = pair;

    // Re-read each producer arg through the consumer's (possibly broadcast)
    // view of the intermediate.
    let expansion = program.view(arg_view).accessor.clone();
    let composed: Vec<BufferViewRef> = inner_args
        .iter()
        .map(|&a| compose_arg_view(program, a, &expansion))
        .collect();

    let (args, outer_to_new, inner_to_new) =
        merge_arg_lists(&program.queue[consumer].args, arg_view, &composed);
    let fused = outer.fuse(&outer_to_new, &inner, &inner_to_new);

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

/// The queue index of the elementwise dispatch producing `arg_view`, if its
/// expression can be spliced into `queue[consumer]` at that view.
fn fusable_producer(
    program: &Program,
    consumer: usize,
    arg_view: BufferViewRef,
) -> Option<usize> {
    let buffer = program.view(arg_view).buffer;
    let producer = program
        .queue
        .iter()
        .position(|d| program.view(d.output).buffer == buffer)?;
    if producer >= consumer {
        return None;
    }
    if !matches!(program.queue[producer].kernel, Kernel::Elementwise(_)) {
        return None;
    }
    // Producer must write the whole buffer densely; consumer must read it as
    // written or as a broadcast of that layout, so coordinates line up.
    let identity = Accessor::dense(program.buffer(buffer).shape.clone(), 0);
    if program.view(program.queue[producer].output).accessor != identity {
        return None;
    }
    if !is_broadcast_of(&program.view(arg_view).accessor, &identity) {
        return None;
    }
    Some(producer)
}

/// Estimated atom count after splicing `inner` into every use of `replaced`.
fn fused_atom_count(
    outer: &RpnExpr,
    outer_args: &[BufferViewRef],
    replaced: BufferViewRef,
    inner: &RpnExpr,
) -> usize {
    let uses = outer
        .atoms
        .iter()
        .filter(|atom| matches!(atom, RpnAtom::Arg(i) if outer_args[*i as usize] == replaced))
        .count();
    outer.atoms.len() + uses * inner.atoms.len()
}

/// Build the fused argument list and the renumbering tables for [`RpnExpr::fuse`].
///
/// Every outer arg equal to `replaced` is dropped (those slots become `inner`);
/// `inner_args` are appended, with duplicates collapsed.
fn merge_arg_lists(
    outer_args: &[BufferViewRef],
    replaced: BufferViewRef,
    inner_args: &[BufferViewRef],
) -> (Vec<BufferViewRef>, Vec<Option<u32>>, Vec<u32>) {
    let mut merged = Vec::new();
    let mut intern = |view: BufferViewRef| -> u32 {
        if let Some(i) = merged.iter().position(|&a| a == view) {
            return i as u32;
        }
        merged.push(view);
        (merged.len() - 1) as u32
    };

    let outer_to_new: Vec<Option<u32>> = outer_args
        .iter()
        .map(|&a| (a != replaced).then(|| intern(a)))
        .collect();
    let inner_to_new: Vec<u32> = inner_args.iter().map(|&a| intern(a)).collect();
    (merged, outer_to_new, inner_to_new)
}

/// Producer arg `arg`, re-addressed through the consumer's `expansion` of the
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

    Program {
        params: program.params.iter().map(|p| BufferRef(buffer_map[&p.0])).collect(),
        sinks: program.sinks.iter().map(|s| BufferViewRef(view_map[&s.0])).collect(),
        queue: program
            .queue
            .into_iter()
            .map(|d| Dispatch {
                kernel: d.kernel,
                args: d.args.iter().map(|a| BufferViewRef(view_map[&a.0])).collect(),
                output: BufferViewRef(view_map[&d.output.0]),
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
