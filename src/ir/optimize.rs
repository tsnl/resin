//! IR→IR optimization passes.
//!
//! Implemented:
//!
//! - **Elementwise RPN fusion** ([`fuse_elementwise`]) — producer→consumer
//!   chains of elementwise dispatches collapse into single kernels, to a
//!   fixed point.
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

use super::{Accessor, BufferRef, BufferView, BufferViewRef, Dispatch, Kernel, Program, RpnExpr};

/// Run middle-end optimization passes on `program`.
pub fn optimize(program: Program) -> Program {
    fuse_elementwise(program)
}

/// Fuse adjacent elementwise dispatches to a fixed point.
///
/// A producer fuses into a consumer when the consumer reads the producer's
/// output buffer through its identity view or a broadcast of it: the
/// producer's RPN expression is spliced into the consumer's
/// ([`RpnExpr::map_args`]) with the producer's argument views composed
/// through the same broadcast. Producers with several consumers fuse into
/// each of them (recomputing is usually cheaper than a round-trip through
/// memory); a producer's own dispatch is deleted once nothing reads its
/// output. Matmul and reduction kernels are fusion boundaries.
pub fn fuse_elementwise(mut program: Program) -> Program {
    while fuse_one(&mut program) {}
    compact(program)
}

/// Stop splicing into expressions past this size; keeps recomputation of
/// shared subexpressions from snowballing.
const MAX_FUSED_ATOMS: usize = 256;

/// Find one producer→consumer pair and fuse it. Returns false at fixed point.
fn fuse_one(program: &mut Program) -> bool {
    for consumer in 0..program.queue.len() {
        let Kernel::Elementwise(outer) = &program.queue[consumer].kernel else {
            continue;
        };
        let outer = outer.clone();
        for slot_of_interest in 0..program.queue[consumer].args.len() {
            let arg_view = program.queue[consumer].args[slot_of_interest];
            let Some(producer) = fusable_producer(program, consumer, arg_view) else {
                continue;
            };
            let inner_dispatch = program.queue[producer].clone();
            let Kernel::Elementwise(inner) = &inner_dispatch.kernel else {
                unreachable!("fusable_producer only returns elementwise dispatches");
            };

            let occurrences = count_arg_atoms(&outer, &program.queue[consumer].args, arg_view);
            let fused_len = outer.atoms.len() + occurrences * inner.atoms.len();
            if fused_len > MAX_FUSED_ATOMS {
                continue;
            }

            // The producer's args, re-read through the consumer's (possibly
            // broadcast) view of the intermediate.
            let expansion = program.view(arg_view).accessor.clone();
            let composed: Vec<BufferViewRef> = inner_dispatch
                .args
                .iter()
                .map(|&a| {
                    let view = program.view(a).clone();
                    let accessor = compose_broadcast(&view.accessor, &expansion);
                    intern_view(program, view.buffer, accessor)
                })
                .collect();

            // Merge argument lists; duplicates collapse.
            let mut args: Vec<BufferViewRef> = Vec::new();
            let slot = |view: BufferViewRef, args: &mut Vec<BufferViewRef>| -> u32 {
                let index = args.iter().position(|&a| a == view).unwrap_or_else(|| {
                    args.push(view);
                    args.len() - 1
                });
                index as u32
            };
            let outer_slots: Vec<Option<u32>> = program.queue[consumer]
                .args
                .iter()
                .map(|&a| (a != arg_view).then(|| slot(a, &mut args)))
                .collect();
            let inner_slots: Vec<u32> =
                composed.iter().map(|&a| slot(a, &mut args)).collect();

            let fused = outer.map_args(&mut |i| match outer_slots[i as usize] {
                Some(new_index) => RpnExpr::arg(new_index),
                None => inner.map_args(&mut |k| RpnExpr::arg(inner_slots[k as usize])),
            });

            program.queue[consumer] = Dispatch {
                kernel: Kernel::Elementwise(fused),
                args,
                output: program.queue[consumer].output,
            };

            // Delete the producer once its output has no readers left.
            let buffer = program.view(arg_view).buffer;
            let still_read = program
                .queue
                .iter()
                .enumerate()
                .any(|(index, dispatch)| {
                    index != producer
                        && dispatch.args.iter().any(|&a| program.view(a).buffer == buffer)
                })
                || program.sinks.iter().any(|&s| program.view(s).buffer == buffer);
            if !still_read {
                program.queue.remove(producer);
            }
            return true;
        }
    }
    false
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
    // The producer must write the whole buffer densely, and the consumer must
    // read it either as written or through a broadcast of that layout — then
    // consumer coordinates map straight onto producer coordinates.
    let identity = Accessor::dense(program.buffer(buffer).shape.clone(), 0);
    if program.view(program.queue[producer].output).accessor != identity {
        return None;
    }
    if !is_broadcast_of(&program.view(arg_view).accessor, &identity) {
        return None;
    }
    Some(producer)
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

/// How many `Arg` atoms of `expr` refer to `target` (via the arg list).
fn count_arg_atoms(expr: &RpnExpr, args: &[BufferViewRef], target: BufferViewRef) -> usize {
    expr.atoms
        .iter()
        .filter(|atom| match atom {
            super::RpnAtom::Arg(i) => args[*i as usize] == target,
            super::RpnAtom::Op(_) => false,
        })
        .count()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;
    use crate::ir::RpnAtom;

    #[test]
    fn map_args_renumbers() {
        let expr = RpnExpr::apply_op(Op::ADD, 2);
        let renumbered = expr.map_args(&mut |i| RpnExpr::arg(i + 10));
        assert_eq!(
            renumbered.atoms,
            vec![RpnAtom::Arg(10), RpnAtom::Arg(11), RpnAtom::Op(Op::ADD)]
        );
    }

    #[test]
    fn map_args_splices_expressions() {
        // outer = a0 * a1; substitute a1 := (b0 + b1)  →  a0 * (b0 + b1)
        let outer = RpnExpr::apply_op(Op::MUL, 2);
        let inner = RpnExpr::apply_op(Op::ADD, 2);
        let fused = outer.map_args(&mut |i| match i {
            0 => RpnExpr::arg(0),
            1 => inner.map_args(&mut |k| RpnExpr::arg(k + 1)),
            _ => unreachable!(),
        });
        assert_eq!(
            fused.atoms,
            vec![
                RpnAtom::Arg(0),
                RpnAtom::Arg(1),
                RpnAtom::Arg(2),
                RpnAtom::Op(Op::ADD),
                RpnAtom::Op(Op::MUL),
            ]
        );
        fused.validate(3).unwrap();
    }
}
