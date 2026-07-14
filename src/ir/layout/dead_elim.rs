//! Drop unreferenced views and buffers, renumbering all references.
//!
//! Tree-shakes the IR after optimization (and as the last step of fusion) so
//! arena packing only sees live storage. Elementwise expressions hold
//! [`BufferViewRef`]s; those are rewritten with the view map.

use std::collections::HashMap;

use crate::ir::{BufferRef, BufferView, BufferViewRef, Dispatch, Expr, Kernel, Program};

/// Keep only views/buffers reachable from params, sinks, or the dispatch queue.
pub fn eliminate_dead(program: Program) -> Program {
    let mut live_views = vec![false; program.views.len()];
    for &param in &program.params {
        live_views[param.0] = true;
    }
    for &sink in &program.sinks {
        live_views[sink.0] = true;
    }
    for dispatch in &program.queue {
        for &arg in &dispatch.args {
            live_views[arg.0] = true;
        }
        live_views[dispatch.output.0] = true;
        if let Kernel::Elementwise { expr } = &dispatch.kernel {
            for v in expr.loads() {
                live_views[v.0] = true;
            }
        }
    }

    let mut live_buffers = vec![false; program.buffers.len()];
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
        params: program.params.iter().map(|p| map_view(*p)).collect(),
        sinks: program.sinks.iter().map(|s| map_view(*s)).collect(),
        queue: program
            .queue
            .into_iter()
            .map(|d| Dispatch {
                kernel: match d.kernel {
                    Kernel::Elementwise { expr } => Kernel::Elementwise {
                        expr: map_expr(expr),
                    },
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
