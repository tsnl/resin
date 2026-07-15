//! Pack every logical buffer into a small set of **arena** buffers.
//!
//! ## Why
//!
//! WebGPU guarantees only eight storage-buffer bindings per shader stage. A
//! 1:1 GPU buffer per IR temporary blows that budget on fused kernels. Packing
//! collapses storage so backends bind O(dtypes × {plain, atomic}) heaps and
//! address regions with views.
//!
//! ## Policy (v1)
//!
//! - **Arenas keyed by `(element_type, atomic)`** — plain storage and RMW
//!   targets never share a heap. Atomics are not a separate IR buffer *kind*;
//!   buffers that are written by atomic scatter-add are **placed** in an
//!   atomic arena so WGSL can declare `array<atomic<u32>>` without forcing
//!   every plain load of that dtype through atomics.
//! - **Size = sum of uses** within each arena; no lifetime reuse.
//! - **Sinks and params are views** into arenas.
//!
//! Call after dead-elim so only live buffers are packed.

use crate::ir::{
    Accessor, Buffer, BufferData, BufferRef, BufferView, Dispatch, Kernel, Program, RemapInfo,
};
use crate::ops::ElementType;

/// Arena identity: element type plus whether GPU storage is atomic-typed.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct ArenaKey {
    etype: ElementType,
    atomic: bool,
}

/// Rewrite `program` so `buffers` is one arena per [`ArenaKey`] and every
/// view addresses absolute offsets into those arenas.
pub fn pack_arenas(program: Program) -> Program {
    if program.buffers.is_empty() {
        return program;
    }
    if is_packed(&program) {
        return program;
    }

    // Logical buffers that are outputs of atomic scatter-add (need RMW storage).
    let mut atomic_target = vec![false; program.buffers.len()];
    for dispatch in &program.queue {
        if let Kernel::Remap {
            info: RemapInfo::ScatterRows { operator: Some(_) },
        } = &dispatch.kernel
        {
            let out_buf = program.view(dispatch.output).buffer.0;
            atomic_target[out_buf] = true;
        }
    }

    // Stable arena order: first-seen (etype, atomic) keys.
    let mut arena_keys: Vec<ArenaKey> = Vec::new();
    let key_of = |i: usize| ArenaKey {
        etype: program.buffers[i].element_type,
        atomic: atomic_target[i],
    };
    for i in 0..program.buffers.len() {
        let k = key_of(i);
        if !arena_keys.contains(&k) {
            arena_keys.push(k);
        }
    }

    let mut base_offset = vec![0usize; program.buffers.len()];
    let mut arena_len = vec![0usize; arena_keys.len()];
    for i in 0..program.buffers.len() {
        let a = arena_keys.iter().position(|&k| k == key_of(i)).unwrap();
        base_offset[i] = arena_len[a];
        arena_len[a] += program.buffers[i].len();
    }

    let new_buffers: Vec<Buffer> = arena_keys
        .iter()
        .enumerate()
        .map(|(a, &key)| {
            let len = arena_len[a].max(1);
            let mut data = match key.etype {
                ElementType::F32 => BufferData::F32(vec![0.0; len].into()),
                ElementType::U32 => BufferData::U32(vec![0; len].into()),
            };
            let mut any_const = false;
            for i in 0..program.buffers.len() {
                if key_of(i) != key {
                    continue;
                }
                let base = base_offset[i];
                if let Some(init) = &program.buffers[i].init {
                    any_const = true;
                    match (&mut data, init) {
                        (BufferData::F32(dst), BufferData::F32(src)) => {
                            dst[base..base + src.len()].copy_from_slice(src);
                        }
                        (BufferData::U32(dst), BufferData::U32(src)) => {
                            dst[base..base + src.len()].copy_from_slice(src);
                        }
                        _ => panic!("arena_pack: init etype mismatch"),
                    }
                }
            }
            Buffer {
                shape: Box::from([len]),
                element_type: key.etype,
                init: if any_const { Some(data) } else { None },
                atomic: key.atomic,
            }
        })
        .collect();

    let arena_of = |old: BufferRef| -> BufferRef {
        let k = key_of(old.0);
        BufferRef(arena_keys.iter().position(|&x| x == k).unwrap())
    };
    let shift = |old: BufferRef, acc: &Accessor| -> Accessor {
        let mut a = acc.clone();
        a.offset += base_offset[old.0];
        a
    };

    let views: Vec<BufferView> = program
        .views
        .iter()
        .map(|v| BufferView {
            buffer: arena_of(v.buffer),
            accessor: shift(v.buffer, &v.accessor),
        })
        .collect();

    let old_views = &program.views;
    let new_queue: Vec<Dispatch> = program
        .queue
        .into_iter()
        .map(|d| {
            let out_buf = old_views[d.output.0].buffer;
            let kernel = match d.kernel {
                // Scatter map is buffer-absolute on the dense output arena.
                Kernel::Remap {
                    info: RemapInfo::ScatterView { map },
                } => Kernel::Remap {
                    info: RemapInfo::ScatterView {
                        map: shift(out_buf, &map),
                    },
                },
                // Gather map is a dense-logical index into the source *view*
                // shape (decoded through the source accessor at run time) — not
                // a raw buffer address, so packing must not shift it.
                other => other,
            };
            Dispatch {
                kernel,
                args: d.args,
                output: d.output,
            }
        })
        .collect();

    Program {
        params: program.params,
        sinks: program.sinks,
        queue: new_queue,
        buffers: new_buffers,
        views,
    }
}

/// Packed iff at most one buffer per `(etype, atomic)` key.
fn is_packed(program: &Program) -> bool {
    let mut seen: Vec<ArenaKey> = Vec::new();
    for b in &program.buffers {
        let k = ArenaKey {
            etype: b.element_type,
            atomic: b.atomic,
        };
        if seen.contains(&k) {
            return false;
        }
        seen.push(k);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::{ScatterOp, Tensor};
    use crate::ir::layout::prepare_for_backend;
    use crate::ir::lower;
    use crate::ops::ElementType;

    #[derive(resin_macros::Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn pack_collapses_f32_to_one_plain_arena() {
        let a = Tensor::parameter(&[4]);
        let b = Tensor::parameter(&[4]);
        let out = a.clone() + b.clone();
        let raw = lower(&Pair { a, b }, &out).unwrap();
        assert!(raw.buffers.len() >= 3);
        let packed = pack_arenas(raw);
        assert_eq!(packed.buffers.len(), 1);
        assert!(!packed.buffers[0].atomic);
        assert!(packed.buffers[0].len() >= 12);
        packed.validate().unwrap();
    }

    #[test]
    fn atomic_scatter_target_gets_separate_arena() {
        let x = Tensor::parameter(&[4]);
        let indices = Tensor::constant_u32(&[4], &[0, 1, 0, 2]);
        let out = x.scatter_rows(&indices, 3, ScatterOp::Add);
        let packed = prepare_for_backend(lower(&x, &out).unwrap());
        let atomics: Vec<_> = packed.buffers.iter().filter(|b| b.atomic).collect();
        let plains: Vec<_> = packed.buffers.iter().filter(|b| !b.atomic).collect();
        assert_eq!(atomics.len(), 1, "one atomic f32 arena for scatter-add out");
        assert_eq!(atomics[0].element_type, ElementType::F32);
        // Source x and maybe indices constant live in plain arenas.
        assert!(plains.iter().any(|b| b.element_type == ElementType::F32));
        assert!(plains.iter().any(|b| b.element_type == ElementType::U32));
        // Sink points at the atomic arena.
        let sink = packed.view(packed.sinks[0]);
        assert!(packed.buffer(sink.buffer).atomic);
        packed.validate().unwrap();
    }

    #[test]
    fn pack_two_dtypes_two_plain_arenas() {
        let x = Tensor::parameter(&[4]);
        let k = Tensor::parameter_typed(&[4], ElementType::U32);
        let out = x.cast(ElementType::U32) + k.clone();
        let packed = pack_arenas(lower(&Pair { a: x, b: k }, &out).unwrap());
        assert_eq!(packed.buffers.len(), 2);
        assert!(packed.buffers.iter().all(|b| !b.atomic));
        packed.validate().unwrap();
    }

    #[test]
    fn pack_is_idempotent() {
        let a = Tensor::parameter(&[4]);
        let out = a.clone() + a.clone();
        let once = pack_arenas(lower(&a, &out).unwrap());
        let twice = pack_arenas(once.clone());
        assert_eq!(once.buffers, twice.buffers);
        assert_eq!(once.views, twice.views);
    }
}
