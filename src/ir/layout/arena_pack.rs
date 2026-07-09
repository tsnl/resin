//! Pack every logical buffer into a small set of **arena** buffers — one per
//! element type present in the program.
//!
//! ## Why
//!
//! WebGPU guarantees only eight storage-buffer bindings per shader stage. A
//! 1:1 GPU buffer per IR buffer blows that budget on fused kernels. Packing
//! does **not** cap fusion; it collapses storage so backends can bind at most
//! one heap per dtype and address regions with views.
//!
//! ## Policy (v1)
//!
//! - **Size = sum of uses**: each logical buffer gets a disjoint slice; no
//!   lifetime reuse / overlapping.
//! - **Sinks and params are views** — both address slices of the arenas after
//!   packing (no private buffer required for either).
//! - Constants' `init` data is copied into the arena at the buffer's base.
//!
//! Call after fusion so fine-grained temps still fuse first; packing only
//! rewrites storage.

use crate::ir::{
    Accessor, Buffer, BufferData, BufferRef, BufferView, Dispatch, Kernel, Program, RemapInfo,
};
use crate::ops::ElementType;

/// Rewrite `program` so `buffers` is one arena per element type and every
/// view (params, sinks, dispatch args/outputs, scatter-view accessors)
/// addresses absolute offsets into those arenas.
pub fn pack_arenas(program: Program) -> Program {
    if program.buffers.is_empty() {
        return program;
    }

    // Already packed: at most one buffer per etype and all etypes unique.
    if is_packed(&program) {
        return program;
    }

    // Stable arena order: first-seen element types.
    let mut arena_etypes: Vec<ElementType> = Vec::new();
    for b in &program.buffers {
        if !arena_etypes.contains(&b.element_type) {
            arena_etypes.push(b.element_type);
        }
    }

    // base_offset[old_buffer] = element index in its arena.
    let mut base_offset = vec![0usize; program.buffers.len()];
    let mut arena_len = vec![0usize; arena_etypes.len()];
    for (i, b) in program.buffers.iter().enumerate() {
        let a = arena_index(&arena_etypes, b.element_type);
        base_offset[i] = arena_len[a];
        arena_len[a] += b.len();
    }

    let new_buffers: Vec<Buffer> = arena_etypes
        .iter()
        .enumerate()
        .map(|(a, &e)| {
            let len = arena_len[a].max(1);
            let mut data = match e {
                ElementType::F32 => BufferData::F32(vec![0.0; len].into()),
                ElementType::U32 => BufferData::U32(vec![0; len].into()),
            };
            let mut any_const = false;
            for (i, b) in program.buffers.iter().enumerate() {
                if b.element_type != e {
                    continue;
                }
                let base = base_offset[i];
                if let Some(init) = &b.init {
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
                element_type: e,
                init: if any_const { Some(data) } else { None },
            }
        })
        .collect();

    let arena_of = |old: BufferRef| -> BufferRef {
        BufferRef(arena_index(&arena_etypes, program.buffers[old.0].element_type))
    };
    let shift = |old: BufferRef, acc: &Accessor| -> Accessor {
        let mut a = acc.clone();
        a.offset += base_offset[old.0];
        a
    };

    // Params and sinks keep their view indices; only view bodies move.
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
                Kernel::Remap {
                    info: RemapInfo::ScatterView { accessor },
                } => Kernel::Remap {
                    info: RemapInfo::ScatterView { accessor: shift(out_buf, &accessor) },
                },
                other => other,
            };
            Dispatch { kernel, args: d.args, output: d.output }
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

fn is_packed(program: &Program) -> bool {
    let mut seen = Vec::new();
    for b in &program.buffers {
        if seen.contains(&b.element_type) {
            return false;
        }
        seen.push(b.element_type);
    }
    true
}

fn arena_index(etypes: &[ElementType], e: ElementType) -> usize {
    etypes.iter().position(|&x| x == e).expect("arena etype")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Tensor;
    use crate::jit::lower::lower;
    use crate::ops::ElementType;

    #[derive(resin_macros::Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn pack_collapses_f32_to_one_arena() {
        let a = Tensor::parameter(&[4]);
        let b = Tensor::parameter(&[4]);
        let out = a.clone() + b.clone();
        let raw = lower(&Pair { a, b }, &out).unwrap();
        assert!(raw.buffers.len() >= 3, "params + output");
        let packed = pack_arenas(raw);
        assert_eq!(packed.buffers.len(), 1);
        assert_eq!(packed.buffers[0].element_type, ElementType::F32);
        assert!(packed.buffers[0].len() >= 12);
        packed.validate().unwrap();
    }

    #[test]
    fn pack_two_dtypes_two_arenas() {
        let x = Tensor::parameter(&[4]);
        let k = Tensor::parameter_typed(&[4], ElementType::U32);
        let out = x.cast(ElementType::U32) + k.clone();
        let raw = lower(&Pair { a: x, b: k }, &out).unwrap();
        let packed = pack_arenas(raw);
        assert_eq!(packed.buffers.len(), 2);
        packed.validate().unwrap();
    }

    #[test]
    fn sinks_are_views_into_arenas() {
        let a = Tensor::parameter(&[2]);
        let out = a.clone() * a.clone();
        let packed = pack_arenas(lower(&a, &out).unwrap());
        let sink = packed.view(packed.sinks[0]);
        assert_eq!(sink.buffer.0, 0);
        assert!(sink.accessor.offset + 2 <= packed.buffers[0].len());
    }

    #[test]
    fn pack_is_idempotent() {
        let a = Tensor::parameter(&[4]);
        let out = a.clone() + a.clone();
        let once = pack_arenas(lower(&a, &out).unwrap());
        let twice = pack_arenas(once.clone());
        assert_eq!(once.buffers.len(), twice.buffers.len());
        assert_eq!(once.views, twice.views);
    }
}
