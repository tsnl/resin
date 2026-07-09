//! Materialize strided sink views into dense buffers.
//!
//! Backends copy each sink out as one contiguous block, so a sink that is a
//! strided view — a function returning a bare transpose, broadcast, or
//! mid-stride slice — has no dense region to copy. This pass appends an
//! identity elementwise copy for each such sink and repoints the sink at the
//! fresh dense output.
//!
//! Dense-with-offset sinks (e.g. a contiguous slice) are left alone: backends
//! address those by offset. Without this the CPU interpreter gathers strided
//! sinks fine while the wgpu backend rejects them — this keeps them equal.

use crate::ir::{Accessor, Buffer, BufferRef, BufferView, BufferViewRef, Dispatch, Expr, Kernel, Program};

/// Rewrite `program` so every sink view is C-contiguous, copying strided ones
/// into a fresh dense buffer. Call before [`super::eliminate_dead`].
///
/// Needed even though every *kernel output* is dense (lowering interns
/// `Accessor::dense` for each dispatch): a sink need not be a kernel output.
/// When a traced function ends in a view node — transpose, broadcast, a
/// mid-stride slice — lowering composes the accessor over the underlying buffer
/// and returns that strided view directly; no kernel runs. The CPU interpreter
/// gathers such a sink at arbitrary strides, but the wgpu backend copies each
/// sink out as one contiguous block and has nothing dense to copy. Giving the
/// view its own dense buffer closes that gap.
pub fn densify_sinks(mut program: Program) -> Program {
    for i in 0..program.sinks.len() {
        let sink = program.sinks[i];
        let (shape, etype) = {
            let view = program.view(sink);
            if view.accessor.is_dense() {
                continue;
            }
            (view.accessor.shape.clone(), program.buffer(view.buffer).element_type)
        };

        // Fresh dense buffer + view for the copy's output.
        let buffer = BufferRef(program.buffers.len());
        program.buffers.push(Buffer {
            shape: shape.clone(),
            element_type: etype,
            init: None,
            atomic: false,
        });
        let output = BufferViewRef(program.views.len());
        program.views.push(BufferView { buffer, accessor: Accessor::dense(shape, 0) });

        // Identity copy: read the strided sink, write the dense buffer.
        program.queue.push(Dispatch {
            kernel: Kernel::elementwise(Expr::Load(sink)),
            args: vec![sink],
            output,
        });
        program.sinks[i] = output;
    }
    program
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Tensor;
    use crate::jit::lower::lower;

    #[test]
    fn transpose_sink_is_materialized() {
        // A bare transpose output is a strided sink.
        let x = Tensor::parameter(&[2, 3]);
        let out = x.transpose();
        let raw = lower(&x, &out).unwrap();
        assert!(!raw.view(raw.sinks[0]).accessor.is_dense());

        let densified = densify_sinks(raw);
        assert!(densified.view(densified.sinks[0]).accessor.is_dense());
        densified.validate().unwrap();
    }

    #[test]
    fn dense_sink_is_untouched() {
        let a = Tensor::parameter(&[4]);
        let out = a.clone() + a.clone();
        let raw = lower(&a, &out).unwrap();
        let before = raw.queue.len();
        let densified = densify_sinks(raw);
        assert_eq!(densified.queue.len(), before, "no copy added for a dense sink");
    }
}
