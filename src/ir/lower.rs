//! Lower a traced DSL graph to an IR program.
//!
//! **Kernels write buffers; views are accessors.** Materializing nodes
//! (constants, parameters, elementwise, matmul, reduction, remap) allocate a
//! buffer and enqueue a dispatch. View nodes (broadcast, permute, unfold,
//! squeeze, index) compose an [`Accessor`] over an existing buffer and cost
//! nothing at run time.
//!
//! **Sinks are dense.** A traced function may end in a strided view (permute,
//! unfold, broadcast, mid-stride slice). Backends copy each sink out as one
//! contiguous block, so lowering densifies non-dense sinks with an identity
//! elementwise copy up-front — in the program graph, before optimize — so
//! fusion (and later passes) can see and potentially elide that copy.
//! Dense-with-offset sinks (e.g. a contiguous slice) are already dense and
//! left alone.

use std::collections::HashMap;

use super::{
    Accessor, Buffer, BufferData, BufferRef, BufferView, BufferViewRef, Dispatch, Error, Expr,
    Kernel, Program, RemapInfo,
};
use crate::dsl::{ConstantData, IndexKeyElement, Remap, ScatterOp, Tensor, TensorKind};
use crate::ops::{AssocOp, ElementType};
use crate::tree::Tree;

/// Lower `sinks` (traced outputs) to an IR program whose parameter slots
/// follow `params`' leaf order. Every sink view is C-contiguous afterwards.
pub fn lower(params: &impl Tree<Tensor>, sinks: &impl Tree<Tensor>) -> Result<Program, Error> {
    let mut builder = Builder::default();
    let mut program = Program {
        params: Vec::new(),
        sinks: Vec::new(),
        queue: Vec::new(),
        buffers: Vec::new(),
        views: Vec::new(),
    };
    for tensor in params.leaves() {
        // Params are dense views of their buffers so arena packing can place
        // them as slices without a separate param-buffer identity.
        let view = builder.view_for(tensor)?;
        program.params.push(view);
    }
    for tensor in sinks.leaves() {
        let view = builder.view_for(tensor)?;
        program.sinks.push(builder.ensure_dense(view));
    }
    program.queue = builder.queue;
    program.buffers = builder.buffers;
    program.views = builder.views;
    program.validate()?;
    Ok(program)
}

#[derive(Default)]
struct Builder {
    buffers: Vec<Buffer>,
    views: Vec<BufferView>,
    queue: Vec<Dispatch>,
    /// Memo for materialized tensors (each owns one buffer).
    materialized: HashMap<Tensor, BufferRef>,
    view_memo: HashMap<BufferView, BufferViewRef>,
}

impl Builder {
    /// The buffer backing a materializing tensor, allocating and enqueuing
    /// its kernel on first visit. View nodes are rejected: they have no
    /// storage of their own.
    fn buffer_for(&mut self, tensor: &Tensor) -> Result<BufferRef, Error> {
        if let Some(&buffer) = self.materialized.get(tensor) {
            return Ok(buffer);
        }

        let etype = tensor.element_type();
        let buffer = match tensor.kind() {
            TensorKind::Constant { values } => {
                let init = match values {
                    ConstantData::F32(v) => BufferData::F32(v.clone()),
                    ConstantData::U32(v) => BufferData::U32(v.clone()),
                };
                self.push_buffer(Buffer {
                    shape: tensor.shape().into(),
                    element_type: etype,
                    init: Some(init),
                    atomic: false,
                })
            }
            TensorKind::Parameter => self.push_buffer(Buffer {
                shape: tensor.shape().into(),
                element_type: etype,
                init: None,
                atomic: false,
            }),
            TensorKind::Elementwise { op, args } => {
                // Resolve each arg, then align it to the output shape
                // (NumPy trailing broadcast) so the kernel iterates uniformly.
                let arg_views = args
                    .iter()
                    .map(|arg| {
                        let (buffer, accessor) = self.resolve_view(arg)?;
                        let aligned = accessor.broadcast_to(tensor.shape())?;
                        Ok(self.intern_view(buffer, aligned))
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                let expr = Expr::new_op(*op, arg_views.iter().copied().map(Expr::Load));
                // Binding list is free loads (deduped); the tree may load one view twice.
                let loads = expr.loads();
                self.push_dispatch(tensor.shape(), etype, Kernel::elementwise(expr), loads)
            }
            TensorKind::Matmul { lhs, rhs } => {
                let args = vec![self.view_for(lhs)?, self.view_for(rhs)?];
                self.push_dispatch(tensor.shape(), etype, Kernel::Matmul, args)
            }
            TensorKind::Reduction { op, axes, arg } => {
                let args = vec![self.view_for(arg)?];
                let kernel = Kernel::Reduction {
                    op: *op,
                    axes: axes.clone(),
                };
                self.push_dispatch(tensor.shape(), etype, kernel, args)
            }
            TensorKind::Remap(remap) => {
                let (info, args) = match remap {
                    Remap::GatherRows { source, indices } => (
                        RemapInfo::GatherRows,
                        vec![self.view_for(source)?, self.view_for(indices)?],
                    ),
                    Remap::ScatterRows {
                        source,
                        indices,
                        op,
                        ..
                    } => {
                        let operator = match op {
                            ScatterOp::Write => None,
                            ScatterOp::Add => Some(AssocOp::Add),
                        };
                        (
                            RemapInfo::ScatterRows { operator },
                            vec![self.view_for(source)?, self.view_for(indices)?],
                        )
                    }
                    Remap::ScatterView {
                        source, map, op, ..
                    } => {
                        let operator = match op {
                            ScatterOp::Write => None,
                            ScatterOp::Add => Some(AssocOp::Add),
                        };
                        (
                            RemapInfo::ScatterView {
                                map: map.clone(),
                                operator,
                            },
                            vec![self.view_for(source)?],
                        )
                    }
                    Remap::GatherView { source, map } => (
                        RemapInfo::GatherView { map: map.clone() },
                        vec![self.view_for(source)?],
                    ),
                };
                self.push_dispatch(tensor.shape(), etype, Kernel::Remap { info }, args)
            }
            TensorKind::Broadcast { .. }
            | TensorKind::Permute { .. }
            | TensorKind::Unfold { .. }
            | TensorKind::Squeeze { .. }
            | TensorKind::Index { .. } => {
                return Err(Error("view node has no buffer".into()));
            }
        };
        self.materialized.insert(tensor.clone(), buffer);
        Ok(buffer)
    }

    /// A logical view of `tensor`: its backing buffer plus a composed accessor.
    fn view_for(&mut self, tensor: &Tensor) -> Result<BufferViewRef, Error> {
        let (buffer, accessor) = self.resolve_view(tensor)?;
        Ok(self.intern_view(buffer, accessor))
    }

    /// Walk view-node chains, composing accessors; materialize at the first
    /// non-view node.
    fn resolve_view(&mut self, tensor: &Tensor) -> Result<(BufferRef, Accessor), Error> {
        match tensor.kind() {
            TensorKind::Broadcast { arg, axes } => {
                let (buffer, accessor) = self.resolve_view(arg)?;
                Ok((buffer, accessor.map_axes(tensor.shape(), axes)?))
            }
            TensorKind::Permute { arg, axes } => {
                let (buffer, accessor) = self.resolve_view(arg)?;
                Ok((buffer, accessor.permute(axes)?))
            }
            TensorKind::Unfold {
                arg,
                axis,
                size,
                step,
            } => {
                let (buffer, accessor) = self.resolve_view(arg)?;
                Ok((buffer, accessor.unfold(*axis, *size, *step)?))
            }
            TensorKind::Squeeze { arg, axes } => {
                let (buffer, accessor) = self.resolve_view(arg)?;
                Ok((buffer, accessor.squeeze(axes)?))
            }
            TensorKind::Index { arg, key } => {
                let (buffer, accessor) = self.resolve_view(arg)?;
                Ok((buffer, compose_index(&accessor, key)))
            }
            _ => {
                let buffer = self.buffer_for(tensor)?;
                Ok((buffer, Accessor::dense(tensor.shape(), 0)))
            }
        }
    }

    /// If `view` is not C-contiguous, enqueue an identity copy into a fresh
    /// dense buffer and return that output. Already-dense views (including
    /// dense-with-offset) are returned unchanged.
    fn ensure_dense(&mut self, view: BufferViewRef) -> BufferViewRef {
        let (shape, etype) = {
            let v = &self.views[view.0];
            if v.accessor.is_dense() {
                return view;
            }
            (
                v.accessor.shape.clone(),
                self.buffers[v.buffer.0].element_type,
            )
        };
        let buffer = self.push_dispatch(
            &shape,
            etype,
            Kernel::elementwise(Expr::Load(view)),
            vec![view],
        );
        self.intern_view(buffer, Accessor::dense(shape, 0))
    }

    fn intern_view(&mut self, buffer: BufferRef, accessor: Accessor) -> BufferViewRef {
        let view = BufferView { buffer, accessor };
        if let Some(&r) = self.view_memo.get(&view) {
            return r;
        }
        let r = BufferViewRef(self.views.len());
        self.views.push(view.clone());
        self.view_memo.insert(view, r);
        r
    }

    fn push_buffer(&mut self, buffer: Buffer) -> BufferRef {
        self.buffers.push(buffer);
        BufferRef(self.buffers.len() - 1)
    }

    /// Allocate an output buffer and enqueue `kernel` writing a dense view of it.
    fn push_dispatch(
        &mut self,
        shape: &[usize],
        element_type: ElementType,
        kernel: Kernel,
        args: Vec<BufferViewRef>,
    ) -> BufferRef {
        let buffer = self.push_buffer(Buffer {
            shape: shape.into(),
            element_type,
            init: None,
            atomic: false,
        });
        let output = self.intern_view(buffer, Accessor::dense(shape, 0));
        self.queue.push(Dispatch {
            kernel,
            args,
            output,
        });
        buffer
    }
}

/// Compose a static index key onto an accessor: pure offset/shape arithmetic.
/// `Single` keeps a size-1 axis (rank is preserved).
fn compose_index(arg_acc: &Accessor, key: &[IndexKeyElement]) -> Accessor {
    debug_assert_eq!(key.len(), arg_acc.rank(), "index key covers every axis");
    let mut offset = arg_acc.offset;
    let mut shape = Vec::with_capacity(key.len());
    for (axis, element) in key.iter().enumerate() {
        let (start, len) = match element {
            IndexKeyElement::Single(i) => (*i, 1usize),
            IndexKeyElement::Slice(range) => (range.start, range.end - range.start),
        };
        offset += start * arg_acc.stride[axis];
        shape.push(len);
    }
    Accessor {
        offset,
        shape: shape.into(),
        stride: arg_acc.stride.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    #[derive(resin_macros::Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn lower_add_of_params() {
        let a = Tensor::parameter(&[2, 3]);
        let b = Tensor::parameter(&[2, 3]);
        let out = a.clone() + b.clone();

        let program = lower(&Pair { a, b }, &out).unwrap();
        assert_eq!(program.buffers.len(), 3);
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(
            program.queue[0].kernel,
            Kernel::Elementwise { .. }
        ));
    }

    #[test]
    fn lower_shared_operand_materializes_once() {
        let a = Tensor::parameter(&[4]);
        let out = a.clone() + a.clone();
        let program = lower(&a, &out).unwrap();

        assert_eq!(program.buffers.len(), 2);
        assert_eq!(program.queue.len(), 1);
        let Kernel::Elementwise { expr, .. } = &program.queue[0].kernel else {
            panic!("expected elementwise");
        };
        assert!(matches!(expr, Expr::Op { op: Op::ADD, .. }));
        // Same view loaded twice; free-load list collapses to one arg.
        assert_eq!(expr.loads().len(), 1);
        assert_eq!(program.queue[0].args.len(), 1);
    }

    #[test]
    fn lower_constant_has_init() {
        let c = Tensor::zeros(&[]);
        let program = lower(&c, &c).unwrap();
        assert_eq!(program.buffers.len(), 1);
        assert!(program.buffers[0].init.is_some());
        assert_eq!(program.queue.len(), 0);
    }

    #[test]
    fn lower_matmul() {
        let a = Tensor::parameter(&[2, 4]);
        let b = Tensor::parameter(&[4, 3]);
        let out = a.matmul(&b);

        let program = lower(&Pair { a, b }, &out).unwrap();
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(program.queue[0].kernel, Kernel::Matmul));
        assert_eq!(&*program.output_shape(&program.queue[0]), &[2, 3]);
    }

    #[test]
    fn lower_reduction_arg_keeps_input_shape() {
        let a = Tensor::parameter(&[8, 10]);
        let out = a.sum_axes(&[0]);
        let program = lower(&a, &out).unwrap();

        let dispatch = &program.queue[0];
        assert!(matches!(dispatch.kernel, Kernel::Reduction { .. }));
        assert_eq!(&*program.view(dispatch.args[0]).accessor.shape, &[8, 10]);
        assert_eq!(&*program.output_shape(dispatch), &[1, 10]);
    }

    #[test]
    fn lower_scalar_mul_broadcasts_arg_accessor() {
        let w = Tensor::parameter(&[4, 3]);
        let out = w.clone() * Tensor::scalar(0.5);
        let program = lower(&w, &out).unwrap();

        let arg1 = program.view(program.queue[0].args[1]);
        assert_eq!(&*arg1.accessor.shape, &[4, 3]);
        assert_eq!(&*arg1.accessor.stride, &[0, 0]);
    }

    #[test]
    fn lower_broadcast_view_densifies_as_sink() {
        // Broadcast is a stride trick; the strided sink is densified with one copy.
        let bias = Tensor::parameter(&[3]);
        let out = bias.broadcast_to(&[2, 3], &[1]);
        let program = lower(&bias, &out).unwrap();

        assert_eq!(program.queue.len(), 1, "only the sink densify copy");
        assert_eq!(program.buffers.len(), 2);
        let sink = program.view(program.sinks[0]);
        assert!(sink.accessor.is_dense());
        assert_eq!(&*sink.accessor.shape, &[2, 3]);
        // The densify reads the strided broadcast view.
        let arg = program.view(program.queue[0].args[0]);
        assert_eq!(&*arg.accessor.shape, &[2, 3]);
        assert_eq!(&*arg.accessor.stride, &[0, 1]);
    }

    #[test]
    fn lower_transpose_view_densifies_as_sink() {
        let a = Tensor::parameter(&[2, 3]);
        let out = a.transpose();
        let program = lower(&a, &out).unwrap();

        assert_eq!(program.queue.len(), 1, "only the sink densify copy");
        let sink = program.view(program.sinks[0]);
        assert!(sink.accessor.is_dense());
        assert_eq!(&*sink.accessor.shape, &[3, 2]);
        let arg = program.view(program.queue[0].args[0]);
        assert_eq!(&*arg.accessor.stride, &[1, 3]);
    }

    #[test]
    fn lower_permute_view_densifies_as_sink() {
        let a = Tensor::parameter(&[2, 3, 4]);
        let out = a.permute(&[2, 0, 1]);
        let program = lower(&a, &out).unwrap();

        assert_eq!(program.queue.len(), 1, "only the sink densify copy");
        let sink = program.view(program.sinks[0]);
        assert!(sink.accessor.is_dense());
        assert_eq!(&*sink.accessor.shape, &[4, 2, 3]);
        let arg = program.view(program.queue[0].args[0]);
        assert_eq!(&*arg.accessor.stride, &[1, 12, 4]);
    }

    #[test]
    fn lower_unfold_view_densifies_as_sink() {
        let a = Tensor::parameter(&[7]);
        let out = a.unfold(0, 3, 1);
        let program = lower(&a, &out).unwrap();

        assert_eq!(program.queue.len(), 1, "only the sink densify copy");
        let sink = program.view(program.sinks[0]);
        assert!(sink.accessor.is_dense());
        assert_eq!(&*sink.accessor.shape, &[5, 3]);
        let arg = program.view(program.queue[0].args[0]);
        assert_eq!(&*arg.accessor.shape, &[5, 3]);
        assert_eq!(&*arg.accessor.stride, &[1, 1]);
    }

    #[test]
    fn lower_squeeze_is_view_not_kernel() {
        // Squeeze of a dense [1, 4] is already dense; no densify copy.
        let a = Tensor::parameter(&[1, 4]);
        let out = a.squeeze(&[0]);
        let program = lower(&a, &out).unwrap();

        assert_eq!(program.queue.len(), 0);
        assert_eq!(&*program.view(program.sinks[0]).accessor.shape, &[4]);
        assert!(program.view(program.sinks[0]).accessor.is_dense());
    }

    #[test]
    fn lower_bias_add_uses_broadcast_view() {
        // y = x @ W + bias: the bias operand is a stride-0 view, not a copy.
        let x = Tensor::parameter(&[2, 4]);
        let w = Tensor::parameter(&[4, 3]);
        let bias = Tensor::parameter(&[3]);
        let y = x.matmul(&w);
        let y = y.clone() + bias.broadcast_to(y.shape(), &[1]);

        let program = lower(&vec![x, w, bias], &y).unwrap();
        assert_eq!(program.queue.len(), 2);
        assert!(matches!(program.queue[0].kernel, Kernel::Matmul));

        let bias_arg = program.view(program.queue[1].args[1]);
        assert_eq!(&*bias_arg.accessor.shape, &[2, 3]);
        assert_eq!(&*bias_arg.accessor.stride, &[0, 1]);
    }

    #[test]
    fn lower_matmul_with_transpose_view() {
        let a = Tensor::parameter(&[2, 4]);
        let b = Tensor::parameter(&[3, 4]);
        let out = a.clone().matmul(&b.transpose());
        let program = lower(&Pair { a, b }, &out).unwrap();

        assert_eq!(program.queue.len(), 1, "transpose is a view on b");
        let b_arg = program.view(program.queue[0].args[1]);
        assert_eq!(&*b_arg.accessor.shape, &[4, 3]);
        assert_eq!(&*b_arg.accessor.stride, &[1, 4]);
    }

    #[test]
    fn lower_gather_rows_is_remap() {
        let x = Tensor::parameter(&[4, 2]);
        let indices = Tensor::constant_u32(&[3], &[0, 2, 1]);
        let out = x.gather_rows(&indices);
        let program = lower(&x, &out).unwrap();
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(
            program.queue[0].kernel,
            Kernel::Remap {
                info: RemapInfo::GatherRows
            }
        ));
        assert_eq!(&*program.output_shape(&program.queue[0]), &[3, 2]);
    }

    #[test]
    fn lower_index_is_view_not_kernel() {
        // Contiguous mid-buffer slice is dense-with-offset; no densify copy.
        let x = Tensor::parameter(&[6]);
        let out = x.index(&[IndexKeyElement::Slice(2..5)]);
        let program = lower(&x, &out).unwrap();
        assert_eq!(program.queue.len(), 0);
        let sink = program.view(program.sinks[0]);
        assert_eq!(&*sink.accessor.shape, &[3]);
        assert_eq!(sink.accessor.offset, 2);
        assert!(sink.accessor.is_dense());
    }

    #[test]
    fn lower_dense_sink_is_untouched() {
        let a = Tensor::parameter(&[4]);
        let out = a.clone() + a.clone();
        let program = lower(&a, &out).unwrap();
        assert_eq!(program.queue.len(), 1, "no densify copy for a dense sink");
        assert!(program.view(program.sinks[0]).accessor.is_dense());
    }
}
