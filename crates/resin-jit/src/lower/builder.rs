//! Lower DSL tensors to IR: **kernels write buffers; views are accessors**.
//!
//! DSL stays node-only (`Tensor` / `TensorKind`). At the IR boundary:
//! - **Materializing** nodes (param, const, elementwise, matmul, reduction, …)
//!   allocate an [`IrBuffer`] and enqueue a kernel.
//! - **View** nodes (broadcast, transpose, squeeze) compose an [`Accessor`] over
//!   an existing buffer and do **not** allocate or dispatch. Broadcasting is
//!   pitch-0, not a densifying copy.

use std::collections::HashMap;

use resin_core::{
    c_contiguous_pitch_for_shape, shape_join, Accessor, BinaryAssocElementOperator, Tree,
};
use resin_dsl::{ElementOperator, IndexKeyElement, ScatterOp, Tensor, TensorKind};
use resin_ir::{
    BufferRef, BufferViewRef, IrBuffer, IrBufferView, IrDispatch, IrElementwiseRpnKernel,
    IrKernel, IrMatmulKernel, IrProgram, IrRasterizeKernel, IrReductionKernel, IrRemapKernel,
    IrTraceRaysKernel, RemapInfo,
};

use crate::error::{tensor_kind_name, CompileError};
use crate::lower::types::{map_element_type, rpn_for_elementwise, shape_u32, ViewKey};

pub(crate) fn lower_program<PT, ST>(
    params: &PT,
    sinks: &ST,
) -> Result<IrProgram<PT::Mapped<BufferRef>, ST::Mapped<BufferViewRef>>, CompileError>
where
    PT: Tree<Tensor>,
    ST: Tree<Tensor>,
{
    let mut builder = ProgramBuilder::default();
    let ir_params = params.try_map(|tensor| builder.buffer_for_tensor(tensor))?;
    let ir_sinks = sinks.try_map(|tensor| builder.view_for_tensor(tensor))?;
    let program = IrProgram {
        params: ir_params,
        sinks: ir_sinks,
        queue: builder.queue,
        buffers: builder.buffers,
        buffer_views: builder.buffer_views,
    };
    program.validate()?;
    Ok(program)
}

#[derive(Default)]
struct ProgramBuilder {
    buffers: Vec<IrBuffer>,
    buffer_views: Vec<IrBufferView>,
    queue: Vec<IrDispatch>,
    /// Memo for materializing tensors only (each owns a buffer).
    tensor_buffers: HashMap<Tensor, BufferRef>,
    view_memo: HashMap<ViewKey, BufferViewRef>,
    /// Memo for densified hardware-node arguments (avoids duplicate copies).
    dense_view_memo: HashMap<Tensor, BufferViewRef>,
    /// Memo for strided-reshape identity copies (one per reshape node, not
    /// one per consumer).
    reshape_copy_memo: HashMap<Tensor, BufferRef>,
}

impl ProgramBuilder {
    /// Buffer backing a materializing tensor. View-only tensors are rejected.
    fn buffer_for_tensor(&mut self, tensor: &Tensor) -> Result<BufferRef, CompileError> {
        if let Some(buffer) = self.tensor_buffers.get(tensor) {
            return Ok(*buffer);
        }
        if is_view_op(tensor.kind()) {
            // Callers that need storage must resolve through `view_for_tensor`.
            // Params tree leaves are always materializing (Parameter / host upload).
            return Err(CompileError::UnsupportedTensorKind(tensor_kind_name(
                tensor.kind(),
            )));
        }
        self.build_materializing_tensor(tensor)
    }

    /// Logical view of `tensor` as a buffer + accessor (view ops compose).
    fn view_for_tensor(&mut self, tensor: &Tensor) -> Result<BufferViewRef, CompileError> {
        let (buffer, accessor) = self.resolve_view(tensor)?;
        self.intern_view(buffer, accessor)
    }

    /// Walk view-op chains; allocate only at materializing leaves.
    fn resolve_view(&mut self, tensor: &Tensor) -> Result<(BufferRef, Accessor), CompileError> {
        match tensor.kind() {
            TensorKind::Broadcast {
                arg,
                target_shape,
                axes,
            } => {
                let (buffer, arg_acc) = self.resolve_view(arg)?;
                let target_shape = shape_u32(target_shape)?;
                let accessor = compose_broadcast_axes(&arg_acc, &target_shape, axes)?;
                Ok((buffer, accessor))
            }
            TensorKind::Transpose { arg } => {
                let (buffer, arg_acc) = self.resolve_view(arg)?;
                Ok((buffer, arg_acc.transpose()?))
            }
            TensorKind::Squeeze { arg, axes } => {
                let (buffer, arg_acc) = self.resolve_view(arg)?;
                Ok((buffer, arg_acc.squeeze(axes)?))
            }
            TensorKind::Index { arg, key } => {
                let (buffer, arg_acc) = self.resolve_view(arg)?;
                Ok((buffer, compose_index(&arg_acc, key)?))
            }
            TensorKind::Reshape { arg, shape } => {
                let new_shape = shape_u32(shape)?;
                // Memoized: a shared strided reshape must materialize its
                // identity copy once, not once per consumer.
                if let Some(&copy_buffer) = self.reshape_copy_memo.get(tensor) {
                    return Ok((copy_buffer, Accessor::dense(new_shape, 0)));
                }
                let (buffer, arg_acc) = self.resolve_view(arg)?;
                let contiguous = arg_acc.pitch.as_ref()
                    == c_contiguous_pitch_for_shape(&arg_acc.shape).as_ref();
                if contiguous {
                    // Row-major reinterpretation is pure accessor arithmetic.
                    let pitch = c_contiguous_pitch_for_shape(&new_shape);
                    Ok((
                        buffer,
                        Accessor {
                            offset: arg_acc.offset,
                            shape: new_shape,
                            pitch,
                        },
                    ))
                } else {
                    // Strided source: materialize an identity copy, then view
                    // the dense copy under the new shape.
                    let element_type = map_element_type(arg.element_type())?;
                    let copy_buffer =
                        self.materialize_dense_copy(buffer, &arg_acc, element_type)?;
                    self.reshape_copy_memo.insert(tensor.clone(), copy_buffer);
                    Ok((copy_buffer, Accessor::dense(new_shape, 0)))
                }
            }
            _ => {
                let buffer = self.buffer_for_tensor(tensor)?;
                let shape = shape_u32(tensor.shape())?;
                Ok((buffer, Accessor::dense(shape, 0)))
            }
        }
    }

    fn intern_view(
        &mut self,
        buffer: BufferRef,
        accessor: Accessor,
    ) -> Result<BufferViewRef, CompileError> {
        let key = ViewKey {
            buffer: buffer.index(),
            accessor: accessor.clone(),
        };
        if let Some(view) = self.view_memo.get(&key) {
            return Ok(*view);
        }
        let view = BufferViewRef::new(self.buffer_views.len());
        self.buffer_views.push(IrBufferView {
            buffer_index: buffer,
            accessor,
        });
        self.view_memo.insert(key, view);
        Ok(view)
    }

    fn build_materializing_tensor(&mut self, tensor: &Tensor) -> Result<BufferRef, CompileError> {
        if let Some(buffer) = self.tensor_buffers.get(tensor) {
            return Ok(*buffer);
        }

        let element_type = map_element_type(tensor.element_type())?;
        let shape = shape_u32(tensor.shape())?;

        let dispatch = match tensor.kind() {
            TensorKind::Constant { bytes } => {
                let buffer = self.push_buffer(IrBuffer {
                    shape: shape.clone(),
                    element_type,
                    init: Some(bytes.clone()),
                    readonly: true,
                });
                self.tensor_buffers.insert(tensor.clone(), buffer);
                return Ok(buffer);
            }
            TensorKind::Parameter => {
                let buffer = self.push_buffer(IrBuffer {
                    shape: shape.clone(),
                    element_type,
                    init: None,
                    readonly: false,
                });
                self.tensor_buffers.insert(tensor.clone(), buffer);
                return Ok(buffer);
            }
            TensorKind::Elementwise { operator, args } => {
                if *operator == ElementOperator::Matmul {
                    let arg_views = self.build_matmul_arg_views(args)?;
                    Some(IrDispatch {
                        kernel: IrKernel::Matmul(IrMatmulKernel {
                            arg_accessors: self.arg_accessors(&arg_views)?,
                            element_type,
                            shape: shape.clone(),
                            clear_output_before_dispatch: false,
                        }),
                        arg_view_indices: arg_views,
                        output_buffer_index: BufferRef::new(self.buffers.len()),
                    })
                } else {
                    let arg_views = self.build_elementwise_arg_views(args, &shape)?;
                    Some(IrDispatch {
                        kernel: IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                            arg_accessors: self.arg_accessors(&arg_views)?,
                            arg_element_types: self.arg_element_types(args)?,
                            element_type,
                            shape: shape.clone(),
                            rpn_expr: rpn_for_elementwise(*operator, args.len())?,
                            clear_output_before_dispatch: false,
                        }),
                        arg_view_indices: arg_views,
                        output_buffer_index: BufferRef::new(self.buffers.len()),
                    })
                }
            }
            TensorKind::Reduction {
                operator,
                axes,
                arg,
            } => {
                let op = match operator {
                    ElementOperator::Add => BinaryAssocElementOperator::Add,
                    other => return Err(CompileError::UnsupportedOperator(*other)),
                };
                // Input view keeps the true input logical shape (keepdims output is `shape`).
                let arg_view = self.view_for_tensor(arg)?;
                Some(IrDispatch {
                    kernel: IrKernel::Reduction(IrReductionKernel {
                        arg_accessors: self.arg_accessors(std::slice::from_ref(&arg_view))?,
                        element_type,
                        shape: shape.clone(),
                        operator: op,
                        axes: shape_u32(axes)?.into(),
                        clear_output_before_dispatch: false,
                    }),
                    arg_view_indices: vec![arg_view],
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::Gather { source, indices } => {
                let arg_views = vec![
                    self.view_for_tensor(source)?,
                    self.view_for_tensor(indices)?,
                ];
                Some(IrDispatch {
                    kernel: IrKernel::Remap(IrRemapKernel {
                        arg_accessors: self.arg_accessors(&arg_views)?,
                        arg_element_types: self
                            .arg_element_types(&[source.clone(), indices.clone()])?,
                        element_type,
                        shape: shape.clone(),
                        info: RemapInfo::GatherRows,
                        clear_output_before_dispatch: false,
                    }),
                    arg_view_indices: arg_views,
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::Scatter {
                source,
                indices,
                op,
            } => {
                let arg_views = vec![
                    self.view_for_tensor(source)?,
                    self.view_for_tensor(indices)?,
                ];
                let operator = match op {
                    ScatterOp::Write => None,
                    ScatterOp::Add => Some(BinaryAssocElementOperator::Add),
                };
                Some(IrDispatch {
                    kernel: IrKernel::Remap(IrRemapKernel {
                        arg_accessors: self.arg_accessors(&arg_views)?,
                        arg_element_types: self
                            .arg_element_types(&[source.clone(), indices.clone()])?,
                        element_type,
                        shape: shape.clone(),
                        info: RemapInfo::ScatterRows { operator },
                        clear_output_before_dispatch: true,
                    }),
                    arg_view_indices: arg_views,
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::ScatterIndex {
                source,
                key,
                target_shape,
            } => {
                let arg_views = vec![self.view_for_tensor(source)?];
                let target_shape = shape_u32(target_shape)?;
                let accessor = region_accessor(&target_shape, key);
                Some(IrDispatch {
                    kernel: IrKernel::Remap(IrRemapKernel {
                        arg_accessors: self.arg_accessors(&arg_views)?,
                        arg_element_types: self.arg_element_types(std::slice::from_ref(source))?,
                        element_type,
                        shape: shape.clone(),
                        info: RemapInfo::ScatterView { accessor },
                        clear_output_before_dispatch: true,
                    }),
                    arg_view_indices: arg_views,
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::TraceRays {
                origins,
                directions,
                t_min,
                t_max,
                vertices,
                triangles,
            } => {
                // Hardware executors consume raw device buffers (acceleration
                // structure builds, vertex pulling), so every argument is
                // densified if it arrives as a strided view.
                let args = [
                    origins.clone(),
                    directions.clone(),
                    t_min.clone(),
                    t_max.clone(),
                    vertices.clone(),
                    triangles.clone(),
                ];
                let arg_views = args
                    .iter()
                    .map(|arg| self.dense_view_for_tensor(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Some(IrDispatch {
                    kernel: IrKernel::TraceRays(IrTraceRaysKernel {
                        arg_accessors: self.arg_accessors(&arg_views)?,
                        arg_element_types: self.arg_element_types(&args)?,
                        element_type,
                        shape: shape.clone(),
                        clear_output_before_dispatch: false,
                    }),
                    arg_view_indices: arg_views,
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::Rasterize {
                clip_positions,
                triangles,
            } => {
                let args = [clip_positions.clone(), triangles.clone()];
                let arg_views = args
                    .iter()
                    .map(|arg| self.dense_view_for_tensor(arg))
                    .collect::<Result<Vec<_>, _>>()?;
                Some(IrDispatch {
                    kernel: IrKernel::Rasterize(IrRasterizeKernel {
                        arg_accessors: self.arg_accessors(&arg_views)?,
                        arg_element_types: self.arg_element_types(&args)?,
                        element_type,
                        shape: shape.clone(),
                        // Background pixels must read (prim=0, hit=0, u=0, v=0).
                        clear_output_before_dispatch: true,
                    }),
                    arg_view_indices: arg_views,
                    output_buffer_index: BufferRef::new(self.buffers.len()),
                })
            }
            TensorKind::Broadcast { .. }
            | TensorKind::Transpose { .. }
            | TensorKind::Squeeze { .. }
            | TensorKind::Index { .. }
            | TensorKind::Reshape { .. } => {
                unreachable!("view ops are handled in resolve_view")
            }
        };

        let buffer = self.push_buffer(IrBuffer {
            shape,
            element_type,
            init: None,
            readonly: false,
        });
        if let Some(dispatch) = dispatch {
            self.queue.push(dispatch);
        }
        self.tensor_buffers.insert(tensor.clone(), buffer);
        Ok(buffer)
    }

    /// Elementwise args: resolve view ops, then NumPy-trailing-align to output shape.
    fn build_elementwise_arg_views(
        &mut self,
        args: &[Tensor],
        output_shape: &[u32],
    ) -> Result<Vec<BufferViewRef>, CompileError> {
        args.iter()
            .map(|arg| {
                let (buffer, acc) = self.resolve_view(arg)?;
                let expanded = expand_accessor_to(&acc, output_shape)?;
                self.intern_view(buffer, expanded)
            })
            .collect()
    }

    fn build_matmul_arg_views(
        &mut self,
        args: &[Tensor],
    ) -> Result<Vec<BufferViewRef>, CompileError> {
        args.iter()
            .map(|arg| self.view_for_tensor(arg))
            .collect()
    }

    /// View of `tensor` guaranteed dense C-contiguous from buffer offset 0.
    /// Strided/broadcast/offset views are materialized with an identity
    /// elementwise copy (hardware nodes consume raw buffers).
    fn dense_view_for_tensor(&mut self, tensor: &Tensor) -> Result<BufferViewRef, CompileError> {
        if let Some(view) = self.dense_view_memo.get(tensor) {
            return Ok(*view);
        }
        let (buffer, accessor) = self.resolve_view(tensor)?;
        let node_shape = accessor.shape.clone();
        let view = if accessor.is_dense_c_contiguous(&node_shape) {
            self.intern_view(buffer, accessor)?
        } else {
            let element_type = map_element_type(tensor.element_type())?;
            let copy_buffer = self.materialize_dense_copy(buffer, &accessor, element_type)?;
            self.intern_view(copy_buffer, Accessor::dense(node_shape, 0))?
        };
        self.dense_view_memo.insert(tensor.clone(), view);
        Ok(view)
    }

    /// Copy `accessor`'s elements (over `buffer`) into a fresh dense buffer
    /// via an identity elementwise kernel — the one way strided data becomes
    /// contiguous storage.
    fn materialize_dense_copy(
        &mut self,
        buffer: BufferRef,
        accessor: &Accessor,
        element_type: resin_core::ElementType,
    ) -> Result<BufferRef, CompileError> {
        let src_view = self.intern_view(buffer, accessor.clone())?;
        let copy_buffer = self.push_buffer(IrBuffer {
            shape: accessor.shape.clone(),
            element_type,
            init: None,
            readonly: false,
        });
        self.queue.push(IrDispatch {
            kernel: IrKernel::ElementwiseRpn(IrElementwiseRpnKernel {
                arg_accessors: vec![accessor.clone()],
                arg_element_types: vec![element_type],
                element_type,
                shape: accessor.shape.clone(),
                rpn_expr: resin_ir::ElementRpnExpr {
                    atoms: vec![resin_ir::RpnAtom::Arg(0)],
                },
                clear_output_before_dispatch: false,
            }),
            arg_view_indices: vec![src_view],
            output_buffer_index: copy_buffer,
        });
        Ok(copy_buffer)
    }

    fn arg_accessors(&self, views: &[BufferViewRef]) -> Result<Vec<Accessor>, CompileError> {
        views
            .iter()
            .map(|view| Ok(self.buffer_views[view.index()].accessor.clone()))
            .collect()
    }

    fn arg_element_types(
        &self,
        args: &[Tensor],
    ) -> Result<Vec<resin_core::ElementType>, CompileError> {
        args.iter()
            .map(|arg| map_element_type(arg.element_type()))
            .collect()
    }

    fn push_buffer(&mut self, buffer: IrBuffer) -> BufferRef {
        let index = BufferRef::new(self.buffers.len());
        self.buffers.push(buffer);
        index
    }
}

fn is_view_op(kind: &TensorKind) -> bool {
    matches!(
        kind,
        TensorKind::Broadcast { .. }
            | TensorKind::Transpose { .. }
            | TensorKind::Squeeze { .. }
            | TensorKind::Index { .. }
            | TensorKind::Reshape { .. }
    )
}

/// Compose a static index key onto an accessor: pure offset/shape arithmetic,
/// no kernel. `Single` keeps a size-1 axis (rank is preserved).
fn compose_index(
    arg_acc: &Accessor,
    key: &[IndexKeyElement],
) -> Result<Accessor, CompileError> {
    debug_assert_eq!(key.len(), arg_acc.rank(), "index key covers every axis");
    let mut offset = arg_acc.offset;
    let mut shape = Vec::with_capacity(key.len());
    for (axis, element) in key.iter().enumerate() {
        let (start, len) = match element {
            IndexKeyElement::Single(i) => (*i, 1usize),
            IndexKeyElement::Slice(range) => (range.start, range.end - range.start),
        };
        offset += u32::try_from(start)
            .map_err(|_| CompileError::ShapeOverflow(start))?
            * arg_acc.pitch[axis];
        shape.push(u32::try_from(len).map_err(|_| CompileError::ShapeOverflow(len))?);
    }
    Ok(Accessor {
        offset,
        shape: shape.into(),
        pitch: arg_acc.pitch.clone(),
    })
}

/// Accessor addressing the `key` region of a dense buffer with `target_shape`.
fn region_accessor(target_shape: &[u32], key: &[IndexKeyElement]) -> Accessor {
    let pitch = c_contiguous_pitch_for_shape(target_shape);
    let mut offset = 0u32;
    let mut shape = Vec::with_capacity(key.len());
    for (axis, element) in key.iter().enumerate() {
        let (start, len) = match element {
            IndexKeyElement::Single(i) => (*i, 1usize),
            IndexKeyElement::Slice(range) => (range.start, range.end - range.start),
        };
        offset += start as u32 * pitch[axis];
        shape.push(len as u32);
    }
    Accessor {
        offset,
        shape: shape.into(),
        pitch,
    }
}

/// Explicit-axis broadcast (`axes[i]` = output axis for source axis `i`).
fn compose_broadcast_axes(
    arg_acc: &Accessor,
    target_shape: &[u32],
    axes: &[usize],
) -> Result<Accessor, CompileError> {
    if axes.len() != arg_acc.rank() {
        return Err(CompileError::Accessor(
            resin_core::ShapeError::IncompatibleShapes {
                lhs: arg_acc.shape.clone(),
                rhs: target_shape.into(),
            },
        ));
    }
    let mut pitch = vec![0u32; target_shape.len()];
    for (arg_axis, &out_axis) in axes.iter().enumerate() {
        if out_axis >= target_shape.len() {
            return Err(CompileError::Accessor(
                resin_core::ShapeError::SqueezeAxisOutOfRange {
                    axis: out_axis,
                    rank: target_shape.len(),
                },
            ));
        }
        // Size-1 dims that expand must keep pitch 0.
        if arg_acc.shape[arg_axis] == 1 && target_shape[out_axis] > 1 {
            pitch[out_axis] = 0;
        } else {
            pitch[out_axis] = arg_acc.pitch[arg_axis];
        }
    }
    Ok(Accessor {
        offset: arg_acc.offset,
        shape: target_shape.into(),
        pitch: pitch.into(),
    })
}

/// NumPy trailing-align expand of an existing view to `target_shape`.
fn expand_accessor_to(
    acc: &Accessor,
    target_shape: &[u32],
) -> Result<Accessor, CompileError> {
    if acc.shape.as_ref() == target_shape {
        return Ok(acc.clone());
    }
    let out_pitch = c_contiguous_pitch_for_shape(target_shape);
    let join = shape_join(&acc.shape, &acc.pitch, target_shape, &out_pitch)?;
    if join.shape.as_ref() != target_shape {
        return Err(CompileError::Accessor(
            resin_core::ShapeError::IncompatibleShapes {
                lhs: acc.shape.clone(),
                rhs: target_shape.into(),
            },
        ));
    }
    Ok(Accessor {
        offset: acc.offset,
        shape: target_shape.into(),
        pitch: join.pitch1,
    })
}

#[cfg(test)]
mod tests {
    use resin_core::ElementOperator as IrElementOperator;
    use resin_dsl::{ElementType, Tensor};
    use resin_ir::{IrKernel, RpnAtom};
    use resin_macros::Tree;

    use super::*;

    #[derive(Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn lower_add_of_params() {
        let a = Tensor::parameter(&[2, 3], ElementType::F32);
        let b = Tensor::parameter(&[2, 3], ElementType::F32);
        let out = a.clone() + b.clone();

        let params = Pair {
            a: a.clone(),
            b: b.clone(),
        };
        let program = lower_program(&params, &out).unwrap();

        program.validate().unwrap();
        assert_eq!(program.buffers.len(), 3);
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(
            program.queue[0].kernel,
            IrKernel::ElementwiseRpn(_)
        ));
    }

    #[test]
    fn lower_shared_param_operand() {
        let a = Tensor::parameter(&[4], ElementType::F32);
        let out = a.clone() + a.clone();
        let params = Pair {
            a: a.clone(),
            b: Tensor::parameter(&[4], ElementType::F32),
        };

        let program = lower_program(&params, &out).unwrap();
        assert_eq!(program.buffers.len(), 3);
        assert_eq!(program.queue.len(), 1);

        let kernel = match &program.queue[0].kernel {
            IrKernel::ElementwiseRpn(kernel) => kernel,
            other => panic!("expected elementwise kernel, got {other:?}"),
        };
        assert_eq!(kernel.arg_accessors.len(), 2);
        assert!(matches!(
            kernel.rpn_expr.atoms.last(),
            Some(RpnAtom::Op(IrElementOperator::Binary(_)))
        ));
    }

    #[test]
    fn lower_scalar_constant() {
        let c = Tensor::zeros(&[], ElementType::F32);
        let program = lower_program(&c, &c).unwrap();
        assert_eq!(program.buffers.len(), 1);
        assert!(program.buffers[0].readonly);
        assert_eq!(program.queue.len(), 0);
    }

    #[test]
    fn lower_matmul() {
        let a = Tensor::parameter(&[2, 4], ElementType::F32);
        let b = Tensor::parameter(&[4, 3], ElementType::F32);
        let out = a.clone().matmul(&b);

        let params = Pair { a, b };
        let program = lower_program(&params, &out).unwrap();
        program.validate().unwrap();

        assert_eq!(program.buffers.len(), 3);
        assert_eq!(program.queue.len(), 1);
        assert!(matches!(program.queue[0].kernel, IrKernel::Matmul(_)));
        assert_eq!(program.buffers[2].shape.as_ref(), &[2, 3]);
    }

    #[test]
    fn lower_reduction_arg_uses_input_shape() {
        let a = Tensor::parameter(&[8, 10], ElementType::F32);
        let out = a.clone().sum_axes(&[0]);
        let program = lower_program(&a, &out).unwrap();
        program.validate().unwrap();

        let kernel = match &program.queue[0].kernel {
            IrKernel::Reduction(k) => k,
            other => panic!("expected reduction, got {other:?}"),
        };
        assert_eq!(kernel.arg_accessors[0].shape.as_ref(), &[8, 10]);
        assert_eq!(kernel.shape.as_ref(), &[1, 10]);
    }

    #[test]
    fn lower_scalar_mul_broadcasts_arg_accessor() {
        let w = Tensor::parameter(&[4, 3], ElementType::F32);
        let lr = Tensor::zeros(&[], ElementType::F32);
        let out = w.clone() * lr;
        let program = lower_program(&w, &out).unwrap();
        program.validate().unwrap();

        let kernel = match &program.queue[0].kernel {
            IrKernel::ElementwiseRpn(k) => k,
            other => panic!("expected elementwise, got {other:?}"),
        };
        assert_eq!(kernel.shape.as_ref(), &[4, 3]);
        assert_eq!(kernel.arg_accessors[1].shape.as_ref(), &[4, 3]);
        assert_eq!(kernel.arg_accessors[1].pitch.as_ref(), &[0, 0]);
    }

    #[test]
    fn lower_broadcast_is_view_not_kernel() {
        let bias = Tensor::parameter(&[3], ElementType::F32);
        let out = bias.broadcast_to(&[2, 3], &[1]);
        let program = lower_program(&bias, &out).unwrap();
        program.validate().unwrap();

        assert_eq!(program.queue.len(), 0, "broadcast must not enqueue a kernel");
        assert_eq!(program.buffers.len(), 1, "only the bias buffer");
        let sink = &program.buffer_views[program.sinks.index()];
        assert_eq!(sink.accessor.shape.as_ref(), &[2, 3]);
        assert_eq!(sink.accessor.pitch.as_ref(), &[0, 1]);
    }

    #[test]
    fn lower_transpose_is_view_not_kernel() {
        let a = Tensor::parameter(&[2, 3], ElementType::F32);
        let out = a.clone().transpose();
        let program = lower_program(&a, &out).unwrap();
        program.validate().unwrap();

        assert_eq!(program.queue.len(), 0);
        assert_eq!(program.buffers.len(), 1);
        let sink = &program.buffer_views[program.sinks.index()];
        assert_eq!(sink.accessor.shape.as_ref(), &[3, 2]);
        assert_eq!(sink.accessor.pitch.as_ref(), &[1, 3]);
    }

    #[test]
    fn lower_squeeze_is_view_not_kernel() {
        let a = Tensor::parameter(&[1, 4], ElementType::F32);
        let out = a.clone().squeeze(&[0]);
        let program = lower_program(&a, &out).unwrap();
        program.validate().unwrap();

        assert_eq!(program.queue.len(), 0);
        assert_eq!(program.buffers.len(), 1);
        let sink = &program.buffer_views[program.sinks.index()];
        assert_eq!(sink.accessor.shape.as_ref(), &[4]);
    }

    #[test]
    fn lower_bias_add_no_materialized_broadcast() {
        // y = x @ W + bias[broadcast]
        let x = Tensor::parameter(&[2, 4], ElementType::F32);
        let w = Tensor::parameter(&[4, 3], ElementType::F32);
        let bias = Tensor::parameter(&[3], ElementType::F32);
        let y = x.clone().matmul(&w);
        let y = y.clone() + bias.broadcast_to(y.shape(), &[1]);

        #[derive(Tree)]
        struct In<T> {
            x: T,
            w: T,
            bias: T,
        }
        let params = In {
            x: x.clone(),
            w: w.clone(),
            bias: bias.clone(),
        };
        let program = lower_program(&params, &y).unwrap();
        program.validate().unwrap();

        // matmul + add only — no identity-copy broadcast kernel
        assert_eq!(program.queue.len(), 2);
        assert!(matches!(program.queue[0].kernel, IrKernel::Matmul(_)));
        assert!(matches!(
            program.queue[1].kernel,
            IrKernel::ElementwiseRpn(_)
        ));
        // buffers: x, w, bias, matmul_out, add_out
        assert_eq!(program.buffers.len(), 5);

        let add = match &program.queue[1].kernel {
            IrKernel::ElementwiseRpn(k) => k,
            other => panic!("expected add, got {other:?}"),
        };
        // bias operand should be a broadcast view (pitch 0 on batch axis)
        assert_eq!(add.arg_accessors[1].shape.as_ref(), &[2, 3]);
        assert_eq!(add.arg_accessors[1].pitch.as_ref(), &[0, 1]);
    }

    #[test]
    fn lower_matmul_with_transpose_view() {
        let a = Tensor::parameter(&[2, 4], ElementType::F32);
        let b = Tensor::parameter(&[3, 4], ElementType::F32);
        let out = a.clone().matmul(&b.transpose());
        let params = Pair {
            a: a.clone(),
            b: b.clone(),
        };
        let program = lower_program(&params, &out).unwrap();
        program.validate().unwrap();

        // only matmul — transpose is a view on b
        assert_eq!(program.queue.len(), 1);
        assert_eq!(program.buffers.len(), 3);
        let mm = match &program.queue[0].kernel {
            IrKernel::Matmul(k) => k,
            other => panic!("expected matmul, got {other:?}"),
        };
        assert_eq!(mm.arg_accessors[1].shape.as_ref(), &[4, 3]);
        assert_eq!(mm.arg_accessors[1].pitch.as_ref(), &[1, 4]);
    }
}
