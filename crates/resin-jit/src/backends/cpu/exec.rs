//! CPU interpreter: kernels consume **views** (buffer + accessor), not densified copies.
//!
//! Broadcast / transpose / squeeze / index are pitch tricks on the accessor;
//! elementwise, matmul, reduction, and remap index storage through those
//! pitches without allocating expanded tensors. RPN evaluation is typed
//! ([`Value`]): each argument is read as its own element type and the result
//! is written as the kernel's output element type.

use resin_core::{
    Accessor, BinaryAssocElementOperator, BinaryBitwiseOperator, BinaryCompareOperator,
    BinaryElementOperator, ElementOperator, ElementType, UnaryElementOperator,
};
use resin_ir::{
    IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel,
    IrReductionKernel, IrRemapKernel, RemapInfo, RpnAtom,
};

use super::program::CpuProgram;
use crate::error::RunError;

/// Typed scalar for RPN evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    F(f32),
    U(u32),
}

impl Value {
    fn as_f32(self) -> Result<f32, RunError> {
        match self {
            Value::F(v) => Ok(v),
            Value::U(_) => Err(RunError::UnsupportedOp(
                "expected f32 operand, got u32".into(),
            )),
        }
    }

    fn as_u32(self) -> Result<u32, RunError> {
        match self {
            Value::U(v) => Ok(v),
            Value::F(_) => Err(RunError::UnsupportedOp(
                "expected u32 operand, got f32".into(),
            )),
        }
    }
}

pub(crate) fn init_storage(program: &CpuProgram) -> Result<Vec<Vec<u8>>, RunError> {
    program
        .buffers
        .iter()
        .map(|buffer| {
            let nbytes = buffer_shape_len(&buffer.shape) * buffer.element_type.nbytes() as usize;
            match &buffer.init {
                Some(init) => {
                    if init.len() != nbytes {
                        return Err(RunError::BufferInitSize {
                            expected: nbytes,
                            got: init.len(),
                        });
                    }
                    Ok(init.to_vec())
                }
                None => Ok(vec![0u8; nbytes]),
            }
        })
        .collect()
}

pub(crate) fn run_program(program: &CpuProgram, storage: &mut [Vec<u8>]) -> Result<(), RunError> {
    for dispatch in &program.queue {
        run_dispatch(dispatch, &program.buffer_views, storage)?;
    }
    Ok(())
}

pub(crate) fn read_sink_bytes(
    program: &CpuProgram,
    storage: &[Vec<u8>],
    view_index: usize,
) -> Result<Vec<u8>, RunError> {
    let view = &program.buffer_views[view_index];
    let buffer = &storage[view.buffer_index.index()];
    gather_view_bytes(buffer, &view.accessor)
}

pub(crate) fn write_param_bytes(
    program: &CpuProgram,
    storage: &mut [Vec<u8>],
    buffer_index: usize,
    bytes: &[u8],
) -> Result<(), RunError> {
    let spec = &program.buffers[buffer_index];
    let expected = buffer_shape_len(&spec.shape) * spec.element_type.nbytes() as usize;
    if bytes.len() != expected {
        return Err(RunError::BufferSizeMismatch {
            expected,
            got: bytes.len(),
        });
    }
    storage[buffer_index].copy_from_slice(bytes);
    Ok(())
}

fn run_dispatch(
    dispatch: &IrDispatch,
    buffer_views: &[IrBufferView],
    storage: &mut [Vec<u8>],
) -> Result<(), RunError> {
    if dispatch.kernel.clear_output_before_dispatch() {
        let out = &mut storage[dispatch.output_buffer_index.index()];
        out.fill(0);
    }

    // Resolve arg views (buffer index + accessor) without densifying.
    let arg_views: Vec<(usize, Accessor)> = dispatch
        .arg_view_indices
        .iter()
        .map(|view_ref| {
            let view = &buffer_views[view_ref.index()];
            (view.buffer_index.index(), view.accessor.clone())
        })
        .collect();

    let out_index = dispatch.output_buffer_index.index();
    match &dispatch.kernel {
        IrKernel::ElementwiseRpn(kernel) => {
            run_elementwise_rpn(kernel, &arg_views, storage, out_index)?
        }
        IrKernel::Matmul(kernel) => run_matmul(kernel, &arg_views, storage, out_index)?,
        IrKernel::Reduction(kernel) => run_reduction(kernel, &arg_views, storage, out_index)?,
        IrKernel::Remap(kernel) => run_remap(kernel, &arg_views, storage, out_index)?,
        IrKernel::TraceRays(kernel) => {
            super::hw::run_trace_rays(kernel, &arg_views, storage, out_index)?
        }
        IrKernel::Rasterize(kernel) => {
            super::hw::run_rasterize(kernel, &arg_views, storage, out_index)?
        }
    }
    Ok(())
}

fn run_elementwise_rpn(
    kernel: &IrElementwiseRpnKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    let count = buffer_shape_len(&kernel.shape);
    let nbytes = element_nbytes(kernel.element_type)?;
    let out_len = storage[out_index].len();
    if out_len != count * nbytes {
        return Err(RunError::BufferSizeMismatch {
            expected: count * nbytes,
            got: out_len,
        });
    }

    // Precompute logical rank for coord walks (reuse one coords buf).
    let rank = kernel.shape.len();
    let mut coords = vec![0u32; rank];

    for linear in 0..count {
        linear_to_coords_into(linear, &kernel.shape, &mut coords);
        let mut stack: Vec<Value> = Vec::with_capacity(4);
        for atom in &kernel.rpn_expr.atoms {
            match atom {
                RpnAtom::Arg(i) => {
                    let (buf_idx, acc) = args
                        .get(*i as usize)
                        .ok_or(RunError::RpnArgOutOfRange { arg: *i })?;
                    let etype = kernel
                        .arg_element_types
                        .get(*i as usize)
                        .copied()
                        .ok_or(RunError::RpnArgOutOfRange { arg: *i })?;
                    // Elementwise arg accessors are aligned to kernel.shape.
                    let value = read_value_view(&storage[*buf_idx], etype, acc, &coords)?;
                    stack.push(value);
                }
                RpnAtom::Op(op) => apply_op(op, &mut stack, kernel.element_type)?,
            }
        }
        let value = stack.pop().ok_or(RunError::RpnEmptyStack)?;
        write_value(
            &mut storage[out_index][linear * nbytes..(linear + 1) * nbytes],
            kernel.element_type,
            value,
        )?;
    }
    Ok(())
}

fn run_matmul(
    kernel: &IrMatmulKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    if args.len() != 2 {
        return Err(RunError::MatmulArgCount { got: args.len() });
    }
    if kernel.element_type != ElementType::F4 {
        return Err(RunError::UnsupportedKernel("matmul is f32-only"));
    }
    let a_acc = &args[0].1;
    let b_acc = &args[1].1;
    let a_shape = &a_acc.shape;
    let b_shape = &b_acc.shape;
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(RunError::UnsupportedKernel("matmul rank < 2"));
    }
    let m = a_shape[a_shape.len() - 2] as usize;
    let k = a_shape[a_shape.len() - 1] as usize;
    let n = b_shape[b_shape.len() - 1] as usize;
    let batch_rank = a_shape.len() - 2;
    let batch: usize = if batch_rank == 0 {
        1
    } else {
        a_shape[..batch_rank]
            .iter()
            .map(|&d| d as usize)
            .product()
    };

    let a_buf = args[0].0;
    let b_buf = args[1].0;
    let rank = a_shape.len();
    // coords layout: [batch..., i, j] for output; A uses [batch..., i, t]; B uses [batch..., t, j]
    let mut a_coords = vec![0u32; rank];
    let mut b_coords = vec![0u32; rank];
    let mut out_coords = vec![0u32; kernel.shape.len()];

    for b in 0..batch {
        // Decode batch indices into leading coords.
        let mut rem = b;
        for axis in (0..batch_rank).rev() {
            let dim = a_shape[axis] as usize;
            let c = (rem % dim) as u32;
            a_coords[axis] = c;
            b_coords[axis] = c;
            out_coords[axis] = c;
            rem /= dim;
        }
        for i in 0..m {
            a_coords[batch_rank] = i as u32;
            out_coords[batch_rank] = i as u32;
            for j in 0..n {
                b_coords[batch_rank + 1] = j as u32;
                out_coords[batch_rank + 1] = j as u32;
                let mut acc = 0.0f32;
                for t in 0..k {
                    a_coords[batch_rank + 1] = t as u32;
                    b_coords[batch_rank] = t as u32;
                    let av = read_f32_view(&storage[a_buf], a_acc, &a_coords)?;
                    let bv = read_f32_view(&storage[b_buf], b_acc, &b_coords)?;
                    acc += av * bv;
                }
                let out_linear = coords_to_linear(&out_coords, &kernel.shape);
                write_f32(
                    &mut storage[out_index][out_linear * 4..(out_linear + 1) * 4],
                    acc,
                );
            }
        }
    }
    Ok(())
}

fn reduction_init(operator: BinaryAssocElementOperator, etype: ElementType) -> Value {
    match etype {
        ElementType::U4 => Value::U(match operator {
            BinaryAssocElementOperator::Add => 0,
            BinaryAssocElementOperator::Mul => 1,
            BinaryAssocElementOperator::Max => u32::MIN,
            BinaryAssocElementOperator::Min => u32::MAX,
        }),
        _ => Value::F(match operator {
            BinaryAssocElementOperator::Add => 0.0,
            BinaryAssocElementOperator::Mul => 1.0,
            BinaryAssocElementOperator::Max => f32::NEG_INFINITY,
            BinaryAssocElementOperator::Min => f32::INFINITY,
        }),
    }
}

fn reduce_values(
    operator: BinaryAssocElementOperator,
    acc: Value,
    value: Value,
) -> Result<Value, RunError> {
    Ok(match (acc, value) {
        (Value::F(a), Value::F(v)) => Value::F(match operator {
            BinaryAssocElementOperator::Add => a + v,
            BinaryAssocElementOperator::Mul => a * v,
            BinaryAssocElementOperator::Max => a.max(v),
            BinaryAssocElementOperator::Min => a.min(v),
        }),
        (Value::U(a), Value::U(v)) => Value::U(match operator {
            BinaryAssocElementOperator::Add => a.wrapping_add(v),
            BinaryAssocElementOperator::Mul => a.wrapping_mul(v),
            BinaryAssocElementOperator::Max => a.max(v),
            BinaryAssocElementOperator::Min => a.min(v),
        }),
        _ => {
            return Err(RunError::UnsupportedOp(
                "mixed-type reduction operands".into(),
            ))
        }
    })
}

fn run_reduction(
    kernel: &IrReductionKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    if args.len() != 1 {
        return Err(RunError::ReductionArgCount { got: args.len() });
    }
    let etype = kernel.element_type;
    let nbytes = element_nbytes(etype)?;
    let (in_buf, in_acc) = &args[0];
    let input_shape = &in_acc.shape;
    let input_count = buffer_shape_len(input_shape);
    let output_count = buffer_shape_len(&kernel.shape);
    if storage[*in_buf].is_empty() && input_count > 0 {
        return Err(RunError::BufferReadOutOfRange { index: 0 });
    }
    if storage[out_index].len() != output_count * nbytes {
        return Err(RunError::BufferSizeMismatch {
            expected: output_count * nbytes,
            got: storage[out_index].len(),
        });
    }

    // Clear output to the operator identity.
    for out_linear in 0..output_count {
        write_value(
            &mut storage[out_index][out_linear * nbytes..(out_linear + 1) * nbytes],
            etype,
            reduction_init(kernel.operator, etype),
        )?;
    }

    let reduced_axes: std::collections::HashSet<usize> =
        kernel.axes.iter().map(|&a| a as usize).collect();
    let mut in_coords = vec![0u32; input_shape.len()];
    let mut out_coords = vec![0u32; kernel.shape.len()];

    for in_linear in 0..input_count {
        linear_to_coords_into(in_linear, input_shape, &mut in_coords);
        out_coords.copy_from_slice(&in_coords);
        for &axis in &reduced_axes {
            out_coords[axis] = 0;
        }
        let out_linear = coords_to_linear(&out_coords, &kernel.shape);
        let value = read_value_view(&storage[*in_buf], etype, in_acc, &in_coords)?;
        let current = read_value(&storage[out_index], etype, out_linear)?;
        let reduced = reduce_values(kernel.operator, current, value)?;
        write_value(
            &mut storage[out_index][out_linear * nbytes..(out_linear + 1) * nbytes],
            etype,
            reduced,
        )?;
    }
    Ok(())
}

/// Gather / scatter data movement. The output buffer is already cleared for
/// scatter variants (see [`run_dispatch`]). Elements move as raw 4-byte
/// cells except scatter-add, which is typed.
fn run_remap(
    kernel: &IrRemapKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    let nbytes = element_nbytes(kernel.element_type)?;
    let (src_buf, src_acc) = &args[0];

    match &kernel.info {
        RemapInfo::GatherRows => {
            let (idx_buf, idx_acc) = &args[1];
            let src_rows = src_acc.shape[0];
            let count = buffer_shape_len(&kernel.shape);
            let mut coords = vec![0u32; kernel.shape.len()];
            for linear in 0..count {
                linear_to_coords_into(linear, &kernel.shape, &mut coords);
                let row = read_u32_view(&storage[*idx_buf], idx_acc, &coords[..1])?;
                let row = row.min(src_rows.saturating_sub(1));
                let out_row = coords[0];
                coords[0] = row;
                let value =
                    read_value_view(&storage[*src_buf], kernel.element_type, src_acc, &coords)?;
                coords[0] = out_row;
                write_value(
                    &mut storage[out_index][linear * nbytes..(linear + 1) * nbytes],
                    kernel.element_type,
                    value,
                )?;
            }
        }
        RemapInfo::ScatterRows { operator } => {
            let (idx_buf, idx_acc) = &args[1];
            let out_rows = kernel.shape[0];
            let count = buffer_shape_len(&src_acc.shape);
            let mut coords = vec![0u32; src_acc.shape.len()];
            let mut out_coords = vec![0u32; kernel.shape.len()];
            for linear in 0..count {
                linear_to_coords_into(linear, &src_acc.shape, &mut coords);
                let row = read_u32_view(&storage[*idx_buf], idx_acc, &coords[..1])?;
                if row >= out_rows {
                    continue; // Out-of-range scatter indices are dropped.
                }
                let value =
                    read_value_view(&storage[*src_buf], kernel.element_type, src_acc, &coords)?;
                out_coords.copy_from_slice(&coords);
                out_coords[0] = row;
                let out_linear = coords_to_linear(&out_coords, &kernel.shape);
                let value = match operator {
                    None => value,
                    Some(BinaryAssocElementOperator::Add) => {
                        let current =
                            read_value(&storage[out_index], kernel.element_type, out_linear)?;
                        reduce_values(BinaryAssocElementOperator::Add, current, value)?
                    }
                    Some(op) => {
                        return Err(RunError::UnsupportedOp(format!(
                            "scatter operator {op:?}"
                        )))
                    }
                };
                write_value(
                    &mut storage[out_index][out_linear * nbytes..(out_linear + 1) * nbytes],
                    kernel.element_type,
                    value,
                )?;
            }
        }
        RemapInfo::ScatterView { accessor } => {
            let count = buffer_shape_len(&src_acc.shape);
            let mut coords = vec![0u32; src_acc.shape.len()];
            for linear in 0..count {
                linear_to_coords_into(linear, &src_acc.shape, &mut coords);
                let value =
                    read_value_view(&storage[*src_buf], kernel.element_type, src_acc, &coords)?;
                let out_offset = accessor_offset(accessor, &coords)?;
                write_value(
                    &mut storage[out_index][out_offset * nbytes..(out_offset + 1) * nbytes],
                    kernel.element_type,
                    value,
                )?;
            }
        }
    }
    Ok(())
}

fn apply_op(
    op: &ElementOperator,
    stack: &mut Vec<Value>,
    out_etype: ElementType,
) -> Result<(), RunError> {
    match op {
        ElementOperator::Unary(unary) => {
            let x = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let y = match unary {
                UnaryElementOperator::Neg => Value::F(-x.as_f32()?),
                UnaryElementOperator::Exp => Value::F(x.as_f32()?.exp()),
                UnaryElementOperator::Log => Value::F(x.as_f32()?.ln()),
                UnaryElementOperator::Relu => Value::F(x.as_f32()?.max(0.0)),
                UnaryElementOperator::Abs => Value::F(x.as_f32()?.abs()),
                UnaryElementOperator::Sqrt => Value::F(x.as_f32()?.sqrt()),
                UnaryElementOperator::Sin => Value::F(x.as_f32()?.sin()),
                UnaryElementOperator::Cos => Value::F(x.as_f32()?.cos()),
                UnaryElementOperator::Floor => Value::F(x.as_f32()?.floor()),
                UnaryElementOperator::Ceil => Value::F(x.as_f32()?.ceil()),
                // 0/1 mask negation (matches WGSL `abs(1 - x)` emission).
                UnaryElementOperator::Not => match x {
                    Value::F(v) => Value::F((1.0 - v).abs()),
                    Value::U(v) => Value::U(1u32.wrapping_sub(v)),
                },
                UnaryElementOperator::Bitcast => match (x, out_etype) {
                    (Value::F(v), ElementType::U4) => Value::U(v.to_bits()),
                    (Value::U(v), ElementType::F4) => Value::F(f32::from_bits(v)),
                    (v, _) => v, // same-type bitcast is the identity
                },
                UnaryElementOperator::Convert => match (x, out_etype) {
                    (Value::F(v), ElementType::U4) => Value::U(v as u32),
                    (Value::U(v), ElementType::F4) => Value::F(v as f32),
                    (v, _) => v, // same-type convert is the identity
                },
            };
            stack.push(y);
        }
        ElementOperator::Binary(binary) => {
            let rhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let lhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let y = match (lhs, rhs) {
                (Value::F(a), Value::F(b)) => Value::F(match binary {
                    BinaryElementOperator::Pow => a.powf(b),
                    BinaryElementOperator::Div => a / b,
                    BinaryElementOperator::Sub => a - b,
                    BinaryElementOperator::Assoc(assoc) => match assoc {
                        BinaryAssocElementOperator::Mul => a * b,
                        BinaryAssocElementOperator::Add => a + b,
                        BinaryAssocElementOperator::Max => a.max(b),
                        BinaryAssocElementOperator::Min => a.min(b),
                    },
                }),
                (Value::U(a), Value::U(b)) => Value::U(match binary {
                    BinaryElementOperator::Pow => {
                        return Err(RunError::UnsupportedOp("pow on u32".into()))
                    }
                    BinaryElementOperator::Div => a.checked_div(b).unwrap_or(0),
                    BinaryElementOperator::Sub => a.wrapping_sub(b),
                    BinaryElementOperator::Assoc(assoc) => match assoc {
                        BinaryAssocElementOperator::Mul => a.wrapping_mul(b),
                        BinaryAssocElementOperator::Add => a.wrapping_add(b),
                        BinaryAssocElementOperator::Max => a.max(b),
                        BinaryAssocElementOperator::Min => a.min(b),
                    },
                }),
                _ => {
                    return Err(RunError::UnsupportedOp(
                        "mixed-type binary operands".into(),
                    ))
                }
            };
            stack.push(y);
        }
        ElementOperator::Compare(compare) => {
            let rhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let lhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let pred = match (lhs, rhs) {
                (Value::F(a), Value::F(b)) => match compare {
                    BinaryCompareOperator::Eq => a == b,
                    BinaryCompareOperator::Ne => a != b,
                    BinaryCompareOperator::Gt => a > b,
                    BinaryCompareOperator::Lt => a < b,
                    BinaryCompareOperator::Ge => a >= b,
                    BinaryCompareOperator::Le => a <= b,
                },
                (Value::U(a), Value::U(b)) => match compare {
                    BinaryCompareOperator::Eq => a == b,
                    BinaryCompareOperator::Ne => a != b,
                    BinaryCompareOperator::Gt => a > b,
                    BinaryCompareOperator::Lt => a < b,
                    BinaryCompareOperator::Ge => a >= b,
                    BinaryCompareOperator::Le => a <= b,
                },
                _ => {
                    return Err(RunError::UnsupportedOp(
                        "mixed-type compare operands".into(),
                    ))
                }
            };
            // Masks are 0/1 in the kernel output element type (WGSL parity).
            stack.push(match out_etype {
                ElementType::U4 => Value::U(pred as u32),
                _ => Value::F(pred as u32 as f32),
            });
        }
        ElementOperator::Bitwise(bitwise) => {
            let rhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let lhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let a = lhs.as_u32()?;
            let b = rhs.as_u32()?;
            let y = match bitwise {
                BinaryBitwiseOperator::Band => a & b,
                BinaryBitwiseOperator::Bor => a | b,
                BinaryBitwiseOperator::Bxor => a ^ b,
                // WGSL masks shift amounts to the bit width.
                BinaryBitwiseOperator::Shl => a.wrapping_shl(b & 31),
                BinaryBitwiseOperator::Shr => a.wrapping_shr(b & 31),
            };
            stack.push(Value::U(y));
        }
    }
    Ok(())
}

/// Dense gather for host sink readback only (4-byte elements, type-agnostic).
fn gather_view_bytes(buffer: &[u8], accessor: &Accessor) -> Result<Vec<u8>, RunError> {
    let count = buffer_shape_len(&accessor.shape);
    let mut out = vec![0u8; count * 4];
    let mut coords = vec![0u32; accessor.shape.len()];
    for linear in 0..count {
        linear_to_coords_into(linear, &accessor.shape, &mut coords);
        let offset = accessor_offset(accessor, &coords)?;
        let start = offset * 4;
        let chunk = buffer
            .get(start..start + 4)
            .ok_or(RunError::BufferReadOutOfRange { index: offset })?;
        out[linear * 4..(linear + 1) * 4].copy_from_slice(chunk);
    }
    Ok(out)
}

fn element_nbytes(etype: ElementType) -> Result<usize, RunError> {
    match etype {
        ElementType::F4 | ElementType::U4 => Ok(4),
        ElementType::F2 => Err(RunError::UnsupportedOp("f16 on cpu".into())),
    }
}

fn read_value_view(
    buffer: &[u8],
    etype: ElementType,
    accessor: &Accessor,
    coords: &[u32],
) -> Result<Value, RunError> {
    let offset = accessor_offset(accessor, coords)?;
    read_value(buffer, etype, offset)
}

fn read_value(bytes: &[u8], etype: ElementType, index: usize) -> Result<Value, RunError> {
    match etype {
        ElementType::F4 => Ok(Value::F(read_f32(bytes, index)?)),
        ElementType::U4 => Ok(Value::U(read_u32(bytes, index)?)),
        ElementType::F2 => Err(RunError::UnsupportedOp("f16 on cpu".into())),
    }
}

fn write_value(bytes: &mut [u8], etype: ElementType, value: Value) -> Result<(), RunError> {
    match (etype, value) {
        (ElementType::F4, Value::F(v)) => {
            bytes.copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        (ElementType::U4, Value::U(v)) => {
            bytes.copy_from_slice(&v.to_le_bytes());
            Ok(())
        }
        _ => Err(RunError::UnsupportedOp(format!(
            "value/type mismatch writing {value:?} as {etype:?}"
        ))),
    }
}

pub(super) fn read_f32_view(buffer: &[u8], accessor: &Accessor, coords: &[u32]) -> Result<f32, RunError> {
    let offset = accessor_offset(accessor, coords)?;
    read_f32(buffer, offset)
}

pub(super) fn read_u32_view(buffer: &[u8], accessor: &Accessor, coords: &[u32]) -> Result<u32, RunError> {
    let offset = accessor_offset(accessor, coords)?;
    read_u32(buffer, offset)
}

fn accessor_offset(accessor: &Accessor, coords: &[u32]) -> Result<usize, RunError> {
    if coords.len() != accessor.shape.len() {
        return Err(RunError::AccessorRank {
            coords: coords.len(),
            shape: accessor.shape.len(),
        });
    }
    let mut offset = accessor.offset as usize;
    for (coord, pitch) in coords.iter().zip(accessor.pitch.iter()) {
        offset += (*coord as usize) * (*pitch as usize);
    }
    Ok(offset)
}

fn linear_to_coords_into(linear: usize, shape: &[u32], coords: &mut [u32]) {
    debug_assert_eq!(coords.len(), shape.len());
    let mut rem = linear;
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis] as usize;
        coords[axis] = if dim == 0 {
            0
        } else {
            (rem % dim) as u32
        };
        if dim != 0 {
            rem /= dim;
        }
    }
}

fn coords_to_linear(coords: &[u32], shape: &[u32]) -> usize {
    let mut linear = 0usize;
    for (&coord, &dim) in coords.iter().zip(shape.iter()) {
        linear = linear * dim as usize + coord as usize;
    }
    linear
}

fn buffer_shape_len(shape: &[u32]) -> usize {
    if shape.is_empty() {
        1
    } else {
        shape.iter().map(|&d| d as usize).product()
    }
}

fn read_f32(bytes: &[u8], index: usize) -> Result<f32, RunError> {
    let start = index * 4;
    let chunk: [u8; 4] = bytes
        .get(start..start + 4)
        .ok_or(RunError::BufferReadOutOfRange { index })?
        .try_into()
        .unwrap();
    Ok(f32::from_le_bytes(chunk))
}

fn read_u32(bytes: &[u8], index: usize) -> Result<u32, RunError> {
    let start = index * 4;
    let chunk: [u8; 4] = bytes
        .get(start..start + 4)
        .ok_or(RunError::BufferReadOutOfRange { index })?
        .try_into()
        .unwrap();
    Ok(u32::from_le_bytes(chunk))
}

pub(super) fn write_f32(bytes: &mut [u8], value: f32) {
    bytes.copy_from_slice(&value.to_le_bytes());
}
