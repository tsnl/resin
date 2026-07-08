//! CPU interpreter: kernels consume **views** (buffer + accessor), not densified copies.
//!
//! Broadcast / transpose / squeeze are pitch tricks on the accessor; elementwise and
//! matmul index storage through those pitches without allocating expanded tensors.

use resin_core::{
    Accessor, BinaryAssocElementOperator, BinaryElementOperator, ElementOperator,
    UnaryElementOperator,
};
use resin_ir::{
    IrBufferView, IrDispatch, IrElementwiseRpnKernel, IrKernel, IrMatmulKernel, IrReductionKernel,
    RpnAtom,
};

use super::program::CpuProgram;
use crate::error::RunError;

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
        IrKernel::Remap(_) => return Err(RunError::UnsupportedKernel("remap")),
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
    let nbytes = kernel.element_type.nbytes() as usize;
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
        let mut stack = Vec::with_capacity(4);
        for atom in &kernel.rpn_expr.atoms {
            match atom {
                RpnAtom::Arg(i) => {
                    let (buf_idx, acc) = args
                        .get(*i as usize)
                        .ok_or(RunError::RpnArgOutOfRange { arg: *i })?;
                    // Elementwise arg accessors are aligned to kernel.shape.
                    let value = read_f32_view(&storage[*buf_idx], acc, &coords)?;
                    stack.push(value);
                }
                RpnAtom::Op(op) => apply_op(op, &mut stack)?,
            }
        }
        let value = stack.pop().ok_or(RunError::RpnEmptyStack)?;
        write_f32(
            &mut storage[out_index][linear * nbytes..(linear + 1) * nbytes],
            value,
        );
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

fn run_reduction(
    kernel: &IrReductionKernel,
    args: &[(usize, Accessor)],
    storage: &mut [Vec<u8>],
    out_index: usize,
) -> Result<(), RunError> {
    if args.len() != 1 {
        return Err(RunError::ReductionArgCount { got: args.len() });
    }
    let (in_buf, in_acc) = &args[0];
    let input_shape = &in_acc.shape;
    let input_count = buffer_shape_len(input_shape);
    let output_count = buffer_shape_len(&kernel.shape);
    if storage[*in_buf].is_empty() && input_count > 0 {
        return Err(RunError::BufferReadOutOfRange { index: 0 });
    }
    if storage[out_index].len() != output_count * 4 {
        return Err(RunError::BufferSizeMismatch {
            expected: output_count * 4,
            got: storage[out_index].len(),
        });
    }

    // Clear output.
    for out_linear in 0..output_count {
        write_f32(
            &mut storage[out_index][out_linear * 4..(out_linear + 1) * 4],
            match kernel.operator {
                BinaryAssocElementOperator::Mul => 1.0,
                BinaryAssocElementOperator::Max => f32::NEG_INFINITY,
                BinaryAssocElementOperator::Min => f32::INFINITY,
                BinaryAssocElementOperator::Add => 0.0,
            },
        );
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
        let value = read_f32_view(&storage[*in_buf], in_acc, &in_coords)?;
        let current = read_f32(&storage[out_index], out_linear)?;
        let reduced = match kernel.operator {
            BinaryAssocElementOperator::Add => current + value,
            BinaryAssocElementOperator::Mul => current * value,
            BinaryAssocElementOperator::Max => current.max(value),
            BinaryAssocElementOperator::Min => current.min(value),
        };
        write_f32(
            &mut storage[out_index][out_linear * 4..(out_linear + 1) * 4],
            reduced,
        );
    }
    Ok(())
}

fn apply_op(op: &ElementOperator, stack: &mut Vec<f32>) -> Result<(), RunError> {
    match op {
        ElementOperator::Unary(unary) => {
            let x = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let y = match unary {
                UnaryElementOperator::Neg => -x,
                UnaryElementOperator::Exp => x.exp(),
                UnaryElementOperator::Log => x.ln(),
                UnaryElementOperator::Relu => x.max(0.0),
                UnaryElementOperator::Abs => x.abs(),
                UnaryElementOperator::Sqrt => x.sqrt(),
                UnaryElementOperator::Sin => x.sin(),
                UnaryElementOperator::Cos => x.cos(),
                UnaryElementOperator::Not
                | UnaryElementOperator::Floor
                | UnaryElementOperator::Ceil
                | UnaryElementOperator::Bitcast => {
                    return Err(RunError::UnsupportedOp(format!("{unary:?}")));
                }
            };
            stack.push(y);
        }
        ElementOperator::Binary(binary) => {
            let rhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let lhs = stack.pop().ok_or(RunError::RpnEmptyStack)?;
            let y = match binary {
                BinaryElementOperator::Pow => lhs.powf(rhs),
                BinaryElementOperator::Div => lhs / rhs,
                BinaryElementOperator::Sub => lhs - rhs,
                BinaryElementOperator::Assoc(assoc) => match assoc {
                    BinaryAssocElementOperator::Mul => lhs * rhs,
                    BinaryAssocElementOperator::Add => lhs + rhs,
                    BinaryAssocElementOperator::Max => lhs.max(rhs),
                    BinaryAssocElementOperator::Min => lhs.min(rhs),
                },
            };
            stack.push(y);
        }
        ElementOperator::Compare(_) | ElementOperator::Bitwise(_) => {
            return Err(RunError::UnsupportedOp(format!("{op:?}")));
        }
    }
    Ok(())
}

/// Dense gather for host sink readback only.
fn gather_view_bytes(buffer: &[u8], accessor: &Accessor) -> Result<Vec<u8>, RunError> {
    let count = buffer_shape_len(&accessor.shape);
    let mut out = vec![0u8; count * 4];
    let mut coords = vec![0u32; accessor.shape.len()];
    for linear in 0..count {
        linear_to_coords_into(linear, &accessor.shape, &mut coords);
        let value = read_f32_view(buffer, accessor, &coords)?;
        write_f32(&mut out[linear * 4..(linear + 1) * 4], value);
    }
    Ok(out)
}

fn read_f32_view(buffer: &[u8], accessor: &Accessor, coords: &[u32]) -> Result<f32, RunError> {
    let offset = accessor_offset(accessor, coords)?;
    read_f32(buffer, offset)
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

fn write_f32(bytes: &mut [u8], value: f32) {
    bytes.copy_from_slice(&value.to_le_bytes());
}
