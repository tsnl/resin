//! CPU interpreter backend.
//!
//! Kernels consume views (buffer + accessor) directly: broadcast, transpose,
//! squeeze, and index are stride tricks, never densifying copies. Storage is
//! typed ([`Slot`]): f32 and u32 buffers stay separate. The artifact is the
//! validated IR program itself.

use super::{ArrayData, Error, HostArray, Jit};
use crate::ir::{
    Accessor, BufferData, BufferViewRef, Dispatch, Expr, Kernel, Program, RemapInfo, element_count,
};
use crate::ops::{AssocOp, BinaryOp, ElementType, Op, UnaryOp};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuJit;

/// Per-buffer host storage.
#[derive(Debug, Clone)]
enum Slot {
    F32(Vec<f32>),
    U32(Vec<u32>),
}

impl Slot {
    fn element_type(&self) -> ElementType {
        match self {
            Slot::F32(_) => ElementType::F32,
            Slot::U32(_) => ElementType::U32,
        }
    }

    fn f32s(&self) -> &[f32] {
        match self {
            Slot::F32(v) => v,
            Slot::U32(_) => panic!("expected f32 slot"),
        }
    }

    fn f32s_mut(&mut self) -> &mut [f32] {
        match self {
            Slot::F32(v) => v,
            Slot::U32(_) => panic!("expected f32 slot"),
        }
    }

    fn u32s(&self) -> &[u32] {
        match self {
            Slot::U32(v) => v,
            Slot::F32(_) => panic!("expected u32 slot"),
        }
    }
}

/// Typed scalar for elementwise evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Value {
    F(f32),
    U(u32),
}

impl Jit for CpuJit {
    type Artifact = Program;
    type Array = HostArray;

    fn lower(&self, program: &Program) -> Result<Program, Error> {
        program.validate()?;
        Ok(program.clone())
    }

    fn alloc(&self, shape: &[usize], element_type: ElementType) -> Result<HostArray, Error> {
        Ok(HostArray::zeros_typed(shape, element_type))
    }

    fn upload(&self, host: &HostArray) -> Result<HostArray, Error> {
        Ok(host.clone())
    }

    fn invoke(
        &self,
        program: &Program,
        params: &[&HostArray],
        outputs: &mut [&mut HostArray],
    ) -> Result<(), Error> {
        if params.len() != program.params.len() || outputs.len() != program.sinks.len() {
            return Err(Error::LeafCount);
        }

        let mut storage: Vec<Slot> = program
            .buffers
            .iter()
            .map(|buffer| match (&buffer.init, buffer.element_type) {
                (Some(BufferData::F32(init)), _) => Slot::F32(init.to_vec()),
                (Some(BufferData::U32(init)), _) => Slot::U32(init.to_vec()),
                (None, ElementType::F32) => Slot::F32(vec![0.0; buffer.len()]),
                (None, ElementType::U32) => Slot::U32(vec![0; buffer.len()]),
            })
            .collect();

        for (array, &param) in params.iter().zip(&program.params) {
            let view = program.view(param);
            let expected_type = program.buffer(view.buffer).element_type;
            if array.element_type() != expected_type {
                return Err(Error::ElementType {
                    expected: expected_type,
                    got: array.element_type(),
                });
            }
            // Dense param views: copy host data into the view's arena region.
            let acc = &view.accessor;
            let count = element_count(&acc.shape);
            if array.as_data().len() != count {
                return Err(Error::Size {
                    expected: count,
                    got: array.as_data().len(),
                });
            }
            scatter_dense(&mut storage[view.buffer.0], acc, array.as_data())?;
        }

        for dispatch in &program.queue {
            run_dispatch(program, dispatch, &mut storage);
        }

        for (array, &sink) in outputs.iter_mut().zip(&program.sinks) {
            let view = program.view(sink);
            let etype = program.buffer(view.buffer).element_type;
            if array.element_type() != etype {
                return Err(Error::ElementType {
                    expected: etype,
                    got: array.element_type(),
                });
            }
            match (etype, array.as_data_mut()) {
                (ElementType::F32, ArrayData::F32(dst)) => {
                    gather_f32(storage[view.buffer.0].f32s(), &view.accessor, dst)?;
                }
                (ElementType::U32, ArrayData::U32(dst)) => {
                    gather_u32(storage[view.buffer.0].u32s(), &view.accessor, dst)?;
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }
}

/// Execute one dispatch. The program is validated, so indexing is in bounds.
fn run_dispatch(program: &Program, dispatch: &Dispatch, storage: &mut [Slot]) {
    let out_view = program.view(dispatch.output);
    let out_buffer = out_view.buffer.0;
    let out = out_view.accessor.clone();
    let out_etype = program.buffer(out_view.buffer).element_type;
    let args: Vec<(usize, Accessor, ElementType)> = dispatch
        .args
        .iter()
        .map(|&r| {
            let view = program.view(r);
            (
                view.buffer.0,
                view.accessor.clone(),
                program.buffer(view.buffer).element_type,
            )
        })
        .collect();

    if matches!(&dispatch.kernel, Kernel::Remap { info } if info.is_scatter()) {
        // Clear only the dense output region (arenas hold many logical buffers).
        clear_dense_region(&mut storage[out_buffer], &out);
    }

    let count = element_count(&out.shape);
    let mut coords = vec![0; out.rank()];

    match &dispatch.kernel {
        Kernel::Elementwise { expr } => {
            let loads = resolve_loads(expr, program);
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                let value = eval_expr(expr, &loads, storage, &coords, out_etype);
                write_slot(&mut storage[out_buffer], out.index(&coords), value);
            }
        }

        Kernel::Matmul => {
            // DSL admits any float etype; CPU implements f32 today (the only float).
            assert!(
                out_etype.is_float(),
                "matmul requires float output, got {out_etype:?}"
            );
            let rank = out.rank();
            let k = args[0].1.shape[rank - 1];
            let mut a_coords = vec![0; rank];
            let mut b_coords = vec![0; rank];
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                a_coords.copy_from_slice(&coords);
                b_coords.copy_from_slice(&coords);
                let mut sum = 0.0;
                for t in 0..k {
                    a_coords[rank - 1] = t;
                    b_coords[rank - 2] = t;
                    sum += storage[args[0].0].f32s()[args[0].1.index(&a_coords)]
                        * storage[args[1].0].f32s()[args[1].1.index(&b_coords)];
                }
                storage[out_buffer].f32s_mut()[out.index(&coords)] = sum;
            }
        }

        Kernel::Reduction { op, axes } => {
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                let id = match out_etype {
                    ElementType::F32 => Value::F(op.identity_f32()),
                    ElementType::U32 => Value::U(op.identity_u32()),
                };
                write_slot(&mut storage[out_buffer], out.index(&coords), id);
            }
            let (in_buffer, input, in_etype) = &args[0];
            let mut in_coords = vec![0; input.rank()];
            for linear in 0..element_count(&input.shape) {
                decode(linear, &input.shape, &mut in_coords);
                coords.copy_from_slice(&in_coords);
                for &axis in axes.iter() {
                    coords[axis] = 0;
                }
                let value = read_slot(&storage[*in_buffer], input.index(&in_coords), *in_etype);
                let slot = out.index(&coords);
                let current = read_slot(&storage[out_buffer], slot, out_etype);
                write_slot(&mut storage[out_buffer], slot, reduce(*op, current, value));
            }
        }

        Kernel::Remap { info } => run_remap(info, &args, storage, out_buffer, &out, out_etype),
    }
}

fn run_remap(
    info: &RemapInfo,
    args: &[(usize, Accessor, ElementType)],
    storage: &mut [Slot],
    out_buffer: usize,
    out: &Accessor,
    out_etype: ElementType,
) {
    let (src_buf, src_acc, _) = &args[0];
    match info {
        RemapInfo::GatherRows => {
            let (idx_buf, idx_acc, _) = &args[1];
            let src_rows = src_acc.shape[0];
            let count = element_count(&out.shape);
            let mut coords = vec![0; out.rank()];
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                let row = storage[*idx_buf].u32s()[idx_acc.index(&coords[..1])] as usize;
                let row = row.min(src_rows.saturating_sub(1));
                let out_row = coords[0];
                coords[0] = row;
                let value = read_slot(&storage[*src_buf], src_acc.index(&coords), out_etype);
                coords[0] = out_row;
                // Dense output: linear index matches out.index for dense views.
                write_slot(&mut storage[out_buffer], out.index(&coords), value);
            }
        }
        RemapInfo::ScatterRows { operator } => {
            let (idx_buf, idx_acc, _) = &args[1];
            let out_rows = out.shape[0];
            let count = element_count(&src_acc.shape);
            let mut coords = vec![0; src_acc.rank()];
            let mut out_coords = vec![0; out.rank()];
            for linear in 0..count {
                decode(linear, &src_acc.shape, &mut coords);
                let row = storage[*idx_buf].u32s()[idx_acc.index(&coords[..1])] as usize;
                if row >= out_rows {
                    continue;
                }
                let value = read_slot(&storage[*src_buf], src_acc.index(&coords), out_etype);
                out_coords.copy_from_slice(&coords);
                out_coords[0] = row;
                let out_i = out.index(&out_coords);
                let value = match operator {
                    None => value,
                    Some(AssocOp::Add) => {
                        let current = read_slot(&storage[out_buffer], out_i, out_etype);
                        reduce(AssocOp::Add, current, value)
                    }
                    Some(op) => panic!("unsupported scatter operator {op:?}"),
                };
                write_slot(&mut storage[out_buffer], out_i, value);
            }
        }
        RemapInfo::ScatterView { map } => {
            let count = element_count(&src_acc.shape);
            let mut coords = vec![0; src_acc.rank()];
            for linear in 0..count {
                decode(linear, &src_acc.shape, &mut coords);
                let value = read_slot(&storage[*src_buf], src_acc.index(&coords), out_etype);
                write_slot(&mut storage[out_buffer], map.index(&coords), value);
            }
        }
        RemapInfo::GatherView { map } => {
            // `map` yields a dense-logical linear index into `src_acc.shape`
            // (dual of scatter's dense-output addressing). Decode → source
            // view address so broadcast/strided sources work. OOR → zero.
            let src_n = element_count(&src_acc.shape);
            let count = element_count(&out.shape);
            let mut out_coords = vec![0; out.rank()];
            let mut src_coords = vec![0; src_acc.rank()];
            for linear in 0..count {
                decode(linear, &out.shape, &mut out_coords);
                let logical = map.index(&out_coords);
                let value = if logical >= src_n {
                    zero_value(out_etype)
                } else {
                    decode(logical, &src_acc.shape, &mut src_coords);
                    read_slot(
                        &storage[*src_buf],
                        src_acc.index(&src_coords),
                        out_etype,
                    )
                };
                write_slot(&mut storage[out_buffer], out.index(&out_coords), value);
            }
        }
    }
}

fn zero_value(etype: ElementType) -> Value {
    match etype {
        ElementType::F32 => Value::F(0.0),
        ElementType::U32 => Value::U(0),
    }
}

fn reduce(op: AssocOp, acc: Value, value: Value) -> Value {
    match (acc, value) {
        (Value::F(a), Value::F(v)) => Value::F(op.apply_f32(a, v)),
        (Value::U(a), Value::U(v)) => Value::U(op.apply_u32(a, v)),
        _ => panic!("mixed-type reduction"),
    }
}

fn resolve_loads(
    expr: &Expr,
    program: &Program,
) -> Vec<(BufferViewRef, usize, Accessor, ElementType)> {
    expr.loads()
        .into_iter()
        .map(|r| {
            let view = program.view(r);
            (
                r,
                view.buffer.0,
                view.accessor.clone(),
                program.buffer(view.buffer).element_type,
            )
        })
        .collect()
}

fn clear_dense_region(slot: &mut Slot, out: &Accessor) {
    let count = element_count(&out.shape);
    let mut coords = vec![0; out.rank()];
    for linear in 0..count {
        decode(linear, &out.shape, &mut coords);
        let i = out.index(&coords);
        match slot {
            Slot::F32(v) => v[i] = 0.0,
            Slot::U32(v) => v[i] = 0,
        }
    }
}

/// Write a dense host array into a (possibly offset) dense view.
fn scatter_dense(slot: &mut Slot, accessor: &Accessor, data: &ArrayData) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if data.len() != count {
        return Err(Error::Size {
            expected: count,
            got: data.len(),
        });
    }
    let mut coords = vec![0; accessor.rank()];
    match (slot, data) {
        (Slot::F32(dst), ArrayData::F32(src)) => {
            for (linear, &value) in src.iter().enumerate() {
                decode(linear, &accessor.shape, &mut coords);
                dst[accessor.index(&coords)] = value;
            }
            Ok(())
        }
        (Slot::U32(dst), ArrayData::U32(src)) => {
            for (linear, &value) in src.iter().enumerate() {
                decode(linear, &accessor.shape, &mut coords);
                dst[accessor.index(&coords)] = value;
            }
            Ok(())
        }
        (Slot::F32(_), ArrayData::U32(_)) => Err(Error::ElementType {
            expected: ElementType::F32,
            got: ElementType::U32,
        }),
        (Slot::U32(_), ArrayData::F32(_)) => Err(Error::ElementType {
            expected: ElementType::U32,
            got: ElementType::F32,
        }),
    }
}

fn gather_f32(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size {
            expected: count,
            got: out.len(),
        });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &accessor.shape, &mut coords);
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn gather_u32(buffer: &[u32], accessor: &Accessor, out: &mut [u32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size {
            expected: count,
            got: out.len(),
        });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &accessor.shape, &mut coords);
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn eval_expr(
    expr: &Expr,
    loads: &[(BufferViewRef, usize, Accessor, ElementType)],
    storage: &[Slot],
    coords: &[usize],
    out_etype: ElementType,
) -> Value {
    match expr {
        Expr::Load(view) => {
            let (_, buffer, accessor, etype) = loads
                .iter()
                .find(|(r, _, _, _)| r == view)
                .expect("load is a free load of the expression");
            read_slot(&storage[*buffer], accessor.index(coords), *etype)
        }
        Expr::Op { op, args } => {
            let mut vals = [Value::F(0.0); 2];
            for (i, arg) in args.iter().enumerate() {
                vals[i] = eval_expr(arg, loads, storage, coords, out_etype);
            }
            apply_op(*op, &vals[..op.arity()], out_etype)
        }
    }
}

fn apply_op(op: Op, args: &[Value], out_etype: ElementType) -> Value {
    match op {
        Op::Unary(u) => {
            let x = args[0];
            match u {
                UnaryOp::Neg => match x {
                    Value::F(v) => Value::F(-v),
                    Value::U(v) => Value::U(v.wrapping_neg()),
                },
                UnaryOp::Exp => Value::F(x.as_f32().exp()),
                UnaryOp::Log => Value::F(x.as_f32().ln()),
                UnaryOp::Relu => Value::F(x.as_f32().max(0.0)),
                UnaryOp::Abs => Value::F(x.as_f32().abs()),
                UnaryOp::Sqrt => Value::F(x.as_f32().sqrt()),
                UnaryOp::Sin => Value::F(x.as_f32().sin()),
                UnaryOp::Cos => Value::F(x.as_f32().cos()),
                UnaryOp::Floor => Value::F(x.as_f32().floor()),
                UnaryOp::Ceil => Value::F(x.as_f32().ceil()),
                UnaryOp::Bitcast { to } => match (x, to) {
                    (Value::F(v), ElementType::U32) => Value::U(v.to_bits()),
                    (Value::U(v), ElementType::F32) => Value::F(f32::from_bits(v)),
                    (v, _) => v,
                },
                UnaryOp::Cast { to } => match (x, to) {
                    (Value::F(v), ElementType::U32) => Value::U(v as u32),
                    (Value::U(v), ElementType::F32) => Value::F(v as f32),
                    (v, _) => v,
                },
            }
        }
        Op::Binary(b) => match (args[0], args[1]) {
            (Value::F(a), Value::F(b_val)) => match b {
                BinaryOp::Assoc(op) => Value::F(op.apply_f32(a, b_val)),
                BinaryOp::Sub => Value::F(a - b_val),
                BinaryOp::Div => Value::F(a / b_val),
                BinaryOp::Pow => Value::F(a.powf(b_val)),
                BinaryOp::CmpEq => mask_f(a == b_val, out_etype),
                BinaryOp::CmpNe => mask_f(a != b_val, out_etype),
                BinaryOp::CmpLt => mask_f(a < b_val, out_etype),
                BinaryOp::CmpLe => mask_f(a <= b_val, out_etype),
                BinaryOp::CmpGt => mask_f(a > b_val, out_etype),
                BinaryOp::CmpGe => mask_f(a >= b_val, out_etype),
                BinaryOp::Band | BinaryOp::Bor | BinaryOp::Bxor | BinaryOp::Shl | BinaryOp::Shr => {
                    panic!("bitwise ops are u32-only")
                }
            },
            (Value::U(a), Value::U(b_val)) => match b {
                BinaryOp::Assoc(op) => Value::U(op.apply_u32(a, b_val)),
                BinaryOp::Sub => Value::U(a.wrapping_sub(b_val)),
                BinaryOp::Div => Value::U(a.checked_div(b_val).unwrap_or(0)),
                BinaryOp::Pow => panic!("pow on u32"),
                BinaryOp::CmpEq => mask_u(a == b_val, out_etype),
                BinaryOp::CmpNe => mask_u(a != b_val, out_etype),
                BinaryOp::CmpLt => mask_u(a < b_val, out_etype),
                BinaryOp::CmpLe => mask_u(a <= b_val, out_etype),
                BinaryOp::CmpGt => mask_u(a > b_val, out_etype),
                BinaryOp::CmpGe => mask_u(a >= b_val, out_etype),
                BinaryOp::Band => Value::U(a & b_val),
                BinaryOp::Bor => Value::U(a | b_val),
                BinaryOp::Bxor => Value::U(a ^ b_val),
                BinaryOp::Shl => Value::U(a.wrapping_shl(b_val & 31)),
                BinaryOp::Shr => Value::U(a.wrapping_shr(b_val & 31)),
            },
            _ => panic!("mixed-type binary operands"),
        },
    }
}

fn mask_f(pred: bool, out_etype: ElementType) -> Value {
    match out_etype {
        ElementType::F32 => Value::F(pred as u32 as f32),
        ElementType::U32 => Value::U(pred as u32),
    }
}

fn mask_u(pred: bool, out_etype: ElementType) -> Value {
    mask_f(pred, out_etype)
}

impl Value {
    fn as_f32(self) -> f32 {
        match self {
            Value::F(v) => v,
            Value::U(v) => v as f32,
        }
    }
}

fn read_slot(slot: &Slot, index: usize, etype: ElementType) -> Value {
    match (slot, etype) {
        (Slot::F32(v), ElementType::F32) => Value::F(v[index]),
        (Slot::U32(v), ElementType::U32) => Value::U(v[index]),
        (s, e) => panic!("slot {:?} vs read {:?}", s.element_type(), e),
    }
}

fn write_slot(slot: &mut Slot, index: usize, value: Value) {
    match (slot, value) {
        (Slot::F32(v), Value::F(x)) => v[index] = x,
        (Slot::U32(v), Value::U(x)) => v[index] = x,
        (s, v) => panic!("slot {:?} vs write {:?}", s.element_type(), v),
    }
}

/// Row-major linear index → coordinates.
fn decode(linear: usize, shape: &[usize], coords: &mut [usize]) {
    debug_assert_eq!(coords.len(), shape.len());
    let mut rem = linear;
    for axis in (0..shape.len()).rev() {
        coords[axis] = rem % shape[axis];
        rem /= shape[axis];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsl::Tensor;
    use crate::dsl::grad_wrt;

    #[derive(resin_macros::Tree, Clone)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn add_runs() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let out = f
            .call(&Pair {
                a: HostArray::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
                b: HostArray::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
            })
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sum_axes_then_squeeze_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.sum_axes(&[0, 1]).squeeze_all());
        let out = f
            .call(&HostArray::from_f32(
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))
            .unwrap();
        assert!((out.scalar() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn scalar_mul_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.clone() * Tensor::scalar(0.5));
        let out = f
            .call(&HostArray::from_f32(&[2, 2], &[2.0, 4.0, 6.0, 8.0]))
            .unwrap();
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bias_add_broadcast_view_runs() {
        #[derive(resin_macros::Tree, Clone)]
        struct LinearIn<T> {
            x: T,
            w: T,
            bias: T,
        }
        let f = CpuJit.jit(|input: &LinearIn<Tensor>| {
            let y = input.x.matmul(&input.w);
            y.clone() + input.bias.broadcast_to(y.shape(), &[1])
        });
        let out = f
            .call(&LinearIn {
                x: HostArray::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                w: HostArray::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                bias: HostArray::from_f32(&[3], &[10.0, 20.0, 30.0]),
            })
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn matmul_transpose_view_runs() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.matmul(&p.b.transpose()));
        let out = f
            .call(&Pair {
                a: HostArray::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                b: HostArray::from_f32(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            })
            .unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn unary_chain_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.exp().log().sqrt());
        let out = f.call(&HostArray::from_f32(&[2], &[4.0, 9.0])).unwrap();
        assert_eq!(out.data(), &[2.0, 3.0]);
    }

    #[test]
    fn gradient_descends_a_quadratic() {
        // One SGD step on loss = mean((x - 3)^2) moves x toward 3.
        let f = CpuJit.jit(|x: &Tensor| {
            let diff = x.clone() - Tensor::scalar(3.0);
            let loss = (diff.clone() * diff).mean_all();
            let grad = grad_wrt(&loss, x).expect("grad");
            x.clone() - grad * Tensor::scalar(0.1)
        });
        let out = f.call(&HostArray::from_f32(&[2], &[0.0, 5.0])).unwrap();
        // grad = 2(x-3)/2 = (x-3); step: 0 → 0.3, 5 → 4.8
        assert!((out.data()[0] - 0.3).abs() < 1e-5, "{:?}", out.data());
        assert!((out.data()[1] - 4.8).abs() < 1e-5, "{:?}", out.data());
    }

    #[test]
    fn u32_add_runs() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let out = f
            .call(&Pair {
                a: HostArray::from_u32(&[3], &[1, 2, 3]),
                b: HostArray::from_u32(&[3], &[10, 20, 30]),
            })
            .unwrap();
        assert_eq!(out.to_u32(), vec![11, 22, 33]);
    }

    #[test]
    fn transpose_sink_runs() {
        // A bare transpose output is a strided sink; layout densifies it.
        let f = CpuJit.jit(|x: &Tensor| x.transpose());
        let out = f
            .call(&HostArray::from_f32(
                &[2, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ))
            .unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_eq!(out.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn permute_sink_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.permute(&[2, 0, 1]));
        let out = f
            .call(&HostArray::from_f32(
                &[2, 2, 3],
                // [0,0,:]=0,1,2  [0,1,:]=3,4,5  [1,0,:]=6,7,8  [1,1,:]=9,10,11
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            ))
            .unwrap();
        // shape [3, 2, 2]; out[c, n, h] = in[n, h, c]
        assert_eq!(out.shape(), &[3, 2, 2]);
        assert_eq!(
            out.data(),
            &[0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
        );
    }

    #[test]
    fn unfold_sink_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.unfold(0, 3, 1));
        let out = f
            .call(&HostArray::from_f32(
                &[5],
                &[10.0, 20.0, 30.0, 40.0, 50.0],
            ))
            .unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.data(),
            &[
                10.0, 20.0, 30.0, // window 0
                20.0, 30.0, 40.0, // window 1
                30.0, 40.0, 50.0, // window 2
            ]
        );
    }

    #[test]
    fn unfold_mul_sum_is_sliding_dot() {
        // 1-d conv: unfold → broadcast kernel → mul → sum over window.
        let f = CpuJit.jit(|p: &Pair<Tensor>| {
            let windows = p.a.unfold(0, 3, 1); // [3, 3]
            let k = p.b.broadcast_to(&[3, 3], &[1]);
            (windows * k).sum_axes(&[1]).squeeze_all()
        });
        let out = f
            .call(&Pair {
                a: HostArray::from_f32(&[5], &[1.0, 2.0, 3.0, 4.0, 5.0]),
                b: HostArray::from_f32(&[3], &[1.0, 0.0, -1.0]),
            })
            .unwrap();
        // dots: 1-3=-2, 2-4=-2, 3-5=-2
        assert_eq!(out.data(), &[-2.0, -2.0, -2.0]);
    }

    #[test]
    fn grad_through_overlapping_unfold() {
        // loss = sum(unfold(x, size=3, step=1)); each interior point appears in 3 windows.
        let f = CpuJit.jit(|x: &Tensor| {
            let loss = x.unfold(0, 3, 1).sum_axes(&[0, 1]).squeeze_all();
            grad_wrt(&loss, x).expect("grad")
        });
        let out = f
            .call(&HostArray::from_f32(&[5], &[0.0, 0.0, 0.0, 0.0, 0.0]))
            .unwrap();
        // windows cover indices: [0,1,2], [1,2,3], [2,3,4] → counts 1,2,3,2,1
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 2.0, 1.0]);
    }
}
