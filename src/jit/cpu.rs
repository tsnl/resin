//! CPU interpreter backend.
//!
//! Kernels consume views (buffer + accessor) directly: broadcast, transpose,
//! and squeeze are pitch tricks, never densifying copies. The artifact is the
//! validated IR program itself. Hot loops use materialised [`Strided`] maps so
//! indexing does not re-walk the accessor tree per element.

use super::{Array, Error, Jit};
use crate::ir::{
    Accessor, BufferViewRef, Dispatch, Expr, Kernel, Program, Strided, element_count,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CpuJit;

impl Jit for CpuJit {
    type Artifact = Program;

    fn lower(&self, program: &Program) -> Result<Program, Error> {
        program.validate()?;
        Ok(program.clone())
    }

    fn invoke(
        &self,
        program: &Program,
        params: &[&Array],
        outputs: &mut [&mut Array],
    ) -> Result<(), Error> {
        if params.len() != program.params.len() || outputs.len() != program.sinks.len() {
            return Err(Error::LeafCount);
        }

        let mut storage: Vec<Vec<f32>> = program
            .buffers
            .iter()
            .map(|buffer| match &buffer.init {
                Some(init) => init.to_vec(),
                None => vec![0.0; buffer.len()],
            })
            .collect();

        for (array, &buffer) in params.iter().zip(&program.params) {
            let slot = &mut storage[buffer.0];
            if array.data().len() != slot.len() {
                return Err(Error::Size { expected: slot.len(), got: array.data().len() });
            }
            slot.copy_from_slice(array.data());
        }

        for dispatch in &program.queue {
            run_dispatch(program, dispatch, &mut storage);
        }

        for (array, &sink) in outputs.iter_mut().zip(&program.sinks) {
            let view = program.view(sink);
            gather(&storage[view.buffer.0], &view.accessor, array.data_mut())?;
        }
        Ok(())
    }
}

/// Execute one dispatch. The program is validated, so indexing is in bounds.
fn run_dispatch(program: &Program, dispatch: &Dispatch, storage: &mut [Vec<f32>]) {
    let out_view = program.view(dispatch.output);
    let out_buffer = out_view.buffer.0;
    let out = out_view.accessor.strided();
    let args: Vec<(usize, Strided)> = dispatch
        .args
        .iter()
        .map(|&r| {
            let view = program.view(r);
            (view.buffer.0, view.accessor.strided())
        })
        .collect();

    let count = element_count(&out.shape);
    let mut coords = vec![0; out.rank()];

    match &dispatch.kernel {
        Kernel::Elementwise { expr } => {
            // Materialise each free load once; the tree indexes this table.
            let loads = resolve_loads(expr, program);
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                let value = eval_expr(expr, &loads, storage, &coords);
                storage[out_buffer][out.index(&coords)] = value;
            }
        }

        Kernel::Matmul => {
            let rank = out.rank();
            let k = args[0].1.shape[rank - 1];
            let mut a_coords = vec![0; rank];
            let mut b_coords = vec![0; rank];
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                a_coords.copy_from_slice(&coords); // [batch…, i, _]
                b_coords.copy_from_slice(&coords); // [batch…, _, j]
                let mut sum = 0.0;
                for t in 0..k {
                    a_coords[rank - 1] = t;
                    b_coords[rank - 2] = t;
                    sum += storage[args[0].0][args[0].1.index(&a_coords)]
                        * storage[args[1].0][args[1].1.index(&b_coords)];
                }
                storage[out_buffer][out.index(&coords)] = sum;
            }
        }

        Kernel::Reduction { op, axes } => {
            for linear in 0..count {
                decode(linear, &out.shape, &mut coords);
                storage[out_buffer][out.index(&coords)] = op.identity();
            }
            let (in_buffer, input) = &args[0];
            let mut in_coords = vec![0; input.rank()];
            for linear in 0..element_count(&input.shape) {
                decode(linear, &input.shape, &mut in_coords);
                coords.copy_from_slice(&in_coords);
                for &axis in axes.iter() {
                    coords[axis] = 0;
                }
                let value = storage[*in_buffer][input.index(&in_coords)];
                let slot = out.index(&coords);
                storage[out_buffer][slot] = op.apply(storage[out_buffer][slot], value);
            }
        }
    }
}

/// Buffer + affine map for each free load in `expr` (one strided() per view).
fn resolve_loads(expr: &Expr, program: &Program) -> Vec<(BufferViewRef, usize, Strided)> {
    expr.loads()
        .into_iter()
        .map(|r| {
            let view = program.view(r);
            (r, view.buffer.0, view.accessor.strided())
        })
        .collect()
}

/// Densify a view into a row-major host slice (sink readback).
fn gather(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let s = accessor.strided();
    let count = element_count(&s.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; s.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &s.shape, &mut coords);
        *slot = buffer[s.index(&coords)];
    }
    Ok(())
}

/// Evaluate an elementwise expression at `coords` using pre-materialised loads.
fn eval_expr(
    expr: &Expr,
    loads: &[(BufferViewRef, usize, Strided)],
    storage: &[Vec<f32>],
    coords: &[usize],
) -> f32 {
    match expr {
        Expr::Load(view) => {
            let (_, buffer, strided) = loads
                .iter()
                .find(|(r, _, _)| r == view)
                .expect("load is a free load of the expression");
            storage[*buffer][strided.index(coords)]
        }
        Expr::Op { op, args } => {
            let mut vals = [0.0f32; 2];
            for (i, arg) in args.iter().enumerate() {
                vals[i] = eval_expr(arg, loads, storage, coords);
            }
            op.apply(&vals[..op.arity()])
        }
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

    #[derive(resin_macros::Tree)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn add_runs() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[4], &[1.0, 2.0, 3.0, 4.0]),
                b: Array::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]),
            })
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn sum_axes_then_squeeze_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.sum_axes(&[0, 1]).squeeze_all());
        let out = f
            .call(&Array::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .unwrap();
        assert!((out.scalar() - 21.0).abs() < 1e-5);
    }

    #[test]
    fn scalar_mul_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.clone() * Tensor::scalar(0.5));
        let out = f.call(&Array::from_f32(&[2, 2], &[2.0, 4.0, 6.0, 8.0])).unwrap();
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn bias_add_broadcast_view_runs() {
        #[derive(resin_macros::Tree)]
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
                x: Array::from_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]),
                w: Array::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                bias: Array::from_f32(&[3], &[10.0, 20.0, 30.0]),
            })
            .unwrap();
        assert_eq!(out.data(), &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn matmul_transpose_view_runs() {
        let f = CpuJit.jit(|p: &Pair<Tensor>| p.a.matmul(&p.b.transpose()));
        let out = f
            .call(&Pair {
                a: Array::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                b: Array::from_f32(&[2, 3], &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            })
            .unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out.data(), &[1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn unary_chain_runs() {
        let f = CpuJit.jit(|x: &Tensor| x.exp().log().sqrt());
        let out = f.call(&Array::from_f32(&[2], &[4.0, 9.0])).unwrap();
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
        let out = f.call(&Array::from_f32(&[2], &[0.0, 5.0])).unwrap();
        // grad = 2(x-3)/2 = (x-3); step: 0 → 0.3, 5 → 4.8
        assert!((out.data()[0] - 0.3).abs() < 1e-5, "{:?}", out.data());
        assert!((out.data()[1] - 4.8).abs() < 1e-5, "{:?}", out.data());
    }
}
