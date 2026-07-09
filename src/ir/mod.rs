//! Intermediate representation: a queue of kernel dispatches over buffers.
//!
//! A [`Program`] owns flat buffers of f32 elements. Kernels never address
//! buffers directly: every read and write goes through a [`BufferView`] — a
//! buffer plus an [`Accessor`] — so broadcast, transpose, and squeeze are
//! pitch tricks rather than copies. A [`Dispatch`] pairs a [`Kernel`] with its
//! argument views and one output view; the output view's shape is the kernel's
//! iteration space.

mod accessor;
mod expr;
pub mod optimize;

pub use accessor::{Accessor, dense_pitch, element_count};
pub use expr::Expr;

use crate::ops::AssocOp;

/// Invalid-IR error. The message is the diagnostic; there is no error taxonomy.
#[derive(Debug, thiserror::Error)]
#[error("invalid IR: {0}")]
pub struct Error(pub String);

/// Index into [`Program::buffers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferRef(pub usize);

/// Index into [`Program::views`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferViewRef(pub usize);

/// A compiled program: kernels to run in order, plus the buffer slots that
/// correspond to the caller's parameter and output leaves (in tree-walk order).
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub params: Vec<BufferRef>,
    pub sinks: Vec<BufferViewRef>,
    pub queue: Vec<Dispatch>,
    pub buffers: Vec<Buffer>,
    pub views: Vec<BufferView>,
}

/// One flat f32 buffer. `init` marks a constant with its contents.
#[derive(Debug, Clone, PartialEq)]
pub struct Buffer {
    pub shape: Box<[usize]>,
    pub init: Option<Box<[f32]>>,
}

impl Buffer {
    /// Element count ("is_empty" is meaningless here: scalars have one).
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        element_count(&self.shape)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BufferView {
    pub buffer: BufferRef,
    pub accessor: Accessor,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch {
    pub kernel: Kernel,
    pub args: Vec<BufferViewRef>,
    pub output: BufferViewRef,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kernel {
    /// Per-element expression over loads, evaluated at every coordinate of
    /// the output view. Load views must match the output shape; `args` is the
    /// free-load list of the expression (binding order for backends).
    Elementwise(Expr),
    /// `args[0] @ args[1]` with matching batch dims.
    Matmul,
    /// Fold `args[0]` with `op` along `axes` (keepdims: output has size 1 there).
    Reduction { op: AssocOp, axes: Box<[usize]> },
}

impl Program {
    pub fn buffer(&self, r: BufferRef) -> &Buffer {
        &self.buffers[r.0]
    }

    pub fn view(&self, r: BufferViewRef) -> &BufferView {
        &self.views[r.0]
    }

    /// Shape of a dispatch's iteration space (its output view's shape).
    pub fn output_shape(&self, dispatch: &Dispatch) -> &[usize] {
        &self.view(dispatch.output).accessor.shape
    }

    pub fn validate(&self) -> Result<(), Error> {
        for (i, view) in self.views.iter().enumerate() {
            if view.buffer.0 >= self.buffers.len() {
                return Err(Error(format!("view {i} buffer {} out of range", view.buffer.0)));
            }
            let accessor = &view.accessor;
            if accessor.shape.len() != accessor.pitch.len() {
                return Err(Error(format!(
                    "view {i} shape {:?} and pitch {:?} rank mismatch",
                    accessor.shape, accessor.pitch
                )));
            }
            // Backends index through views unchecked; prove them in bounds here.
            if element_count(&accessor.shape) > 0 {
                let max_index = accessor.offset
                    + accessor
                        .shape
                        .iter()
                        .zip(&accessor.pitch)
                        .map(|(&d, &p)| (d - 1) * p)
                        .sum::<usize>();
                let len = self.buffer(view.buffer).len();
                if max_index >= len {
                    return Err(Error(format!(
                        "view {i} reaches element {max_index} of a {len}-element buffer"
                    )));
                }
            }
        }
        for r in &self.params {
            if r.0 >= self.buffers.len() {
                return Err(Error(format!("param buffer {} out of range", r.0)));
            }
        }
        for r in &self.sinks {
            if r.0 >= self.views.len() {
                return Err(Error(format!("sink view {} out of range", r.0)));
            }
        }
        for (i, buffer) in self.buffers.iter().enumerate() {
            if let Some(init) = &buffer.init
                && init.len() != buffer.len()
            {
                return Err(Error(format!(
                    "buffer {i} init has {} elements, shape {:?} wants {}",
                    init.len(),
                    buffer.shape,
                    buffer.len()
                )));
            }
        }
        for (i, dispatch) in self.queue.iter().enumerate() {
            self.validate_dispatch(dispatch)
                .map_err(|Error(msg)| Error(format!("dispatch {i}: {msg}")))?;
        }
        Ok(())
    }

    fn validate_dispatch(&self, dispatch: &Dispatch) -> Result<(), Error> {
        for r in dispatch.args.iter().chain([&dispatch.output]) {
            if r.0 >= self.views.len() {
                return Err(Error(format!("view {} out of range", r.0)));
            }
        }
        let out = &self.view(dispatch.output).accessor;
        let arg = |i: usize| &self.view(dispatch.args[i]).accessor;

        match &dispatch.kernel {
            Kernel::Elementwise(expr) => {
                expr.validate()?;
                if expr.loads() != dispatch.args {
                    return Err(Error(
                        "elementwise args must be the free loads of the expression".into(),
                    ));
                }
                for (i, &r) in dispatch.args.iter().enumerate() {
                    let shape = &self.view(r).accessor.shape;
                    if shape != &out.shape {
                        return Err(Error(format!(
                            "elementwise arg {i} shape {shape:?} != output shape {:?}",
                            out.shape
                        )));
                    }
                }
            }
            Kernel::Matmul => {
                if dispatch.args.len() != 2 {
                    return Err(Error(format!("matmul takes 2 args, got {}", dispatch.args.len())));
                }
                let (a, b) = (arg(0), arg(1));
                let rank = a.rank();
                if rank < 2 || b.rank() != rank || out.rank() != rank {
                    return Err(Error(format!(
                        "matmul ranks must match and be >= 2: {:?} @ {:?} -> {:?}",
                        a.shape, b.shape, out.shape
                    )));
                }
                if a.shape[rank - 1] != b.shape[rank - 2] {
                    return Err(Error(format!(
                        "matmul inner dims: {} != {}",
                        a.shape[rank - 1],
                        b.shape[rank - 2]
                    )));
                }
                let expected: Box<[usize]> = a.shape[..rank - 1]
                    .iter()
                    .chain([&b.shape[rank - 1]])
                    .copied()
                    .collect();
                if a.shape[..rank - 2] != b.shape[..rank - 2] || out.shape != expected {
                    return Err(Error(format!(
                        "matmul shapes: {:?} @ {:?} -> {:?}",
                        a.shape, b.shape, out.shape
                    )));
                }
            }
            Kernel::Reduction { axes, .. } => {
                if dispatch.args.len() != 1 {
                    return Err(Error(format!(
                        "reduction takes 1 arg, got {}",
                        dispatch.args.len()
                    )));
                }
                let input = arg(0);
                if input.rank() != out.rank() {
                    return Err(Error(format!(
                        "reduction rank mismatch: {:?} -> {:?}",
                        input.shape, out.shape
                    )));
                }
                for axis in 0..out.rank() {
                    let want = if axes.contains(&axis) { 1 } else { input.shape[axis] };
                    if out.shape[axis] != want {
                        return Err(Error(format!(
                            "reduction axis {axis}: {:?} over {axes:?} -> {:?}",
                            input.shape, out.shape
                        )));
                    }
                }
                if let Some(axis) = axes.iter().find(|&&a| a >= out.rank()) {
                    return Err(Error(format!("reduction axis {axis} out of range")));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    fn dense_view(program: &mut Program, buffer: BufferRef) -> BufferViewRef {
        let shape = program.buffer(buffer).shape.clone();
        program.views.push(BufferView {
            buffer,
            accessor: Accessor::dense(shape, 0),
        });
        BufferViewRef(program.views.len() - 1)
    }

    fn empty_program() -> Program {
        Program {
            params: vec![],
            sinks: vec![],
            queue: vec![],
            buffers: vec![],
            views: vec![],
        }
    }

    fn push_buffer(program: &mut Program, shape: &[usize]) -> BufferRef {
        program.buffers.push(Buffer { shape: shape.into(), init: None });
        BufferRef(program.buffers.len() - 1)
    }

    #[test]
    fn elementwise_add_validates() {
        let mut p = empty_program();
        let a = push_buffer(&mut p, &[2, 3]);
        let b = push_buffer(&mut p, &[2, 3]);
        let out = push_buffer(&mut p, &[2, 3]);
        let (va, vb, vout) = (
            dense_view(&mut p, a),
            dense_view(&mut p, b),
            dense_view(&mut p, out),
        );
        p.queue.push(Dispatch {
            kernel: Kernel::Elementwise(Expr::apply_op(Op::ADD, [va, vb])),
            args: vec![va, vb],
            output: vout,
        });
        p.validate().unwrap();
    }

    #[test]
    fn elementwise_rejects_shape_mismatch() {
        let mut p = empty_program();
        let a = push_buffer(&mut p, &[2, 3]);
        let out = push_buffer(&mut p, &[3, 2]);
        let (va, vout) = (dense_view(&mut p, a), dense_view(&mut p, out));
        p.queue.push(Dispatch {
            kernel: Kernel::Elementwise(Expr::apply_op(Op::NEG, [va])),
            args: vec![va],
            output: vout,
        });
        assert!(p.validate().is_err());
    }

    #[test]
    fn matmul_validates() {
        let mut p = empty_program();
        let a = push_buffer(&mut p, &[2, 4]);
        let b = push_buffer(&mut p, &[4, 3]);
        let out = push_buffer(&mut p, &[2, 3]);
        let (va, vb, vout) = (
            dense_view(&mut p, a),
            dense_view(&mut p, b),
            dense_view(&mut p, out),
        );
        p.queue.push(Dispatch {
            kernel: Kernel::Matmul,
            args: vec![va, vb],
            output: vout,
        });
        p.validate().unwrap();
    }

    #[test]
    fn reduction_validates() {
        let mut p = empty_program();
        let a = push_buffer(&mut p, &[2, 3, 4]);
        let out = push_buffer(&mut p, &[2, 1, 4]);
        let (va, vout) = (dense_view(&mut p, a), dense_view(&mut p, out));
        p.queue.push(Dispatch {
            kernel: Kernel::Reduction { op: AssocOp::Add, axes: Box::from([1]) },
            args: vec![va],
            output: vout,
        });
        p.validate().unwrap();
    }

    #[test]
    fn rejects_out_of_range_view_buffer() {
        let mut p = empty_program();
        p.views.push(BufferView {
            buffer: BufferRef(0),
            accessor: Accessor::dense([4], 0),
        });
        assert!(p.validate().is_err());
    }

}
