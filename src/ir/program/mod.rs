//! A program: buffers, views, and a queue of kernel dispatches.
//!
//! Kernels never address buffers directly: every read and write goes through
//! a [`BufferView`] — a buffer plus an [`Accessor`] — so broadcast, transpose,
//! and squeeze are pitch tricks rather than copies. A [`Dispatch`] pairs a
//! [`Kernel`] with its argument views and one output view; the output view's
//! shape is the kernel's iteration space.
//!
//! Elementwise bodies are [`Expr`] trees ([`expr`]) whose leaves are stable
//! [`BufferViewRef`]s.
//!
//! # Dense kernel outputs
//!
//! Every dispatch's **output** is [`Accessor::Dense`] (C-contiguous in element
//! units). Shape and pitch count **elements**, not bytes.
//!
//! **Arguments** may use any accessor (broadcast, transpose, strided gather,
//! …). Layout conversion into a denser domain is always **read non-dense →
//! write dense** (a materializing copy when needed), never a non-dense store.
//!
//! Lowering already emits dense outputs; passes must preserve the rule.
//! [`Program::validate`] rejects any dispatch that does not.

mod expr;

pub use expr::Expr;

use super::accessor::{Accessor, element_count};
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

/// One kernel invocation: read `args`, write `output`.
///
/// `output` must be [`Accessor::Dense`] (see module-level dense-output law).
/// `args` may be non-dense.
#[derive(Debug, Clone, PartialEq)]
pub struct Dispatch {
    pub kernel: Kernel,
    pub args: Vec<BufferViewRef>,
    /// Iteration space and write view; must be dense.
    pub output: BufferViewRef,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kernel {
    /// Per-element expression over loads, evaluated at every coordinate of
    /// the output view. Load views must match the output shape; `args` is the
    /// free-load list of the expression (binding order for backends).
    Elementwise { expr: Expr },
    /// `args[0] @ args[1]` with matching batch dims.
    Matmul,
    /// Fold `args[0]` with `op` along `axes` (keepdims: output has size 1 there).
    Reduction { op: AssocOp, axes: Box<[usize]> },
}

impl Kernel {
    /// Elementwise kernel over `expr`.
    pub fn elementwise(expr: Expr) -> Self {
        Kernel::Elementwise { expr }
    }
}

impl Program {
    pub fn buffer(&self, r: BufferRef) -> &Buffer {
        &self.buffers[r.0]
    }

    pub fn view(&self, r: BufferViewRef) -> &BufferView {
        &self.views[r.0]
    }

    /// Shape of a dispatch's iteration space (its output view's shape).
    pub fn output_shape(&self, dispatch: &Dispatch) -> Box<[usize]> {
        self.view(dispatch.output).accessor.shape()
    }

    pub fn validate(&self) -> Result<(), Error> {
        for (i, view) in self.views.iter().enumerate() {
            if view.buffer.0 >= self.buffers.len() {
                return Err(Error(format!("view {i} buffer {} out of range", view.buffer.0)));
            }
            let strided = view.accessor.strided();
            if strided.shape.len() != strided.pitch.len() {
                return Err(Error(format!(
                    "view {i} shape {:?} and pitch {:?} rank mismatch",
                    strided.shape, strided.pitch
                )));
            }
            // Backends index through views unchecked; prove them in bounds here.
            if element_count(&strided.shape) > 0 {
                let max_index = strided.offset
                    + strided
                        .shape
                        .iter()
                        .zip(&strided.pitch)
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
        // Dense-output law: kernels write C-contiguous element grids only.
        if !self.view(dispatch.output).accessor.is_dense() {
            return Err(Error("kernel output must be dense".into()));
        }
        let out_shape = self.view(dispatch.output).accessor.shape();
        let arg_shape = |i: usize| self.view(dispatch.args[i]).accessor.shape();

        match &dispatch.kernel {
            Kernel::Elementwise { expr } => {
                // Expr arity is checked at construction ([`Expr::new_op`]).
                if expr.loads() != dispatch.args {
                    return Err(Error(
                        "elementwise args must be the free loads of the expression".into(),
                    ));
                }
                for (i, &r) in dispatch.args.iter().enumerate() {
                    let shape = self.view(r).accessor.shape();
                    if shape != out_shape {
                        return Err(Error(format!(
                            "elementwise arg {i} shape {shape:?} != output shape {out_shape:?}"
                        )));
                    }
                }
            }
            Kernel::Matmul => {
                if dispatch.args.len() != 2 {
                    return Err(Error(format!("matmul takes 2 args, got {}", dispatch.args.len())));
                }
                let (a, b) = (arg_shape(0), arg_shape(1));
                let rank = a.len();
                if rank < 2 || b.len() != rank || out_shape.len() != rank {
                    return Err(Error(format!(
                        "matmul ranks must match and be >= 2: {:?} @ {:?} -> {:?}",
                        a, b, out_shape
                    )));
                }
                if a[rank - 1] != b[rank - 2] {
                    return Err(Error(format!(
                        "matmul inner dims: {} != {}",
                        a[rank - 1],
                        b[rank - 2]
                    )));
                }
                let expected: Box<[usize]> = a[..rank - 1]
                    .iter()
                    .chain([&b[rank - 1]])
                    .copied()
                    .collect();
                if a[..rank - 2] != b[..rank - 2] || out_shape.as_ref() != expected.as_ref() {
                    return Err(Error(format!(
                        "matmul shapes: {:?} @ {:?} -> {:?}",
                        a, b, out_shape
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
                let input = arg_shape(0);
                if input.len() != out_shape.len() {
                    return Err(Error(format!(
                        "reduction rank mismatch: {:?} -> {:?}",
                        input, out_shape
                    )));
                }
                for axis in 0..out_shape.len() {
                    let want = if axes.contains(&axis) { 1 } else { input[axis] };
                    if out_shape[axis] != want {
                        return Err(Error(format!(
                            "reduction axis {axis}: {:?} over {axes:?} -> {:?}",
                            input, out_shape
                        )));
                    }
                }
                if let Some(axis) = axes.iter().find(|&&a| a >= out_shape.len()) {
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
            kernel: Kernel::elementwise(Expr::new_op(Op::ADD, [Expr::Load(va), Expr::Load(vb)])),
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
            kernel: Kernel::elementwise(Expr::new_op(Op::NEG, [Expr::Load(va)])),
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

    #[test]
    fn rejects_non_dense_kernel_output() {
        let mut p = empty_program();
        let a = push_buffer(&mut p, &[2, 3]);
        let out = push_buffer(&mut p, &[2, 3]);
        let va = dense_view(&mut p, a);
        // Transposed write view — valid as a *read*, not as a kernel output.
        p.views.push(BufferView {
            buffer: out,
            accessor: Accessor::dense([3, 2], 0).transpose().unwrap(),
        });
        let vout = BufferViewRef(p.views.len() - 1);
        p.queue.push(Dispatch {
            kernel: Kernel::elementwise(Expr::new_op(Op::NEG, [Expr::Load(va)])),
            args: vec![va],
            output: vout,
        });
        let err = p.validate().unwrap_err();
        assert!(err.0.contains("dense"), "{err}");
    }
}
