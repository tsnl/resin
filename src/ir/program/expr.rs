//! Elementwise expressions: trees with stable load leaves.
//!
//! An elementwise kernel body is an [`Expr`]. Leaves are [`BufferViewRef`]s —
//! not dense argument slots — so fusion is substitution of one tree into
//! another, with no renumbering. Free loads of the result become the
//! dispatch's argument list.
//!
//! Well-formedness is checked at construction ([`Expr::new_op`]), not by a
//! separate validate pass.

use super::BufferViewRef;
use crate::ops::Op;

/// Tree expression over buffer views and scalar operators.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Read one element from this view at the kernel's output coordinates.
    Load(BufferViewRef),
    /// Apply `op` to `args` (`args.len() == op.arity()`). Build with [`Expr::new_op`].
    Op { op: Op, args: Box<[Expr]> },
}

impl Expr {
    /// Construct `op(args…)`. Arity is checked here — the only intended way
    /// to build [`Expr::Op`] nodes.
    pub fn new_op(op: Op, args: impl IntoIterator<Item = Expr>) -> Self {
        let args: Box<[Expr]> = args.into_iter().collect();
        assert_eq!(
            args.len(),
            op.arity(),
            "op {op:?} wants {} args, got {}",
            op.arity(),
            args.len()
        );
        Expr::Op { op, args }
    }

    /// Free loads in first-seen order (depth-first, left-to-right).
    pub fn loads(&self) -> Vec<BufferViewRef> {
        let mut out = Vec::new();
        self.collect_loads(&mut out);
        out
    }

    fn collect_loads(&self, out: &mut Vec<BufferViewRef>) {
        match self {
            Expr::Load(v) => {
                if !out.contains(v) {
                    out.push(*v);
                }
            }
            Expr::Op { args, .. } => {
                for arg in args.iter() {
                    arg.collect_loads(out);
                }
            }
        }
    }

    /// Number of nodes (loads and ops). Used as a soft fusion size budget.
    pub fn node_count(&self) -> usize {
        match self {
            Expr::Load(_) => 1,
            Expr::Op { args, .. } => 1 + args.iter().map(Expr::node_count).sum::<usize>(),
        }
    }

    /// Rewrite every load with `f`. Ops are rebuilt via [`Expr::new_op`].
    pub fn map_loads(&self, f: &mut dyn FnMut(BufferViewRef) -> Expr) -> Expr {
        match self {
            Expr::Load(v) => f(*v),
            Expr::Op { op, args } => {
                Expr::new_op(*op, args.iter().map(|a| a.map_loads(f)))
            }
        }
    }

    /// Inline `replacement` at every load of `target`.
    pub fn substitute(&self, target: BufferViewRef, replacement: &Expr) -> Expr {
        self.map_loads(&mut |v| {
            if v == target {
                replacement.clone()
            } else {
                Expr::Load(v)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::Op;

    fn v(i: usize) -> BufferViewRef {
        BufferViewRef(i)
    }

    #[test]
    fn new_op_builds_tree() {
        let expr = Expr::new_op(Op::ADD, [Expr::Load(v(0)), Expr::Load(v(1))]);
        assert_eq!(
            expr,
            Expr::Op {
                op: Op::ADD,
                args: Box::from([Expr::Load(v(0)), Expr::Load(v(1))]),
            }
        );
        assert_eq!(expr.loads(), vec![v(0), v(1)]);
    }

    #[test]
    fn substitute_inlines_a_load() {
        // outer: v0 * v1 ; replace v1 with (v2 + v3)
        let outer = Expr::new_op(Op::MUL, [Expr::Load(v(0)), Expr::Load(v(1))]);
        let inner = Expr::new_op(Op::ADD, [Expr::Load(v(2)), Expr::Load(v(3))]);
        let fused = outer.substitute(v(1), &inner);
        assert_eq!(
            fused,
            Expr::Op {
                op: Op::MUL,
                args: Box::from([
                    Expr::Load(v(0)),
                    Expr::Op {
                        op: Op::ADD,
                        args: Box::from([Expr::Load(v(2)), Expr::Load(v(3))]),
                    },
                ]),
            }
        );
        assert_eq!(fused.loads(), vec![v(0), v(2), v(3)]);
    }

    #[test]
    fn map_loads_rewrites_leaves() {
        let expr = Expr::new_op(Op::NEG, [Expr::Load(v(0))]);
        let mapped = expr.map_loads(&mut |x| Expr::Load(BufferViewRef(x.0 + 10)));
        assert_eq!(mapped.loads(), vec![v(10)]);
    }

    #[test]
    #[should_panic(expected = "wants 2 args, got 1")]
    fn new_op_rejects_arity_mismatch() {
        let _ = Expr::new_op(Op::ADD, [Expr::Load(v(0))]);
    }
}
