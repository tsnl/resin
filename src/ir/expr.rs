//! Elementwise expressions: trees with stable load leaves.
//!
//! An elementwise kernel body is an [`Expr`]. Leaves are [`BufferViewRef`]s —
//! not dense argument slots — so fusion is substitution of one tree into
//! another, with no renumbering. Free loads of the result become the
//! dispatch's argument list.

use super::{BufferViewRef, Error};
use crate::ops::Op;

/// Tree expression over buffer views and scalar operators.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Read one element from this view at the kernel's output coordinates.
    Load(BufferViewRef),
    /// Apply `op` to `args` (`args.len() == op.arity()`).
    Op { op: Op, args: Box<[Expr]> },
}

impl Expr {
    /// A single operator over loads: `op(Load(v0), …, Load(vN-1))`.
    pub fn apply_op(op: Op, loads: impl IntoIterator<Item = BufferViewRef>) -> Self {
        let args: Box<[Expr]> = loads.into_iter().map(Expr::Load).collect();
        debug_assert_eq!(args.len(), op.arity());
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

    /// Rewrite every load with `f`. Ops are rebuilt around the new children.
    pub fn map_loads(&self, f: &mut dyn FnMut(BufferViewRef) -> Expr) -> Expr {
        match self {
            Expr::Load(v) => f(*v),
            Expr::Op { op, args } => Expr::Op {
                op: *op,
                args: args.iter().map(|a| a.map_loads(f)).collect(),
            },
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

    /// Check operator arities.
    pub fn validate(&self) -> Result<(), Error> {
        match self {
            Expr::Load(_) => Ok(()),
            Expr::Op { op, args } => {
                if args.len() != op.arity() {
                    return Err(Error(format!(
                        "op {op:?} wants {} args, got {}",
                        op.arity(),
                        args.len()
                    )));
                }
                for arg in args.iter() {
                    arg.validate()?;
                }
                Ok(())
            }
        }
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
    fn apply_op_builds_loads() {
        let expr = Expr::apply_op(Op::ADD, [v(0), v(1)]);
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
        let outer = Expr::apply_op(Op::MUL, [v(0), v(1)]);
        let inner = Expr::apply_op(Op::ADD, [v(2), v(3)]);
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
        fused.validate().unwrap();
    }

    #[test]
    fn map_loads_rewrites_leaves() {
        let expr = Expr::apply_op(Op::NEG, [v(0)]);
        let mapped = expr.map_loads(&mut |x| Expr::Load(BufferViewRef(x.0 + 10)));
        assert_eq!(mapped.loads(), vec![v(10)]);
    }

    #[test]
    fn validate_rejects_arity_mismatch() {
        let bad = Expr::Op {
            op: Op::ADD,
            args: Box::from([Expr::Load(v(0))]),
        };
        assert!(bad.validate().is_err());
    }
}
