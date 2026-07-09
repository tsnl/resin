//! Reverse-polish expressions over kernel arguments.
//!
//! An elementwise kernel is an [`RpnExpr`]: a flat list of argument loads and
//! scalar operators in postfix order. Fusion of two elementwise kernels is
//! pure list splicing — no expression tree — via [`RpnExpr::map_args`] and
//! [`RpnExpr::fuse`].

use crate::ops::Op;

use super::Error;

/// Reverse-polish expression over argument indices and scalar operators.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RpnExpr {
    pub atoms: Vec<RpnAtom>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RpnAtom {
    /// Index into the dispatch argument list.
    Arg(u32),
    Op(Op),
}

impl RpnExpr {
    /// The trivial expression `op(arg0, …, argN-1)`.
    pub fn apply_op(op: Op, arity: usize) -> Self {
        let mut atoms: Vec<RpnAtom> = (0..arity as u32).map(RpnAtom::Arg).collect();
        atoms.push(RpnAtom::Op(op));
        Self { atoms }
    }

    /// The expression that is just a reference to argument `i`.
    pub fn arg(i: u32) -> Self {
        Self { atoms: vec![RpnAtom::Arg(i)] }
    }

    /// Substitute every `Arg(i)` with `f(i)`, leaving operators in place.
    ///
    /// This is the primitive behind RPN–RPN fusion: substitute another
    /// kernel's expression for an argument to splice the two; substitute
    /// [`RpnExpr::arg`] to renumber.
    pub fn map_args(&self, f: &mut dyn FnMut(u32) -> RpnExpr) -> RpnExpr {
        let mut atoms = Vec::with_capacity(self.atoms.len());
        for atom in &self.atoms {
            match atom {
                RpnAtom::Arg(i) => atoms.extend(f(*i).atoms),
                RpnAtom::Op(op) => atoms.push(RpnAtom::Op(*op)),
            }
        }
        RpnExpr { atoms }
    }

    /// Fuse `inner` into this expression.
    ///
    /// `outer_to_new[i]`:
    /// - `Some(j)` — keep this expression's `Arg(i)` as `Arg(j)`
    /// - `None` — replace it with `inner`, renumbering `inner`'s `Arg(k)` to
    ///   `Arg(inner_to_new[k])`
    ///
    /// The two maps together describe a merged argument list; this method is
    /// the pure algebraic step and does not know about buffers or views.
    pub fn fuse(
        &self,
        outer_to_new: &[Option<u32>],
        inner: &RpnExpr,
        inner_to_new: &[u32],
    ) -> RpnExpr {
        self.map_args(&mut |i| match outer_to_new[i as usize] {
            Some(j) => RpnExpr::arg(j),
            None => inner.map_args(&mut |k| RpnExpr::arg(inner_to_new[k as usize])),
        })
    }

    /// Check stack discipline: every op has its operands, one value remains.
    /// All `Arg` indices must be below `num_args`.
    pub fn validate(&self, num_args: usize) -> Result<(), Error> {
        let mut depth = 0usize;
        for atom in &self.atoms {
            match atom {
                RpnAtom::Arg(i) => {
                    if *i as usize >= num_args {
                        return Err(Error(format!("rpn arg {i} out of range ({num_args} args)")));
                    }
                    depth += 1;
                }
                RpnAtom::Op(op) => {
                    if depth < op.arity() {
                        return Err(Error(format!("rpn stack underflow at {op:?}")));
                    }
                    depth = depth - op.arity() + 1;
                }
            }
        }
        if depth != 1 {
            return Err(Error(format!("rpn leaves {depth} values on the stack, want 1")));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_args_renumbers() {
        let expr = RpnExpr::apply_op(Op::ADD, 2);
        let renumbered = expr.map_args(&mut |i| RpnExpr::arg(i + 10));
        assert_eq!(
            renumbered.atoms,
            vec![RpnAtom::Arg(10), RpnAtom::Arg(11), RpnAtom::Op(Op::ADD)]
        );
    }

    #[test]
    fn fuse_splices_inner_at_a_slot() {
        // outer = a0 * a1; replace a1 with (b0 + b1) renumbered as a1, a2
        // → a0 * (a1 + a2)
        let outer = RpnExpr::apply_op(Op::MUL, 2);
        let inner = RpnExpr::apply_op(Op::ADD, 2);
        let fused = outer.fuse(&[Some(0), None], &inner, &[1, 2]);
        assert_eq!(
            fused.atoms,
            vec![
                RpnAtom::Arg(0),
                RpnAtom::Arg(1),
                RpnAtom::Arg(2),
                RpnAtom::Op(Op::ADD),
                RpnAtom::Op(Op::MUL),
            ]
        );
        fused.validate(3).unwrap();
    }

    #[test]
    fn validate_catches_underflow_and_leftovers() {
        assert!(RpnExpr { atoms: vec![RpnAtom::Op(Op::ADD)] }.validate(0).is_err());
        assert!(
            RpnExpr { atoms: vec![RpnAtom::Arg(0), RpnAtom::Arg(0)] }
                .validate(1)
                .is_err()
        );
        assert!(RpnExpr::apply_op(Op::ADD, 2).validate(2).is_ok());
        assert!(RpnExpr::apply_op(Op::ADD, 2).validate(1).is_err());
    }
}
