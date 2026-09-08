//! Finish checking by replacing every inference handle with a concrete type.
//! The solver is dropped after this boundary; IR lowering cannot query it.
use super::super::typed::{Annotation, MatchArm, Statement, StatementKind, Term, TermKind};
use super::{Result, Type};
use crate::ir::typecheck::infer::solver::Solver;

impl Annotation<Type> {
    pub(super) fn resolve(self, solver: &Solver) -> Result<Annotation> {
        Ok(Annotation {
            ty: solver.require(&self.ty, self.span)?,
            span: self.span,
        })
    }
}

fn child(term: Box<Term<Type>>, solver: &Solver) -> Result<Box<Term>> {
    Ok(Box::new(term.resolve(solver)?))
}

impl Term<Type> {
    pub(super) fn resolve(self, solver: &Solver) -> Result<Term> {
        let kind = match self.kind {
            TermKind::Error(error) => return Err(error),
            TermKind::Unit => TermKind::Unit,
            TermKind::None => TermKind::None,
            TermKind::Num { value } => TermKind::Num { value },
            TermKind::String { value } => TermKind::String { value },
            TermKind::Var { name } => TermKind::Var { name },
            TermKind::Type { ty } => TermKind::Type {
                ty: ty.resolve(solver)?,
            },
            TermKind::Unwrap { value } => TermKind::Unwrap {
                value: child(value, solver)?,
            },
            TermKind::Try { value } => TermKind::Try {
                value: child(value, solver)?,
            },
            TermKind::Match { value, arms } => TermKind::Match {
                value: child(value, solver)?,
                arms: arms
                    .into_iter()
                    .map(|arm| {
                        Ok(MatchArm {
                            variant: arm.variant.map(|ty| ty.resolve(solver)).transpose()?,
                            failure: arm.failure,
                            binding: arm.binding,
                            body: arm.body.resolve(solver)?,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
            TermKind::If { cond, then, els } => TermKind::If {
                cond: child(cond, solver)?,
                then: child(then, solver)?,
                els: child(els, solver)?,
            },
            TermKind::While { cond, body } => TermKind::While {
                cond: child(cond, solver)?,
                body: child(body, solver)?,
            },
            TermKind::Block { stmts, tail } => TermKind::Block {
                stmts: stmts
                    .into_iter()
                    .map(|stmt| stmt.resolve(solver))
                    .collect::<Result<_>>()?,
                tail: child(tail, solver)?,
            },
            TermKind::Record { fields } => TermKind::Record {
                fields: fields
                    .into_iter()
                    .map(|(name, term)| Ok((name, term.resolve(solver)?)))
                    .collect::<Result<_>>()?,
            },
            TermKind::Array { elems } => TermKind::Array {
                elems: elems
                    .into_iter()
                    .map(|term| term.resolve(solver))
                    .collect::<Result<_>>()?,
            },
            TermKind::Builtin { name, args } => TermKind::Builtin {
                name,
                args: args
                    .into_iter()
                    .map(|term| term.resolve(solver))
                    .collect::<Result<_>>()?,
            },
            TermKind::MethodCall {
                receiver,
                receiver_type,
                name,
                arg,
            } => TermKind::MethodCall {
                receiver: receiver.map(|term| child(term, solver)).transpose()?,
                receiver_type: receiver_type.resolve(solver)?,
                name,
                arg: child(arg, solver)?,
            },
            TermKind::Call { func, arg } => TermKind::Call {
                func: child(func, solver)?,
                arg: child(arg, solver)?,
            },
            TermKind::Ascribe { ty, arg } => TermKind::Ascribe {
                ty: ty.resolve(solver)?,
                arg: child(arg, solver)?,
            },
            TermKind::Result { failure, arg } => TermKind::Result {
                failure,
                arg: child(arg, solver)?,
            },
            TermKind::Absurd { arg } => TermKind::Absurd {
                arg: child(arg, solver)?,
            },
            TermKind::Layout { ty, size } => TermKind::Layout {
                ty: ty.resolve(solver)?,
                size,
            },
            TermKind::Assign { place, value } => TermKind::Assign {
                place: child(place, solver)?,
                value: child(value, solver)?,
            },
            TermKind::Address { place } => TermKind::Address {
                place: child(place, solver)?,
            },
            TermKind::Deref { pointer } => TermKind::Deref {
                pointer: child(pointer, solver)?,
            },
            TermKind::Field { base, name } => TermKind::Field {
                base: child(base, solver)?,
                name,
            },
        };
        Ok(Term {
            span: self.span,
            context: self.context,
            ty: solver.require(&self.ty, self.span)?,
            kind,
        })
    }
}

impl Statement<Type> {
    fn resolve(self, solver: &Solver) -> Result<Statement> {
        let kind = match self.kind {
            StatementKind::Error(error) => return Err(error),
            StatementKind::Define {
                binding,
                name,
                init,
            } => StatementKind::Define {
                binding,
                name,
                init: init.resolve(solver)?,
            },
            StatementKind::Declare { binding, name, ty } => StatementKind::Declare {
                binding,
                name,
                ty: ty.resolve(solver)?,
            },
            StatementKind::TypeDefinition => StatementKind::TypeDefinition,
            StatementKind::Expr { term } => StatementKind::Expr {
                term: term.resolve(solver)?,
            },
        };
        Ok(Statement {
            context: self.context,
            kind,
        })
    }
}
