//! Typing services used while generation plans expressions.
//! Inference handles never escape into the final IR.

pub(in crate::ir) mod constraints;
pub(in crate::ir) mod solver;
pub(in crate::ir) mod types;

use crate::{
    ast::{Ident, Span},
    ir::{GenerateError, GenerateErrorKind},
};
use std::sync::Arc;
type Result<T> = std::result::Result<T, GenerateError>;
fn error(span: Span, message: impl Into<Arc<str>>) -> GenerateError {
    GenerateError::inference(span, message)
}
pub(in crate::ir) fn check_binding_name(name: &Ident) -> Result<()> {
    if matches!(
        name.val.as_ref(),
        "fmt" | "print" | "ok" | "err" | "size_of" | "align_of" | "absurd"
    ) {
        return Err(GenerateError {
            span: name.span,
            kind: GenerateErrorKind::ReservedBuiltin {
                name: name.val.clone(),
            },
        });
    }
    Ok(())
}

use crate::ir::TyperContext;
use constraints::Constraint;
use solver::Solver;
use types::{Head, Type};

/// Constraint services injected into generation; this layer never traverses expressions.
pub(in crate::ir) struct Inference<'a> {
    pub typer: &'a mut TyperContext,
    pub solver: Solver,
    pub constraints: Vec<(Span, Constraint)>,
}
impl<'a> Inference<'a> {
    pub fn new(typer: &'a mut TyperContext) -> Self {
        Self {
            typer,
            solver: Solver::default(),
            constraints: vec![],
        }
    }
    pub fn result_parts(&mut self, ty: &Type, span: Span) -> Result<(Type, Type)> {
        match self.solver.head(ty) {
            Type::Node(Head::Result, parts) => Ok((parts[0].clone(), parts[1].clone())),
            Type::Variable(_) => {
                let value = self.solver.fresh();
                let errors = self.solver.fresh();
                self.solver.errors(&errors, span)?;
                self.solver
                    .unify(ty, &Type::result(value.clone(), errors.clone()), span)
                    .map_err(|_| error(span, "ok, err, and ? require a Result type"))?;
                Ok((value, errors))
            }
            _ => Err(error(
                span,
                "ok, err, and ? require a Result type; ? also requires a Result return type",
            )),
        }
    }
}
