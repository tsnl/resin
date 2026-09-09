//! Typing services used by the source-checking pass.
//! Inference handles are resolved before the typed tree reaches IR lowering.

pub(crate) mod constraints;
pub(crate) mod solver;
pub(crate) mod types;

use resin_ast::{Ident, Span};
use resin_common::diagnostic::{GenerateError, GenerateErrorKind};
use std::sync::Arc;
type Result<T> = std::result::Result<T, GenerateError>;
fn error(span: Span, message: impl Into<Arc<str>>) -> GenerateError {
    GenerateError::inference(span, message)
}
pub(crate) fn check_binding_name(name: &Ident) -> Result<()> {
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

use crate::lower::context::Context;
use constraints::Constraint;
use solver::{Solver, VariableId};
use types::{Head, Type};

/// Identity of one typing operation; its owned variables are private to Inference.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rule(usize);

#[derive(Clone)]
pub(crate) struct Equation {
    span: Span,
    owner: Rule,
    relation: Constraint,
}

/// Constraint services injected into source checking; this layer never traverses expressions.
pub(crate) struct Inference<'a> {
    pub typer: &'a mut Context,
    pub solver: Solver,
    pub constraints: Vec<Equation>,
    owners: Vec<Vec<VariableId>>,
}
impl<'a> Inference<'a> {
    pub fn new(typer: &'a mut Context) -> Self {
        Self {
            typer,
            solver: Solver::default(),
            constraints: vec![],
            owners: vec![],
        }
    }
    fn rule(&mut self, variables: Vec<VariableId>) -> Rule {
        let rule = Rule(self.owners.len());
        self.owners.push(variables);
        rule
    }
    pub fn expression(&mut self) -> (Rule, Type) {
        let variable = self.solver.fresh_variable();
        (self.rule(vec![variable]), variable.ty())
    }
    pub fn constrain(&mut self, owner: Rule, (span, relation): (Span, Constraint)) {
        self.constraints.push(Equation {
            span,
            owner,
            relation,
        });
    }
    pub fn depends(&mut self, owner: Rule, span: Span, input: Type) {
        self.constrain(owner, (span, Constraint::Depends(input)));
    }
    pub fn infer_from(&mut self, holes: &[(Span, VariableId)], span: Span, input: Type) {
        let owner = self.rule(holes.iter().map(|(_, variable)| *variable).collect());
        self.depends(owner, span, input);
    }
    pub fn fail(&mut self, rule: Rule) {
        for variable in &self.owners[rule.0] {
            self.solver.invalidate(*variable);
        }
    }
    pub fn result_parts(&mut self, owner: Rule, ty: &Type, span: Span) -> Result<(Type, Type)> {
        match self.solver.head(ty) {
            Type::Node(Head::Result, parts) => Ok((parts[0].clone(), parts[1].clone())),
            Type::Variable(_) => {
                let value = self.solver.fresh();
                let errors = self.solver.fresh();
                self.solver.errors(&errors, span)?;
                self.constrain(
                    owner,
                    (
                        span,
                        Constraint::Equal(ty.clone(), Type::result(value.clone(), errors.clone())),
                    ),
                );
                Ok((value, errors))
            }
            _ => Err(error(
                span,
                "ok, err, and ? require a Result type; ? also requires a Result return type",
            )),
        }
    }
}

#[cfg(test)]
mod tests;
