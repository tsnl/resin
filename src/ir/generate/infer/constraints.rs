use std::{collections::HashSet, sync::Arc};

use super::super::GenerateError;
use super::{
    Result,
    check::Checker,
    error,
    types::{Head, Type},
};
use crate::{ast::Span, ir::Ty};

pub(super) enum Constraint {
    Boolean(Type),
    Deref(Type, Type),
    Field(Type, Arc<str>, Type),
    Call(Type, Type, Type),
    Ascribe(Type, Type),
    Record(Vec<(Arc<str>, Type)>, Type),
    Builtin(Arc<str>, Vec<Type>, Type),
}

impl Checker<'_> {
    pub fn solve(&mut self) -> Result<()> {
        loop {
            let before = self.solver.revision;
            let pending = std::mem::take(&mut self.constraints);
            for (span, constraint) in pending {
                if !self.constraint(&constraint, span)? {
                    self.constraints.push((span, constraint));
                }
            }
            if self.solver.revision != before {
                continue;
            }
            // Give contextual record layouts priority over source field order.
            let mut seeded = false;
            for (span, constraint) in &self.constraints {
                if let Constraint::Record(fields, out) = constraint
                    && matches!(self.solver.head(out), Type::Variable(_))
                {
                    self.solver
                        .unify(out, &Type::record(fields.clone()), *span)?;
                    seeded = true;
                }
            }
            if seeded || self.solver.default_numbers() {
                continue;
            }
            if let Some((span, _)) = self.constraints.first() {
                return Err(error(
                    *span,
                    "cannot infer this operation; annotate its operand or result",
                ));
            }
            return Ok(());
        }
    }

    fn shape(&self, ty: &Type, deref: bool, span: Span) -> Result<Type> {
        let mut ty = self.solver.head(ty);
        let mut visited = HashSet::new();
        loop {
            ty = match &ty {
                Type::Node(Head::Atom(t @ Ty::Defined { definition }), _) => {
                    if !visited.insert(*definition) {
                        return Err(error(span, "recursive type has no usable shape"));
                    }
                    self.typer
                        .body(t)
                        .map_err(|e| GenerateError::typing(span, e))?
                        .into()
                }
                Type::Node(Head::Pointer, children) if deref => self.solver.head(&children[0]),
                _ => return Ok(ty),
            };
        }
    }

    fn constraint(&mut self, constraint: &Constraint, span: Span) -> Result<bool> {
        match constraint {
            Constraint::Boolean(input) => {
                let shape = self.shape(input, false, span)?;
                if matches!(shape, Type::Variable(_)) {
                    return Ok(false);
                }
                self.solver.unify(&shape, &Ty::Bool.into(), span)?;
            }
            Constraint::Deref(input, out) => {
                let shape = self.shape(input, false, span)?;
                match shape {
                    Type::Variable(_) => return Ok(false),
                    Type::Node(Head::Pointer, children) => {
                        self.solver.unify(out, &children[0], span)?
                    }
                    _ => return Err(error(span, "dereference requires a pointer")),
                }
            }
            Constraint::Field(input, name, out) => {
                let shape = self.shape(input, true, span)?;
                match shape {
                    Type::Variable(_) => return Ok(false),
                    Type::Node(Head::Record(names), children) => {
                        let index = names
                            .iter()
                            .position(|n| n == name)
                            .ok_or_else(|| error(span, format!("unknown field `{name}`")))?;
                        self.solver.unify(out, &children[index], span)?;
                    }
                    _ => return Err(error(span, "field access requires a record")),
                }
            }
            Constraint::Call(func, arg, out) => {
                let shape = self.shape(func, false, span)?;
                match shape {
                    Type::Variable(_) => return Ok(false),
                    Type::Node(Head::Function, children) => {
                        self.solver.unify(arg, &children[0], span)?;
                        self.solver.unify(out, &children[1], span)?;
                    }
                    _ => return Err(error(span, "call requires a function")),
                }
            }
            Constraint::Record(fields, out) => {
                let Type::Node(Head::Record(names), types) = self.solver.head(out) else {
                    if matches!(self.solver.head(out), Type::Variable(_)) {
                        return Ok(false);
                    }
                    return Err(error(span, "record initializer requires a record type"));
                };
                let unique: HashSet<_> = fields.iter().map(|(name, _)| name).collect();
                if unique.len() != fields.len() || fields.len() != names.len() {
                    return Err(error(span, "record fields do not match the expected type"));
                }
                for (name, ty) in names.iter().zip(types) {
                    let found = fields
                        .iter()
                        .find(|(n, _)| n == name)
                        .ok_or_else(|| error(span, format!("missing field `{name}`")))?;
                    self.solver.unify(&found.1, &ty, span)?;
                }
            }
            Constraint::Ascribe(from, to) => {
                if let (Some(from), Some(to)) = (self.solver.resolve(from), self.solver.resolve(to))
                {
                    if !from.pointer_cast(&to) {
                        self.typer
                            .ascribe(&from, &to)
                            .map_err(|e| GenerateError::typing(span, e))?;
                    }
                } else {
                    let source = self.solver.head(from);
                    let target = self.solver.head(to);
                    match (&source, &target) {
                        (_, Type::Variable(_)) => self.solver.unify(to, from, span)?,
                        (Type::Variable(_), _) => {
                            let context = if let Type::Node(Head::Atom(t @ Ty::Defined { .. }), _) =
                                &target
                            {
                                self.typer
                                    .body(t)
                                    .map_err(|e| GenerateError::typing(span, e))?
                                    .into()
                            } else {
                                target
                            };
                            self.solver.unify(from, &context, span)?;
                        }
                        (Type::Node(Head::Pointer, _), Type::Node(Head::Pointer, _)) => {
                            self.solver.unify(from, to, span)?
                        }
                        _ => {}
                    }
                    return Ok(false);
                }
            }
            Constraint::Builtin(name, args, out) => {
                if matches!(name.as_ref(), "&&" | "||" | "!") {
                    self.solver.unify(out, &Ty::Bool.into(), span)?;
                    for arg in args {
                        if !self.constraint(&Constraint::Boolean(arg.clone()), span)? {
                            return Ok(false);
                        }
                    }
                } else {
                    let comparison = matches!(name.as_ref(), "==" | "!=" | "<" | "<=" | ">" | ">=");
                    if comparison {
                        self.solver.unify(out, &Ty::Bool.into(), span)?;
                    } else if let Some(first) = args.first() {
                        self.solver.unify(out, first, span)?;
                    }
                    if let [left, right] = args.as_slice() {
                        let pointer_op = matches!(name.as_ref(), "+" | "-");
                        let left_shape = self.solver.head(left);
                        if pointer_op && matches!(left_shape, Type::Variable(_)) {
                            return Ok(false);
                        }
                        if !(pointer_op && matches!(left_shape, Type::Node(Head::Pointer, _))) {
                            self.solver.unify(left, right, span)?;
                        }
                    }
                }
                let Some(args) = args
                    .iter()
                    .map(|t| self.solver.resolve(t))
                    .collect::<Option<Vec<_>>>()
                else {
                    return Ok(false);
                };
                let call = self
                    .typer
                    .type_builtin_call(name, &args)
                    .map_err(|e| GenerateError::typing(span, e))?;
                self.solver.unify(out, &call.result.into(), span)?;
            }
        }
        Ok(true)
    }
}
