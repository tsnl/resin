use std::{collections::HashSet, sync::Arc};

use super::super::GenerateError;
use super::{
    Result, error,
    expressions::Checker,
    types::{Head, Type},
};
use crate::{ast::Span, ir::Ty};

pub(super) enum Constraint {
    Coerce(Type, Type),
    ExcludeNone(Type, Type),
    Layout(Type),
    Errors(Type, Type),
    Variant(Type, Pattern, Type),
    Boolean(Type),
    Deref(Type, Type),
    Field(Type, Arc<str>, Type),
    Call(Type, Type, Type),
    Method(Type, Arc<str>, Type, Type, bool),
    Ascribe(Type, Type, bool),
    Record(Vec<(Arc<str>, Type)>, Type),
    Builtin(Arc<str>, Vec<Type>, Type),
}

pub(super) enum Pattern {
    Ok,
    Err,
    Type(Type),
}

impl Checker<'_> {
    pub fn solve(&mut self, roots: &[Type]) -> Result<()> {
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
            if self.solver.finish_errors(roots) {
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
        let mut ty = self.solver.shape_hint(ty);
        if deref {
            while let Type::Node(Head::Pointer, children) = &ty {
                ty = self.solver.shape_hint(&children[0]);
            }
        }
        if let Type::Node(Head::Atom(t @ Ty::Defined { .. }), _) = &ty {
            ty = self
                .typer
                .body(t)
                .map_err(|e| GenerateError::typing(span, e))?
                .into();
        }
        Ok(ty)
    }

    fn constraint(&mut self, constraint: &Constraint, span: Span) -> Result<bool> {
        match constraint {
            Constraint::Method(base, name, arg, out, associated) => {
                let Some(base) = self.solver.resolve(base) else {
                    return Ok(false);
                };
                let method = self
                    .typer
                    .method(&base, name)
                    .ok_or_else(|| error(span, format!("unknown method `{name}`")))?;
                let params = method.arguments(&base, *associated).ok_or_else(|| {
                    error(span, "method receiver does not match the first parameter")
                })?;
                let a = self
                    .solver
                    .coerce(arg, &Ty::parameter(params).into(), span)?;
                let b = self
                    .solver
                    .coerce(&method.result.clone().into(), out, span)?;
                return Ok(a && b);
            }
            Constraint::Layout(ty) => {
                let Some(ty) = self.solver.resolve(ty) else {
                    return Ok(false);
                };
                crate::ir::layout::layout(self.typer.definitions(), &ty)
                    .map_err(|e| error(span, e.to_string()))?;
            }
            Constraint::ExcludeNone(input, out) => {
                let Some(input) = self.solver.resolve(input) else {
                    return Ok(false);
                };
                let remaining = input
                    .without_none()
                    .ok_or_else(|| error(span, "postfix ! requires a type containing None"))?;
                self.solver.unify(out, &remaining.into(), span)?;
            }
            Constraint::Coerce(from, to) => return self.solver.coerce(from, to, span),
            Constraint::Errors(from, to) => return self.solver.include(from, to, span),
            Constraint::Variant(input, variant, out) => match (self.solver.head(input), variant) {
                (Type::Variable(_), _) => return Ok(false),
                (Type::Node(Head::Result, parts), Pattern::Ok) => {
                    self.solver.unify(out, &parts[0], span)?
                }
                (Type::Node(Head::Result, parts), Pattern::Err) => {
                    self.solver.unify(out, &parts[1], span)?
                }
                (_, Pattern::Type(ty)) => self.solver.unify(out, ty, span)?,
                _ => return Err(error(span, "match pattern does not belong to this type")),
            },
            Constraint::Boolean(input) => {
                let Some(ty) = self.solver.resolve(input) else {
                    return Ok(false);
                };
                self.typer
                    .as_bool(&ty)
                    .map_err(|e| GenerateError::typing(span, e))?;
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
                if let Type::Node(Head::Function, _) = &shape
                    && name.as_ref() == "spirv"
                {
                    self.solver.unify(out, &Ty::shader().into(), span)?;
                    return Ok(true);
                }
                if let Type::Node(Head::Span, children) = &shape {
                    let ty = match name.as_ref() {
                        "data" => Type::pointer(children[0].clone()),
                        "length" => Ty::UInt64.into(),
                        _ => return Err(error(span, "unknown Span field")),
                    };
                    self.solver.unify(out, &ty, span)?;
                    return Ok(true);
                }

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
                if let Type::Node(Head::Span | Head::Array(_), children) = &shape {
                    self.solver
                        .unify(out, &Type::pointer(children[0].clone()), span)?;
                    let Some(index) = self.solver.resolve(arg) else {
                        return Ok(false);
                    };
                    if !index.is_integer() {
                        return Err(error(span, "index must be an integer"));
                    }
                    return Ok(true);
                }

                match shape {
                    Type::Variable(_) => return Ok(false),
                    Type::Node(Head::Function, children) => {
                        let a = self.solver.coerce(arg, &children[0], span)?;
                        let b = self.solver.coerce(&children[1], out, span)?;
                        return Ok(a && b);
                    }
                    _ => return Err(error(span, "call requires a function")),
                }
            }
            Constraint::Record(fields, out) => {
                let Type::Node(Head::Record(names), types) = self.solver.head(out) else {
                    if matches!(self.solver.head(out), Type::Variable(_)) {
                        return Ok(false);
                    }
                    let record = Type::record(fields.clone());
                    if self.solver.resolve(&record).is_none() {
                        return Ok(false);
                    }
                    self.solver.unify(&record, out, span)?;
                    unreachable!("a record cannot equal a non-record");
                };
                let unique: HashSet<_> = fields.iter().map(|(name, _)| name).collect();
                if unique.len() != fields.len() || fields.len() != names.len() {
                    return Err(error(span, "record fields do not match the expected type"));
                }
                let mut complete = true;
                for (name, ty) in names.iter().zip(types) {
                    let found = fields
                        .iter()
                        .find(|(n, _)| n == name)
                        .ok_or_else(|| error(span, format!("missing field `{name}`")))?;
                    complete &= self.solver.coerce(&found.1, &ty, span)?;
                }
                return Ok(complete);
            }
            Constraint::Ascribe(from, to, literal) => {
                if let Type::Node(Head::Span, children) = self.solver.head(to) {
                    if matches!(self.solver.head(from), Type::Node(Head::Span, _)) {
                        return self.solver.unify(from, to, span).map(|_| true);
                    }
                    let repr = Type::record(vec![
                        ("data".into(), Type::pointer(children[0].clone())),
                        ("length".into(), Ty::UInt64.into()),
                    ]);
                    self.solver.unify(from, &repr, span)?;
                    return Ok(true);
                }
                if matches!(self.solver.head(to), Type::Node(Head::Result, _)) {
                    return self.solver.coerce(from, to, span);
                }
                if let (Some(from), Some(to)) = (self.solver.resolve(from), self.solver.resolve(to))
                {
                    let empty = from == Ty::Unit
                        && matches!(self.typer.body(&to), Ok(Ty::Record { fields }) if fields.is_empty());
                    if !(empty
                        || from.pointer_cast(&to)
                        || from.widens_to(&to)
                        || from.is_numeric() && to.is_numeric())
                    {
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
                            if !literal && self.solver.resolve(to).is_some_and(|ty| ty.is_numeric())
                            {
                                return Ok(false); // A runtime cast must not choose source storage.
                            }
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
                            self.solver.cast(from, to, span)?
                        }
                        _ => {}
                    }
                    return Ok(false);
                }
            }
            Constraint::Builtin(name, args, out) => {
                if name.as_ref() == "print" {
                    self.solver.unify(out, &Ty::Unit.into(), span)?;
                } else if matches!(name.as_ref(), "&&" | "||" | "!") {
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
                        self.solver.unify(left, right, span)?;
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
