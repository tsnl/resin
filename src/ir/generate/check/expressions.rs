use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use super::super::{Evaluator, GenerateError, GenerateErrorKind, Generator, Scopes};
use super::{
    Result,
    constraints::{Constraint, Pattern},
    error,
    solver::Solver,
    types::{Head, Type},
};
use crate::{
    ast::{self, Ident, Span, StmtKind, Term, TermKind, TypeKind},
    ir::{Ty, TypeId, TyperContext},
};

pub(super) struct Checker<'a> {
    pub typer: &'a mut TyperContext,
    pub solver: Solver,
    scopes: Scopes,
    locals: Vec<BTreeMap<Arc<str>, Type>>,
    pub functions: BTreeMap<Arc<str>, Type>,
    pub holes: Vec<(*const ast::Type, Span, Type)>,
    pub expressions: Vec<(*const Term, Span, Type)>,
    pub definitions: HashMap<*const Ident, TypeId>,
    pub constraints: Vec<(Span, Constraint)>,
    pub result: Type,
    in_defer: bool,
}

impl<'a> Checker<'a> {
    pub fn new(typer: &'a mut TyperContext, scopes: Scopes) -> Self {
        Self {
            typer,
            scopes,
            solver: Solver::default(),
            locals: vec![],
            functions: BTreeMap::new(),
            holes: vec![],
            expressions: vec![],
            definitions: HashMap::new(),
            constraints: vec![],
            result: Ty::Unit.into(),
            in_defer: false,
        }
    }

    pub fn annotation(&mut self, ann: &ast::Type, infer: bool) -> Result<Type> {
        Ok(match &ann.val {
            TypeKind::Result { value, error } => {
                let value = self.annotation(value, infer)?;
                let error = self.annotation(error, infer)?;
                self.solver.errors(&error, ann.span)?;
                Type::result(value, error)
            }
            TypeKind::Infer => {
                if !infer {
                    return Err(error(
                        ann.span,
                        "type holes are only allowed in local annotations and function results",
                    ));
                }
                let ty = self.solver.fresh();
                self.holes.push((ann, ann.span, ty.clone()));
                ty
            }
            TypeKind::App { head, arg } => {
                let head = match head.val.as_ref() {
                    "Ptr" => Head::Pointer,
                    "Span" => Head::Span,
                    _ => return Err(error(head.span, "unknown type former")),
                };
                Type::Node(head, vec![self.annotation(arg, infer)?])
            }
            TypeKind::Func { from, to } => {
                Type::function(self.annotation(from, infer)?, self.annotation(to, infer)?)
            }
            TypeKind::Record { fields } => Type::record(
                fields
                    .iter()
                    .map(|(name, ty)| Ok((name.val.clone(), self.annotation(ty, infer)?)))
                    .collect::<Result<_>>()?,
            ),
            _ => Evaluator {
                scopes: &self.scopes,
                typer: self.typer,
                checked: None,
            }
            .ty(ann)?
            .into(),
        })
    }

    pub fn push(&mut self) {
        self.scopes.push();
        self.locals.push(BTreeMap::new());
    }
    pub fn pop(&mut self) {
        self.scopes.pop();
        self.locals.pop();
    }

    pub fn bind(&mut self, name: &Ident, ty: Type) -> Result<()> {
        Generator::check_binding_name(name)?;
        let frame = self.locals.last_mut().unwrap();
        if frame.insert(name.val.clone(), ty).is_some() {
            return Err(GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue {
                    name: name.val.clone(),
                },
            });
        }
        Ok(())
    }

    fn value(&self, name: &Ident) -> Result<Type> {
        for frame in self.locals.iter().rev() {
            if let Some(ty) = frame.get(&name.val) {
                return Ok(ty.clone());
            }
        }
        if let Some(ty) = self.functions.get(&name.val) {
            return Ok(ty.clone());
        }
        self.scopes
            .lookup_value(&name.val)
            .and_then(|b| b.ty.clone())
            .map(Type::from)
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })
    }

    pub fn term(&mut self, term: &Term, expected: Option<Type>) -> Result<Type> {
        // Control-flow branches share their result context. Other expressions
        // first produce their own value, which may then widen at the consumer.
        let propagate = matches!(
            term.val,
            TermKind::If { .. } | TermKind::Match { .. } | TermKind::Block { .. }
        ) || matches!(&term.val, TermKind::Call { func, .. } if matches!(&func.val, TermKind::Var { name } if matches!(name.val.as_ref(), "ok" | "err" | "absurd")));
        let contextual = propagate && expected.is_some();
        let out = if propagate {
            expected.clone().unwrap_or_else(|| self.solver.fresh())
        } else {
            self.solver.fresh()
        };
        let span = term.span;
        let mut equate = None;
        match &term.val {
            TermKind::Hole { .. } | TermKind::FieldHole { .. } => {
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            TermKind::Unit => equate = Some(Ty::Unit.into()),
            TermKind::None => equate = Some(Ty::None.into()),
            TermKind::Unwrap { value } => {
                let payload = self.solver.fresh();
                let input = self.term(value, None)?;
                self.constraints
                    .push((span, Constraint::ExcludeNone(input, payload.clone())));
                equate = Some(payload);
            }
            TermKind::Try { value } => {
                if self.in_defer {
                    return Err(error(
                        span,
                        "postfix ? is not allowed in a deferred expression; handle the error locally",
                    ));
                }
                let input = self.term(value, None)?;
                let (value, errors) = self.result_parts(&input, span)?;
                let (_, target_errors) = self.result_parts(&self.result.clone(), span)?;
                self.constraints
                    .push((span, Constraint::Errors(errors, target_errors)));
                equate = Some(value);
            }
            TermKind::Match { value, arms } => {
                let input = self.term(value, None)?;
                for arm in arms {
                    self.push();
                    let payload = self.solver.fresh();
                    let variant = match &arm.variant {
                        ast::MatchVariant::Ok => Pattern::Ok,
                        ast::MatchVariant::Err => Pattern::Err,
                        ast::MatchVariant::Type(ty) => Pattern::Type(self.annotation(ty, false)?),
                    };
                    self.constraints.push((
                        arm.body.span,
                        Constraint::Variant(input.clone(), variant, payload.clone()),
                    ));
                    if let Some(name) = &arm.name {
                        self.bind(name, payload)?;
                    }
                    self.term(&arm.body, Some(out.clone()))?;
                    self.pop();
                }
            }
            TermKind::Num { value } => equate = Some(self.solver.number(value)),
            TermKind::String { value } => {
                equate = Some(
                    Ty::Array {
                        element: Box::new(Ty::UInt8),
                        length: value.len(),
                    }
                    .into(),
                )
            }
            TermKind::Var { name } => equate = Some(self.value(name)?),
            TermKind::Type { ty } => {
                self.annotation(ty, true)?;
                equate = Some(Ty::Type.into());
            }
            TermKind::If { cond, then, els } => {
                let condition = self.term(cond, None)?;
                self.constraints
                    .push((cond.span, Constraint::Boolean(condition)));
                self.term(then, Some(out.clone()))?;
                self.term(els, Some(out.clone()))?;
            }
            TermKind::While { cond, body } => {
                let condition = self.term(cond, None)?;
                self.constraints
                    .push((cond.span, Constraint::Boolean(condition)));
                self.term(body, None)?;
                equate = Some(Ty::Unit.into());
            }
            TermKind::Block { stmts, tail } => {
                self.push();
                for stmt in stmts {
                    match &stmt.val {
                        StmtKind::Define { name, init } => {
                            let ty = self.solver.fresh();
                            self.bind(name, ty.clone())?;
                            self.term(init, Some(ty))?;
                        }
                        StmtKind::Declare { name, ann } => {
                            let ty = self.annotation(ann, true)?;
                            self.bind(name, ty)?;
                        }
                        StmtKind::Struct { name, body: init } => {
                            let definition = self.typer.reserve_type(name.val.clone());
                            self.scopes
                                .define_type(name.val.clone(), definition)
                                .map_err(|name| GenerateError {
                                    span,
                                    kind: GenerateErrorKind::DuplicateType { name },
                                })?;
                            let ty = self.annotation(init, false)?;
                            let ty = self.solver.require(&ty, init.span)?;
                            self.typer
                                .define_type(definition, ty)
                                .map_err(|e| GenerateError::typing(init.span, e))?;
                            self.definitions.insert(name, definition);
                        }
                        StmtKind::DefineType { name, init } => {
                            let ty = self.annotation(init, false)?;
                            let ty = self.solver.require(&ty, init.span)?;
                            self.scopes
                                .define_alias(name.val.clone(), ty)
                                .map_err(|name| GenerateError {
                                    span,
                                    kind: GenerateErrorKind::DuplicateType { name },
                                })?;
                        }
                        StmtKind::Expr { term } => {
                            self.term(term, None)?;
                        }
                        StmtKind::Defer { body } => {
                            let before = std::mem::replace(&mut self.in_defer, true);
                            self.term(body, None)?;
                            self.in_defer = before;
                        }
                        _ => {
                            return Err(error(
                                stmt.span,
                                "unexpected declaration in function body",
                            ));
                        }
                    }
                }
                self.term(tail, Some(out.clone()))?;
                self.pop();
            }
            TermKind::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, term)| Ok((name.val.clone(), self.term(term, None)?)))
                    .collect::<Result<_>>()?;
                self.constraints
                    .push((span, Constraint::Record(fields, out.clone())));
            }
            TermKind::Array { elems } => {
                let element = self.solver.fresh();
                for elem in elems {
                    self.term(elem, Some(element.clone()))?;
                }
                equate = Some(Type::Node(Head::Array(elems.len()), vec![element]));
            }
            TermKind::Builtin { name, args } => {
                let args = args
                    .iter()
                    .map(|t| self.term(t, None))
                    .collect::<Result<_>>()?;
                self.constraints
                    .push((span, Constraint::Builtin(name.clone(), args, out.clone())));
            }
            TermKind::MethodCall { base, name, arg } => {
                let (receiver, associated) = if let TermKind::Type { ty } = &base.val {
                    (self.annotation(ty, false)?, true)
                } else {
                    (self.term(base, None)?, false)
                };
                let arg = self.term(arg, None)?;
                self.constraints.push((
                    span,
                    Constraint::Method(receiver, name.val.clone(), arg, out.clone(), associated),
                ));
            }
            TermKind::Call { func, arg } => {
                if let TermKind::Var { name } = &func.val
                    && name.val.as_ref() == "absurd"
                {
                    self.term(arg, Some(Ty::union([]).into()))?;
                    // The enclosing expected type constrains `out`; an unconstrained
                    // elimination is ambiguous, never a value of an arbitrary type.
                } else if let TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "size_of" | "align_of")
                {
                    let ty = if let TermKind::Type { ty } = &arg.val {
                        self.annotation(ty, false)?
                    } else {
                        self.term(arg, None)?
                    };
                    self.constraints.push((span, Constraint::Layout(ty)));
                    equate = Some(Ty::UInt64.into());
                } else if let TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "ok" | "err")
                {
                    let (value, errors) = self.result_parts(&out, span)?;
                    if name.val.as_ref() == "ok" {
                        self.term(arg, Some(value))?;
                    } else {
                        let payload = self.term(arg, None)?;
                        self.constraints
                            .push((span, Constraint::Errors(payload, errors)));
                    }
                } else if let TermKind::Type { ty } = &func.val {
                    let to = self.annotation(ty, true)?;
                    let from = self.term(arg, None)?;
                    self.constraints
                        .push((span, Constraint::Ascribe(from, to.clone(), matches!(arg.val, TermKind::Num { .. }) || matches!(&arg.val, TermKind::Builtin { name, args } if matches!(name.as_ref(), "+" | "-") && matches!(args.as_slice(), [Term { val: TermKind::Num { .. }, .. }])))));
                    equate = Some(to);
                } else if let TermKind::Var { name } = &func.val
                    && name.val.as_ref() == "print"
                {
                    let arg = self.term(arg, None)?;
                    self.constraints.push((
                        span,
                        Constraint::Builtin("print".into(), vec![arg], out.clone()),
                    ));
                } else {
                    let func = self.term(func, None)?;
                    let arg = self.term(arg, None)?;
                    self.constraints
                        .push((span, Constraint::Call(func, arg, out.clone())));
                }
            }
            TermKind::Assign { place, value } => {
                let place = self.term(place, None)?;
                self.term(value, Some(place.clone()))?;
                equate = Some(place);
            }
            TermKind::Address { place } => equate = Some(Type::pointer(self.term(place, None)?)),
            TermKind::Deref { pointer } => {
                let pointer = self.term(pointer, None)?;
                self.constraints
                    .push((span, Constraint::Deref(pointer, out.clone())));
            }
            TermKind::Field { base, name } => {
                let base = self.term(base, None)?;
                self.constraints
                    .push((span, Constraint::Field(base, name.val.clone(), out.clone())));
            }
        }
        if let Some(ty) = equate {
            if contextual {
                self.constraints
                    .push((span, Constraint::Coerce(ty, out.clone())));
            } else {
                self.solver.unify(&ty, &out, span)?;
            }
        }
        if !propagate && let Some(expected) = expected {
            self.constraints
                .push((span, Constraint::Coerce(out.clone(), expected)));
        }
        self.expressions.push((term, span, out.clone()));
        Ok(out)
    }

    fn result_parts(&mut self, ty: &Type, span: Span) -> Result<(Type, Type)> {
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
