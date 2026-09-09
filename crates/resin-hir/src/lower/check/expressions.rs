use super::super::GenerateError;
use super::super::typed::{StatementKind, TermKind};
use super::super::{scope::Cursor, semantic::DefinitionKind};
use super::{Annotation, Checker, Head, MatchArm, Result, Statement, Term, Type};
use crate::lower::infer::{
    Rule,
    constraints::{Constraint, Pattern},
};
use resin_ast::StmtKind;
use resin_common::diagnostic::GenerateErrorKind;
use resin_common::types::Ty;

impl Checker<'_> {
    pub fn term(&mut self, term: &resin_ast::Term, expected: Option<Type>) -> (Rule, Term) {
        let (rule, out) = self.typing.expression();
        let term = Expression {
            checker: self,
            rule,
        }
        .check(term, expected, out);
        (rule, term)
    }
}

/// One expression owns all of its equations and automatically depends on its children.
struct Expression<'p, 'a> {
    checker: &'p mut Checker<'a>,
    rule: Rule,
}

impl Expression<'_, '_> {
    fn child(&mut self, term: &resin_ast::Term, expected: Option<Type>) -> Term {
        let (_, child) = self.checker.term(term, expected);
        self.checker
            .typing
            .depends(self.rule, term.span, child.ty.clone());
        child
    }

    fn annotation(&mut self, ann: &resin_ast::Type, infer: bool) -> Annotation {
        let annotation = self.checker.ann(ann, infer);
        self.checker
            .typing
            .depends(self.rule, ann.span, annotation.ty.clone());
        annotation
    }

    fn constrain(&mut self, constraint: (resin_ast::Span, Constraint)) {
        self.checker.typing.constrain(self.rule, constraint);
    }

    fn result_parts(&mut self, ty: &Type, span: resin_ast::Span) -> Result<(Type, Type)> {
        self.checker.typing.result_parts(self.rule, ty, span)
    }

    fn check(&mut self, term: &resin_ast::Term, expected: Option<Type>, out: Type) -> Term {
        let context = self.checker.scopes.capture();
        let checked = self.term_inner(term, expected, out.clone(), context);
        let checked = match checked {
            Ok(checked) => checked,
            Err(error) => {
                self.checker.scopes.restore(context);
                self.checker.errors.push(error.clone());
                self.checker.typing.fail(self.rule);
                Term {
                    context,
                    span: term.span,
                    ty: out.clone(),
                    kind: TermKind::Error(error),
                }
            }
        };
        self.checker.expressions.push((term.span, out.clone()));
        self.checker.scopes.record_inferred(term.span, out);
        checked
    }

    fn term_inner(
        &mut self,
        term: &resin_ast::Term,
        expected: Option<Type>,
        out: Type,
        context: Cursor,
    ) -> Result<Term> {
        let propagate = matches!(
            term.val,
            resin_ast::TermKind::If { .. }
                | resin_ast::TermKind::Match { .. }
                | resin_ast::TermKind::Block { .. }
        ) || matches!(&term.val, resin_ast::TermKind::Call { func, .. } if matches!(&func.val, resin_ast::TermKind::Var { name } if matches!(name.val.as_ref(), "ok" | "err" | "absurd")));
        let contextual = propagate && expected.is_some();
        if propagate && let Some(expected) = &expected {
            self.constrain((term.span, Constraint::Equal(expected.clone(), out.clone())));
        }
        let span = term.span;
        let mut equate = None;
        let kind = match &term.val {
            resin_ast::TermKind::Hole { children } => {
                for child in children {
                    self.child(child, None);
                }
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            resin_ast::TermKind::FieldHole { base } => {
                let base = self.child(base, None);
                let (ty, associated) = match &base.kind {
                    TermKind::Type { ty } => (ty.ty.clone(), true),
                    TermKind::Var { name } if self.checker.scopes.is_shader(&name.val) => {
                        (crate::lower::builtins::shader_properties().into(), false)
                    }
                    _ => (base.ty.clone(), false),
                };
                self.checker.scopes.record_members(
                    resin_ast::Span {
                        start: span.end,
                        end: span.end,
                    },
                    ty,
                    associated,
                );
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            resin_ast::TermKind::Unit => {
                equate = Some(Ty::Unit.into());
                TermKind::Unit
            }
            resin_ast::TermKind::None => {
                equate = Some(Ty::None.into());
                TermKind::None
            }
            resin_ast::TermKind::Unwrap { value } => {
                let input = self.child(value, None);
                self.constrain((span, Constraint::ExcludeNone(input.ty.clone(), out.clone())));
                TermKind::Unwrap {
                    value: Box::new(input),
                }
            }
            resin_ast::TermKind::Num { value } => {
                equate = Some(self.checker.typing.solver.number(value));
                TermKind::Num {
                    value: value.clone(),
                }
            }
            resin_ast::TermKind::String { value } => {
                equate = Some(Ty::byte_span().into());
                TermKind::String {
                    value: value.clone(),
                }
            }
            resin_ast::TermKind::Var { name } => {
                equate = Some(self.checker.value(name)?);
                TermKind::Var { name: name.clone() }
            }
            resin_ast::TermKind::Type { ty } => {
                let ann = self.annotation(ty, true);
                equate = Some(Ty::Type.into());
                TermKind::Type {
                    ty: ann.into_tree(),
                }
            }
            resin_ast::TermKind::Try { value } => {
                let input = self.child(value, None);
                let (value, errors) = self.result_parts(&input.ty, span)?;
                let result = self.checker.result.clone();
                let (_, target_errors) = self.result_parts(&result, span)?;
                self.constrain((span, Constraint::Errors(errors, target_errors)));
                equate = Some(value);
                TermKind::Try {
                    value: Box::new(input),
                }
            }
            resin_ast::TermKind::Match { value, arms } => {
                let input = self.child(value, None);
                let mut checked = vec![];
                for arm in arms {
                    self.checker.scopes.push_at(arm.body.span);
                    let payload = self.checker.typing.solver.fresh();
                    let (variant, pattern) = match &arm.variant {
                        resin_ast::MatchVariant::Ok => (None, Pattern::Ok),
                        resin_ast::MatchVariant::Err => (None, Pattern::Err),
                        resin_ast::MatchVariant::Type(ty) => {
                            let ann = self.annotation(ty, false);
                            let ty = ann.ty.clone();
                            (Some(ann.into_tree()), Pattern::Type(ty))
                        }
                    };
                    self.constrain((
                        arm.body.span,
                        Constraint::Variant(input.ty.clone(), pattern, payload.clone()),
                    ));
                    let binding = arm.name.as_ref().and_then(|name| {
                        self.checker
                            .bind(name, payload, DefinitionKind::Variable)
                            .map_err(|error| self.checker.errors.push(error))
                            .ok()
                    });
                    let body = self.child(&arm.body, Some(out.clone()));
                    checked.push(MatchArm {
                        binding,
                        variant,
                        failure: matches!(arm.variant, resin_ast::MatchVariant::Err),
                        body,
                    });
                    self.checker.scopes.pop();
                }
                TermKind::Match {
                    value: Box::new(input),
                    arms: checked,
                }
            }
            resin_ast::TermKind::If { cond, then, els } => {
                let cond = self.child(cond, None);
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let then = self.child(then, Some(out.clone()));
                let els = self.child(els, Some(out.clone()));
                TermKind::If {
                    cond: Box::new(cond),
                    then: Box::new(then),
                    els: Box::new(els),
                }
            }
            resin_ast::TermKind::While { cond, body } => {
                let cond = self.child(cond, None);
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let body = self.child(body, None);
                equate = Some(Ty::Unit.into());
                TermKind::While {
                    cond: Box::new(cond),
                    body: Box::new(body),
                }
            }
            resin_ast::TermKind::Block { stmts, tail } => {
                self.checker.scopes.push_at(term.span);
                let stmts = stmts
                    .iter()
                    .map(|stmt| self.statement(stmt))
                    .collect::<Vec<_>>();
                let tail = self.child(tail, Some(out.clone()));
                self.checker.scopes.pop();
                TermKind::Block {
                    stmts,
                    tail: Box::new(tail),
                }
            }
            resin_ast::TermKind::Record { fields } => {
                let fields = fields
                    .iter()
                    .map(|(name, term)| (name.clone(), self.child(term, None)))
                    .collect::<Vec<_>>();
                self.constrain((
                    span,
                    Constraint::Record(
                        fields
                            .iter()
                            .map(|(name, term)| (name.val.clone(), term.ty.clone()))
                            .collect(),
                        out.clone(),
                    ),
                ));
                TermKind::Record { fields }
            }
            resin_ast::TermKind::Array { elems } => {
                let element = self.checker.typing.solver.fresh();
                let elems = elems
                    .iter()
                    .map(|elem| self.child(elem, Some(element.clone())))
                    .collect::<Vec<_>>();
                equate = Some(Type::Node(Head::Array(elems.len()), vec![element]));
                TermKind::Array { elems }
            }
            resin_ast::TermKind::Builtin { name, args } => {
                let args = args
                    .iter()
                    .map(|arg| self.child(arg, None))
                    .collect::<Vec<_>>();
                self.constrain((
                    span,
                    Constraint::Builtin(
                        name.clone(),
                        args.iter().map(|arg| arg.ty.clone()).collect(),
                        out.clone(),
                    ),
                ));
                TermKind::Builtin {
                    name: name.clone(),
                    args,
                }
            }
            resin_ast::TermKind::MethodCall {
                receiver,
                name,
                arg,
            } => {
                let (receiver, annotation, receiver_type, associated) =
                    if let resin_ast::TermKind::Type { ty } = &receiver.val {
                        let annotation = self.annotation(ty, false);
                        let ty = annotation.ty.clone();
                        (None, Some(annotation), ty, true)
                    } else {
                        let receiver = self.child(receiver, None);
                        let ty = receiver.ty.clone();
                        (Some(receiver), None, ty, false)
                    };
                self.checker
                    .scopes
                    .record_members(name.span, receiver_type.clone(), associated);
                let arg = self.child(arg, None);
                self.constrain((
                    span,
                    Constraint::Method(
                        receiver_type.clone(),
                        name.val.clone(),
                        arg.ty.clone(),
                        out.clone(),
                        associated,
                    ),
                ));
                TermKind::MethodCall {
                    receiver: receiver.map(Box::new),
                    receiver_type: annotation.map(Annotation::into_tree).unwrap_or(
                        super::super::typed::Annotation {
                            ty: receiver_type,
                            span,
                        },
                    ),
                    name: name.clone(),
                    arg: Box::new(arg),
                }
            }
            resin_ast::TermKind::Call { func, arg } => {
                if let resin_ast::TermKind::Var { name } = &func.val
                    && name.val.as_ref() == "absurd"
                {
                    let arg = self.child(arg, Some(Ty::union([]).into()));
                    TermKind::Absurd { arg: Box::new(arg) }
                } else if let resin_ast::TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "size_of" | "align_of")
                {
                    // Check the operand for typing only. Never execute its effects or read its locals.
                    let ann = if let resin_ast::TermKind::Type { ty } = &arg.val {
                        self.annotation(ty, false)
                    } else {
                        let term = self.child(arg, None);
                        Annotation {
                            holes: Vec::new(),
                            ty: term.ty,
                            span: term.span,
                        }
                    };
                    self.constrain((span, Constraint::Layout(ann.ty.clone())));
                    equate = Some(Ty::UInt64.into());
                    let size = name.val.as_ref() == "size_of";
                    TermKind::Layout {
                        ty: ann.into_tree(),
                        size,
                    }
                } else if let resin_ast::TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "ok" | "err")
                {
                    let (value, errors) = self.result_parts(&out, span)?;
                    let failure = name.val.as_ref() == "err";
                    let arg = self.child(arg, if failure { None } else { Some(value) });
                    if failure {
                        self.constrain((span, Constraint::Errors(arg.ty.clone(), errors)));
                    }
                    TermKind::Result {
                        failure,
                        arg: Box::new(arg),
                    }
                } else if let resin_ast::TermKind::Type { ty } = &func.val {
                    let ann = self.annotation(ty, true);
                    let arg = if let Type::Node(Head::Arc, parts) = &ann.ty {
                        let context = if matches!(
                            arg.val,
                            resin_ast::TermKind::Record { .. } | resin_ast::TermKind::Unit
                        ) {
                            let payload = self.checker.typing.solver.require(&parts[0], span)?;
                            let body = self
                                .checker
                                .typing
                                .typer
                                .body(&payload)
                                .map_err(|e| GenerateError::typing(span, e))?;
                            if matches!(arg.val, resin_ast::TermKind::Unit)
                                && matches!(&body, Ty::Record { fields } if fields.is_empty())
                            {
                                Ty::Unit.into()
                            } else {
                                body.into()
                            }
                        } else {
                            parts[0].clone()
                        };
                        self.child(arg, Some(context))
                    } else {
                        let literal = matches!(arg.val, resin_ast::TermKind::Num { .. })
                            || matches!(&arg.val, resin_ast::TermKind::Builtin { name, args } if matches!(name.as_ref(), "+" | "-") && matches!(args.as_slice(), [resin_ast::Term { val: resin_ast::TermKind::Num { .. }, .. }]));
                        let arg = self.child(arg, None);
                        self.constrain((
                            span,
                            Constraint::Ascribe(arg.ty.clone(), ann.ty.clone(), literal),
                        ));
                        arg
                    };
                    equate = Some(ann.ty.clone());
                    TermKind::Ascribe {
                        ty: ann.into_tree(),
                        arg: Box::new(arg),
                    }
                } else if let resin_ast::TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "print" | "fmt")
                {
                    let arg = self.child(arg, None);
                    self.constrain((
                        span,
                        Constraint::Builtin(name.val.clone(), vec![arg.ty.clone()], out.clone()),
                    ));
                    TermKind::Builtin {
                        name: name.val.clone(),
                        args: vec![arg],
                    }
                } else {
                    let func = self.child(func, None);
                    let arg = self.child(arg, None);
                    self.constrain((
                        span,
                        Constraint::Call(func.ty.clone(), arg.ty.clone(), out.clone()),
                    ));
                    TermKind::Call {
                        func: Box::new(func),
                        arg: Box::new(arg),
                    }
                }
            }
            resin_ast::TermKind::Assign { place, value } => {
                let place = self.child(place, None);
                let value = self.child(value, Some(place.ty.clone()));
                equate = Some(place.ty.clone());
                TermKind::Assign {
                    place: Box::new(place),
                    value: Box::new(value),
                }
            }
            resin_ast::TermKind::Address { place } => {
                let place = self.child(place, None);
                equate = Some(Type::pointer(place.ty.clone()));
                TermKind::Address {
                    place: Box::new(place),
                }
            }
            resin_ast::TermKind::Deref { pointer } => {
                let pointer = self.child(pointer, None);
                self.constrain((span, Constraint::Deref(pointer.ty.clone(), out.clone())));
                TermKind::Deref {
                    pointer: Box::new(pointer),
                }
            }
            resin_ast::TermKind::Field { base, name } => {
                let base = self.child(base, None);
                let (receiver, associated) = match &base.kind {
                    TermKind::Type { ty } => (ty.ty.clone(), true),
                    TermKind::Var { name } if self.checker.scopes.is_shader(&name.val) => {
                        (crate::lower::builtins::shader_properties().into(), false)
                    }
                    _ => (base.ty.clone(), false),
                };
                self.checker
                    .scopes
                    .record_members(name.span, receiver, associated);
                self.constrain((
                    span,
                    Constraint::Field(base.ty.clone(), name.val.clone(), out.clone()),
                ));
                TermKind::Field {
                    base: Box::new(base),
                    name: name.clone(),
                }
            }
        };
        if let Some(ty) = equate {
            if contextual {
                self.constrain((span, Constraint::Coerce(ty, out.clone())));
            } else {
                self.constrain((span, Constraint::Equal(ty, out.clone())));
            }
        }
        if !propagate && let Some(expected) = expected {
            self.constrain((span, Constraint::Coerce(out.clone(), expected)));
        }
        Ok(Term {
            context,
            span,
            ty: out,
            kind,
        })
    }

    fn statement(&mut self, stmt: &resin_ast::Stmt) -> Statement {
        let context = self.checker.scopes.capture();
        let result = self.statement_inner(stmt);
        let kind = match result {
            Ok(statement) => statement,
            Err(error) => {
                match &stmt.val {
                    StmtKind::Declare { name, .. } => {
                        let _ = self
                            .checker
                            .bind(name, Type::Invalid, DefinitionKind::Variable);
                    }
                    StmtKind::DefineType { name, .. } => {
                        self.checker.scopes.define_invalid_type(name);
                    }
                    _ => {}
                }
                self.checker.errors.push(error.clone());
                StatementKind::Error(error)
            }
        };
        Statement { context, kind }
    }

    fn statement_inner(&mut self, stmt: &resin_ast::Stmt) -> Result<StatementKind<Type>> {
        let span = stmt.span;
        Ok(match &stmt.val {
            StmtKind::Define { name, init } => {
                let ty = self.checker.typing.solver.fresh();
                let binding = self
                    .checker
                    .bind(name, ty.clone(), DefinitionKind::Variable)
                    .map_err(|error| self.checker.errors.push(error))
                    .ok();
                let init = self.child(init, Some(ty));
                if let Some(binding) = binding {
                    self.checker.scopes.set_inferred(binding, init.ty.clone());
                }
                StatementKind::Define {
                    binding,
                    name: name.clone(),
                    init,
                }
            }
            StmtKind::Declare { name, ann } => {
                let ann = self.annotation(ann, true);
                let binding = self
                    .checker
                    .bind(name, ann.ty.clone(), DefinitionKind::Variable)?;
                StatementKind::Declare {
                    binding,
                    name: name.clone(),
                    ty: ann.into_tree(),
                }
            }
            StmtKind::Struct { name, body } => {
                let definition = self.checker.typing.typer.declare_type(
                    name.val.clone(),
                    crate::lower::namespaces::SourceOrigin {
                        module: self.checker.source_module,
                        span: name.span,
                    },
                );
                self.checker
                    .scopes
                    .define_type(name, definition)
                    .map_err(|name| GenerateError {
                        span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                let ann = self.annotation(body, false);
                let ty = self.checker.typing.solver.require(&ann.ty, ann.span)?;
                self.checker
                    .typing
                    .typer
                    .define_type(definition, ty)
                    .map_err(|e| GenerateError::typing(ann.span, e))?;
                StatementKind::TypeDefinition
            }
            StmtKind::DefineType { name, init } => {
                let ann = self.annotation(init, false);
                let ty = self.checker.typing.solver.require(&ann.ty, ann.span)?;
                self.checker
                    .scopes
                    .define_alias(name, ty)
                    .map_err(|name| GenerateError {
                        span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                StatementKind::TypeDefinition
            }
            StmtKind::Expr { term } => {
                let term = self.child(term, None);
                StatementKind::Expr { term }
            }
            _ => {
                return Err(GenerateError::inference(
                    span,
                    "unexpected declaration in function body",
                ));
            }
        })
    }
}
