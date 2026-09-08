use super::super::GenerateError;
use super::super::{scope::Cursor, semantic::DefinitionKind};
use super::{Annotation, Emit, Form, Head, MatchArm, Planner, Result, Statement, Term, Type};
use crate::ir::typecheck::infer::{
    Rule,
    constraints::{Constraint, Pattern},
};
use crate::{
    ast::{self, StmtKind, TermKind},
    ir::{GenerateErrorKind, Instr, Ty, Value},
};
use std::rc::Rc;

impl Planner<'_> {
    pub fn term(&mut self, term: &ast::Term, expected: Option<Type>) -> Term {
        let (rule, out) = self.typing.expression();
        Expression {
            planner: self,
            rule,
        }
        .plan(term, expected, out)
    }
}

/// One expression owns all of its equations and automatically depends on its children.
struct Expression<'p, 'a> {
    planner: &'p mut Planner<'a>,
    rule: Rule,
}

impl Expression<'_, '_> {
    fn child(&mut self, term: &ast::Term, expected: Option<Type>) -> Term {
        let child = self.planner.term(term, expected);
        self.planner
            .typing
            .depends(self.rule, term.span, child.ty.clone());
        child
    }

    fn annotation(&mut self, ann: &ast::Type, infer: bool) -> Annotation {
        let annotation = self.planner.ann(ann, infer);
        self.planner
            .typing
            .depends(self.rule, ann.span, annotation.ty.clone());
        annotation
    }

    fn constrain(&mut self, constraint: (ast::Span, Constraint)) {
        self.planner.typing.constrain(self.rule, constraint);
    }

    fn result_parts(&mut self, ty: &Type, span: ast::Span) -> Result<(Type, Type)> {
        self.planner.typing.result_parts(self.rule, ty, span)
    }

    fn plan(&mut self, term: &ast::Term, expected: Option<Type>, out: Type) -> Term {
        let context = self.planner.scopes.capture();
        let planned = self.term_inner(term, expected, out.clone(), context);
        let planned = match planned {
            Ok(planned) => planned,
            Err(error) => {
                self.planner.scopes.restore(context);
                self.planner.errors.push(error.clone());
                self.planner.typing.fail(self.rule);
                Term {
                    rule: self.rule,
                    context,
                    span: term.span,
                    ty: out.clone(),
                    form: Form::Other,
                    emit: Rc::new(move |_, _| Err(error.clone())),
                }
            }
        };
        self.planner.expressions.push((term.span, out.clone()));
        self.planner.scopes.record_inferred(term.span, out);
        planned
    }

    fn term_inner(
        &mut self,
        term: &ast::Term,
        expected: Option<Type>,
        out: Type,
        context: Cursor,
    ) -> Result<Term> {
        let propagate = matches!(
            term.val,
            TermKind::If { .. } | TermKind::Match { .. } | TermKind::Block { .. }
        ) || matches!(&term.val, TermKind::Call { func, .. } if matches!(&func.val, TermKind::Var { name } if matches!(name.val.as_ref(), "ok" | "err" | "absurd")));
        let contextual = propagate && expected.is_some();
        if propagate && let Some(expected) = &expected {
            self.constrain((term.span, Constraint::Equal(expected.clone(), out.clone())));
        }
        let span = term.span;
        let mut equate = None;
        let mut form = Form::Other;
        let emit: Emit = match &term.val {
            TermKind::Hole { children } => {
                for child in children {
                    self.child(child, None);
                }
                return Err(GenerateError {
                    span,
                    kind: GenerateErrorKind::IncompleteSyntax,
                });
            }
            TermKind::FieldHole { base } => {
                let base = self.child(base, None);
                let (ty, associated) = match &base.form {
                    Form::Type { ty } => (ty.clone(), true),
                    Form::Var { name } if self.planner.scopes.is_shader(&name.val) => {
                        (Ty::shader_properties().into(), false)
                    }
                    _ => (base.ty.clone(), false),
                };
                self.planner.scopes.record_members(
                    ast::Span {
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
            TermKind::Unit => {
                equate = Some(Ty::Unit.into());
                form = Form::Unit;
                Rc::new(|g, _| {
                    g.emit(Instr::Push { value: Value::Unit });
                    Ok(Ty::Unit)
                })
            }
            TermKind::None => {
                equate = Some(Ty::None.into());
                Rc::new(|g, _| {
                    g.emit(Instr::Push { value: Value::None });
                    Ok(Ty::None)
                })
            }
            TermKind::Unwrap { value } => {
                let input = self.child(value, None);
                self.constrain((span, Constraint::ExcludeNone(input.ty.clone(), out.clone())));
                Rc::new(move |g, expected| {
                    g.gen_term(&input, None)?;
                    g.emit(Instr::ExcludeNone);
                    Ok(expected.clone())
                })
            }
            TermKind::Num { value } => {
                equate = Some(self.planner.typing.solver.number(value));
                let value = value.clone();
                form = Form::Num {
                    value: value.clone(),
                };
                Rc::new(move |g, expected| {
                    let (value, ty) = g.evaluator().number(span, &value, Some(expected))?;
                    g.emit(Instr::Push { value });
                    Ok(ty)
                })
            }
            TermKind::String { value } => {
                equate = Some(Ty::byte_span().into());
                let value = value.clone();
                Rc::new(move |g, _| {
                    g.emit(Instr::Push {
                        value: Value::Bytes {
                            value: value.as_bytes().into(),
                        },
                    });
                    Ok(Ty::byte_span())
                })
            }
            TermKind::Var { name } => {
                equate = Some(self.planner.value(name)?);
                let name = name.clone();
                form = Form::Var { name: name.clone() };
                Rc::new(move |g, _| g.gen_var(&name))
            }
            TermKind::Type { ty } => {
                let ann = self.annotation(ty, true);
                form = Form::Type { ty: ann.ty.clone() };
                equate = Some(Ty::Type.into());
                Rc::new(move |g, _| {
                    let ty = ann.resolve(g)?;
                    g.emit(Instr::Push {
                        value: Value::Type { ty },
                    });
                    Ok(Ty::Type)
                })
            }
            TermKind::Try { value } => {
                let input = self.child(value, None);
                let (value, errors) = self.result_parts(&input.ty, span)?;
                let result = self.planner.result.clone();
                let (_, target_errors) = self.result_parts(&result, span)?;
                self.constrain((span, Constraint::Errors(errors, target_errors)));
                equate = Some(value);
                Rc::new(move |g, _| g.gen_try(span, &input))
            }
            TermKind::Match { value, arms } => {
                let input = self.child(value, None);
                let mut planned = vec![];
                for arm in arms {
                    self.planner.scopes.push_at(arm.body.span);
                    let payload = self.planner.typing.solver.fresh();
                    let (variant, pattern) = match &arm.variant {
                        ast::MatchVariant::Ok => (None, Pattern::Ok),
                        ast::MatchVariant::Err => (None, Pattern::Err),
                        ast::MatchVariant::Type(ty) => {
                            let ann = self.annotation(ty, false);
                            let ty = ann.ty.clone();
                            (Some(ann), Pattern::Type(ty))
                        }
                    };
                    self.constrain((
                        arm.body.span,
                        Constraint::Variant(input.ty.clone(), pattern, payload.clone()),
                    ));
                    let binding = arm.name.as_ref().and_then(|name| {
                        self.planner
                            .bind(name, payload, DefinitionKind::Variable)
                            .map_err(|error| self.planner.errors.push(error))
                            .ok()
                    });
                    let body = self.child(&arm.body, Some(out.clone()));
                    planned.push(MatchArm {
                        binding,
                        variant,
                        failure: matches!(arm.variant, ast::MatchVariant::Err),
                        body,
                    });
                    self.planner.scopes.pop();
                }
                Rc::new(move |g, expected| g.gen_match(span, &input, &planned, expected))
            }
            TermKind::If { cond, then, els } => {
                let cond = self.child(cond, None);
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let then = self.child(then, Some(out.clone()));
                let els = self.child(els, Some(out.clone()));
                Rc::new(move |g, expected| g.gen_if(&cond, &then, &els, expected))
            }
            TermKind::While { cond, body } => {
                let cond = self.child(cond, None);
                self.constrain((cond.span, Constraint::Boolean(cond.ty.clone())));
                let body = self.child(body, None);
                equate = Some(Ty::Unit.into());
                Rc::new(move |g, _| g.gen_while(&cond, &body))
            }
            TermKind::Block { stmts, tail } => {
                self.planner.scopes.push_at(term.span);
                let stmts = stmts
                    .iter()
                    .map(|stmt| self.statement(stmt))
                    .collect::<Vec<_>>();
                let tail = self.child(tail, Some(out.clone()));
                self.planner.scopes.pop();
                Rc::new(move |g, expected| g.gen_block(&stmts, &tail, expected))
            }
            TermKind::Record { fields } => {
                form = Form::Record;
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
                Rc::new(move |g, expected| g.gen_record(&fields, expected))
            }
            TermKind::Array { elems } => {
                let element = self.planner.typing.solver.fresh();
                let elems = elems
                    .iter()
                    .map(|elem| self.child(elem, Some(element.clone())))
                    .collect::<Vec<_>>();
                equate = Some(Type::Node(Head::Array(elems.len()), vec![element]));
                Rc::new(move |g, expected| g.gen_array(&elems, expected))
            }
            TermKind::Builtin { name, args } => {
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
                let name = name.clone();
                Rc::new(move |g, expected| g.gen_builtin(span, &name, &args, expected))
            }
            TermKind::MethodCall {
                receiver,
                name,
                arg,
            } => {
                let (receiver, annotation, receiver_type, associated) =
                    if let TermKind::Type { ty } = &receiver.val {
                        let annotation = self.annotation(ty, false);
                        let ty = annotation.ty.clone();
                        (None, Some(annotation), ty, true)
                    } else {
                        let receiver = self.child(receiver, None);
                        let ty = receiver.ty.clone();
                        (Some(receiver), None, ty, false)
                    };
                self.planner
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
                let name = name.clone();
                Rc::new(move |g, _| {
                    let ty = if let Some(annotation) = &annotation {
                        annotation.resolve(g)?
                    } else {
                        g.solver.require(&receiver_type, span)?
                    };
                    g.gen_method_call(receiver.as_ref(), &ty, &name, &arg)
                })
            }
            TermKind::Call { func, arg } => {
                if let TermKind::Var { name } = &func.val
                    && name.val.as_ref() == "absurd"
                {
                    let arg = self.child(arg, Some(Ty::union([]).into()));
                    Rc::new(move |g, expected| {
                        g.gen_term(&arg, Some(&Ty::union([])))?;
                        g.emit(Instr::Eliminate {
                            result: expected.clone(),
                        });
                        Ok(expected.clone())
                    })
                } else if let TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "size_of" | "align_of")
                {
                    // Plan the operand for typing only. Never execute its effects or read its locals.
                    let ann = if let TermKind::Type { ty } = &arg.val {
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
                    Rc::new(move |g, _| {
                        let ty = ann.resolve(g)?;
                        let layout = crate::ir::layout::layout(g.typer.definitions(), &ty)
                            .map_err(|e| GenerateError::inference(span, e.to_string()))?;
                        g.emit(Instr::Push {
                            value: Value::UInt64 {
                                value: if size { layout.size } else { layout.align } as u64,
                            },
                        });
                        Ok(Ty::UInt64)
                    })
                } else if let TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "ok" | "err")
                {
                    let (value, errors) = self.result_parts(&out, span)?;
                    let failure = name.val.as_ref() == "err";
                    let arg = self.child(arg, if failure { None } else { Some(value) });
                    if failure {
                        self.constrain((span, Constraint::Errors(arg.ty.clone(), errors)));
                    }
                    Rc::new(move |g, expected| g.gen_result(span, failure, &arg, expected))
                } else if let TermKind::Type { ty } = &func.val {
                    let ann = self.annotation(ty, true);
                    let arg = if let Type::Node(Head::Arc, parts) = &ann.ty {
                        let context = if matches!(arg.val, TermKind::Record { .. } | TermKind::Unit)
                        {
                            let payload = self.planner.typing.solver.require(&parts[0], span)?;
                            let body = self
                                .planner
                                .typing
                                .typer
                                .body(&payload)
                                .map_err(|e| GenerateError::typing(span, e))?;
                            if matches!(arg.val, TermKind::Unit)
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
                        let literal = matches!(arg.val, TermKind::Num { .. })
                            || matches!(&arg.val, TermKind::Builtin { name, args } if matches!(name.as_ref(), "+" | "-") && matches!(args.as_slice(), [ast::Term { val: TermKind::Num { .. }, .. }]));
                        let arg = self.child(arg, None);
                        self.constrain((
                            span,
                            Constraint::Ascribe(arg.ty.clone(), ann.ty.clone(), literal),
                        ));
                        arg
                    };
                    equate = Some(ann.ty.clone());
                    Rc::new(move |g, _| {
                        let ty = ann.resolve(g)?;
                        g.gen_ascription(span, &ty, &arg)
                    })
                } else if let TermKind::Var { name } = &func.val
                    && matches!(name.val.as_ref(), "print" | "fmt")
                {
                    let arg = self.child(arg, None);
                    self.constrain((
                        span,
                        Constraint::Builtin(name.val.clone(), vec![arg.ty.clone()], out.clone()),
                    ));
                    let name = name.val.clone();
                    Rc::new(move |g, expected| {
                        g.gen_builtin(span, &name, std::slice::from_ref(&arg), expected)
                    })
                } else {
                    let func = self.child(func, None);
                    let arg = self.child(arg, None);
                    self.constrain((
                        span,
                        Constraint::Call(func.ty.clone(), arg.ty.clone(), out.clone()),
                    ));
                    Rc::new(move |g, _| g.gen_call(&func, &arg))
                }
            }
            TermKind::Assign { place, value } => {
                let place = self.child(place, None);
                let value = self.child(value, Some(place.ty.clone()));
                equate = Some(place.ty.clone());
                Rc::new(move |g, _| g.gen_assign(&place, &value))
            }
            TermKind::Address { place } => {
                let place = self.child(place, None);
                equate = Some(Type::pointer(place.ty.clone()));
                Rc::new(move |g, _| g.gen_place(&place))
            }
            TermKind::Deref { pointer } => {
                let pointer = self.child(pointer, None);
                self.constrain((span, Constraint::Deref(pointer.ty.clone(), out.clone())));
                form = Form::Deref {
                    pointer: Rc::new(pointer.clone()),
                };
                Rc::new(move |g, expected| {
                    if matches!(g.solver.require(&pointer.ty, pointer.span)?, Ty::Arc { .. }) {
                        g.hold_arc_address(&pointer)?;
                    } else {
                        g.gen_term(&pointer, None)?;
                    }
                    g.emit(Instr::Load);
                    Ok(expected.clone())
                })
            }
            TermKind::Field { base, name } => {
                let base = self.child(base, None);
                let (receiver, associated) = match &base.form {
                    Form::Type { ty } => (ty.clone(), true),
                    Form::Var { name } if self.planner.scopes.is_shader(&name.val) => {
                        (Ty::shader_properties().into(), false)
                    }
                    _ => (base.ty.clone(), false),
                };
                self.planner
                    .scopes
                    .record_members(name.span, receiver, associated);
                self.constrain((
                    span,
                    Constraint::Field(base.ty.clone(), name.val.clone(), out.clone()),
                ));
                let name = name.clone();
                form = Form::Field {
                    base: Rc::new(base.clone()),
                    name: name.clone(),
                };
                Rc::new(move |g, _| g.gen_field_value(span, &base, &name))
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
            rule: self.rule,
            context,
            span,
            ty: out,
            form,
            emit,
        })
    }

    fn statement(&mut self, stmt: &ast::Stmt) -> Statement {
        let context = self.planner.scopes.capture();
        let result = self.statement_inner(stmt);
        let statement: Statement = match result {
            Ok(statement) => statement,
            Err(error) => {
                match &stmt.val {
                    StmtKind::Declare { name, .. } => {
                        let _ = self
                            .planner
                            .bind(name, Type::Invalid, DefinitionKind::Variable);
                    }
                    StmtKind::DefineType { name, .. } => {
                        self.planner.scopes.define_invalid_type(name);
                    }
                    _ => {}
                }
                self.planner.errors.push(error.clone());
                Rc::new(move |_| Err(error.clone()))
            }
        };
        Rc::new(move |g| g.with_context(context, |g| statement(g)))
    }

    fn statement_inner(&mut self, stmt: &ast::Stmt) -> Result<Statement> {
        let span = stmt.span;
        Ok(match &stmt.val {
            StmtKind::Define { name, init } => {
                let ty = self.planner.typing.solver.fresh();
                let binding = self
                    .planner
                    .bind(name, ty.clone(), DefinitionKind::Variable)
                    .map_err(|error| self.planner.errors.push(error))
                    .ok();
                let init = self.child(init, Some(ty));
                if let Some(binding) = binding {
                    self.planner.scopes.set_inferred(binding, init.ty.clone());
                }
                let name = name.clone();
                Rc::new(move |g| g.gen_define(binding.expect("valid declaration"), &name, &init))
            }
            StmtKind::Declare { name, ann } => {
                let ann = self.annotation(ann, true);
                let binding = self
                    .planner
                    .bind(name, ann.ty.clone(), DefinitionKind::Variable)?;
                let name = name.clone();
                Rc::new(move |g| {
                    let ty = ann.resolve(g)?;
                    g.gen_declare(binding, &name, ty)
                })
            }
            StmtKind::Struct { name, body } => {
                let definition = self.planner.typing.typer.declare_type(
                    name.val.clone(),
                    crate::ir::typecheck::SourceOrigin {
                        module: self.planner.source_module,
                        span: name.span,
                    },
                );
                self.planner
                    .scopes
                    .define_type(name, definition)
                    .map_err(|name| GenerateError {
                        span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                let ann = self.annotation(body, false);
                let ty = self.planner.typing.solver.require(&ann.ty, ann.span)?;
                self.planner
                    .typing
                    .typer
                    .define_type(definition, ty)
                    .map_err(|e| GenerateError::typing(ann.span, e))?;
                Rc::new(|_| Ok(()))
            }
            StmtKind::DefineType { name, init } => {
                let ann = self.annotation(init, false);
                let ty = self.planner.typing.solver.require(&ann.ty, ann.span)?;
                self.planner
                    .scopes
                    .define_alias(name, ty)
                    .map_err(|name| GenerateError {
                        span,
                        kind: GenerateErrorKind::DuplicateType { name },
                    })?;
                Rc::new(|_| Ok(()))
            }
            StmtKind::Expr { term } => {
                let term = self.child(term, None);
                Rc::new(move |g| {
                    g.gen_term(&term, None)?;
                    g.emit(Instr::Discard);
                    Ok(())
                })
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
