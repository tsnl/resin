//! Discard source contexts and express sugar using the HIR language.
use super::elaborate_annotation;
use super::eval::Evaluator;
use super::{Generator, typed};
use crate::ReceiverConversion;
use crate::lower::context::FunctionBody;
use crate::{Arguments, MatchArm, Statement, Term, TermKind};
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
use resin_types::ExplicitConversion;
use resin_types::prelude::*;

type Result<T> = std::result::Result<T, GenerateError>;

impl Generator {
    pub(super) fn elaborate(&mut self, source: &typed::Term) -> Result<Term> {
        let before = self.scopes.select(source.context);
        let kind = self.elaborate_kind(source);
        self.scopes.select(before);
        Ok(Term {
            span: source.span,
            ty: source.ty.clone(),
            kind: kind?,
        })
    }

    fn boxed(&mut self, source: &typed::Term) -> Result<Box<Term>> {
        self.elaborate(source).map(Box::new)
    }

    fn elaborate_kind(&mut self, source: &typed::Term) -> Result<TermKind> {
        Ok(match &source.kind {
            typed::TermKind::Error(error) => return Err(error.clone()),
            typed::TermKind::Unit => TermKind::Constant { value: Value::Unit },
            typed::TermKind::None => TermKind::Constant { value: Value::None },
            typed::TermKind::Num { value } => self.number(source, value)?,
            typed::TermKind::String { value } => TermKind::Constant {
                value: Value::Str {
                    value: value.as_bytes().into(),
                },
            },
            typed::TermKind::Type { ty } => TermKind::Constant {
                value: Value::Type { ty: ty.ty.clone() },
            },
            typed::TermKind::Var { name } => self.reference(name)?,
            typed::TermKind::Layout { ty, size } => self.layout(ty, *size)?,
            typed::TermKind::Unwrap { value } => TermKind::Unwrap {
                value: self.boxed(value)?,
            },
            typed::TermKind::Try { value } => TermKind::Try {
                value: self.boxed(value)?,
            },
            typed::TermKind::Match { value, arms } => {
                self.match_expression(source.span, value, arms)?
            }
            typed::TermKind::If { cond, then, els } => TermKind::If {
                cond: self.boxed(cond)?,
                then: self.boxed(then)?,
                els: self.boxed(els)?,
            },
            typed::TermKind::While { cond, body } => TermKind::While {
                cond: self.boxed(cond)?,
                body: self.boxed(body)?,
            },
            typed::TermKind::Block { stmts, tail } => TermKind::Block {
                stmts: stmts
                    .iter()
                    .filter_map(|stmt| self.statement(stmt).transpose())
                    .collect::<Result<_>>()?,
                tail: self.boxed(tail)?,
            },
            typed::TermKind::Record { fields } => TermKind::Record {
                fields: fields
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), self.elaborate(value)?)))
                    .collect::<Result<_>>()?,
            },
            typed::TermKind::Array { elems } => TermKind::Array {
                elems: elems
                    .iter()
                    .map(|e| self.elaborate(e))
                    .collect::<Result<_>>()?,
            },
            typed::TermKind::Builtin { name, args } => self.builtin(source, name, args)?,
            typed::TermKind::MethodCall {
                receiver,
                receiver_type,
                name,
                arg,
            } => self.method(receiver.as_deref(), &receiver_type.ty, name, arg)?,
            typed::TermKind::Call { func, arg } => self.call(func, arg)?,
            typed::TermKind::Ascribe { ty, arg } => self.ascription(source.span, &ty.ty, arg)?,
            typed::TermKind::Result { failure, arg } => TermKind::Result {
                failure: *failure,
                arg: self.boxed(arg)?,
            },
            typed::TermKind::Absurd { arg } => TermKind::Absurd {
                arg: self.boxed(arg)?,
            },
            typed::TermKind::Assign { place, value } => TermKind::Assign {
                place: self.boxed(place)?,
                value: self.boxed(value)?,
            },
            typed::TermKind::Address { place } => TermKind::Address {
                place: self.boxed(place)?,
            },
            typed::TermKind::Deref { pointer } => TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            typed::TermKind::Field { base, name } => self.field(source.span, base, name)?,
        })
    }

    fn reference(&self, name: &Ident) -> Result<TermKind> {
        let binding = self
            .scopes
            .lookup(&name.val, false)
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })?;
        Ok(match self.function_bindings.get(&binding).copied() {
            Some(function) => TermKind::Function { function },
            None => TermKind::Local {
                binding,
                name: name.clone(),
            },
        })
    }

    fn number(&self, source: &typed::Term, text: &str) -> Result<TermKind> {
        let evaluator = Evaluator {
            scopes: &self.scopes,
            typer: &self.typer,
        };
        let (value, _) = evaluator.number(source.span, text, Some(&source.ty))?;
        Ok(TermKind::Constant { value })
    }

    fn layout(&self, ty: &typed::Annotation, size: bool) -> Result<TermKind> {
        let layout = resin_types::layout::layout(self.typer.definitions(), &ty.ty)
            .map_err(|e| GenerateError::inference(ty.span, e.to_string()))?;
        Ok(TermKind::Constant {
            value: Value::UInt64 {
                value: if size { layout.size } else { layout.align } as u64,
            },
        })
    }

    fn builtin(
        &mut self,
        source: &typed::Term,
        name: &str,
        args: &[typed::Term],
    ) -> Result<TermKind> {
        if matches!(name, "&&" | "||") {
            return self.short_circuit(source.span, name, args);
        }
        if matches!(name, "+" | "-")
            && let [
                typed::Term {
                    kind: typed::TermKind::Num { value },
                    ..
                },
            ] = args
        {
            return self.number(
                source,
                &format!("{}{value}", if name == "-" { "-" } else { "" }),
            );
        }
        Ok(TermKind::Builtin {
            name: name.into(),
            args: args
                .iter()
                .map(|arg| self.elaborate(arg))
                .collect::<Result<_>>()?,
        })
    }

    fn short_circuit(&mut self, span: Span, name: &str, args: &[typed::Term]) -> Result<TermKind> {
        let fixed = Box::new(Term {
            span,
            ty: Ty::Bool,
            kind: TermKind::Constant {
                value: Value::Bool {
                    value: name == "||",
                },
            },
        });
        let right = self.boxed(&args[1])?;
        let (then, els) = if name == "&&" {
            (right, fixed)
        } else {
            (fixed, right)
        };
        Ok(TermKind::If {
            cond: self.boxed(&args[0])?,
            then,
            els,
        })
    }

    fn field(&mut self, span: Span, base: &typed::Term, name: &Ident) -> Result<TermKind> {
        let base = self.boxed(base)?;
        if name.val.as_ref() == "spirv"
            && let TermKind::Function { function } = base.kind
        {
            return self.shader(span, function);
        }
        let mut shape = &base.ty;
        while let Some(pointee) = shape.deref_target() {
            shape = pointee;
        }
        let access = self
            .typer
            .type_field(shape, &name.val)
            .map_err(|e| GenerateError::typing(span, e))?;
        Ok(TermKind::Field { base, access })
    }

    fn shader(&mut self, span: Span, function: FunctionId) -> Result<TermKind> {
        let entry = self
            .module
            .shaders
            .get_mut(&function)
            .ok_or_else(|| GenerateError {
                span,
                kind: GenerateErrorKind::InvalidShader {
                    message: "`.spirv` requires a function with a shader decorator".into(),
                },
            })?;
        entry.embedded = true;
        Ok(TermKind::Shader {
            function,
            stage: entry.stage.clone(),
        })
    }

    fn method(
        &mut self,
        receiver: Option<&typed::Term>,
        receiver_ty: &Ty,
        name: &Ident,
        argument: &typed::Term,
    ) -> Result<TermKind> {
        if name.val.as_ref() == "project"
            && let Some(receiver) = receiver
        {
            let shader = self.elaborate(receiver)?;
            if let TermKind::Function { function } = shader.kind
                && self.module.shaders.contains_key(&function)
            {
                let Ty::Record { fields } = &argument.ty else {
                    unreachable!("checked projection arguments")
                };
                let allocator = self
                    .typer
                    .gpu_allocator(&fields[0].ty)
                    .expect("checked projection allocator");
                return Ok(TermKind::GpuProject {
                    allocator,
                    shader: function,
                    args: Arguments {
                        receiver: None,
                        params: fields.iter().map(|field| field.ty.clone()).collect(),
                        argument: self.boxed(argument)?,
                    },
                });
            }
        }
        let declaration = self
            .typer
            .method_call(receiver_ty, &name.val, &argument.ty, receiver.is_none())
            .ok_or_else(|| GenerateError::inference(name.span, "unknown method"))?;
        let receiver = receiver
            .map(|r| self.adapt(r, receiver_ty, &declaration.params[0]))
            .transpose()?;
        let params = declaration.params[usize::from(receiver.is_some())..].to_vec();
        let args = Arguments {
            receiver,
            argument: self.boxed(argument)?,
            params,
        };
        Ok(match declaration.body {
            FunctionBody::Intrinsic(op) => TermKind::Intrinsic { op, args },
            FunctionBody::GpuNew { allocator } => TermKind::GpuNew { allocator, args },
            FunctionBody::GpuAllocate { allocator } => TermKind::GpuAllocate { allocator, args },
            FunctionBody::Defined(function) => {
                let param = Ty::parameter(&declaration.params);
                let ty = Ty::Function {
                    param: Box::new(param.clone()),
                    result: Box::new(declaration.result),
                };
                let func = Box::new(Term {
                    span: name.span,
                    ty,
                    kind: TermKind::Function { function },
                });
                let arg = if args.receiver.is_some() {
                    Box::new(Term {
                        span: argument.span,
                        ty: param,
                        kind: TermKind::Pack { args },
                    })
                } else {
                    args.argument
                };
                TermKind::Call { func, arg }
            }
        })
    }

    fn adapt(&mut self, source: &typed::Term, from: &Ty, to: &Ty) -> Result<Box<Term>> {
        let conversion = ReceiverConversion::between(from, to).expect("checked receiver");
        Ok(Box::new(Term {
            span: source.span,
            ty: to.clone(),
            kind: TermKind::Adapt {
                conversion,
                arg: self.boxed(source)?,
            },
        }))
    }

    fn call(&mut self, func: &typed::Term, arg: &typed::Term) -> Result<TermKind> {
        let shape = self
            .typer
            .body(&func.ty)
            .map_err(|e| GenerateError::typing(func.span, e))?;
        let to = match shape {
            Ty::Array { .. } => Some(Ty::Pointer {
                pointee: Box::new(func.ty.clone()),
            }),
            Ty::Str | Ty::Span { .. } | Ty::GpuPointer { .. } | Ty::GpuSpan { .. } => {
                Some(func.ty.clone())
            }
            _ => None,
        };
        if let Some(to) = to {
            let args = Arguments {
                receiver: Some(self.adapt(func, &func.ty, &to)?),
                argument: self.boxed(arg)?,
                params: vec![arg.ty.clone()],
            };
            return Ok(TermKind::Intrinsic {
                op: if matches!(func.ty, Ty::GpuPointer { .. } | Ty::GpuSpan { .. }) {
                    Intrinsic::GpuIndex
                } else {
                    Intrinsic::Index
                },
                args,
            });
        }
        Ok(TermKind::Call {
            func: self.boxed(func)?,
            arg: self.boxed(arg)?,
        })
    }

    fn statement(&mut self, stmt: &typed::Statement) -> Result<Option<Statement>> {
        Ok(Some(match &stmt.kind {
            typed::StatementKind::Error(error) => return Err(error.clone()),
            typed::StatementKind::TypeDefinition => return Ok(None),
            typed::StatementKind::Define {
                binding,
                name,
                init,
            } => Statement::Define {
                binding: binding.expect("checked declaration"),
                name: name.clone(),
                init: self.elaborate(init)?,
            },
            typed::StatementKind::Declare { binding, name, ty } => Statement::Declare {
                binding: *binding,
                name: name.clone(),
                ty: elaborate_annotation(ty),
            },
            typed::StatementKind::Expr { term } => Statement::Expr {
                term: self.elaborate(term)?,
            },
        }))
    }

    fn match_expression(
        &mut self,
        span: Span,
        value: &typed::Term,
        arms: &[typed::MatchArm],
    ) -> Result<TermKind> {
        let tags = match &value.ty {
            Ty::Result { .. } => vec![Case::Ok, Case::Err],
            ty => ty.members().into_iter().map(Case::Type).collect(),
        };
        let mut seen = vec![];
        let mut checked = vec![];
        for arm in arms {
            let tag = pattern(arm, &value.ty)?;
            if !tags.contains(&tag) || seen.contains(&tag) {
                return Err(GenerateError::inference(
                    arm.body.span,
                    "unknown or duplicate match variant",
                ));
            }
            seen.push(tag.clone());
            checked.push(MatchArm {
                tag,
                binding: arm.binding,
                body: self.elaborate(&arm.body)?,
            });
        }
        if tags.len() != seen.len() || tags.is_empty() {
            return Err(GenerateError::inference(
                span,
                "match must cover every variant exactly once",
            ));
        }
        Ok(TermKind::Match {
            value: self.boxed(value)?,
            arms: checked,
        })
    }
}

fn pattern(arm: &typed::MatchArm, ty: &Ty) -> Result<Case> {
    match (&arm.variant, ty) {
        (None, Ty::Result { .. }) => Ok(if arm.failure { Case::Err } else { Case::Ok }),
        (Some(ann), ty) if !matches!(ty, Ty::Result { .. }) => {
            if matches!(ann.ty, Ty::Union { .. }) {
                return Err(GenerateError::inference(
                    ann.span,
                    "union patterns must name a single member type",
                ));
            }
            Ok(Case::Type(ann.ty.clone()))
        }
        _ => Err(GenerateError::inference(
            arm.body.span,
            "pattern does not belong to this match type",
        )),
    }
}

impl Generator {
    pub(super) fn ascription(
        &mut self,
        span: Span,
        to: &Ty,
        source: &typed::Term,
    ) -> Result<TermKind> {
        if let Ty::Arc { pointee } = to {
            return Ok(TermKind::ArcNew {
                value: Box::new(self.shared_payload(pointee, source)?),
            });
        }
        if let Ty::Weak { pointee } = to
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(TermKind::WeakEmpty {
                pointee: *pointee.clone(),
            });
        }
        let value = self.constructor_argument(to, source)?;
        self.conversion(span, value, to)
    }

    fn constructor_argument(&mut self, to: &Ty, source: &typed::Term) -> Result<Term> {
        let body = self
            .typer
            .body(to)
            .map_err(|e| GenerateError::typing(source.span, e))?;
        let body = body.view_record().unwrap_or(body);
        if matches!(&body, Ty::Record { fields } if fields.is_empty())
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(Term {
                span: source.span,
                ty: body,
                kind: TermKind::Record { fields: vec![] },
            });
        }
        self.elaborate(source)
    }

    fn shared_payload(&mut self, to: &Ty, source: &typed::Term) -> Result<Term> {
        if !matches!(
            source.kind,
            typed::TermKind::Record { .. } | typed::TermKind::Unit
        ) {
            return self.elaborate(source);
        }
        let value = self.constructor_argument(to, source)?;
        if &value.ty == to {
            return Ok(value);
        }
        let kind = self.conversion(source.span, value, to)?;
        Ok(Term {
            span: source.span,
            ty: to.clone(),
            kind,
        })
    }

    fn conversion(&self, span: Span, value: Term, to: &Ty) -> Result<TermKind> {
        let conversion = self
            .typer
            .explicit_conversion(&value.ty, to)
            .map_err(|e| GenerateError::typing(span, e))?;
        if let ExplicitConversion::Ascribe(steps) = &conversion {
            self.check_unwrap(span, steps)?;
        }
        Ok(TermKind::Convert {
            conversion,
            arg: Box::new(value),
        })
    }

    fn check_unwrap(&self, span: Span, steps: &[Conv]) -> Result<()> {
        if steps.iter().any(|step| {
            matches!(step, Conv::Unwrap { definition }
            if self.typer.definition(*definition).unwrap().drop_hook().is_some())
        }) {
            return Err(GenerateError::inference(
                span,
                "cannot unwrap a type with drop; access its fields through a pointer or use Ptr.replace",
            ));
        }
        Ok(())
    }
}
