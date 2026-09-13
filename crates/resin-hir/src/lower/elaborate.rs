//! Complete solved expressions directly into public HIR.
//! This is the last part of HIR construction; no concrete private tree is retained.
use super::infer::{Rule, Solver, Type};
use super::scope::DeclarationId;
use super::{typed, types};
use crate::ReceiverConversion;
use crate::lower::context::{FunctionBody, FunctionDecl};
use crate::{Arguments, MatchArm, Statement, Term, TermKind};
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
use resin_types::ExplicitConversion;
use resin_types::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};

type Result<T> = std::result::Result<T, GenerateError>;

pub(super) struct CompletedBody {
    pub body: Term,
    pub shaders: BTreeSet<FunctionId>,
}

pub(super) fn function(
    source: &typed::Term,
    parameters: &[Option<DeclarationId>],
    solver: &Solver,
    methods: &BTreeMap<Rule, FunctionDecl>,
    typer: &TyperContext,
    function_bindings: &HashMap<DeclarationId, FunctionId>,
    shaders: &BTreeMap<FunctionId, ShaderEntry>,
) -> Result<CompletedBody> {
    let mut completion = Completion {
        solver,
        methods,
        typer,
        function_bindings,
        shaders,
        embedded: BTreeSet::new(),
        initialization: parameters
            .iter()
            .flatten()
            .map(|id| (*id, Initialization::Initialized))
            .collect(),
    };
    let body = completion.elaborate(source)?;
    Ok(CompletedBody {
        body,
        shaders: completion.embedded,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Initialization {
    Uninitialized,
    Initializing,
    Initialized,
}

struct Completion<'a> {
    solver: &'a Solver,
    methods: &'a BTreeMap<Rule, FunctionDecl>,
    typer: &'a TyperContext,
    function_bindings: &'a HashMap<DeclarationId, FunctionId>,
    shaders: &'a BTreeMap<FunctionId, ShaderEntry>,
    embedded: BTreeSet<FunctionId>,
    initialization: BTreeMap<DeclarationId, Initialization>,
}

impl Completion<'_> {
    fn elaborate(&mut self, source: &typed::Term) -> Result<Term> {
        Ok(Term {
            span: source.span,
            ty: types::ty(&self.ty(source)?),
            kind: self.elaborate_kind(source)?,
        })
    }

    fn ty(&self, source: &typed::Term) -> Result<Ty> {
        self.solver.require(&source.ty, source.span)
    }

    fn annotation(&self, source: &typed::Annotation<Type>) -> Result<typed::Annotation> {
        Ok(typed::Annotation {
            ty: self.solver.require(&source.ty, source.span)?,
            span: source.span,
        })
    }

    fn boxed(&mut self, source: &typed::Term) -> Result<Box<Term>> {
        self.elaborate(source).map(Box::new)
    }

    fn elaborate_kind(&mut self, source: &typed::Term) -> Result<TermKind> {
        Ok(match &source.kind {
            typed::TermKind::Error(error) => return Err(error.clone()),
            typed::TermKind::Unit => TermKind::Constant {
                value: crate::Constant::Unit,
            },
            typed::TermKind::None => TermKind::Constant {
                value: crate::Constant::None,
            },
            typed::TermKind::Num { value } => self.number(source, value)?,
            typed::TermKind::String { value } => TermKind::Constant {
                value: crate::Constant::Str {
                    value: value.as_bytes().into(),
                },
            },
            typed::TermKind::Type { ty } => TermKind::Constant {
                value: crate::Constant::Type {
                    ty: types::ty(&self.annotation(ty)?.ty),
                },
            },
            typed::TermKind::Var { declaration, name } => {
                self.reference(*declaration, name, true)?
            }
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
            typed::TermKind::If { cond, then, els } => self.if_expression(cond, then, els)?,
            typed::TermKind::While { cond, body } => self.while_expression(cond, body)?,
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
                rule,
                receiver,
                receiver_type,
                name,
                arg,
            } => self.method(
                self.methods.get(rule).expect("solved method").clone(),
                receiver.as_deref(),
                &self.annotation(receiver_type)?.ty,
                name,
                arg,
            )?,
            typed::TermKind::Call { func, arg } => self.call(func, arg)?,
            typed::TermKind::Ascribe { ty, arg } => {
                self.ascription(source.span, &self.annotation(ty)?.ty, arg)?
            }
            typed::TermKind::Result { failure, arg } => TermKind::Result {
                failure: *failure,
                arg: self.boxed(arg)?,
            },
            typed::TermKind::Absurd { arg } => TermKind::Absurd {
                arg: self.boxed(arg)?,
            },
            typed::TermKind::Assign { place, value } => self.assign(place, value)?,
            typed::TermKind::Address { place } => TermKind::Address {
                place: self.place(place)?,
            },
            typed::TermKind::Deref { pointer } => TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            typed::TermKind::Field { base, name } => self.field(source.span, base, name)?,
        })
    }

    fn reference(&self, declaration: DeclarationId, name: &Ident, read: bool) -> Result<TermKind> {
        if let Some(&function) = self.function_bindings.get(&declaration) {
            return Ok(TermKind::Function {
                function,
                type_args: vec![],
            });
        }
        let kind = match self.initialization.get(&declaration) {
            None => Some(GenerateErrorKind::UnboundValue {
                name: name.val.clone(),
            }),
            Some(Initialization::Initializing) => Some(GenerateErrorKind::EagerRecursion {
                name: name.val.clone(),
            }),
            Some(Initialization::Uninitialized) if read => {
                Some(GenerateErrorKind::UninitializedValue {
                    name: name.val.clone(),
                })
            }
            _ => None,
        };
        if let Some(kind) = kind {
            return Err(GenerateError {
                span: name.span,
                kind,
            });
        }
        Ok(TermKind::Local {
            binding: declaration,
            name: name.clone(),
        })
    }

    fn place(&mut self, source: &typed::Term) -> Result<Box<Term>> {
        if let typed::TermKind::Var { declaration, name } = &source.kind {
            return Ok(Box::new(Term {
                span: source.span,
                ty: types::ty(&self.ty(source)?),
                kind: self.reference(*declaration, name, false)?,
            }));
        }
        // Field access and pointer dereference need their base initialized even
        // when the resulting place will be written rather than read.
        self.boxed(source)
    }

    fn assign(&mut self, place: &typed::Term, value: &typed::Term) -> Result<TermKind> {
        let place = self.place(place)?;
        let value = self.boxed(value)?;
        if let TermKind::Local { binding, .. } = place.kind {
            self.initialization
                .insert(binding, Initialization::Initialized);
        }
        Ok(TermKind::Assign { place, value })
    }

    fn if_expression(
        &mut self,
        cond: &typed::Term,
        then: &typed::Term,
        els: &typed::Term,
    ) -> Result<TermKind> {
        let cond = self.boxed(cond)?;
        let before = self.initialization.clone();
        let then = self.boxed(then)?;
        let after_then = std::mem::replace(&mut self.initialization, before);
        let els = self.boxed(els)?;
        self.intersect_initialization(&after_then);
        Ok(TermKind::If { cond, then, els })
    }

    fn while_expression(&mut self, cond: &typed::Term, body: &typed::Term) -> Result<TermKind> {
        let cond = self.boxed(cond)?;
        let after_condition = self.initialization.clone();
        let body = self.boxed(body)?;
        self.initialization = after_condition;
        Ok(TermKind::While { cond, body })
    }

    fn intersect_initialization(&mut self, other: &BTreeMap<DeclarationId, Initialization>) {
        for (binding, state) in &mut self.initialization {
            if other.get(binding) != Some(state) {
                *state = Initialization::Uninitialized;
            }
        }
    }

    fn number(&self, source: &typed::Term, text: &str) -> Result<TermKind> {
        let (value, _) =
            super::eval::number(self.typer, source.span, text, Some(&self.ty(source)?))?;
        Ok(TermKind::Constant {
            value: types::constant(&value),
        })
    }

    fn layout(&self, ty: &typed::Annotation<Type>, size: bool) -> Result<TermKind> {
        let layout =
            resin_types::layout::layout(self.typer.definitions(), &self.annotation(ty)?.ty)
                .map_err(|e| GenerateError::inference(ty.span, e.to_string()))?;
        Ok(TermKind::Constant {
            value: crate::Constant::UInt64 {
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
            ty: crate::Type::Bool,
            kind: TermKind::Constant {
                value: crate::Constant::Bool {
                    value: name == "||",
                },
            },
        });
        let cond = self.boxed(&args[0])?;
        let before_right = self.initialization.clone();
        let right = self.boxed(&args[1])?;
        self.intersect_initialization(&before_right);
        let (then, els) = if name == "&&" {
            (right, fixed)
        } else {
            (fixed, right)
        };
        Ok(TermKind::If { cond, then, els })
    }

    fn field(&mut self, span: Span, base: &typed::Term, name: &Ident) -> Result<TermKind> {
        let base_type = self.ty(base)?;
        let base = self.boxed(base)?;
        if name.val.as_ref() == "spirv"
            && let TermKind::Function { function, .. } = base.kind
        {
            return self.shader(span, function);
        }
        let mut shape = &base_type;
        while let Some(pointee) = shape.deref_target() {
            shape = pointee;
        }
        let access = self
            .typer
            .type_field(shape, &name.val)
            .map_err(|e| GenerateError::typing(span, e))?;
        Ok(TermKind::Field {
            base,
            access: types::field(access),
        })
    }

    fn shader(&mut self, span: Span, function: FunctionId) -> Result<TermKind> {
        let entry = self.shaders.get(&function).ok_or_else(|| GenerateError {
            span,
            kind: GenerateErrorKind::InvalidShader {
                message: "`.spirv` requires a function with a shader decorator".into(),
            },
        })?;
        self.embedded.insert(function);
        Ok(TermKind::Shader {
            function,
            stage: entry.stage.clone(),
        })
    }

    fn method(
        &mut self,
        declaration: FunctionDecl,
        receiver: Option<&typed::Term>,
        receiver_ty: &Ty,
        name: &Ident,
        argument: &typed::Term,
    ) -> Result<TermKind> {
        if let FunctionBody::GpuPipelineFactory { factory, graphics } = declaration.body {
            return self.pipeline_create(
                receiver,
                argument,
                &declaration.params,
                factory,
                graphics,
            );
        }
        let receiver = receiver
            .map(|r| self.adapt(r, receiver_ty, &declaration.params[0]))
            .transpose()?;
        let params = declaration.params[usize::from(receiver.is_some())..].to_vec();
        let args = Arguments {
            receiver,
            argument: self.boxed(argument)?,
            params: params.iter().map(types::ty).collect(),
        };
        Ok(match declaration.body {
            FunctionBody::Intrinsic(op) => TermKind::Intrinsic { op, args },
            FunctionBody::GpuNew { allocator } => TermKind::GpuNew { allocator, args },
            FunctionBody::GpuAllocate { allocator } => TermKind::GpuAllocate { allocator, args },
            FunctionBody::GpuPipelineDispatch {
                context,
                allocator,
                record,
            } => TermKind::GpuPipelineDispatch {
                context,
                allocator,
                record,
                args,
            },
            FunctionBody::GpuPipelineFactory { .. } | FunctionBody::GpuPipelineRecord { .. } => {
                unreachable!("specialized pipeline bridge")
            }
            FunctionBody::Defined(function) => {
                let param = Ty::parameter(&declaration.params);
                let ty = Ty::Function {
                    param: Box::new(param.clone()),
                    result: Box::new(declaration.result),
                };
                let func = Box::new(Term {
                    span: name.span,
                    ty: types::ty(&ty),
                    kind: TermKind::Function {
                        function,
                        type_args: vec![],
                    },
                });
                let arg = if args.receiver.is_some() {
                    Box::new(Term {
                        span: argument.span,
                        ty: types::ty(&param),
                        kind: TermKind::Pack { args },
                    })
                } else {
                    args.argument
                };
                TermKind::Call { func, arg }
            }
        })
    }

    fn pipeline_create(
        &mut self,
        receiver: Option<&typed::Term>,
        argument: &typed::Term,
        params: &[Ty],
        factory: FunctionId,
        graphics: bool,
    ) -> Result<TermKind> {
        let count = params.len() - usize::from(receiver.is_some());
        let terms = if count == 1 {
            vec![argument]
        } else {
            let typed::TermKind::Record { fields } = &argument.kind else {
                return Err(GenerateError::inference(
                    argument.span,
                    "pipeline creation requires shader declarations as direct arguments",
                ));
            };
            fields.iter().map(|(_, term)| term).collect()
        };
        let stages = if graphics {
            &["vertex", "fragment"][..]
        } else {
            &["compute"][..]
        };
        let mut shaders = Vec::new();
        for (shader, stage) in terms[usize::from(receiver.is_none())..].iter().zip(stages) {
            let term = self.elaborate(shader)?;
            let TermKind::Function { function, .. } = term.kind else {
                return Err(GenerateError::inference(
                    shader.span,
                    "pipeline creation requires direct shader declarations; runtime aliases are unsupported",
                ));
            };
            let entry = self.shaders.get(&function).ok_or_else(|| {
                GenerateError::inference(
                    shader.span,
                    "pipeline creation requires a decorated shader declaration",
                )
            })?;
            if entry.stage.as_ref() != *stage {
                return Err(GenerateError::inference(
                    shader.span,
                    format!("pipeline requires a @{stage}_shader declaration"),
                ));
            }
            self.embedded.insert(function);
            shaders.push(function);
        }
        let args = if let Some(receiver) = receiver {
            Arguments {
                receiver: Some(self.adapt(receiver, &self.ty(receiver)?, &params[0])?),
                argument: Box::new(Term {
                    span: argument.span,
                    ty: crate::Type::Unit,
                    kind: TermKind::Constant {
                        value: crate::Constant::Unit,
                    },
                }),
                params: vec![],
            }
        } else {
            Arguments {
                receiver: None,
                argument: self.boxed(terms[0])?,
                params: vec![types::ty(&params[0])],
            }
        };
        Ok(TermKind::GpuPipelineCreate {
            factory,
            shaders,
            args,
        })
    }

    fn adapt(&mut self, source: &typed::Term, from: &Ty, to: &Ty) -> Result<Box<Term>> {
        let conversion = ReceiverConversion::between(from, to).expect("checked receiver");
        Ok(Box::new(Term {
            span: source.span,
            ty: types::ty(to),
            kind: TermKind::Adapt {
                conversion,
                arg: self.boxed(source)?,
            },
        }))
    }

    fn call(&mut self, func: &typed::Term, arg: &typed::Term) -> Result<TermKind> {
        let function_type = self.ty(func)?;
        let shape = self
            .typer
            .body(&function_type)
            .map_err(|e| GenerateError::typing(func.span, e))?;
        let to = match shape {
            Ty::Array { .. } => Some(Ty::Pointer {
                pointee: Box::new(function_type.clone()),
            }),
            Ty::Str | Ty::Span { .. } | Ty::GpuPointer { .. } | Ty::GpuSpan { .. } => {
                Some(function_type.clone())
            }
            _ => None,
        };
        if let Some(to) = to {
            let args = Arguments {
                receiver: Some(self.adapt(func, &function_type, &to)?),
                argument: self.boxed(arg)?,
                params: vec![types::ty(&self.ty(arg)?)],
            };
            return Ok(TermKind::Intrinsic {
                op: if matches!(function_type, Ty::GpuPointer { .. } | Ty::GpuSpan { .. }) {
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
            } => {
                let binding = binding.expect("checked declaration");
                self.initialization
                    .insert(binding, Initialization::Initializing);
                let init = self.elaborate(init)?;
                self.initialization
                    .insert(binding, Initialization::Initialized);
                Statement::Define {
                    binding,
                    name: name.clone(),
                    init,
                }
            }
            typed::StatementKind::Declare { binding, name, ty } => {
                self.initialization
                    .insert(*binding, Initialization::Uninitialized);
                Statement::Declare {
                    binding: *binding,
                    name: name.clone(),
                    ty: crate::Annotation {
                        ty: types::ty(&self.annotation(ty)?.ty),
                        span: ty.span,
                    },
                }
            }
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
        let value_type = self.ty(value)?;
        let tags = match &value_type {
            Ty::Result { .. } => vec![Case::Ok, Case::Err],
            ty => ty.members().into_iter().map(Case::Type).collect(),
        };
        let value = self.boxed(value)?;
        let before = self.initialization.clone();
        let mut after = None;
        let mut seen = vec![];
        let mut checked = vec![];
        for arm in arms {
            let tag = self.pattern(arm, &value_type)?;
            if !tags.contains(&tag) || seen.contains(&tag) {
                return Err(GenerateError::inference(
                    arm.body.span,
                    "unknown or duplicate match variant",
                ));
            }
            seen.push(tag.clone());
            self.initialization = before.clone();
            if let Some(binding) = arm.binding {
                self.initialization
                    .insert(binding, Initialization::Initialized);
            }
            let body = self.elaborate(&arm.body)?;
            if let Some(previous) = &after {
                self.intersect_initialization(previous);
            }
            after = Some(self.initialization.clone());
            checked.push(MatchArm {
                tag: types::case(&tag),
                binding: arm.binding,
                body,
            });
        }
        if tags.len() != seen.len() || tags.is_empty() {
            return Err(GenerateError::inference(
                span,
                "match must cover every variant exactly once",
            ));
        }
        self.initialization = after.expect("nonempty exhaustive match");
        Ok(TermKind::Match {
            value,
            arms: checked,
        })
    }
}

impl Completion<'_> {
    fn pattern(&self, arm: &typed::MatchArm, ty: &Ty) -> Result<Case> {
        match (&arm.variant, ty) {
            (None, Ty::Result { .. }) => Ok(if arm.failure { Case::Err } else { Case::Ok }),
            (Some(ann), ty) if !matches!(ty, Ty::Result { .. }) => {
                let annotation = self.annotation(ann)?;
                if matches!(annotation.ty, Ty::Union { .. }) {
                    return Err(GenerateError::inference(
                        ann.span,
                        "union patterns must name a single member type",
                    ));
                }
                Ok(Case::Type(annotation.ty))
            }
            _ => Err(GenerateError::inference(
                arm.body.span,
                "pattern does not belong to this match type",
            )),
        }
    }
}

impl Completion<'_> {
    fn ascription(&mut self, span: Span, to: &Ty, source: &typed::Term) -> Result<TermKind> {
        if let Ty::Arc { pointee } = to {
            return Ok(TermKind::ArcNew {
                value: Box::new(self.shared_payload(pointee, source)?),
            });
        }
        if let Ty::Weak { pointee } = to
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(TermKind::WeakEmpty {
                pointee: types::ty(pointee),
            });
        }
        let (from, value) = self.constructor_argument(to, source)?;
        self.conversion(span, value, &from, to)
    }

    fn constructor_argument(&mut self, to: &Ty, source: &typed::Term) -> Result<(Ty, Term)> {
        let body = self
            .typer
            .body(to)
            .map_err(|e| GenerateError::typing(source.span, e))?;
        let body = body.view_record().unwrap_or(body);
        if matches!(&body, Ty::Record { fields } if fields.is_empty())
            && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok((
                body.clone(),
                Term {
                    span: source.span,
                    ty: types::ty(&body),
                    kind: TermKind::Record { fields: vec![] },
                },
            ));
        }
        Ok((self.ty(source)?, self.elaborate(source)?))
    }

    fn shared_payload(&mut self, to: &Ty, source: &typed::Term) -> Result<Term> {
        if !matches!(
            source.kind,
            typed::TermKind::Record { .. } | typed::TermKind::Unit
        ) {
            return self.elaborate(source);
        }
        let (from, value) = self.constructor_argument(to, source)?;
        if &from == to {
            return Ok(value);
        }
        let kind = self.conversion(source.span, value, &from, to)?;
        Ok(Term {
            span: source.span,
            ty: types::ty(to),
            kind,
        })
    }

    fn conversion(&self, span: Span, value: Term, from: &Ty, to: &Ty) -> Result<TermKind> {
        let conversion = self
            .typer
            .explicit_conversion(from, to)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lower::{
        context::Context,
        infer::{Constraint, Inference},
    };

    #[test]
    fn completion_uses_the_selected_method_after_its_namespace_is_gone() {
        let span = Span { start: 0, end: 0 };
        let (solver, methods, rule, ty) = {
            let mut context = Context::with_builtins();
            let mut inference = Inference::new(&mut context);
            let (rule, ty) = inference.expression();
            inference.constrain(
                rule,
                (
                    span,
                    Constraint::Method(
                        Ty::Str.into(),
                        "at".into(),
                        Ty::UInt64.into(),
                        ty.clone(),
                        false,
                        vec![],
                    ),
                ),
            );
            assert!(inference.solve(std::slice::from_ref(&ty)).is_empty());
            (inference.solver, inference.methods, rule, ty)
        };
        let source = typed::Term {
            span,
            ty,
            kind: typed::TermKind::MethodCall {
                rule,
                receiver: Some(Box::new(typed::Term {
                    span,
                    ty: Ty::Str.into(),
                    kind: typed::TermKind::String {
                        value: "bytes".into(),
                    },
                })),
                receiver_type: typed::Annotation {
                    span,
                    ty: Ty::Str.into(),
                },
                // Names remain diagnostic metadata; completion must not resolve it again.
                name: Ident::new("no_such_method".into(), span),
                arg: Box::new(typed::Term {
                    span,
                    ty: Ty::UInt64.into(),
                    kind: typed::TermKind::Num { value: "0".into() },
                }),
            },
        };
        let completed = function(
            &source,
            &[],
            &solver,
            &methods,
            &TyperContext::new(),
            &HashMap::new(),
            &BTreeMap::new(),
        )
        .unwrap();
        assert!(matches!(
            completed.body.kind,
            TermKind::Intrinsic {
                op: Intrinsic::Index,
                ..
            }
        ));
        assert_eq!(
            completed.body.ty,
            crate::Type::Pointer {
                pointee: Box::new(crate::Type::UInt8)
            }
        );
        assert!(completed.shaders.is_empty());
    }
}
