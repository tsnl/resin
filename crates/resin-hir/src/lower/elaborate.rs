//! Complete solved expressions directly into public HIR.
//! This is the last part of HIR construction; no concrete private tree is retained.
use super::infer::{ResolvedMethod, Rule, Solver, Type};
use super::scope::DeclarationId;
use super::{typed, types};
use crate::ReceiverConversion;
use crate::lower::context::{FunctionBody, FunctionDecl};
use crate::{Arguments, MatchArm, Statement, Term, TermKind};
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
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
    methods: &BTreeMap<Rule, ResolvedMethod>,
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
        reachable: true,
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
    reachable: bool,
    solver: &'a Solver,
    methods: &'a BTreeMap<Rule, ResolvedMethod>,
    typer: &'a TyperContext,
    function_bindings: &'a HashMap<DeclarationId, FunctionId>,
    shaders: &'a BTreeMap<FunctionId, ShaderEntry>,
    embedded: BTreeSet<FunctionId>,
    initialization: BTreeMap<DeclarationId, Initialization>,
}

impl Completion<'_> {
    fn convert_method_result(
        &self,
        source: &typed::Term,
        result: crate::Type,
        kind: TermKind,
    ) -> Result<TermKind> {
        Ok(
            if result == self.solver.require_complete(&source.actual, source.span)? {
                kind
            } else {
                TermKind::Convert {
                    arg: Box::new(Term {
                        span: source.span,
                        ty: result,
                        kind,
                    }),
                }
            },
        )
    }

    fn elaborate(&mut self, source: &typed::Term) -> Result<Term> {
        let term = Term {
            span: source.span,
            ty: self.solver.require_complete(&source.actual, source.span)?,
            kind: self.elaborate_kind(source)?,
        };
        let target = self.solver.require_complete(&source.ty, source.span)?;
        self.consume(term, target)
    }

    fn consume(&self, term: Term, target: crate::Type) -> Result<Term> {
        if term.ty == target {
            return Ok(term);
        }
        if matches!(target, crate::Type::Reference { .. }) && !reference_place(&term) {
            return Err(GenerateError::inference(
                term.span,
                "reference binding requires an initialized place, not a temporary value",
            ));
        }
        Ok(Term {
            span: term.span,
            ty: target,
            kind: TermKind::Use {
                arg: Box::new(term),
            },
        })
    }

    fn argument(&mut self, source: &typed::Term, target: &Type) -> Result<Term> {
        let value = self.elaborate(source)?;
        self.consume(value, self.solver.require_complete(target, source.span)?)
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
            typed::TermKind::SizeOf { ty } => TermKind::SizeOf {
                of: self.solver.require_complete(&ty.ty, ty.span)?,
            },
            typed::TermKind::Constant { value } => value.kind.clone(),
            typed::TermKind::Error(error) => return Err(error.clone()),
            typed::TermKind::Unit => TermKind::Constant {
                value: crate::Constant::Unit,
            },
            typed::TermKind::Bool { value } => TermKind::Constant {
                value: crate::Constant::Bool { value: *value },
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
                    ty: self.solver.require_complete(&ty.ty, ty.span)?,
                },
            },
            typed::TermKind::Var {
                declaration,
                name,
                type_args,
            } => self.reference(*declaration, name, type_args, true)?,
            typed::TermKind::Layout { ty, size } => self.layout(ty, *size)?,
            typed::TermKind::Unwrap { value } => TermKind::Unwrap {
                value: self.boxed(value)?,
            },
            typed::TermKind::Return { value } => {
                let value = self.boxed(value)?;
                self.reachable = false;
                TermKind::Return { value }
            }
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
            typed::TermKind::Builtin {
                rule,
                name,
                name_span,
                args,
            } => match self.methods.get(rule).cloned() {
                Some(ResolvedMethod::Dependent { signature }) => TermKind::DependentMethodCall {
                    lookup: self.method_lookup(&signature, source.span)?,
                    receiver: None,
                    args: args
                        .iter()
                        .enumerate()
                        .map(|(index, arg)| {
                            self.argument(
                                arg,
                                &Type::Node(
                                    super::infer::Head::FunctionParameter { index },
                                    vec![signature.clone()],
                                ),
                            )
                        })
                        .collect::<Result<_>>()?,
                },
                Some(method @ ResolvedMethod::Source { .. }) => self.source_method_call(
                    &method,
                    None,
                    &Ident::new(name.clone(), *name_span),
                    args,
                )?,
                None => self.builtin(source, name, args)?,
                _ => unreachable!("operator resolution"),
            },
            typed::TermKind::MethodCall {
                rule,
                receiver,
                receiver_type,
                name,
                args,
            } => {
                let method = self.methods.get(rule).expect("solved method").clone();
                match method {
                    ResolvedMethod::Dependent { signature } => TermKind::DependentMethodCall {
                        lookup: self.method_lookup(&signature, name.span)?,
                        receiver: receiver
                            .as_deref()
                            .map(|receiver| self.boxed(receiver))
                            .transpose()?,
                        args: args
                            .iter()
                            .enumerate()
                            .map(|(index, arg)| {
                                self.argument(
                                    arg,
                                    &Type::Node(
                                        super::infer::Head::FunctionParameter { index },
                                        vec![signature.clone()],
                                    ),
                                )
                            })
                            .collect::<Result<_>>()?,
                    },
                    ResolvedMethod::GpuPipeline { method } => {
                        self.source_pipeline(method, receiver.as_deref(), name, args)?
                    }
                    ResolvedMethod::Intrinsic {
                        signature,
                        receiver_conversion,
                    } => {
                        let result = self
                            .solver
                            .require_complete(&signature.result, source.span)?;
                        let kind = self.intrinsic_method(
                            signature,
                            receiver_conversion,
                            receiver.as_deref(),
                            args,
                            source.span,
                        )?;
                        self.convert_method_result(source, result, kind)?
                    }
                    ResolvedMethod::Compiler { declaration } => {
                        let result = types::ty(&declaration.result);
                        let kind = self.method(
                            declaration,
                            receiver.as_deref(),
                            &self.annotation(receiver_type)?.ty,
                            name,
                            args,
                        )?;
                        self.convert_method_result(source, result, kind)?
                    }
                    source => self.source_method_call(&source, receiver.as_deref(), name, args)?,
                }
            }
            typed::TermKind::MethodReference { rule, name } => {
                match self.methods.get(rule).expect("solved method reference") {
                    ResolvedMethod::Source {
                        declaration,
                        type_args,
                        ..
                    } => self.reference(*declaration, name, type_args, true)?,
                    ResolvedMethod::Dependent { signature } => TermKind::DependentMethod {
                        lookup: self.method_lookup(signature, name.span)?,
                    },
                    ResolvedMethod::Compiler { .. }
                    | ResolvedMethod::GpuPipeline { .. }
                    | ResolvedMethod::Intrinsic { .. } => {
                        unreachable!("source method reference")
                    }
                }
            }
            typed::TermKind::Call { func, args } => self.call(func, args)?,
            typed::TermKind::Ascribe { ty, arg } => {
                if let Some(to) = self.solver.resolve(&ty.ty) {
                    self.ascription(source.span, &to, arg)?
                } else {
                    let target = self.solver.require_complete(&ty.ty, ty.span)?;
                    self.symbolic_ascription(&target, arg)?
                }
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
            typed::TermKind::Field { base, name } => TermKind::Field {
                base: self.boxed(base)?,
                name: name.val.clone(),
            },
        })
    }

    fn reference(
        &self,
        declaration: DeclarationId,
        name: &Ident,
        type_args: &[Type],
        read: bool,
    ) -> Result<TermKind> {
        if let Some(&function) = self.function_bindings.get(&declaration) {
            return Ok(TermKind::Function {
                function,
                type_args: type_args
                    .iter()
                    .map(|ty| self.solver.require_complete(ty, name.span))
                    .collect::<Result<_>>()?,
            });
        }
        let kind = match self.initialization.get(&declaration) {
            None => Some(GenerateErrorKind::UnboundValue {
                name: name.val.clone(),
            }),
            Some(Initialization::Initializing) => Some(GenerateErrorKind::EagerRecursion {
                name: name.val.clone(),
            }),
            Some(Initialization::Uninitialized) if read && self.reachable => {
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
        if constant_place(source) {
            return Err(GenerateError::inference(
                source.span,
                "a constant has no mutable storage or address",
            ));
        }
        if let typed::TermKind::Var {
            declaration,
            name,
            type_args,
        } = &source.kind
            && !matches!(
                self.solver.head(&source.actual),
                Type::Node(super::infer::Head::Reference, _)
            )
        {
            return Ok(Box::new(Term {
                span: source.span,
                ty: self.solver.require_complete(&source.ty, source.span)?,
                kind: self.reference(*declaration, name, type_args, false)?,
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
        let reachable = self.reachable;
        let then = self.boxed(then)?;
        let then_reachable = self.reachable;
        let after_then = std::mem::replace(&mut self.initialization, before);
        self.reachable = reachable;
        let els = self.boxed(els)?;
        if then_reachable {
            if self.reachable {
                self.intersect_initialization(&after_then);
            } else {
                self.initialization = after_then;
            }
        }
        self.reachable |= then_reachable;
        Ok(TermKind::If { cond, then, els })
    }

    fn while_expression(&mut self, cond: &typed::Term, body: &typed::Term) -> Result<TermKind> {
        let cond = self.boxed(cond)?;
        let after_condition = self.initialization.clone();
        let reachable = self.reachable;
        let body = self.boxed(body)?;
        self.initialization = after_condition;
        self.reachable = reachable;
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
        if let Some(ty) = self.solver.resolve(&source.actual) {
            super::eval::number(self.typer, source.span, text, Some(&ty))?;
        }
        Ok(TermKind::Numeric { text: text.into() })
    }

    fn layout(&self, ty: &typed::Annotation<Type>, size: bool) -> Result<TermKind> {
        Ok(TermKind::Layout {
            of: self.solver.require_complete(&ty.ty, ty.span)?,
            size,
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
        let reachable = self.reachable;
        let right = self.boxed(&args[1])?;
        if self.reachable {
            self.intersect_initialization(&before_right);
        } else {
            self.initialization = before_right;
        }
        self.reachable = reachable;
        let (then, els) = if name == "&&" {
            (right, fixed)
        } else {
            (fixed, right)
        };
        Ok(TermKind::If { cond, then, els })
    }

    fn method_lookup(&self, signature: &Type, span: Span) -> Result<crate::MethodLookup> {
        let crate::Type::Method { lookup } = self.solver.require_complete(signature, span)? else {
            unreachable!("dependent method signature");
        };
        Ok(*lookup)
    }

    fn source_method_call(
        &mut self,
        method: &ResolvedMethod,
        receiver: Option<&typed::Term>,
        name: &Ident,
        arguments: &[typed::Term],
    ) -> Result<TermKind> {
        let ResolvedMethod::Source {
            declaration,
            type_args,
            params,
            result,
            receiver_conversion,
        } = method
        else {
            unreachable!("source method completion");
        };
        let function_type = Type::function(params.clone(), result.clone());
        let func = Box::new(Term {
            span: name.span,
            ty: self.solver.require_complete(&function_type, name.span)?,
            kind: self.reference(*declaration, name, type_args, true)?,
        });
        let receiver = receiver
            .map(|receiver| {
                if matches!(
                    self.solver.head(&params[0]),
                    Type::Node(super::infer::Head::Reference, _)
                ) {
                    self.argument(receiver, &params[0]).map(Box::new)
                } else {
                    Ok(Box::new(Term {
                        span: receiver.span,
                        ty: self.solver.require_complete(&params[0], receiver.span)?,
                        kind: TermKind::Adapt {
                            conversion: receiver_conversion.expect("checked source receiver"),
                            arg: self.boxed(receiver)?,
                        },
                    }))
                }
            })
            .transpose()?;
        let offset = usize::from(receiver.is_some());
        let args = receiver
            .into_iter()
            .map(|term| *term)
            .chain(
                arguments
                    .iter()
                    .zip(&params[offset..])
                    .map(|(arg, param)| self.argument(arg, param))
                    .collect::<Result<Vec<_>>>()?,
            )
            .collect();
        Ok(TermKind::Call { func, args })
    }

    fn intrinsic_method(
        &mut self,
        signature: super::context::IntrinsicMethod,
        conversion: Option<crate::ReceiverConversion>,
        receiver: Option<&typed::Term>,
        arguments: &[typed::Term],
        span: Span,
    ) -> Result<TermKind> {
        let receiver = receiver
            .map(|receiver| {
                Ok::<_, GenerateError>(Term {
                    span: receiver.span,
                    ty: self
                        .solver
                        .require_complete(&signature.params[0], receiver.span)?,
                    kind: TermKind::Adapt {
                        conversion: conversion.expect("checked primitive receiver"),
                        arg: self.boxed(receiver)?,
                    },
                })
            })
            .transpose()?;
        let offset = usize::from(receiver.is_some());
        let values = receiver
            .into_iter()
            .chain(
                arguments
                    .iter()
                    .zip(&signature.params[offset..])
                    .map(|(arg, param)| self.argument(arg, param))
                    .collect::<Result<Vec<_>>>()?,
            )
            .collect();
        let params = signature
            .params
            .iter()
            .map(|ty| self.solver.require_complete(ty, span))
            .collect::<Result<Vec<_>>>()?;
        Ok(TermKind::Intrinsic {
            op: signature.op,
            type_args: vec![],
            args: Arguments { values, params },
        })
    }

    fn source_pipeline(
        &mut self,
        method: super::gpu::PipelineMethod,
        receiver: Option<&typed::Term>,
        _name: &Ident,
        arguments: &[typed::Term],
    ) -> Result<TermKind> {
        if let FunctionBody::GpuPipelineFactory { factory, graphics } = method.body {
            let receiver_type = self
                .solver
                .resolve(&Type::from_hir(&method.params[0]))
                .expect("fixed native GPU receiver");
            return self.pipeline_create(receiver, arguments, &[receiver_type], factory, graphics);
        }
        let receiver_type = self
            .solver
            .resolve(&Type::from_hir(&method.params[0]))
            .expect("fixed native GPU receiver");
        let receiver = receiver
            .map(|value| self.adapt(value, &self.ty(value)?, &receiver_type))
            .transpose()?;
        let values = receiver
            .into_iter()
            .map(|value| *value)
            .chain(
                arguments
                    .iter()
                    .map(|value| self.elaborate(value))
                    .collect::<Result<Vec<_>>>()?,
            )
            .collect();
        let args = Arguments {
            values,
            params: method.params,
        };
        let FunctionBody::GpuPipelineDispatch {
            context,
            allocator,
            record,
        } = method.body
        else {
            unreachable!("completed pipeline bridge")
        };
        Ok(TermKind::GpuPipelineDispatch {
            context,
            allocator,
            record,
            args,
        })
    }

    fn method(
        &mut self,
        declaration: FunctionDecl,
        receiver: Option<&typed::Term>,
        receiver_ty: &Ty,
        name: &Ident,
        arguments: &[typed::Term],
    ) -> Result<TermKind> {
        if let FunctionBody::GpuPipelineFactory { factory, graphics } = declaration.body {
            return self.pipeline_create(
                receiver,
                arguments,
                &declaration.params,
                factory,
                graphics,
            );
        }
        let receiver = receiver
            .map(|r| self.adapt(r, receiver_ty, &declaration.params[0]))
            .transpose()?;
        let offset = usize::from(receiver.is_some());
        let values = receiver
            .into_iter()
            .map(|term| *term)
            .chain(
                arguments
                    .iter()
                    .zip(&declaration.params[offset..])
                    .map(|(arg, param)| self.argument(arg, &param.clone().into()))
                    .collect::<Result<Vec<_>>>()?,
            )
            .collect();
        let args = Arguments {
            values,
            params: declaration.params.iter().map(types::ty).collect(),
        };
        Ok(match declaration.body {
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
                let ty = Ty::Function {
                    params: declaration.params.clone(),
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
                TermKind::Call {
                    func,
                    args: args.values,
                }
            }
        })
    }

    fn pipeline_create(
        &mut self,
        receiver: Option<&typed::Term>,
        arguments: &[typed::Term],
        params: &[Ty],
        factory: FunctionId,
        graphics: bool,
    ) -> Result<TermKind> {
        let terms = arguments;
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
        let owner = if let Some(receiver) = receiver {
            *self.adapt(receiver, &self.ty(receiver)?, &params[0])?
        } else {
            self.elaborate(&terms[0])?
        };
        let args = Arguments {
            values: vec![owner],
            params: vec![types::ty(&params[0])],
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

    fn call(&mut self, func: &typed::Term, args: &[typed::Term]) -> Result<TermKind> {
        if matches!(
            self.solver.head(&func.ty),
            Type::Node(super::infer::Head::Function, _)
        ) {
            let Type::Node(_, signature) = self.solver.head(&func.ty) else {
                unreachable!()
            };
            return Ok(TermKind::Call {
                func: self.boxed(func)?,
                args: args
                    .iter()
                    .zip(&signature[1..])
                    .map(|(arg, param)| self.argument(arg, param))
                    .collect::<Result<_>>()?,
            });
        }
        let function_type = self.solver.require_complete(&func.ty, func.span)?;
        let receiver = match &function_type {
            crate::Type::Array { .. } => Some((
                crate::Type::Pointer {
                    pointee: Box::new(function_type.clone()),
                },
                ReceiverConversion::Address,
            )),
            crate::Type::Str => Some((function_type.clone(), ReceiverConversion::Value)),
            _ => None,
        };
        if let Some((ty, conversion)) = receiver {
            let args = Arguments {
                params: vec![
                    ty.clone(),
                    self.solver.require_complete(&args[0].ty, args[0].span)?,
                ],
                values: vec![
                    Term {
                        span: func.span,
                        ty,
                        kind: TermKind::Adapt {
                            conversion,
                            arg: self.boxed(func)?,
                        },
                    },
                    self.elaborate(&args[0])?,
                ],
            };
            return Ok(TermKind::Intrinsic {
                type_args: vec![],
                op: Intrinsic::Index,
                args,
            });
        }
        Ok(TermKind::Call {
            func: self.boxed(func)?,
            args: args
                .iter()
                .enumerate()
                .map(|(index, arg)| {
                    self.argument(
                        arg,
                        &Type::Node(
                            super::infer::Head::FunctionParameter { index },
                            vec![func.ty.clone()],
                        ),
                    )
                })
                .collect::<Result<_>>()?,
        })
    }

    fn statement(&mut self, stmt: &typed::Statement) -> Result<Option<Statement>> {
        Ok(Some(match &stmt.kind {
            typed::StatementKind::Error(error) => return Err(error.clone()),
            typed::StatementKind::CompileTimeDefinition => return Ok(None),
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
                if matches!(
                    self.solver.head(&ty.ty),
                    Type::Node(super::infer::Head::Reference, _)
                ) {
                    return Err(GenerateError::inference(
                        name.span,
                        "reference locals require an initializer",
                    ));
                }
                self.initialization
                    .insert(*binding, Initialization::Uninitialized);
                Statement::Declare {
                    binding: *binding,
                    name: name.clone(),
                    ty: crate::Annotation {
                        ty: self.solver.require_complete(&ty.ty, ty.span)?,
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
        let value_type = self.solver.require_complete(&value.ty, value.span)?;
        let tags = match &value_type {
            crate::Type::Result { .. } => vec![crate::Case::Ok, crate::Case::Err],
            crate::Type::Union { variants } => variants
                .iter()
                .cloned()
                .map(|ty| crate::Case::Type { ty })
                .collect(),
            ty => vec![crate::Case::Type { ty: ty.clone() }],
        };
        let value = self.boxed(value)?;
        let before = self.initialization.clone();
        let reachable = self.reachable;
        let mut after = None;
        let mut seen = vec![];
        let mut checked = vec![];
        for (index, arm) in arms.iter().enumerate() {
            let tag = self.pattern(arm, &value_type)?;
            if arm.wildcard && (index + 1 != arms.len() || seen.len() == tags.len()) {
                return Err(GenerateError::inference(
                    arm.body.span,
                    "wildcard must be the final arm and cover a remaining variant",
                ));
            }
            if !arm.wildcard && (!tags.contains(&tag) || seen.contains(&tag)) {
                return Err(GenerateError::inference(
                    arm.body.span,
                    "unknown or duplicate match variant",
                ));
            }
            seen.push(tag.clone());
            self.initialization = before.clone();
            self.reachable = reachable;
            if let Some(binding) = arm.binding {
                self.initialization
                    .insert(binding, Initialization::Initialized);
            }
            let body = self.elaborate(&arm.body)?;
            if self.reachable {
                if let Some(previous) = &after {
                    self.intersect_initialization(previous);
                }
                after = Some(self.initialization.clone());
            }
            checked.push(MatchArm {
                tag,
                binding: arm.binding,
                body,
            });
        }
        if (!seen.contains(&crate::Case::Wildcard) && tags.len() != seen.len()) || tags.is_empty() {
            return Err(GenerateError::inference(
                span,
                "match must cover every variant exactly once",
            ));
        }
        self.reachable = after.is_some();
        self.initialization = after.unwrap_or(before);
        Ok(TermKind::Match {
            value,
            arms: checked,
        })
    }
}

impl Completion<'_> {
    fn pattern(&self, arm: &typed::MatchArm, ty: &crate::Type) -> Result<crate::Case> {
        if arm.wildcard {
            return Ok(crate::Case::Wildcard);
        }
        match (&arm.variant, ty) {
            (None, crate::Type::Result { .. }) => Ok(if arm.failure {
                crate::Case::Err
            } else {
                crate::Case::Ok
            }),
            (Some(ann), ty) if !matches!(ty, crate::Type::Result { .. }) => {
                let ty = self.solver.require_complete(&ann.ty, ann.span)?;
                if matches!(ty, crate::Type::Union { .. }) {
                    return Err(GenerateError::inference(
                        ann.span,
                        "union patterns must name a single member type",
                    ));
                }
                Ok(crate::Case::Type { ty })
            }
            _ => Err(GenerateError::inference(
                arm.body.span,
                "pattern does not belong to this match type",
            )),
        }
    }
}

impl Completion<'_> {
    fn symbolic_ascription(&mut self, to: &crate::Type, source: &typed::Term) -> Result<TermKind> {
        Ok(TermKind::Convert {
            arg: Box::new(self.nominal_argument(to, source)?),
        })
    }

    fn nominal_argument(&mut self, to: &crate::Type, source: &typed::Term) -> Result<Term> {
        if matches!(to, crate::Type::Defined { .. }) && matches!(source.kind, typed::TermKind::Unit)
        {
            return Ok(Term {
                span: source.span,
                ty: crate::Type::Record { fields: vec![] },
                kind: TermKind::Record { fields: vec![] },
            });
        }
        self.elaborate(source)
    }

    fn ascription(&mut self, span: Span, to: &Ty, source: &typed::Term) -> Result<TermKind> {
        if self.typer.body(to).is_err() {
            return self.symbolic_ascription(&types::ty(to), source);
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
                ty: types::ty(&body),
                kind: TermKind::Record { fields: vec![] },
            });
        }
        self.elaborate(source)
    }

    fn conversion(&self, span: Span, value: Term, to: &Ty) -> Result<TermKind> {
        if let Some(from) = self.solver.resolve(&Type::from_hir(&value.ty)) {
            self.typer
                .explicit_conversion(&from, to)
                .map_err(|e| GenerateError::typing(span, e))?;
        }
        Ok(TermKind::Convert {
            arg: Box::new(value),
        })
    }
}

fn constant_place(term: &typed::Term) -> bool {
    match &term.kind {
        typed::TermKind::Constant { .. } => true,
        typed::TermKind::Field { base, .. } => constant_place(base),
        _ => false,
    }
}

// A reference value already carries a location. Reading its referent preserves
// a place until storage lowering reaches an actual value consumer.
fn reference_place(term: &Term) -> bool {
    if matches!(
        term.ty,
        crate::Type::Reference { .. }
            | crate::Type::Value { .. }
            | crate::Type::FunctionResult { .. }
    ) {
        return true;
    }
    match &term.kind {
        TermKind::Local { .. } | TermKind::Deref { .. } => true,
        TermKind::Use { arg } => reference_place(arg),
        TermKind::Field { base, .. } => {
            reference_place(base) || matches!(base.ty, crate::Type::Pointer { .. })
        }
        _ => false,
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
            let mut context = Context::new();
            let mut inference = Inference::new(&mut context);
            let (rule, ty) = inference.expression();
            inference.constrain(
                rule,
                (
                    span,
                    Constraint::Method {
                        receiver: Ty::Str.into(),
                        name: "at".into(),
                        type_args: None,
                        args: vec![Ty::UInt64.into()],
                        out: ty.clone(),
                        associated: false,
                    },
                ),
            );
            assert!(inference.solve(std::slice::from_ref(&ty)).is_empty());
            (inference.solver, inference.methods, rule, ty)
        };
        let source = typed::Term {
            span,
            actual: ty.clone(),
            ty,
            kind: typed::TermKind::MethodCall {
                rule,
                receiver: Some(Box::new(typed::Term {
                    span,
                    ty: Ty::Str.into(),
                    actual: Ty::Str.into(),
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
                args: vec![typed::Term {
                    span,
                    ty: Ty::UInt64.into(),
                    actual: Ty::UInt64.into(),
                    kind: typed::TermKind::Num { value: "0".into() },
                }],
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
            crate::Type::Reference {
                referent: Box::new(crate::Type::UInt8)
            }
        );
        assert!(completed.shaders.is_empty());
    }
}
