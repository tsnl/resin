//! Complete solved expressions directly into public HIR.
//! This is the last part of HIR construction; no concrete private tree is retained.
use super::infer::{ResolvedMethod, Rule, Solver, Type};
use super::scope::DeclarationId;
use super::{typed, types};
use crate::ReceiverConversion;
use crate::lower::context::FunctionBody;
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
    parameters: impl IntoIterator<Item = (DeclarationId, bool)>,
    solver: &Solver,
    methods: &BTreeMap<Rule, ResolvedMethod>,
    typer: &super::context::Context,
    function_bindings: &HashMap<DeclarationId, FunctionId>,
    shaders: &BTreeMap<FunctionId, ShaderEntry>,
) -> Result<CompletedBody> {
    let mutable: BTreeMap<_, _> = parameters.into_iter().collect();
    let mut completion = Completion {
        solver,
        methods,
        typer,
        function_bindings,
        shaders,
        reachable: true,
        loops: Vec::new(),
        embedded: BTreeSet::new(),
        initialization: mutable
            .keys()
            .map(|id| (*id, Initialization::Initialized))
            .collect(),
        written: mutable.keys().copied().collect(),
        mutable,
        moved: BTreeMap::new(),
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

#[derive(Default)]
struct LoopStates {
    breaks: Vec<OwnershipState>,
    continues: Vec<OwnershipState>,
}

struct Completion<'a> {
    loops: Vec<LoopStates>,
    reachable: bool,
    solver: &'a Solver,
    methods: &'a BTreeMap<Rule, ResolvedMethod>,
    typer: &'a super::context::Context,
    function_bindings: &'a HashMap<DeclarationId, FunctionId>,
    shaders: &'a BTreeMap<FunctionId, ShaderEntry>,
    embedded: BTreeSet<FunctionId>,
    initialization: BTreeMap<DeclarationId, Initialization>,
    mutable: BTreeMap<DeclarationId, bool>,
    written: BTreeSet<DeclarationId>,
    moved: BTreeMap<DeclarationId, BTreeSet<Vec<std::sync::Arc<str>>>>,
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
        let target = self.solver.require_complete(&source.ty, source.span)?;
        self.elaborate_as(source, target)
    }

    fn elaborate_as(&mut self, source: &typed::Term, target: crate::Type) -> Result<Term> {
        let term = Term {
            span: source.span,
            ty: self.solver.require_complete(&source.actual, source.span)?,
            kind: self.elaborate_kind(source)?,
        };
        self.consume(term, target)
    }

    fn consume(&mut self, mut term: Term, target: crate::Type) -> Result<Term> {
        self.require_available(&term)?;
        let value_type = match &term.ty {
            crate::Type::Reference { referent, .. } => referent.as_ref(),
            ty => ty,
        }
        .clone();
        if let crate::Type::Reference { mutable, .. } = target {
            require_reference_place(&term)?;
            if mutable {
                require_mutable_place(&term)?;
            }
        } else if let copyability = self.typer.copyability(&value_type)
            && copyability != Some(true)
        {
            let span = term.span;
            let ty = term.ty.clone();
            if !reference_place(&term) {
                self.require_movable(&term)?;
            } else if let Some((binding, path)) = owned_path(&term) {
                self.require_movable(&term)?;
                self.moved.entry(binding).or_default().insert(path);
                term = Term {
                    span,
                    ty,
                    kind: TermKind::Move {
                        place: Box::new(term),
                    },
                };
            } else if copyability.is_none()
                || (self.solver.resolve(&Type::from_hir(&target)).is_none()
                    && !matches!(target, crate::Type::Defined { .. }))
            {
                term = Term {
                    span,
                    ty,
                    kind: TermKind::Read {
                        place: Box::new(term),
                    },
                };
            } else {
                return Err(GenerateError::inference(
                    span,
                    "cannot move a value through a reference or pointer; replace its contents instead",
                ));
            }
        }
        if term.ty == target {
            return Ok(term);
        }
        Ok(Term {
            span: term.span,
            ty: target,
            kind: TermKind::Use {
                arg: Box::new(term),
            },
        })
    }

    fn require_movable(&self, term: &Term) -> Result<()> {
        if let TermKind::Field { base, .. } = &term.kind {
            if let crate::Type::Defined { definition, .. } = base.ty
                && self
                    .typer
                    .definition(definition)
                    .ok()
                    .is_some_and(|definition| definition.drop_hook().is_some())
            {
                return Err(GenerateError::inference(
                    term.span,
                    "cannot move a field out of a type with a drop hook",
                ));
            }
            self.require_movable(base)?;
        }
        Ok(())
    }

    fn require_available(&self, term: &Term) -> Result<()> {
        if !self.reachable {
            return Ok(());
        }
        let Some((binding, path)) = owned_path(term) else {
            return Ok(());
        };
        if self.initialization.get(&binding) != Some(&Initialization::Initialized) {
            return Err(GenerateError::inference(
                term.span,
                "use of an uninitialized value",
            ));
        }
        if self.moved.get(&binding).is_some_and(|moved| {
            moved
                .iter()
                .any(|other| path.starts_with(other) || other.starts_with(&path))
        }) {
            return Err(GenerateError::inference(term.span, "use of a moved value"));
        }
        Ok(())
    }

    fn argument(&mut self, source: &typed::Term, target: &Type) -> Result<Term> {
        let target = self.solver.require_complete(target, source.span)?;
        let term = Term {
            span: source.span,
            ty: self.solver.require_complete(&source.actual, source.span)?,
            kind: self.elaborate_kind(source)?,
        };
        self.consume(term, target)
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
            typed::TermKind::Break | typed::TermKind::Continue => {
                if self.reachable {
                    let state = self.state();
                    let exits = self.loops.last_mut().expect("checked loop exit");
                    if matches!(source.kind, typed::TermKind::Break) {
                        exits.breaks.push(state);
                    } else {
                        exits.continues.push(state);
                    }
                }
                self.reachable = false;
                if matches!(source.kind, typed::TermKind::Break) {
                    TermKind::Break
                } else {
                    TermKind::Continue
                }
            }
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
                Some(ResolvedMethod::Operation { signature }) => {
                    self.operation_call(&signature, args, source.span)?
                }
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
                Some(ResolvedMethod::Intrinsic { signature, .. }) => {
                    let result = self
                        .solver
                        .require_complete(&signature.result, source.span)?;
                    let kind = self.intrinsic_method(signature, None, None, args, source.span)?;
                    self.convert_method_result(source, result, kind)?
                }
                Some(ResolvedMethod::GpuPipeline { method }) => {
                    self.source_pipeline(method, None, &Ident::new(name.clone(), *name_span), args)?
                }
                None => self.builtin(source, name, args)?,
            },
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
                    ResolvedMethod::Operation { .. }
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
            typed::TermKind::Absurd { arg } => TermKind::Absurd {
                arg: self.boxed(arg)?,
            },
            typed::TermKind::Assign { place, value } => self.assign(place, value)?,
            typed::TermKind::Address { place } => {
                let place = self.place(place)?;
                if reference_borrow(&place) {
                    return Err(GenerateError::inference(
                        source.span,
                        "cannot take the address of a Ref; accept or return a Ptr when an address is required",
                    ));
                }
                if local_address(&place) {
                    return Err(GenerateError::inference(
                        source.span,
                        "cannot take the address of a local value; borrow it with Ref or use explicitly allocated storage",
                    ));
                }
                TermKind::Address { place }
            }
            typed::TermKind::Deref { pointer } => TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            typed::TermKind::Field { base, name } => TermKind::Field {
                base: if constant_place(base) {
                    self.boxed(base)?
                } else {
                    self.place(base)?
                },
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
            mutable: self.mutable.get(&declaration).copied().unwrap_or(false),
        })
    }

    fn place(&mut self, source: &typed::Term) -> Result<Box<Term>> {
        if constant_place(source) {
            return Err(GenerateError::inference(
                source.span,
                "a constant has no mutable storage or address",
            ));
        }
        let kind = match &source.kind {
            typed::TermKind::Var {
                declaration,
                name,
                type_args,
            } => self.reference(*declaration, name, type_args, false)?,
            typed::TermKind::Field { base, name } => TermKind::Field {
                base: self.place(base)?,
                name: name.val.clone(),
            },
            typed::TermKind::Deref { pointer } => TermKind::Deref {
                pointer: self.boxed(pointer)?,
            },
            // Accessing a field or assigning through a returned reference must
            // preserve its place; consuming it here would read the whole value.
            _ => self.elaborate_kind(source)?,
        };
        let place = Term {
            span: source.span,
            ty: self.solver.require_complete(&source.actual, source.span)?,
            kind,
        };
        Ok(Box::new(
            if let crate::Type::Reference { referent, .. } = &place.ty {
                Term {
                    span: source.span,
                    ty: *referent.clone(),
                    kind: TermKind::Use {
                        arg: Box::new(place),
                    },
                }
            } else {
                place
            },
        ))
    }

    fn assign(&mut self, place: &typed::Term, value: &typed::Term) -> Result<TermKind> {
        let place = self.place(place)?;
        let destination = owned_path(&place);
        if destination.is_none() {
            if !reference_place(&place) {
                return Err(GenerateError {
                    span: place.span,
                    kind: GenerateErrorKind::NotAPlace,
                });
            }
            require_mutable_place(&place)?;
        }
        if let Some((binding, path)) = &destination {
            if !self.mutable.get(binding).copied().unwrap_or(false)
                && (self.written.contains(binding) || !path.is_empty())
            {
                return Err(GenerateError::inference(
                    place.span,
                    "cannot assign to an immutable binding; declare it with `mut`",
                ));
            }
            if self.moved.get(binding).is_some_and(|moved| {
                moved
                    .iter()
                    .any(|ancestor| path.starts_with(ancestor) && path.len() > ancestor.len())
            }) {
                return Err(GenerateError::inference(
                    place.span,
                    "initialize the moved value before assigning one of its fields",
                ));
            }
            if !path.is_empty()
                && self.initialization.get(binding) != Some(&Initialization::Initialized)
            {
                return Err(GenerateError::inference(
                    place.span,
                    "cannot assign a field of an uninitialized value",
                ));
            }
        }
        let value = self.boxed(value)?;
        if let Some((binding, path)) = destination {
            self.written.insert(binding);
            self.initialization
                .insert(binding, Initialization::Initialized);
            self.moved
                .entry(binding)
                .or_default()
                .retain(|moved| !moved.starts_with(&path));
        }
        Ok(TermKind::Assign { place, value })
    }

    fn state(&self) -> OwnershipState {
        OwnershipState {
            initialization: self.initialization.clone(),
            written: self.written.clone(),
            moved: self.moved.clone(),
        }
    }

    fn restore(&mut self, state: OwnershipState) {
        self.initialization = state.initialization;
        self.written = state.written;
        self.moved = state.moved;
    }

    fn intersect(&mut self, other: &OwnershipState) {
        for (binding, state) in &mut self.initialization {
            if other.initialization.get(binding) != Some(state) {
                *state = Initialization::Uninitialized;
            }
        }
        self.written.extend(&other.written);
        for (binding, moved) in &other.moved {
            self.moved
                .entry(*binding)
                .or_default()
                .extend(moved.iter().cloned());
        }
    }

    fn if_expression(
        &mut self,
        cond: &typed::Term,
        then: &typed::Term,
        els: &typed::Term,
    ) -> Result<TermKind> {
        let cond = self.boxed(cond)?;
        let before = self.state();
        let reachable = self.reachable;
        let then = self.boxed(then)?;
        let then_reachable = self.reachable;
        let after_then = {
            let state = self.state();
            self.restore(before);
            state
        };
        self.reachable = reachable;
        let els = self.boxed(els)?;
        if then_reachable {
            if self.reachable {
                self.intersect(&after_then);
            } else {
                self.restore(after_then);
            }
        }
        self.reachable |= then_reachable;
        Ok(TermKind::If { cond, then, els })
    }

    fn while_expression(&mut self, cond: &typed::Term, body: &typed::Term) -> Result<TermKind> {
        let entry = self.state();
        let reachable = self.reachable;
        loop {
            let before = self.state();
            self.reachable = reachable;
            self.loops.push(LoopStates::default());
            let condition = self.boxed(cond)?;
            let exit = self.state();
            let condition_reachable = self.reachable;
            let checked_body = self.boxed(body)?;
            let mut exits = self.loops.pop().expect("current loop");
            if self.reachable {
                exits.continues.push(self.state());
            }
            self.restore(entry.clone());
            for backedge in &exits.continues {
                self.intersect(backedge);
            }
            self.restrict_state(&entry);
            if self.state() == before {
                self.restore(exit);
                for exit in &exits.breaks {
                    self.intersect(exit);
                }
                self.restrict_state(&entry);
                self.reachable = condition_reachable || !exits.breaks.is_empty();
                return Ok(TermKind::While {
                    cond: condition,
                    body: checked_body,
                });
            }
        }
    }

    fn restrict_state(&mut self, entry: &OwnershipState) {
        self.initialization
            .retain(|binding, _| entry.initialization.contains_key(binding));
        self.written
            .retain(|binding| entry.initialization.contains_key(binding));
        self.moved
            .retain(|binding, _| entry.initialization.contains_key(binding));
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
        let before_right = self.state();
        let reachable = self.reachable;
        let right = self.boxed(&args[1])?;
        if self.reachable {
            self.intersect(&before_right);
        } else {
            self.restore(before_right);
        }
        self.reachable = reachable;
        let (then, els) = if name == "&&" {
            (right, fixed)
        } else {
            (fixed, right)
        };
        Ok(TermKind::If { cond, then, els })
    }

    fn operation_call(
        &mut self,
        signature: &Type,
        args: &[typed::Term],
        span: Span,
    ) -> Result<TermKind> {
        let crate::Type::Operation { lookup } = self.solver.require_complete(signature, span)?
        else {
            unreachable!("operation query");
        };
        let args = args
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
            .collect::<Result<_>>()?;
        Ok(TermKind::OperationCall {
            lookup: *lookup,
            args,
        })
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
                    Type::Node(super::infer::Head::Reference { .. }, _)
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
        let sources = receiver
            .into_iter()
            .chain(arguments.iter())
            .collect::<Vec<_>>();
        let values = sources
            .iter()
            .zip(&method.params)
            .map(|(source, parameter)| self.argument(source, &Type::from_hir(parameter)))
            .collect::<Result<Vec<_>>>()?;
        if let FunctionBody::GpuPipelineFactory { factory, graphics } = method.body {
            let stages = if graphics {
                &["vertex", "fragment"][..]
            } else {
                &["compute"][..]
            };
            let mut shaders = Vec::new();
            for (term, stage) in values[1..].iter().zip(stages) {
                let TermKind::Function { function, .. } = term.kind else {
                    return Err(GenerateError::inference(
                        term.span,
                        "pipeline creation requires direct shader declarations; runtime aliases are unsupported",
                    ));
                };
                let entry = self.shaders.get(&function).ok_or_else(|| {
                    GenerateError::inference(
                        term.span,
                        "pipeline creation requires a decorated shader declaration",
                    )
                })?;
                if entry.stage.as_ref() != *stage {
                    return Err(GenerateError::inference(
                        term.span,
                        format!("pipeline requires a @{stage}_shader declaration"),
                    ));
                }
                self.embedded.insert(function);
                shaders.push(function);
            }
            return Ok(TermKind::GpuPipelineCreate {
                factory,
                shaders,
                args: Arguments {
                    values: vec![values.into_iter().next().unwrap()],
                    params: vec![method.params[0].clone()],
                },
            });
        }
        let FunctionBody::GpuPipelineDispatch {
            context,
            allocator,
            record,
        } = method.body
        else {
            unreachable!("completed pipeline bridge");
        };
        Ok(TermKind::GpuPipelineDispatch {
            context,
            allocator,
            record,
            args: Arguments {
                values,
                params: method.params,
            },
        })
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
                crate::Type::Reference {
                    mutable: false,
                    referent: Box::new(function_type.clone()),
                },
                ReceiverConversion::Borrow,
            )),
            crate::Type::Str => Some((function_type.clone(), ReceiverConversion::Value)),
            _ => None,
        };
        if let Some((ty, conversion)) = receiver {
            let base = if conversion == ReceiverConversion::Borrow {
                let place = self.place(func)?;
                self.require_available(&place)?;
                require_reference_place(&place)?;
                place
            } else {
                self.boxed(func)?
            };
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
                            arg: base,
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
                mutable,
                binding,
                name,
                init,
            } => {
                let binding = binding.expect("checked declaration");
                self.mutable.insert(binding, *mutable);
                self.moved.remove(&binding);
                self.written.insert(binding);
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
            typed::StatementKind::Declare {
                mutable,
                binding,
                name,
                ty,
            } => {
                self.mutable.insert(*binding, *mutable);
                self.moved.remove(binding);
                self.written.remove(binding);
                if matches!(
                    self.solver.head(&ty.ty),
                    Type::Node(super::infer::Head::Reference { .. }, _)
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
            crate::Type::Union { variants } => variants
                .iter()
                .cloned()
                .map(|ty| crate::Case::Type { ty })
                .collect(),
            ty => vec![crate::Case::Type { ty: ty.clone() }],
        };
        let value = self.boxed(value)?;
        let before = self.state();
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
            let matched: Vec<_> = if matches!(tag, crate::Case::Error { .. }) {
                tags.iter()
                    .filter(|tag| {
                        matches!(
                            tag,
                            crate::Case::Type {
                                ty: crate::Type::Error { .. }
                            }
                        )
                    })
                    .cloned()
                    .collect()
            } else {
                vec![tag.clone()]
            };
            if !arm.wildcard
                && (matched.is_empty()
                    || matched
                        .iter()
                        .any(|tag| !tags.contains(tag) || seen.contains(tag)))
            {
                return Err(GenerateError::inference(
                    arm.body.span,
                    "unknown or duplicate match variant",
                ));
            }
            seen.extend(matched);
            self.restore(before.clone());
            self.reachable = reachable;
            if let Some(binding) = arm.binding {
                self.mutable.insert(binding, arm.mutable);
                self.moved.remove(&binding);
                self.written.insert(binding);
                self.initialization
                    .insert(binding, Initialization::Initialized);
            }
            let body = self.elaborate(&arm.body)?;
            if self.reachable {
                if let Some(previous) = &after {
                    self.intersect(previous);
                }
                after = Some(self.state());
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
        self.restore(after.unwrap_or(before));
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
        if arm.error {
            let members = match ty {
                crate::Type::Union { variants } => variants.clone(),
                ty => vec![ty.clone()],
            };
            let payloads = members
                .into_iter()
                .filter_map(|ty| match ty {
                    crate::Type::Error { payload } => Some(Type::from_hir(&payload)),
                    _ => None,
                })
                .collect();
            let payload = self
                .solver
                .require_complete(&self.solver.union(payloads), arm.body.span)?;
            return Ok(crate::Case::Error { payload });
        }

        match (&arm.variant, ty) {
            (Some(ann), _) => {
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

fn require_reference_place(term: &Term) -> Result<()> {
    if reference_place(term) {
        Ok(())
    } else {
        Err(GenerateError::inference(
            term.span,
            "reference binding requires an initialized place; bind the temporary to a local first",
        ))
    }
}

fn require_mutable_place(term: &Term) -> Result<()> {
    if mutable_place(term) {
        Ok(())
    } else {
        Err(GenerateError::inference(
            term.span,
            "writable access requires RefMut or a mutable place; declare local storage with `let mut`",
        ))
    }
}

fn mutable_place(term: &Term) -> bool {
    if let crate::Type::Reference { mutable, .. } = &term.ty {
        return *mutable;
    }
    if matches!(
        term.ty,
        crate::Type::Parameter { .. }
            | crate::Type::Member { .. }
            | crate::Type::Operation { .. }
            | crate::Type::Method { .. }
            | crate::Type::FunctionParameter { .. }
            | crate::Type::FunctionResult { .. }
            | crate::Type::Value { .. }
    ) {
        return true;
    }
    match &term.kind {
        TermKind::Local { mutable, .. } => *mutable,
        TermKind::Use { arg } => mutable_place(arg),
        TermKind::Deref { pointer } => mutable_access(&pointer.ty).unwrap_or(false),
        TermKind::Field { base, .. } => {
            mutable_access(&base.ty).unwrap_or_else(|| mutable_place(base))
        }
        _ => false,
    }
}

// Loading a pointer field starts access under that pointer's own contract.
fn mutable_access(ty: &crate::Type) -> Option<bool> {
    match ty {
        crate::Type::Pointer { pointee } => Some(mutable_access(pointee).unwrap_or(true)),
        crate::Type::Reference { referent, mutable } => {
            Some(mutable_access(referent).unwrap_or(*mutable))
        }
        _ => None,
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

// A dependent field receiver might specialize to a pointer. Its address contract
// is checked again during specialization, before references lose their distinction.
fn known_value_receiver(ty: &crate::Type) -> bool {
    use crate::Type;
    !matches!(
        ty,
        Type::Pointer { .. }
            | Type::Parameter { .. }
            | Type::Member { .. }
            | Type::Operation { .. }
            | Type::Method { .. }
            | Type::FunctionParameter { .. }
            | Type::FunctionResult { .. }
            | Type::Value { .. }
    )
}

fn local_address(term: &Term) -> bool {
    match &term.kind {
        TermKind::Local { .. } => !matches!(term.ty, crate::Type::Reference { .. }),
        TermKind::Use { arg } => local_address(arg),
        TermKind::Field { base, .. } => known_value_receiver(&base.ty) && local_address(base),
        _ => false,
    }
}

// A reference grants access to its referent, but not its address. Dereferencing
// a pointer read through a reference starts a new, independently addressable place.
fn reference_borrow(term: &Term) -> bool {
    if matches!(term.ty, crate::Type::Reference { .. }) {
        return true;
    }
    match &term.kind {
        TermKind::Use { arg } => reference_borrow(arg),
        TermKind::Field { base, .. } => {
            let ty = match &base.ty {
                crate::Type::Reference { referent, .. } => referent.as_ref(),
                ty => ty,
            };
            known_value_receiver(ty) && reference_borrow(base)
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
    fn completion_uses_the_selected_operation_after_lookup_is_gone() {
        let span = Span { start: 0, end: 0 };
        let (solver, methods, rule, ty) = {
            let mut context = Context::new();
            let mut inference = Inference::new(&mut context);
            let (rule, ty) = inference.expression();
            inference.constrain(
                rule,
                (
                    span,
                    Constraint::Overload {
                        lookup: super::super::infer::Overload {
                            name: "at".into(),
                            candidates: vec![],
                            primitive: Some("at".into()),
                            expected: None,
                            type_args: None,
                            args: Some(vec![Ty::Str.into(), Ty::UInt64.into()]),
                            out: ty.clone(),
                        },
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
            kind: typed::TermKind::Builtin {
                rule,
                // Completion consumes the chosen operation, never its spelling.
                name: "no_such_operation".into(),
                name_span: span,
                args: vec![
                    typed::Term {
                        span,
                        ty: Ty::Str.into(),
                        actual: Ty::Str.into(),
                        kind: typed::TermKind::String {
                            value: "bytes".into(),
                        },
                    },
                    typed::Term {
                        span,
                        ty: Ty::UInt64.into(),
                        actual: Ty::UInt64.into(),
                        kind: typed::TermKind::Num { value: "0".into() },
                    },
                ],
            },
        };
        let completed = function(
            &source,
            [],
            &solver,
            &methods,
            &super::super::context::Context::new(),
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
                mutable: false,
                referent: Box::new(crate::Type::UInt8)
            }
        );
        assert!(completed.shaders.is_empty());
    }
}

#[derive(Clone, PartialEq, Eq)]
struct OwnershipState {
    initialization: BTreeMap<DeclarationId, Initialization>,
    written: BTreeSet<DeclarationId>,
    moved: BTreeMap<DeclarationId, BTreeSet<Vec<std::sync::Arc<str>>>>,
}

fn owned_path(term: &Term) -> Option<(DeclarationId, Vec<std::sync::Arc<str>>)> {
    if matches!(term.ty, crate::Type::Reference { .. }) {
        return None;
    }
    match &term.kind {
        TermKind::Local { binding, .. } => Some((*binding, vec![])),
        TermKind::Field { base, name }
            if !matches!(
                base.ty,
                crate::Type::Pointer { .. } | crate::Type::Reference { .. }
            ) =>
        {
            let (binding, mut path) = owned_path(base)?;
            path.push(name.clone());
            Some((binding, path))
        }
        _ => None,
    }
}
