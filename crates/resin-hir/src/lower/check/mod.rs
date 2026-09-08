//! First expression pass: resolve declarations and types, producing a typed tree.
//! No IR instructions or deferred code-generation operations are created here.
use crate::DefinitionKind;
use crate::lower::{
    context::Context,
    context::SourceModuleId,
    infer::{
        Constraint, Equation, Head, Inference, Pattern, Rule, Solver, Type, VariableId,
        check_binding_name,
    },
    scope::{ContextView, Cursor, DeclarationId, Scopes},
    typed::{self, StatementKind, TermKind},
};
use crate::{GenerateError, GenerateErrorKind};
use resin_ast::{SourceFile, StmtKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

//
// Checking state and signatures
//

type Result<T> = std::result::Result<T, GenerateError>;
type Term = typed::Term<Type>;
type Statement = typed::Statement<Type>;
type MatchArm = typed::MatchArm<Type>;
pub(super) struct Annotation {
    holes: Vec<(Span, VariableId)>,
    pub ty: Type,
    pub span: Span,
}
impl Annotation {
    fn into_tree(self) -> typed::Annotation<Type> {
        typed::Annotation {
            ty: self.ty,
            span: self.span,
        }
    }
}
pub(super) struct Signature {
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
impl Checker<'_> {
    fn ann(&mut self, ann: &resin_ast::Type, infer: bool) -> Annotation {
        let scoped = !matches!(ann.val, resin_ast::TypeKind::Unit);
        if scoped {
            self.scopes.push_at(ann.span);
        }
        let checkpoint = self.typing.solver.clone();
        let result = super::eval::Decoder {
            solver: &mut self.typing.solver,
            holes: Vec::new(),
            resolve: &mut |name| {
                if name.val.as_ref() == "String" {
                    return Ok(self
                        .typing
                        .typer
                        .string_type()
                        .cloned()
                        .expect("builtin String")
                        .into());
                }
                self.scopes.resolve_type(name)
            },
        }
        .decode(ann, infer);
        if scoped {
            self.scopes.pop();
        }
        let decoded = result.unwrap_or_else(|error| {
            self.errors.push(error);
            self.typing.solver = checkpoint;
            super::eval::Decoded {
                ty: Type::Invalid,
                holes: Vec::new(),
            }
        });
        self.holes.extend_from_slice(&decoded.holes);
        Annotation {
            ty: decoded.ty,
            holes: decoded.holes,
            span: ann.span,
        }
    }

    fn signature(
        &mut self,
        params: &[(Ident, resin_ast::Type)],
        result: &resin_ast::Type,
        infer: bool,
    ) -> Signature {
        let params = params
            .iter()
            .map(|(name, ann)| (name.clone(), self.ann(ann, false)))
            .collect();
        let result = self.ann(result, infer);
        Signature {
            params,
            result,
            declaration: None,
            parameters: vec![],
        }
    }
}

pub(super) struct CheckedFile {
    pub context: ContextView,
    pub signatures: BTreeMap<Arc<str>, typed::Signature>,
    pub bodies: BTreeMap<Arc<str>, typed::Term>,
    pub errors: Vec<GenerateError>,
}
struct Checker<'a> {
    typing: Inference<'a>,
    scopes: Scopes,
    errors: Vec<GenerateError>,
    dependencies: BTreeSet<Arc<str>>,
    holes: Vec<(Span, VariableId)>,
    expressions: Vec<(Span, Type)>,
    result: Type,
    source_module: SourceModuleId,
}
impl<'a> Checker<'a> {
    fn new(typer: &'a mut Context, scopes: Scopes, source_module: SourceModuleId) -> Self {
        Self {
            typing: Inference::new(typer),
            scopes,
            errors: vec![],
            dependencies: BTreeSet::new(),
            holes: vec![],
            expressions: vec![],
            result: Ty::Unit.into(),
            source_module,
        }
    }
}

//
// Local name lookup and bindings
//

impl Checker<'_> {
    pub fn bind(&mut self, name: &Ident, ty: Type, kind: DefinitionKind) -> Result<DeclarationId> {
        check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, kind)
            .map_err(|duplicate| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: duplicate },
            })
    }

    pub fn value(&mut self, name: &Ident) -> Result<Type> {
        self.scopes
            .lookup_inferred(&name.val)
            .map(|(ty, function)| {
                if function {
                    self.dependencies.insert(name.val.clone());
                }
                ty
            })
            .ok_or_else(|| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::UnboundValue {
                    name: name.val.clone(),
                },
            })
    }
}

//
// Declarations, bodies, and dependency solving
//

type Signatures = BTreeMap<Arc<str>, Signature>;
type SourceBodies<'a> = Vec<(&'a Ident, &'a resin_ast::Term)>;

struct Body {
    name: Arc<str>,
    term: Term,
    dependencies: BTreeSet<Arc<str>>,
    constraints: Vec<Equation>,
    expressions: Vec<(Span, Type)>,
}

pub(in crate::lower) fn file(
    file: &SourceFile,
    typer: &mut Context,
    scopes: Scopes,
    source_module: SourceModuleId,
    methods: BTreeMap<Arc<str>, DeclarationId>,
) -> CheckedFile {
    let mut checker = Checker::new(typer, scopes, source_module);
    let (mut signatures, sources) = checker.declarations(file, methods);
    let mut bodies = checker.bodies(&mut signatures, sources);
    checker.solve_functions(&signatures, &mut bodies);
    checker.require_holes();
    checker.finish(signatures, bodies)
}

impl Checker<'_> {
    fn declarations<'s>(
        &mut self,
        file: &'s SourceFile,
        mut methods: BTreeMap<Arc<str>, DeclarationId>,
    ) -> (Signatures, SourceBodies<'s>) {
        let mut signatures = BTreeMap::new();
        let mut bodies = vec![];
        for stmt in file.declarations() {
            if let Some((name, body, signature)) = self.declaration(&stmt.val, &mut methods) {
                if let Some(body) = body {
                    bodies.push((name, body));
                }
                signatures.insert(name.val.clone(), signature);
            }
        }
        (signatures, bodies)
    }

    fn declaration<'s>(
        &mut self,
        stmt: &'s StmtKind,
        methods: &mut BTreeMap<Arc<str>, DeclarationId>,
    ) -> Option<(&'s Ident, Option<&'s resin_ast::Term>, Signature)> {
        let (name, params, result, body) = match stmt {
            StmtKind::Function {
                name,
                params,
                result,
                body,
                ..
            } => (name, params, result, Some(body)),
            StmtKind::ForeignFunction {
                name,
                params,
                result,
                ..
            } => (name, params, result, None),
            _ => return None,
        };
        let mut signature = self.signature(params, result, body.is_some());
        match self.declare(name, signature.ty(), methods.remove(&name.val)) {
            Ok(id) => signature.declaration = Some(id),
            Err(error) => {
                self.errors.push(error);
                return None;
            }
        }
        if is_shader(stmt) {
            self.scopes.mark_shader(name);
        }
        Some((name, body, signature))
    }

    fn declare(
        &mut self,
        name: &Ident,
        ty: Type,
        method: Option<DeclarationId>,
    ) -> Result<DeclarationId> {
        if let Some(id) = method {
            self.scopes.set_inferred(id, ty);
            return Ok(id);
        }
        crate::lower::infer::check_binding_name(name)?;
        self.scopes
            .define_inferred(name, ty, DefinitionKind::Function)
            .map_err(|name_| GenerateError {
                span: name.span,
                kind: GenerateErrorKind::DuplicateValue { name: name_ },
            })
    }

    fn bodies(&mut self, signatures: &mut Signatures, sources: SourceBodies<'_>) -> Vec<Body> {
        sources
            .into_iter()
            .map(|(name, body)| {
                let term = self.function_body(body, signatures.get_mut(&name.val).unwrap());
                Body {
                    name: name.val.clone(),
                    term,
                    dependencies: std::mem::take(&mut self.dependencies),
                    constraints: std::mem::take(&mut self.typing.constraints),
                    expressions: std::mem::take(&mut self.expressions),
                }
            })
            .collect()
    }

    fn function_body(&mut self, source: &resin_ast::Term, signature: &mut Signature) -> Term {
        let errors_before = self.errors.len();
        self.scopes.push_at(source.span);
        self.result = signature.result.ty.clone();
        self.bind_parameters(signature);
        let (rule, term) = self.term(source, Some(signature.result.ty.clone()));
        self.scopes.pop();
        if self.errors.len() != errors_before {
            self.typing.fail(rule);
        }
        self.typing.infer_from(
            &signature.result.holes,
            signature.result.span,
            term.ty.clone(),
        );
        term
    }

    fn bind_parameters(&mut self, signature: &mut Signature) {
        for (name, ann) in &signature.params {
            let binding = self
                .bind(name, ann.ty.clone(), DefinitionKind::Parameter)
                .map_err(|error| self.errors.push(error))
                .ok();
            signature.parameters.push(binding);
        }
    }

    fn solve_functions(&mut self, signatures: &Signatures, bodies: &mut [Body]) {
        for group in groups(&dependencies(bodies)) {
            let roots = self.group_roots(signatures, bodies, &group);
            self.errors.extend(self.typing.solve(&roots));
            for index in group {
                self.require_function(
                    &signatures[&bodies[index].name].result,
                    &bodies[index].expressions,
                );
            }
        }
    }

    fn group_roots(
        &mut self,
        signatures: &Signatures,
        bodies: &mut [Body],
        group: &[usize],
    ) -> Vec<Type> {
        let mut roots = vec![];
        for &index in group {
            let body = &mut bodies[index];
            self.typing.constraints.append(&mut body.constraints);
            roots.extend(body.expressions.iter().map(|(_, ty)| ty.clone()));
            roots.push(signatures[&body.name].result.ty.clone());
        }
        roots
    }

    fn require_function(&mut self, result: &Annotation, expressions: &[(Span, Type)]) {
        for (span, ty) in std::iter::once(&(result.span, result.ty.clone())).chain(expressions) {
            if let Err(error) = self.require(ty, *span) {
                self.errors.push(error);
                if ty == &result.ty {
                    for (_, variable) in &result.holes {
                        self.typing.solver.invalidate(*variable);
                    }
                }
            }
        }
    }

    fn require(&self, ty: &Type, span: Span) -> Result<()> {
        if self.typing.solver.invalid(ty) {
            return Ok(());
        }
        self.typing
            .solver
            .require(ty, span)
            .map(|_| ())
            .map_err(|error| {
                if matches!(self.typing.solver.head(ty), Type::Node(Head::Array(0), _)) {
                    GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::EmptyArrayNeedsElementType,
                        },
                    )
                } else {
                    error
                }
            })
    }

    fn require_holes(&mut self) {
        for (span, variable) in &self.holes {
            let ty = variable.ty();
            if !self.typing.solver.invalid(&ty)
                && let Err(error) = self.typing.solver.require(&ty, *span)
            {
                self.errors.push(error);
            }
        }
    }

    fn finish(mut self, signatures: Signatures, bodies: Vec<Body>) -> CheckedFile {
        self.scopes
            .resolve_inferred(&self.typing.solver, self.typing.typer);
        let signatures = self.resolve_signatures(signatures);
        let bodies = self.resolve_bodies(bodies);
        CheckedFile {
            context: self.scopes.finish(),
            signatures,
            bodies,
            errors: self.errors,
        }
    }

    fn resolve_signatures(
        &mut self,
        signatures: Signatures,
    ) -> BTreeMap<Arc<str>, typed::Signature> {
        signatures
            .into_iter()
            .filter_map(|(name, signature)| {
                let resolved = signature.resolve(&self.typing.solver);
                self.record(resolved).map(|signature| (name, signature))
            })
            .collect()
    }

    fn resolve_bodies(&mut self, bodies: Vec<Body>) -> BTreeMap<Arc<str>, typed::Term> {
        bodies
            .into_iter()
            .filter_map(|body| {
                if self.typing.solver.invalid(&body.term.ty) {
                    return None;
                }
                let resolved = body.term.resolve(&self.typing.solver);
                self.record(resolved).map(|term| (body.name, term))
            })
            .collect()
    }

    fn record<T>(&mut self, result: Result<T>) -> Option<T> {
        result
            .map_err(|error| {
                if !self.errors.contains(&error) {
                    self.errors.push(error);
                }
            })
            .ok()
    }
}

impl Signature {
    fn ty(&self) -> Type {
        Type::function(
            Type::parameter(self.params.iter().map(|(_, ann)| ann.ty.clone()).collect()),
            self.result.ty.clone(),
        )
    }

    fn resolve(self, solver: &Solver) -> Result<typed::Signature> {
        Ok(typed::Signature {
            declaration: self.declaration,
            parameters: self.parameters,
            params: self
                .params
                .into_iter()
                .map(|(name, ann)| Ok((name, ann.into_tree().resolve(solver)?)))
                .collect::<Result<_>>()?,
            result: self.result.into_tree().resolve(solver)?,
        })
    }
}

fn dependencies(bodies: &[Body]) -> Vec<Vec<usize>> {
    let names: BTreeMap<_, _> = bodies
        .iter()
        .enumerate()
        .map(|(i, body)| (&body.name, i))
        .collect();
    bodies
        .iter()
        .map(|body| {
            body.dependencies
                .iter()
                .filter_map(|name| names.get(name).copied())
                .collect()
        })
        .collect()
}

fn is_shader(stmt: &StmtKind) -> bool {
    matches!(stmt, StmtKind::Function { decorators, .. } if decorators.len() == 1
        && matches!(decorators[0].val.as_ref(), "compute_shader" | "vertex_shader" | "fragment_shader"))
}

//
// Expression constraints and typed tree construction
//

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

    fn constrain(&mut self, constraint: (Span, Constraint)) {
        self.checker.typing.constrain(self.rule, constraint);
    }

    fn result_parts(&mut self, ty: &Type, span: Span) -> Result<(Type, Type)> {
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
                        (crate::lower::context::shader_properties().into(), false)
                    }
                    _ => (base.ty.clone(), false),
                };
                self.checker.scopes.record_members(
                    Span {
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
                equate = Some(Ty::Str.into());
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
                        typed::Annotation {
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
                        (crate::lower::context::shader_properties().into(), false)
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
                    crate::lower::context::SourceOrigin {
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

//
// Concrete tree resolution
//

impl typed::Annotation<Type> {
    fn resolve(self, solver: &Solver) -> Result<typed::Annotation> {
        Ok(typed::Annotation {
            ty: solver.require(&self.ty, self.span)?,
            span: self.span,
        })
    }
}

fn child(term: Box<typed::Term<Type>>, solver: &Solver) -> Result<Box<typed::Term>> {
    Ok(Box::new(term.resolve(solver)?))
}

impl typed::Term<Type> {
    fn resolve(self, solver: &Solver) -> Result<typed::Term> {
        let kind = match self.kind {
            TermKind::Error(error) => return Err(error),
            TermKind::Unit => TermKind::Unit,
            TermKind::None => TermKind::None,
            TermKind::Num { value } => TermKind::Num { value },
            TermKind::String { value } => TermKind::String { value },
            TermKind::Var { name } => TermKind::Var { name },
            TermKind::Type { ty } => TermKind::Type {
                ty: ty.resolve(solver)?,
            },
            TermKind::Unwrap { value } => TermKind::Unwrap {
                value: child(value, solver)?,
            },
            TermKind::Try { value } => TermKind::Try {
                value: child(value, solver)?,
            },
            TermKind::Match { value, arms } => TermKind::Match {
                value: child(value, solver)?,
                arms: arms
                    .into_iter()
                    .map(|arm| {
                        Ok(typed::MatchArm {
                            variant: arm.variant.map(|ty| ty.resolve(solver)).transpose()?,
                            failure: arm.failure,
                            binding: arm.binding,
                            body: arm.body.resolve(solver)?,
                        })
                    })
                    .collect::<Result<_>>()?,
            },
            TermKind::If { cond, then, els } => TermKind::If {
                cond: child(cond, solver)?,
                then: child(then, solver)?,
                els: child(els, solver)?,
            },
            TermKind::While { cond, body } => TermKind::While {
                cond: child(cond, solver)?,
                body: child(body, solver)?,
            },
            TermKind::Block { stmts, tail } => TermKind::Block {
                stmts: stmts
                    .into_iter()
                    .map(|stmt| stmt.resolve(solver))
                    .collect::<Result<_>>()?,
                tail: child(tail, solver)?,
            },
            TermKind::Record { fields } => TermKind::Record {
                fields: fields
                    .into_iter()
                    .map(|(name, term)| Ok((name, term.resolve(solver)?)))
                    .collect::<Result<_>>()?,
            },
            TermKind::Array { elems } => TermKind::Array {
                elems: elems
                    .into_iter()
                    .map(|term| term.resolve(solver))
                    .collect::<Result<_>>()?,
            },
            TermKind::Builtin { name, args } => TermKind::Builtin {
                name,
                args: args
                    .into_iter()
                    .map(|term| term.resolve(solver))
                    .collect::<Result<_>>()?,
            },
            TermKind::MethodCall {
                receiver,
                receiver_type,
                name,
                arg,
            } => TermKind::MethodCall {
                receiver: receiver.map(|term| child(term, solver)).transpose()?,
                receiver_type: receiver_type.resolve(solver)?,
                name,
                arg: child(arg, solver)?,
            },
            TermKind::Call { func, arg } => TermKind::Call {
                func: child(func, solver)?,
                arg: child(arg, solver)?,
            },
            TermKind::Ascribe { ty, arg } => TermKind::Ascribe {
                ty: ty.resolve(solver)?,
                arg: child(arg, solver)?,
            },
            TermKind::Result { failure, arg } => TermKind::Result {
                failure,
                arg: child(arg, solver)?,
            },
            TermKind::Absurd { arg } => TermKind::Absurd {
                arg: child(arg, solver)?,
            },
            TermKind::Layout { ty, size } => TermKind::Layout {
                ty: ty.resolve(solver)?,
                size,
            },
            TermKind::Assign { place, value } => TermKind::Assign {
                place: child(place, solver)?,
                value: child(value, solver)?,
            },
            TermKind::Address { place } => TermKind::Address {
                place: child(place, solver)?,
            },
            TermKind::Deref { pointer } => TermKind::Deref {
                pointer: child(pointer, solver)?,
            },
            TermKind::Field { base, name } => TermKind::Field {
                base: child(base, solver)?,
                name,
            },
        };
        Ok(typed::Term {
            span: self.span,
            context: self.context,
            ty: solver.require(&self.ty, self.span)?,
            kind,
        })
    }
}

impl typed::Statement<Type> {
    fn resolve(self, solver: &Solver) -> Result<typed::Statement> {
        let kind = match self.kind {
            StatementKind::Error(error) => return Err(error),
            StatementKind::Define {
                binding,
                name,
                init,
            } => StatementKind::Define {
                binding,
                name,
                init: init.resolve(solver)?,
            },
            StatementKind::Declare { binding, name, ty } => StatementKind::Declare {
                binding,
                name,
                ty: ty.resolve(solver)?,
            },
            StatementKind::TypeDefinition => StatementKind::TypeDefinition,
            StatementKind::Expr { term } => StatementKind::Expr {
                term: term.resolve(solver)?,
            },
        };
        Ok(typed::Statement {
            context: self.context,
            kind,
        })
    }
}

//
// Dependency groups
//

/// Strongly connected groups in dependency-first order.
fn groups(edges: &[Vec<usize>]) -> Vec<Vec<usize>> {
    struct Walk<'a> {
        edges: &'a [Vec<usize>],
        index: Vec<Option<usize>>,
        low: Vec<usize>,
        active: Vec<bool>,
        stack: Vec<usize>,
        next: usize,
        groups: Vec<Vec<usize>>,
    }
    impl Walk<'_> {
        fn visit(&mut self, node: usize) {
            self.index[node] = Some(self.next);
            self.low[node] = self.next;
            self.next += 1;
            self.stack.push(node);
            self.active[node] = true;
            for &dependency in &self.edges[node] {
                if self.index[dependency].is_none() {
                    self.visit(dependency);
                    self.low[node] = self.low[node].min(self.low[dependency]);
                } else if self.active[dependency] {
                    self.low[node] = self.low[node].min(self.index[dependency].unwrap());
                }
            }
            if self.low[node] == self.index[node].unwrap() {
                let mut group = Vec::new();
                loop {
                    let member = self.stack.pop().unwrap();
                    self.active[member] = false;
                    group.push(member);
                    if member == node {
                        break;
                    }
                }
                self.groups.push(group);
            }
        }
    }
    let n = edges.len();
    let mut walk = Walk {
        edges,
        index: vec![None; n],
        low: vec![0; n],
        active: vec![false; n],
        stack: vec![],
        next: 0,
        groups: vec![],
    };
    for node in 0..n {
        if walk.index[node].is_none() {
            walk.visit(node);
        }
    }
    walk.groups
}

#[cfg(test)]
mod tests;
