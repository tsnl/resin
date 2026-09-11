//! Inference variables, unification, and expression constraints.
//! All handles are resolved before the typed tree reaches LIR lowering.
use crate::lower::context::Context;
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{collections::HashSet, sync::Arc};

//
// Inference types
//

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Head {
    Atom(Ty),
    Pointer,
    GpuPointer,
    GpuSpan,
    Arc,
    Weak,
    Span,
    Array(usize),
    Record(Vec<Arc<str>>),
    Function,
    Result,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Type {
    Invalid,
    Variable(usize),
    Node(Head, Vec<Type>),
}

impl Type {
    /// The inference representation of Ty::deref_target; Weak is not dereferenceable.
    pub fn deref_target(&self) -> Option<&Type> {
        match self {
            Self::Node(Head::Pointer | Head::GpuPointer | Head::Arc, children) => children.first(),
            _ => None,
        }
    }

    pub fn view_element(&self) -> Option<Type> {
        match self {
            Self::Node(Head::Atom(Ty::Str), _) => Some(Ty::UInt8.into()),
            Self::Node(Head::Span | Head::GpuSpan, children) => children.first().cloned(),
            _ => None,
        }
    }

    pub fn index_element(&self) -> Option<Type> {
        match self {
            Self::Node(Head::Array(_) | Head::GpuPointer, children) => children.first().cloned(),
            _ => self.view_element(),
        }
    }

    pub fn result(value: Type, error: Type) -> Self {
        Self::Node(Head::Result, vec![value, error])
    }
    pub fn pointer(pointee: Type) -> Self {
        Self::Node(Head::Pointer, vec![pointee])
    }

    pub fn gpu_pointer(pointee: Type) -> Self {
        Self::Node(Head::GpuPointer, vec![pointee])
    }

    pub fn function(param: Type, result: Type) -> Self {
        Self::Node(Head::Function, vec![param, result])
    }

    pub fn record(fields: Vec<(Arc<str>, Type)>) -> Self {
        let (names, types) = fields.into_iter().unzip();
        Self::Node(Head::Record(names), types)
    }

    pub fn parameter(types: Vec<Type>) -> Self {
        match types.len() {
            0 => Ty::Unit.into(),
            1 => types.into_iter().next().unwrap(),
            _ => Self::record(
                types
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| (format!("_{i}").into(), t))
                    .collect(),
            ),
        }
    }
}

impl From<Ty> for Type {
    fn from(ty: Ty) -> Self {
        match ty {
            Ty::Arc { pointee } => Self::Node(Head::Arc, vec![(*pointee).into()]),
            Ty::Weak { pointee } => Self::Node(Head::Weak, vec![(*pointee).into()]),
            Ty::Pointer { pointee } => Self::pointer((*pointee).into()),
            Ty::Span { element } => Self::Node(Head::Span, vec![(*element).into()]),
            Ty::GpuPointer { pointee } => Self::gpu_pointer((*pointee).into()),
            Ty::GpuSpan { element } => Self::Node(Head::GpuSpan, vec![(*element).into()]),
            Ty::Array { element, length } => {
                Self::Node(Head::Array(length), vec![(*element).into()])
            }
            Ty::Record { fields } => {
                Self::record(fields.into_iter().map(|f| (f.name, f.ty.into())).collect())
            }
            Ty::Function { param, result } => Self::function((*param).into(), (*result).into()),
            Ty::Result { value, error } => Self::result((*value).into(), (*error).into()),
            atom => Self::Node(Head::Atom(atom), vec![]),
        }
    }
}

impl Head {
    pub fn concrete(&self, children: Vec<Ty>) -> Ty {
        let mut children = children.into_iter();
        match self {
            Self::Atom(ty) => ty.clone(),
            Self::Arc => Ty::Arc {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::Weak => Ty::Weak {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::Pointer => Ty::Pointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuPointer => Ty::GpuPointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuSpan => Ty::GpuSpan {
                element: Box::new(children.next().unwrap()),
            },
            Self::Span => Ty::Span {
                element: Box::new(children.next().unwrap()),
            },
            Self::Array(length) => Ty::Array {
                element: Box::new(children.next().unwrap()),
                length: *length,
            },
            Self::Record(names) => Ty::Record {
                fields: names
                    .iter()
                    .cloned()
                    .zip(children)
                    .map(|(name, ty)| RecordField { name, ty })
                    .collect(),
            },
            Self::Function => Ty::Function {
                param: Box::new(children.next().unwrap()),
                result: Box::new(children.next().unwrap()),
            },
            Self::Result => Ty::Result {
                value: Box::new(children.next().unwrap()),
                error: Box::new(children.next().unwrap()),
            },
        }
    }
}

//
// Inference variables and unification
//

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Any,
    Number,
    Float,
    Errors,
}

#[derive(Clone)]
struct Variable {
    value: Option<Type>,
    class: Class,
    variants: Vec<TypeId>,
}

/// An allocated inference variable, retained independently of its resolved type.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VariableId(usize);

impl VariableId {
    pub fn ty(self) -> Type {
        Type::Variable(self.0)
    }
}

#[derive(Clone, Default)]
pub(crate) struct Solver {
    variables: Vec<Variable>,
    pub revision: usize,
}

impl Solver {
    pub fn fresh(&mut self) -> Type {
        self.fresh_variable().ty()
    }

    pub fn fresh_variable(&mut self) -> VariableId {
        self.variable(Class::Any)
    }

    pub fn number(&mut self, text: &str) -> Type {
        if let (_, Some(ty)) = resin_types::literal::split(text) {
            return ty.into();
        }
        self.variable(
            if resin_types::literal::unsuffixed_type(text).is_integer() {
                Class::Number
            } else {
                Class::Float
            },
        )
        .ty()
    }

    fn variable(&mut self, class: Class) -> VariableId {
        let id = self.variables.len();
        self.variables.push(Variable {
            value: None,
            class,
            variants: vec![],
        });
        VariableId(id)
    }

    pub fn errors(&mut self, ty: &Type, span: Span) -> Result<()> {
        match self.head(ty) {
            Type::Variable(id)
                if matches!(self.variables[id].class, Class::Any | Class::Errors) =>
            {
                self.variables[id].class = Class::Errors;
                Ok(())
            }
            Type::Node(Head::Atom(ty), _) if ty.variants().is_some() => Ok(()),
            _ => Err(error(
                span,
                "Result errors must be structs or unions of structs",
            )),
        }
    }

    pub fn include(&mut self, from: &Type, to: &Type, span: Span) -> Result<bool> {
        self.errors(to, span)?;
        let variants = match self.head(from) {
            Type::Variable(id) if self.variables[id].class == Class::Errors => {
                self.variables[id].variants.clone()
            }
            Type::Variable(_) => return Ok(false),
            Type::Node(Head::Atom(ty), _) => ty
                .variants()
                .ok_or_else(|| error(span, "error and union payloads must be nominal structs"))?,
            _ => {
                return Err(error(
                    span,
                    "error and union payloads must be nominal structs",
                ));
            }
        };
        match self.head(to) {
            Type::Variable(id) => {
                for variant in variants {
                    if !self.variables[id].variants.contains(&variant) {
                        self.variables[id].variants.push(variant);
                        self.revision += 1;
                    }
                }
                Ok(false)
            }
            Type::Node(Head::Atom(target), _) => {
                if !variants
                    .iter()
                    .all(|id| target.variants().unwrap().contains(id))
                {
                    return Err(error(
                        span,
                        "the destination error set does not include every propagated error",
                    ));
                }
                Ok(self.resolve(from).is_some())
            }
            _ => unreachable!(),
        }
    }

    pub fn coerce(&mut self, from: &Type, to: &Type, span: Span) -> Result<bool> {
        match (self.head(from), self.head(to)) {
            (Type::Node(Head::Record(a), aa), Type::Node(Head::Record(b), bb)) if a == b => {
                let mut complete = true;
                for (from, to) in aa.iter().zip(&bb) {
                    complete &= self.coerce(from, to, span)?;
                }
                Ok(complete)
            }
            (Type::Node(Head::Result, a), Type::Node(Head::Result, b)) => {
                self.unify(&a[0], &b[0], span)?;
                self.include(&a[1], &b[1], span)
            }
            (_, Type::Node(Head::Atom(target @ Ty::Union { .. }), _)) => {
                // Literal context may select one numeric member, but pointers and
                // other mutable storage remain invariant inside union members.
                if let Type::Variable(id) = self.head(from) {
                    let class = self.variables[id].class;
                    if matches!(class, Class::Number | Class::Float) {
                        let candidates: Vec<_> = target
                            .members()
                            .into_iter()
                            .filter(|ty| {
                                if class == Class::Float {
                                    matches!(ty, Ty::Float32 | Ty::Float64)
                                } else {
                                    ty.is_numeric()
                                }
                            })
                            .collect();
                        if let [ty] = candidates.as_slice() {
                            self.unify(from, &ty.clone().into(), span)?;
                        }
                    }
                }
                let Some(source) = self.resolve(from) else {
                    return Ok(false);
                };
                if source.widens_to(&target) {
                    Ok(true)
                } else {
                    Err(GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: target,
                                found: source,
                            },
                        },
                    ))
                }
            }
            (Type::Node(Head::Atom(Ty::Union { variants }), _), _) if variants.is_empty() => {
                Ok(true)
            }
            _ => {
                self.unify(from, to, span)?;
                Ok(true)
            }
        }
    }

    fn variables_in(&self, roots: &[Type]) -> Vec<usize> {
        fn collect(solver: &Solver, ty: &Type, ids: &mut Vec<usize>) {
            match solver.head(ty) {
                Type::Invalid => {}
                Type::Variable(id) => {
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                Type::Node(_, args) => {
                    for arg in args {
                        collect(solver, &arg, ids);
                    }
                }
            }
        }
        let mut ids = vec![];
        for root in roots {
            collect(self, root, &mut ids);
        }
        ids
    }

    pub fn finish_errors(&mut self, roots: &[Type]) -> bool {
        let before = self.revision;
        for id in self.variables_in(roots) {
            let variable = &mut self.variables[id];
            if variable.class == Class::Errors {
                variable.value = Some(Ty::union(variable.variants.clone()).into());
                self.revision += 1;
            }
        }
        self.revision != before
    }

    pub fn head(&self, ty: &Type) -> Type {
        let mut ty = ty;
        while let Type::Variable(id) = ty {
            match &self.variables[*id].value {
                Some(value) => ty = value,
                None => break,
            }
        }
        ty.clone()
    }

    pub fn shape_hint(&self, ty: &Type) -> Type {
        let head = self.head(ty);
        if let Type::Variable(id) = head
            && self.variables[id].class == Class::Errors
            && !self.variables[id].variants.is_empty()
        {
            return Ty::union(self.variables[id].variants.clone()).into();
        }
        head
    }

    pub fn resolve(&self, ty: &Type) -> Option<Ty> {
        match self.head(ty) {
            Type::Invalid | Type::Variable(_) => None,
            Type::Node(head, args) => Some(
                head.concrete(
                    args.iter()
                        .map(|t| self.resolve(t))
                        .collect::<Option<_>>()?,
                ),
            ),
        }
    }

    pub fn invalid(&self, ty: &Type) -> bool {
        match self.head(ty) {
            Type::Invalid => true,
            Type::Node(_, children) => children.iter().any(|ty| self.invalid(ty)),
            Type::Variable(_) => false,
        }
    }

    pub fn invalidate(&mut self, variable: VariableId) {
        self.variables[variable.0].value = Some(Type::Invalid);
    }

    pub fn require(&self, ty: &Type, span: Span) -> Result<Ty> {
        self.resolve(ty)
            .ok_or_else(|| error(span, "cannot infer this type; add an explicit annotation"))
    }

    pub fn unify(&mut self, left: &Type, right: &Type, span: Span) -> Result<()> {
        let left = self.head(left);
        let right = self.head(right);
        if left == right {
            return Ok(());
        }
        match (&left, &right) {
            (Type::Variable(id), _) => self.bind(*id, right, span),
            (_, Type::Variable(id)) => self.bind(*id, left, span),
            (Type::Node(a, aa), Type::Node(b, bb)) if a == b && aa.len() == bb.len() => {
                for (a, b) in aa.iter().zip(bb) {
                    self.unify(a, b, span)?;
                }
                Ok(())
            }
            _ => {
                if let (Some(found), Some(expected)) = (self.resolve(&left), self.resolve(&right)) {
                    Err(GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::TypeMismatch { expected, found },
                        },
                    ))
                } else {
                    Err(error(
                        span,
                        format!("incompatible inferred types: {left:?} and {right:?}"),
                    ))
                }
            }
        }
    }

    // Explicit pointer casts relate holes only where their pointee shapes agree.
    // A cast from Ptr<[T; N]> to Ptr<U> must not equate the array with U.
    pub fn cast(&mut self, from: &Type, to: &Type, span: Span) -> Result<()> {
        match (self.head(from), self.head(to)) {
            (_, Type::Variable(_)) => self.unify(to, from, span),
            (Type::Node(a, aa), Type::Node(b, bb)) if a == b => {
                for (a, b) in aa.iter().zip(bb.iter()) {
                    self.cast(a, b, span)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind(&mut self, id: usize, ty: Type, span: Span) -> Result<()> {
        if self.occurs(id, &ty) {
            return Err(error(span, "inference would create an infinite type"));
        }
        let class = self.variables[id].class;
        if let Type::Variable(other) = ty {
            let other_class = self.variables[other].class;
            if (class == Class::Errors && !matches!(other_class, Class::Any | Class::Errors))
                || (other_class == Class::Errors && !matches!(class, Class::Any | Class::Errors))
            {
                return Err(error(span, "an error set cannot be a numeric type"));
            }
            self.variables[other].class = match (class, other_class) {
                (Class::Errors, _) | (_, Class::Errors) => Class::Errors,
                (Class::Float, _) | (_, Class::Float) => Class::Float,
                (Class::Number, _) | (_, Class::Number) => Class::Number,
                _ => Class::Any,
            };
            let variants = self.variables[id].variants.clone();
            for variant in variants {
                if !self.variables[other].variants.contains(&variant) {
                    self.variables[other].variants.push(variant);
                }
            }
        } else if class == Class::Errors {
            self.errors(&ty, span)?;
            let Type::Node(Head::Atom(ref concrete), _) = ty else {
                unreachable!()
            };
            if !self.variables[id]
                .variants
                .iter()
                .all(|v| concrete.variants().unwrap().contains(v))
            {
                return Err(error(
                    span,
                    "inferred errors are not included in the annotated error set",
                ));
            }
        } else if class != Class::Any {
            let numeric = matches!(&ty, Type::Node(Head::Atom(t), _) if t.is_numeric());
            let float = matches!(&ty, Type::Node(Head::Atom(Ty::Float32 | Ty::Float64), _));
            if !numeric || (class == Class::Float && !float) {
                if let Some(expected) = self.resolve(&ty) {
                    let found = if class == Class::Float {
                        Ty::Float64
                    } else {
                        Ty::Int64
                    };
                    return Err(GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::TypeMismatch { expected, found },
                        },
                    ));
                }
                return Err(error(
                    span,
                    "numeric literal has an incompatible inferred type",
                ));
            }
        }
        self.variables[id].value = Some(ty);
        self.revision += 1;
        Ok(())
    }

    fn occurs(&self, id: usize, ty: &Type) -> bool {
        match self.head(ty) {
            Type::Invalid => false,
            Type::Variable(other) => id == other,
            Type::Node(_, args) => args.iter().any(|arg| self.occurs(id, arg)),
        }
    }

    pub fn default_numbers(&mut self, roots: &[Type]) -> bool {
        let before = self.revision;
        for id in self.variables_in(roots) {
            self.default_number(id);
        }
        self.revision != before
    }

    fn default_number(&mut self, id: usize) {
        let variable = &mut self.variables[id];
        if variable.value.is_none() {
            let ty = match variable.class {
                Class::Any | Class::Errors => return,
                Class::Number => Ty::Int64,
                Class::Float => Ty::Float64,
            };
            variable.value = Some(ty.into());
            self.revision += 1;
        }
    }
}

//
// Constraint ownership and expression rules
//

type Result<T> = std::result::Result<T, GenerateError>;
fn error(span: Span, message: impl Into<Arc<str>>) -> GenerateError {
    GenerateError::inference(span, message)
}
pub(crate) fn check_binding_name(name: &Ident) -> Result<()> {
    if matches!(
        name.val.as_ref(),
        "fmt" | "print" | "ok" | "err" | "size_of" | "align_of" | "absurd"
    ) {
        return Err(GenerateError {
            span: name.span,
            kind: GenerateErrorKind::ReservedBuiltin {
                name: name.val.clone(),
            },
        });
    }
    Ok(())
}

/// Identity of one typing operation; its owned variables are private to Inference.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rule(usize);

#[derive(Clone)]
pub(crate) struct Equation {
    span: Span,
    owner: Rule,
    relation: Constraint,
}

/// Constraint services injected into source checking; this layer never traverses expressions.
pub(crate) struct Inference<'a> {
    pub typer: &'a mut Context,
    pub solver: Solver,
    pub constraints: Vec<Equation>,
    owners: Vec<Vec<VariableId>>,
}
impl<'a> Inference<'a> {
    pub fn new(typer: &'a mut Context) -> Self {
        Self {
            typer,
            solver: Solver::default(),
            constraints: vec![],
            owners: vec![],
        }
    }
    fn rule(&mut self, variables: Vec<VariableId>) -> Rule {
        let rule = Rule(self.owners.len());
        self.owners.push(variables);
        rule
    }
    pub fn expression(&mut self) -> (Rule, Type) {
        let variable = self.solver.fresh_variable();
        (self.rule(vec![variable]), variable.ty())
    }
    pub fn constrain(&mut self, owner: Rule, (span, relation): (Span, Constraint)) {
        self.constraints.push(Equation {
            span,
            owner,
            relation,
        });
    }
    pub fn depends(&mut self, owner: Rule, span: Span, input: Type) {
        self.constrain(owner, (span, Constraint::Depends(input)));
    }
    pub fn infer_from(&mut self, holes: &[(Span, VariableId)], span: Span, input: Type) {
        let owner = self.rule(holes.iter().map(|(_, variable)| *variable).collect());
        self.depends(owner, span, input);
    }
    pub fn fail(&mut self, rule: Rule) {
        for variable in &self.owners[rule.0] {
            self.solver.invalidate(*variable);
        }
    }
    pub fn result_parts(&mut self, owner: Rule, ty: &Type, span: Span) -> Result<(Type, Type)> {
        match self.solver.head(ty) {
            Type::Node(Head::Result, parts) => Ok((parts[0].clone(), parts[1].clone())),
            Type::Variable(_) => {
                let value = self.solver.fresh();
                let errors = self.solver.fresh();
                self.solver.errors(&errors, span)?;
                self.constrain(
                    owner,
                    (
                        span,
                        Constraint::Equal(ty.clone(), Type::result(value.clone(), errors.clone())),
                    ),
                );
                Ok((value, errors))
            }
            _ => Err(error(
                span,
                "ok, err, and ? require a Result type; ? also requires a Result return type",
            )),
        }
    }
}

//
// Constraint solving
//

#[derive(Clone)]
pub(crate) enum AddressOrigin {
    /// Explicit `.*` dereferences exactly one pointer.
    Deref { pointer: Type },
    /// Field lookup implicitly follows every pointer and Arc receiver.
    Field { base: Type },
}

impl AddressOrigin {
    fn input(&self) -> &Type {
        match self {
            Self::Deref { pointer } => pointer,
            Self::Field { base } => base,
        }
    }
}

#[derive(Clone)]
pub(crate) enum Constraint {
    Equal(Type, Type),
    Depends(Type),
    Coerce(Type, Type),
    ExcludeNone(Type, Type),
    Layout(Type),
    Errors(Type, Type),
    Variant(Type, Pattern, Type),
    Boolean(Type),
    Deref(Type, Type),
    Address(Vec<AddressOrigin>, Type, Type),
    Field(Type, Arc<str>, Type),
    Call(Type, Type, Type),
    Method(Type, Arc<str>, Type, Type, bool, Vec<AddressOrigin>),
    GpuProject(Type, Type, Type),
    Ascribe(Type, Type, bool),
    Record(Vec<(Arc<str>, Type)>, Type),
    Builtin(Arc<str>, Vec<Type>, Type),
}

#[derive(Clone)]
pub(crate) enum Pattern {
    Ok,
    Err,
    Type(Type),
}

impl Inference<'_> {
    /// Retry equations from a clean SCC snapshot after discarding a failed
    /// producer. Pending constraints may have mutated the solver on earlier
    /// attempts, so rolling back only the final attempt is insufficient.
    pub fn solve(&mut self, roots: &[Type]) -> Vec<GenerateError> {
        let baseline = self.solver.clone();
        let equations = std::mem::take(&mut self.constraints);
        let mut failed = Vec::new();
        let mut errors = Vec::new();
        'retry: loop {
            self.solver = baseline.clone();
            for owner in &failed {
                self.fail(*owner);
            }
            self.constraints = equations
                .iter()
                .filter(|equation| !failed.contains(&equation.owner))
                .cloned()
                .collect();
            loop {
                let before = self.solver.revision;
                let pending = std::mem::take(&mut self.constraints);
                for Equation {
                    span,
                    owner,
                    relation: constraint,
                } in pending
                {
                    if constraint.inputs().iter().any(|ty| self.solver.invalid(ty)) {
                        failed.push(owner);
                        continue 'retry;
                    }
                    match self.constraint(&constraint, span) {
                        Ok(true) => {}
                        Ok(false) => self.constraints.push(Equation {
                            span,
                            owner,
                            relation: constraint,
                        }),
                        Err(error) => {
                            errors.push(error);
                            failed.push(owner);
                            continue 'retry;
                        }
                    }
                }
                if self.solver.revision != before {
                    continue;
                }
                let mut seeded = false;
                for Equation {
                    span,
                    relation: constraint,
                    ..
                } in &self.constraints
                {
                    if let Constraint::Record(fields, out) = constraint
                        && matches!(self.solver.head(out), Type::Variable(_))
                    {
                        self.solver
                            .unify(out, &Type::record(fields.clone()), *span)
                            .expect("fresh record result");
                        seeded = true;
                    }
                }
                if seeded {
                    continue;
                }
                // Receiver defaults select the parameter types before argument defaults.
                for Equation {
                    relation: constraint,
                    ..
                } in &self.constraints
                {
                    if let Constraint::Method(receiver, ..) = constraint {
                        seeded |= self.solver.default_numbers(std::slice::from_ref(receiver));
                    }
                }
                if seeded || self.solver.default_numbers(roots) {
                    continue;
                }
                if self.solver.finish_errors(roots) {
                    continue;
                }
                if let Some(Equation { span, owner, .. }) = self.constraints.first() {
                    errors.push(error(
                        *span,
                        "cannot infer this operation; annotate its operand or result",
                    ));
                    failed.push(*owner);
                    continue 'retry;
                }
                return errors;
            }
        }
    }

    fn shape(&self, ty: &Type, deref: bool, span: Span) -> Result<Type> {
        let mut ty = self.solver.shape_hint(ty);
        if deref {
            while let Some(pointee) = ty.deref_target() {
                ty = self.solver.shape_hint(pointee);
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

    fn require_gpu_element(&self, element: &Ty, span: Span) -> Result<()> {
        if element.gpu_element(self.typer.definitions()) {
            return Ok(());
        }
        Err(error(
            span,
            "GPU elements require a shared host/device layout without pointers, owners, or drop hooks",
        ))
    }

    fn gpu_address(&self, origins: &[AddressOrigin]) -> Option<bool> {
        let mut gpu = false;
        for origin in origins {
            let mut current = self.solver.head(origin.input());
            loop {
                match current {
                    Type::Variable(_) => return None,
                    Type::Node(Head::GpuPointer, _) => gpu = true,
                    _ => {}
                }
                if matches!(origin, AddressOrigin::Deref { .. }) {
                    break;
                }
                let Some(pointee) = current.deref_target() else {
                    break;
                };
                current = self.solver.head(pointee);
            }
        }
        Some(gpu)
    }

    fn constraint(&mut self, constraint: &Constraint, span: Span) -> Result<bool> {
        match constraint {
            Constraint::Method(receiver_type, name, arg, out, associated, origins) => {
                let Some(receiver_type) = self.solver.resolve(receiver_type) else {
                    return Ok(false);
                };
                if !associated
                    && name.as_ref() == "new"
                    && let Some(allocator) = self.typer.gpu_allocator(&receiver_type)
                {
                    let result = Type::result(
                        Type::gpu_pointer(arg.clone()),
                        self.typer.gpu_error(allocator).into(),
                    );
                    self.solver.coerce(&result, out, span)?;
                    let Some(element) = self.solver.resolve(arg) else {
                        return Ok(false);
                    };
                    self.require_gpu_element(&element, span)?;
                    return Ok(true);
                }
                let method = if *associated
                    && matches!(
                        (&receiver_type, name.as_ref()),
                        (Ty::GpuPointer { .. }, "new") | (Ty::GpuSpan { .. }, "allocate")
                    ) {
                    let Type::Node(Head::Record(_), parts) = self.solver.head(arg) else {
                        return Ok(false);
                    };
                    let Some(gpu) = parts.first().and_then(|ty| self.solver.resolve(ty)) else {
                        return Ok(false);
                    };
                    let element = match &receiver_type {
                        Ty::GpuPointer { pointee } => &**pointee,
                        Ty::GpuSpan { element } => &**element,
                        _ => unreachable!(),
                    };
                    self.require_gpu_element(element, span)?;
                    let argument = Ty::parameter(&[gpu, element.clone()]);
                    self.typer
                        .method_call(&receiver_type, name, &argument, true)
                } else if *associated
                    && name.as_ref() == "allocate_native"
                    && receiver_type
                        == (Ty::GpuPointer {
                            pointee: Box::new(Ty::UInt8),
                        })
                {
                    let Type::Node(Head::Record(_), parts) = self.solver.head(arg) else {
                        return Ok(false);
                    };
                    let Some(prefix) = parts
                        .iter()
                        .take(2)
                        .map(|ty| self.solver.resolve(ty))
                        .collect::<Option<Vec<_>>>()
                    else {
                        return Ok(false);
                    };
                    if prefix.len() != 2 {
                        return Err(error(
                            span,
                            "native GPU allocation requires a handle and owner",
                        ));
                    }
                    let argument = Ty::parameter(&[
                        prefix[0].clone(),
                        prefix[1].clone(),
                        Ty::UInt64,
                        Ty::UInt64,
                        Ty::Int32,
                    ]);
                    self.typer
                        .method_call(&receiver_type, name, &argument, true)
                } else {
                    self.typer.method(&receiver_type, name)
                }
                .ok_or_else(|| error(span, format!("unknown method `{name}`")))?;
                if !associated
                    && let Some(first) = method.params.first()
                    && crate::ReceiverConversion::between(&receiver_type, first)
                        == Some(crate::ReceiverConversion::Address)
                {
                    let Some(gpu) = self.gpu_address(origins) else {
                        return Ok(false);
                    };
                    if gpu != matches!(first, Ty::GpuPointer { .. }) {
                        return Err(error(
                            span,
                            if gpu {
                                "GPU storage requires a GpuPtr receiver; it cannot be borrowed as a raw Ptr"
                            } else {
                                "a GpuPtr receiver requires an address in GPU storage"
                            },
                        ));
                    }
                }
                let params = method
                    .arguments(&receiver_type, *associated)
                    .ok_or_else(|| {
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
            Constraint::Depends(_) => {}
            Constraint::Equal(from, to) => {
                if !self.solver.invalid(to) {
                    self.solver.unify(from, to, span)?;
                }
            }
            Constraint::Layout(ty) => {
                let Some(ty) = self.solver.resolve(ty) else {
                    return Ok(false);
                };
                resin_types::layout::layout(self.typer.definitions(), &ty)
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
            Constraint::Coerce(from, to) => {
                if !self.solver.invalid(to) {
                    return self.solver.coerce(from, to, span);
                }
            }
            Constraint::Errors(from, to) => {
                if !self.solver.invalid(to) {
                    return self.solver.include(from, to, span);
                }
            }
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

            Constraint::Address(origins, pointee, out) => {
                let Some(gpu) = self.gpu_address(origins) else {
                    return Ok(false);
                };
                let pointer = if gpu {
                    Type::gpu_pointer(pointee.clone())
                } else {
                    Type::pointer(pointee.clone())
                };
                self.solver.unify(out, &pointer, span)?;
            }
            Constraint::Deref(input, out) => {
                let shape = self.shape(input, false, span)?;
                if matches!(shape, Type::Variable(_)) {
                    return Ok(false);
                }
                let pointee = shape
                    .deref_target()
                    .ok_or_else(|| error(span, "dereference requires a pointer"))?;
                self.solver.unify(out, pointee, span)?;
            }
            Constraint::Field(input, name, out) => {
                let shape = self.shape(input, true, span)?;
                if let Type::Node(Head::Function, _) = &shape
                    && name.as_ref() == "spirv"
                {
                    self.solver.unify(out, &Ty::shader().into(), span)?;
                    return Ok(true);
                }
                if let Some(element) = shape.view_element() {
                    let ty = match name.as_ref() {
                        "data" if matches!(shape, Type::Node(Head::GpuSpan, _)) => {
                            Type::gpu_pointer(element)
                        }
                        "data" => Type::pointer(element),
                        "length" => Ty::UInt64.into(),
                        _ => return Err(error(span, "unknown string or span field")),
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
            Constraint::GpuProject(func, arg, out) => {
                let Some(Ty::Function { param, .. }) = self.solver.resolve(func) else {
                    return Ok(false);
                };
                let Ty::Record { fields } = *param else {
                    return Err(error(
                        span,
                        "projection requires a shader with a root pointer",
                    ));
                };
                let Some(RecordField {
                    ty: Ty::Pointer { pointee },
                    ..
                }) = fields.get(1)
                else {
                    return Err(error(
                        span,
                        "projection requires a shader with a root pointer",
                    ));
                };
                let input = pointee.gpu_projection(self.typer.definitions()).ok_or_else(||
                    error(span, "shader root projection requires shared values and pointers or spans to plain GPU elements"))?;
                let Type::Node(Head::Record(names), arguments) = self.solver.head(arg) else {
                    if matches!(self.solver.head(arg), Type::Variable(_)) {
                        return Ok(false);
                    }
                    return Err(error(
                        span,
                        "projection takes a GPU and a host argument record",
                    ));
                };
                if names.as_slice() != [Arc::from("_0"), Arc::from("_1")] {
                    return Err(error(
                        span,
                        "projection takes a GPU and a host argument record",
                    ));
                }
                let Some(gpu) = self.solver.resolve(&arguments[0]) else {
                    return Ok(false);
                };
                let allocator = self.typer.gpu_allocator(&gpu).ok_or_else(|| {
                    error(
                        span,
                        "projection requires a GPU with a registered allocator",
                    )
                })?;
                self.solver.unify(
                    out,
                    &Type::result(
                        Ty::GpuArguments.into(),
                        self.typer.gpu_error(allocator).into(),
                    ),
                    span,
                )?;
                return self.solver.coerce(&arguments[1], &input.into(), span);
            }
            Constraint::Call(func, arg, out) => {
                let shape = self.shape(func, false, span)?;
                if let Some(element) = shape.index_element() {
                    let pointer =
                        if matches!(shape, Type::Node(Head::GpuPointer | Head::GpuSpan, _)) {
                            Type::gpu_pointer(element)
                        } else {
                            Type::pointer(element)
                        };
                    self.solver.unify(out, &pointer, span)?;
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
                if matches!(self.solver.head(to), Type::Node(Head::Weak, _))
                    && self.solver.resolve(from) == Some(Ty::Unit)
                {
                    return Ok(true);
                }
                if let Type::Node(Head::Span, children) = self.solver.head(to) {
                    // The source may still become str, Span, or a field record.
                    if matches!(self.solver.head(from), Type::Variable(_)) {
                        return Ok(false);
                    }
                    if self.solver.resolve(from) == Some(Ty::Str) {
                        self.solver.unify(&children[0], &Ty::UInt8.into(), span)?;
                        return Ok(true);
                    }
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
                    if !empty {
                        self.typer
                            .explicit_conversion(&from, &to)
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
                use resin_types::BuiltinRule;
                let rule = BuiltinRule::lookup(name, args.len())
                    .map_err(|e| GenerateError::typing(span, e))?;
                match rule {
                    BuiltinRule::Print => self.solver.unify(out, &Ty::Unit.into(), span)?,
                    BuiltinRule::Format | BuiltinRule::StringFromBytes => self.solver.unify(
                        out,
                        &self
                            .typer
                            .string_type()
                            .cloned()
                            .expect("builtin String")
                            .into(),
                        span,
                    )?,
                    BuiltinRule::Boolean => {
                        self.solver.unify(out, &Ty::Bool.into(), span)?;
                        for arg in args {
                            if !self.constraint(&Constraint::Boolean(arg.clone()), span)? {
                                return Ok(false);
                            }
                        }
                    }
                    BuiltinRule::Arithmetic | BuiltinRule::Comparison => {
                        if rule == BuiltinRule::Comparison {
                            self.solver.unify(out, &Ty::Bool.into(), span)?;
                        } else {
                            self.solver.unify(out, &args[0], span)?;
                        }
                        for arg in &args[1..] {
                            self.solver.unify(&args[0], arg, span)?;
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

impl Constraint {
    fn inputs(&self) -> Vec<&Type> {
        match self {
            Self::Depends(from)
            | Self::ExcludeNone(from, _)
            | Self::Equal(from, _)
            | Self::Coerce(from, _)
            | Self::Errors(from, _)
            | Self::Layout(from)
            | Self::Boolean(from)
            | Self::Deref(from, _)
            | Self::Field(from, _, _)
            | Self::Ascribe(from, _, _)
            | Self::Variant(from, _, _) => vec![from],
            Self::Call(func, arg, _) | Self::GpuProject(func, arg, _) => vec![func, arg],
            Self::Method(func, _, arg, _, _, origins) => origins
                .iter()
                .map(AddressOrigin::input)
                .chain([func, arg])
                .collect(),
            Self::Address(origins, pointee, _) => origins
                .iter()
                .map(AddressOrigin::input)
                .chain(std::iter::once(pointee))
                .collect(),
            Self::Record(fields, _) => fields.iter().map(|(_, ty)| ty).collect(),
            Self::Builtin(_, args, _) => args.iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests;
