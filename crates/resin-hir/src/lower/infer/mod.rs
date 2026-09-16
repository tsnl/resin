//! Inference variables, unification, and expression constraints.
//! All handles are resolved before the typed tree reaches LIR lowering.
use crate::lower::context::{Context, FunctionBody, FunctionDecl};
use crate::lower::scope::DeclarationId;
use crate::{GenerateError, GenerateErrorKind};
use resin_source::prelude::*;
use resin_types::prelude::*;
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

//
// Inference types
//

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Head {
    Operation {
        name: Arc<str>,
        candidates: Vec<FunctionId>,
        explicit: Option<usize>,
        primitive: Option<Arc<str>>,
    },
    Atom(Ty),
    Parameter {
        id: crate::TypeParameterId,
    },
    Member {
        name: Arc<str>,
    },
    Method {
        name: crate::MethodName,
        associated: bool,
    },
    FunctionParameter {
        index: usize,
    },
    FunctionResult,
    Nominal {
        definition: TypeId,
    },
    Pointer,
    Reference,
    Value,
    Array(usize),
    Record(Vec<Arc<str>>),
    // Result first, followed by the parameter types in declaration order.
    Function,
    Error,
    Union,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Type {
    Invalid,
    Variable(usize),
    Node(Head, Vec<Type>),
    /// Preserve a definition's unresolved weak variables under this call's substitution.
    Apply {
        body: Box<Type>,
        arguments: Vec<(crate::TypeParameterId, Type)>,
    },
}

impl Type {
    fn function_parameter(function: Self, index: usize) -> Self {
        Self::Node(Head::FunctionParameter { index }, vec![function])
    }

    fn function_result(function: Self) -> Self {
        Self::Node(Head::FunctionResult, vec![function])
    }

    /// The symbolic counterpart of `Ty::deref_target` for primitive pointers.
    pub fn deref_target(&self) -> Option<&Type> {
        match self {
            Self::Node(Head::Pointer, children) => children.first(),
            _ => None,
        }
    }

    pub fn view_element(&self) -> Option<Type> {
        match self {
            Self::Node(Head::Atom(Ty::Str), _) => Some(Ty::UInt8.into()),
            _ => None,
        }
    }

    pub fn index_element(&self) -> Option<Type> {
        match self {
            Self::Node(Head::Array(_), children) => children.first().cloned(),
            _ => self.view_element(),
        }
    }

    pub fn reference(referent: Type) -> Self {
        Self::Node(Head::Reference, vec![referent])
    }
    pub fn value(of: Type) -> Self {
        Self::Node(Head::Value, vec![of])
    }
    pub fn pointer(pointee: Type) -> Self {
        Self::Node(Head::Pointer, vec![pointee])
    }

    pub fn function(params: Vec<Type>, result: Type) -> Self {
        Self::Node(
            Head::Function,
            std::iter::once(result).chain(params).collect(),
        )
    }

    pub fn record(fields: Vec<(Arc<str>, Type)>) -> Self {
        let (names, types) = fields.into_iter().unzip();
        Self::Node(Head::Record(names), types)
    }
}

impl Type {
    pub fn from_hir(source: &crate::Type) -> Self {
        match source {
            crate::Type::Operation { lookup } => Self::Node(
                Head::Operation {
                    primitive: lookup.primitive.clone(),
                    name: lookup.name.clone(),
                    candidates: lookup.candidates.clone(),
                    explicit: lookup.type_args.as_ref().map(Vec::len),
                },
                lookup
                    .type_args
                    .iter()
                    .flatten()
                    .chain(&lookup.arguments)
                    .map(Self::from_hir)
                    .collect(),
            ),
            crate::Type::Type => Ty::Type.into(),
            crate::Type::Unit => Ty::Unit.into(),
            crate::Type::None => Ty::None.into(),
            crate::Type::Bool => Ty::Bool.into(),
            crate::Type::Int8 => Ty::Int8.into(),
            crate::Type::Int16 => Ty::Int16.into(),
            crate::Type::Int32 => Ty::Int32.into(),
            crate::Type::Int64 => Ty::Int64.into(),
            crate::Type::UInt8 => Ty::UInt8.into(),
            crate::Type::UInt16 => Ty::UInt16.into(),
            crate::Type::UInt32 => Ty::UInt32.into(),
            crate::Type::UInt64 => Ty::UInt64.into(),
            crate::Type::Float32 => Ty::Float32.into(),
            crate::Type::Float64 => Ty::Float64.into(),
            crate::Type::Str => Ty::Str.into(),
            crate::Type::GpuView => Ty::GpuView.into(),
            crate::Type::GpuPipelineContract => Ty::GpuPipelineContract.into(),
            crate::Type::GpuArguments => Ty::GpuArguments.into(),
            crate::Type::StrongOwner => Ty::StrongOwner.into(),
            crate::Type::WeakOwner => Ty::WeakOwner.into(),
            crate::Type::Foreign { name } => Ty::Foreign { name: name.clone() }.into(),
            crate::Type::Defined {
                definition,
                arguments,
            } if arguments.is_empty() => Ty::Defined {
                definition: *definition,
            }
            .into(),
            crate::Type::Defined {
                definition,
                arguments,
            } => Self::Node(
                Head::Nominal {
                    definition: *definition,
                },
                arguments.iter().map(Self::from_hir).collect(),
            ),
            crate::Type::Parameter { parameter } => {
                Self::Node(Head::Parameter { id: *parameter }, vec![])
            }
            crate::Type::Member { base, name } => Self::Node(
                Head::Member { name: name.clone() },
                vec![Self::from_hir(base)],
            ),
            crate::Type::Method { lookup } => Self::Node(
                Head::Method {
                    name: lookup.name.clone(),
                    associated: lookup.associated,
                },
                std::iter::once(&lookup.receiver)
                    .chain(&lookup.type_args)
                    .map(Self::from_hir)
                    .collect(),
            ),
            crate::Type::FunctionParameter { function, index } => {
                Self::function_parameter(Self::from_hir(function), *index)
            }
            crate::Type::FunctionResult { function } => {
                Self::function_result(Self::from_hir(function))
            }
            crate::Type::Reference { referent } => Self::reference(Self::from_hir(referent)),
            crate::Type::Value { of } => Self::value(Self::from_hir(of)),
            crate::Type::Pointer { pointee } => {
                Self::Node(Head::Pointer, vec![Self::from_hir(pointee)])
            }
            crate::Type::Function { params, result } => Self::function(
                params.iter().map(Self::from_hir).collect(),
                Self::from_hir(result),
            ),
            crate::Type::Error { payload } => {
                Self::Node(Head::Error, vec![Self::from_hir(payload)])
            }
            crate::Type::Array { element, length } => {
                Self::Node(Head::Array(*length), vec![Self::from_hir(element)])
            }
            crate::Type::Record { fields } => Self::record(
                fields
                    .iter()
                    .map(|f| (f.name.clone(), Self::from_hir(&f.ty)))
                    .collect(),
            ),
            crate::Type::Union { variants } => {
                let parts: Vec<_> = variants.iter().map(Self::from_hir).collect();
                let solver = Solver::default();
                if let Some(parts) = parts
                    .iter()
                    .map(|ty| solver.resolve(ty))
                    .collect::<Option<Vec<_>>>()
                {
                    Ty::union_of(parts).into()
                } else {
                    Self::Node(Head::Union, parts)
                }
            }
        }
    }
}

impl From<Ty> for Type {
    fn from(ty: Ty) -> Self {
        match ty {
            Ty::Pointer { pointee } => Self::pointer((*pointee).into()),
            Ty::Array { element, length } => {
                Self::Node(Head::Array(length), vec![(*element).into()])
            }
            Ty::Record { fields } => {
                Self::record(fields.into_iter().map(|f| (f.name, f.ty.into())).collect())
            }
            Ty::Function { params, result } => Self::function(
                params.into_iter().map(Self::from).collect(),
                (*result).into(),
            ),
            Ty::Error { payload } => Self::Node(Head::Error, vec![(*payload).into()]),
            atom => Self::Node(Head::Atom(atom), vec![]),
        }
    }
}

impl Head {
    /// These expressions determine a type without revealing its structural shape.
    fn determining(&self) -> bool {
        matches!(self, Self::Parameter { .. }) || self.projection()
    }

    /// A projection is not injective: its result cannot infer its input.
    fn projection(&self) -> bool {
        matches!(
            self,
            Self::Member { .. }
                | Self::Method { .. }
                | Self::Operation { .. }
                | Self::FunctionParameter { .. }
                | Self::FunctionResult
                | Self::Value
        )
    }

    pub fn concrete(&self, children: Vec<Ty>) -> Option<Ty> {
        let mut children = children.into_iter();
        Some(match self {
            Self::Parameter { .. }
            | Self::Member { .. }
            | Self::Method { .. }
            | Self::Operation { .. }
            | Self::FunctionParameter { .. }
            | Self::FunctionResult
            | Self::Nominal { .. }
            | Self::Reference
            | Self::Value => return None,
            Self::Atom(ty) => ty.clone(),
            Self::Union => Ty::union_of(children),
            Self::Pointer => Ty::Pointer {
                pointee: Box::new(children.next().unwrap()),
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
                result: Box::new(children.next().unwrap()),
                params: children.collect(),
            },
            Self::Error => Ty::Error {
                payload: Box::new(children.next().unwrap()),
            },
        })
    }
}

/// Substitution can reveal nested unions and equal members before a match is checked.
fn completed_union(members: Vec<crate::Type>) -> crate::Type {
    let mut variants = vec![];
    let mut pending = members;
    while let Some(member) = pending.pop() {
        match member {
            crate::Type::Union { variants } => pending.extend(variants),
            member => variants.push(member),
        }
    }
    variants.sort();
    variants.dedup();
    if variants.len() == 1 {
        variants.pop().unwrap()
    } else {
        crate::Type::Union { variants }
    }
}

fn completed_widens_to(source: &crate::Type, target: &crate::Type) -> bool {
    use crate::Type;
    if source == target {
        return true;
    }
    match (source, target) {
        (Type::Union { variants }, _) => variants.iter().all(|ty| completed_widens_to(ty, target)),
        (_, Type::Union { variants }) => variants.iter().any(|ty| completed_widens_to(source, ty)),
        (Type::Error { payload: from }, Type::Error { payload: to }) => {
            completed_widens_to(from, to)
        }
        _ => false,
    }
}

impl Head {
    fn completed(&self, children: Vec<crate::Type>) -> crate::Type {
        let mut children = children.into_iter();
        match self {
            Self::Operation {
                name,
                candidates,
                explicit,
                primitive,
            } => crate::Type::Operation {
                lookup: Box::new(crate::OperationLookup {
                    primitive: primitive.clone(),
                    name: name.clone(),
                    candidates: candidates.clone(),
                    type_args: explicit.map(|count| children.by_ref().take(count).collect()),
                    arguments: children.collect(),
                }),
            },
            Self::Atom(ty) => super::types::ty(ty),
            Self::Union => completed_union(children.collect()),
            Self::Parameter { id } => crate::Type::Parameter { parameter: *id },
            Self::Nominal { definition } => crate::Type::Defined {
                definition: *definition,
                arguments: children.collect(),
            },
            Self::Member { name } => crate::Type::Member {
                base: Box::new(children.next().unwrap()),
                name: name.clone(),
            },
            Self::Method { name, associated } => crate::Type::Method {
                lookup: Box::new(crate::MethodLookup {
                    receiver: children.next().unwrap(),
                    name: name.clone(),
                    type_args: children.collect(),
                    associated: *associated,
                }),
            },
            Self::FunctionParameter { index } => crate::Type::FunctionParameter {
                function: Box::new(children.next().unwrap()),
                index: *index,
            },
            Self::FunctionResult => crate::Type::FunctionResult {
                function: Box::new(children.next().unwrap()),
            },
            Self::Reference => crate::Type::Reference {
                referent: Box::new(children.next().unwrap()),
            },
            Self::Value => crate::Type::Value {
                of: Box::new(children.next().unwrap()),
            },
            Self::Pointer => crate::Type::Pointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::Array(length) => crate::Type::Array {
                element: Box::new(children.next().unwrap()),
                length: *length,
            },
            Self::Record(names) => crate::Type::Record {
                fields: names
                    .iter()
                    .cloned()
                    .zip(children)
                    .map(|(name, ty)| crate::RecordField { name, ty })
                    .collect(),
            },
            Self::Function => crate::Type::Function {
                result: Box::new(children.next().unwrap()),
                params: children.collect(),
            },
            Self::Error => crate::Type::Error {
                payload: Box::new(children.next().unwrap()),
            },
        }
    }
}

//
// Inference variables and unification
//

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    // Only an expression result may still turn out to be a reference. Ordinary
    // inference holes and generic value arguments always describe value types.
    Expression,
    Any,
    Number,
    Float,
    Errors,
}

#[derive(Clone)]
struct Variable {
    value: Option<Type>,
    class: Class,
    variants: Vec<Type>,
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
    pub fn apply(
        &mut self,
        signature: Type,
        parameters: &[crate::TypeParameter],
        explicit: Option<Vec<Type>>,
        span: Span,
    ) -> Result<(Type, Vec<Type>)> {
        let arguments =
            explicit.unwrap_or_else(|| parameters.iter().map(|_| self.fresh()).collect());
        if parameters.len() != arguments.len() {
            return Err(error(
                span,
                format!(
                    "expected {} type arguments, found {}",
                    parameters.len(),
                    arguments.len()
                ),
            ));
        }
        let substitution = parameters
            .iter()
            .zip(&arguments)
            .map(|(p, a)| (p.id, a.clone()))
            .collect();
        Ok((
            Type::Apply {
                body: Box::new(signature),
                arguments: substitution,
            },
            arguments,
        ))
    }

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
                if matches!(
                    self.variables[id].class,
                    Class::Any | Class::Expression | Class::Errors
                ) =>
            {
                self.variables[id].class = Class::Errors;
                Ok(())
            }
            Type::Apply { .. } => Ok(()),
            Type::Node(Head::Union, members) => {
                for member in members {
                    self.errors(&member, span)?;
                }
                Ok(())
            }
            Type::Node(head, _) if head != Head::Reference => Ok(()),
            _ => Err(error(span, "error payloads must be value types")),
        }
    }

    fn error_members(&self, ty: &Type, span: Span) -> Result<Option<Vec<Type>>> {
        match self.head(ty) {
            Type::Variable(id) if self.variables[id].class == Class::Errors => {
                Ok(Some(self.variables[id].variants.clone()))
            }
            Type::Variable(_) => Ok(None),
            Type::Apply { body, arguments } => {
                Ok(self.error_members(&body, span)?.map(|members| {
                    members
                        .into_iter()
                        .map(|body| {
                            self.head(&Type::Apply {
                                body: Box::new(body),
                                arguments: arguments.clone(),
                            })
                        })
                        .collect()
                }))
            }
            // A value read of an unresolved recursive call is an inclusion edge,
            // not a new payload variant. Preserve the error set's fixed point.
            Type::Node(Head::Value, parts) => self.error_members(&parts[0], span),
            Type::Node(Head::Atom(ty), _) => {
                Ok(Some(ty.members().into_iter().map(Type::from).collect()))
            }
            Type::Node(Head::Union, members) => {
                let mut result = vec![];
                for member in members {
                    let Some(members) = self.error_members(&member, span)? else {
                        return Ok(None);
                    };
                    result.extend(members);
                }
                Ok(Some(result))
            }
            Type::Node(head, children) if head != Head::Reference => {
                Ok(Some(vec![Type::Node(head, children)]))
            }
            _ => Err(error(span, "error payloads must be value types")),
        }
    }

    pub fn include(&mut self, from: &Type, to: &Type, span: Span) -> Result<bool> {
        self.errors(to, span)?;
        let Some(variants) = self.error_members(from, span)? else {
            return Ok(false);
        };
        if let Type::Variable(id) = self.head(to) {
            for variant in variants {
                if !self.variables[id].variants.contains(&variant) {
                    self.variables[id].variants.push(variant);
                    self.revision += 1;
                }
            }
            return Ok(false);
        }
        if let (Some(source), Some(target)) = (self.resolve(from), self.resolve(to)) {
            if !source.widens_to(&target) {
                return Err(error(
                    span,
                    "the destination error set does not include every propagated error",
                ));
            }
            return Ok(true);
        }
        // Propagation retains this ground inclusion in the error operations. It
        // does not equate independent error parameters or choose a new argument.
        Ok(self.complete(from).is_some() && self.complete(to).is_some())
    }

    pub fn union(&self, members: Vec<Type>) -> Type {
        if let Some(types) = members
            .iter()
            .map(|ty| self.resolve(ty))
            .collect::<Option<Vec<_>>>()
        {
            return Ty::union_of(types).into();
        }
        let mut unique = vec![];
        let mut pending = members;
        while let Some(member) = pending.pop() {
            match self.head(&member) {
                Type::Node(Head::Union, members) => pending.extend(members),
                member if !unique.contains(&member) => unique.push(member),
                _ => {}
            }
        }
        match unique.len() {
            0 => Ty::union([]).into(),
            1 => unique.pop().unwrap(),
            _ => Type::Node(Head::Union, unique),
        }
    }

    fn union_parts(&self, ty: &Type) -> Vec<Type> {
        let mut pending = vec![ty.clone()];
        let mut members = vec![];
        while let Some(ty) = pending.pop() {
            match self.head(&ty) {
                Type::Node(Head::Union, children) => pending.extend(children.into_iter().rev()),
                Type::Node(Head::Atom(ty @ Ty::Union { .. }), _) => {
                    pending.extend(ty.members().into_iter().rev().map(Type::from))
                }
                ty => members.push(ty),
            }
        }
        members
    }

    fn known_union_members(&self, ty: &Type) -> Option<Vec<Type>> {
        let members = self.union_parts(ty);
        for member in &members {
            match member {
                Type::Invalid | Type::Variable(_) | Type::Apply { .. } => return None,
                Type::Node(head, _) if head.projection() => return None,
                _ => {}
            }
        }
        Some(members)
    }

    pub fn coerce(&mut self, from: &Type, to: &Type, span: Span) -> Result<bool> {
        // Reference use is invariant. Every other consumer reads a value, even
        // when a dependent call's result will only be known at specialization.
        let value = Type::value(from.clone());
        if let Type::Node(Head::Reference, parts) = self.head(to) {
            return self.unify(&value, &parts[0], span);
        }
        let from = &value;
        if let Type::Variable(id) = self.head(to)
            && self.variables[id].class == Class::Errors
        {
            return self.include(from, to, span);
        }
        if let Type::Node(Head::Error, source) = self.head(from)
            && let Some(target) = self.error_target(to)
        {
            return self.coerce(&source[0], &target, span);
        }
        self.union_context(from, to, span)?;
        if matches!(self.head(to), Type::Node(Head::Union, _)) {
            if let Some(target) = self.resolve(to) {
                return self.coerce(from, &target.into(), span);
            }
            let (Some(source), Some(target)) = (self.complete(from), self.complete(to)) else {
                return Ok(false);
            };
            // Closed nominal applications can be compared without materializing
            // layouts. Projections and bound parameters remain for specialization.
            if !self.dependent(from)
                && !self.dependent(to)
                && !completed_widens_to(&source, &target)
            {
                return Err(error(
                    span,
                    "the value's type is not included in the destination union",
                ));
            }
            return Ok(true);
        }
        match (self.head(from), self.head(to)) {
            (Type::Node(Head::Record(a), aa), Type::Node(Head::Record(b), bb)) if a == b => {
                let mut complete = true;
                for (from, to) in aa.iter().zip(&bb) {
                    complete &= self.coerce(from, to, span)?;
                }
                Ok(complete)
            }
            (_, Type::Node(Head::Atom(target @ Ty::Union { .. }), _)) => {
                let Some(source) = self.resolve(from) else {
                    return Ok(self.complete(from).is_some());
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
            _ => self.unify(from, to, span),
        }
    }

    fn error_target(&self, to: &Type) -> Option<Type> {
        match self.head(to) {
            Type::Node(Head::Error, target) => Some(target[0].clone()),
            _ => {
                let members = self.union_parts(to);
                let errors: Vec<_> = members
                    .into_iter()
                    .filter_map(|member| match self.head(&member) {
                        Type::Node(Head::Error, parts) => Some(parts[0].clone()),
                        _ => None,
                    })
                    .collect();
                if errors.len() == 1 {
                    errors.into_iter().next()
                } else {
                    None
                }
            }
        }
    }

    fn union_context(&mut self, from: &Type, to: &Type, span: Span) -> Result<()> {
        let members = self.union_parts(to);
        self.infer_success_hole(from, &members, span)?;
        if let Type::Variable(id) = self.head(from) {
            let class = self.variables[id].class;
            if matches!(class, Class::Number | Class::Float) {
                let candidates: Vec<_> = members
                    .iter()
                    .filter_map(|member| self.resolve(member))
                    .filter(|ty| {
                        if class == Class::Float {
                            matches!(ty, Ty::Float32 | Ty::Float64)
                        } else {
                            ty.is_numeric()
                        }
                    })
                    .collect();
                if let [candidate] = candidates.as_slice() {
                    self.unify(from, &candidate.clone().into(), span)?;
                }
            }
        }
        if members
            .iter()
            .any(|member| matches!(self.head(member), Type::Node(Head::Error, _)))
            && let Some(source) = self.known_union_members(from)
            && source.len() > 1
        {
            for member in source {
                if matches!(self.head(&member), Type::Node(Head::Error, _)) {
                    self.coerce(&member, to, span)?;
                }
            }
        }
        Ok(())
    }

    fn infer_success_hole(&mut self, from: &Type, target: &[Type], span: Span) -> Result<()> {
        if !target
            .iter()
            .any(|ty| matches!(self.head(ty), Type::Node(Head::Error, _)))
        {
            return Ok(());
        }
        let successes: Vec<_> = target
            .iter()
            .filter(|ty| !matches!(self.head(ty), Type::Node(Head::Error, _)))
            .collect();
        let [success] = successes.as_slice() else {
            return Ok(());
        };
        if !matches!(self.head(success), Type::Variable(_)) {
            return Ok(());
        }
        let values: Vec<_> = self
            .union_parts(from)
            .into_iter()
            .filter(|ty| !matches!(self.head(ty), Type::Node(Head::Error, _)))
            .collect();
        if !values.is_empty() {
            self.unify(success, &self.union(values), span)?;
        }
        Ok(())
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
                Type::Apply { body, arguments } => {
                    collect(solver, &body, ids);
                    for (_, argument) in arguments {
                        collect(solver, &argument, ids);
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
            if self.variables[id].class == Class::Errors {
                self.variables[id].value = Some(self.union(self.variables[id].variants.clone()));
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
        if let Type::Node(Head::Value, parts) = ty {
            return match self.head(&parts[0]) {
                Type::Node(Head::Reference, parts) => self.head(&parts[0]),
                Type::Node(head, parts)
                    if !head.projection()
                        || matches!(head, Head::Member { .. } | Head::Method { .. }) =>
                {
                    Type::Node(head, parts)
                }
                Type::Variable(id) if self.variables[id].class != Class::Expression => {
                    Type::Variable(id)
                }
                Type::Invalid => Type::Invalid,
                _ => ty.clone(),
            };
        }
        let Type::Apply { body, arguments } = ty else {
            return ty.clone();
        };
        match self.head(body) {
            Type::Invalid => Type::Invalid,
            Type::Node(Head::Parameter { id }, _) => arguments
                .iter()
                .find(|(parameter, _)| *parameter == id)
                .map(|(_, argument)| self.head(argument))
                .unwrap_or_else(|| Type::Node(Head::Parameter { id }, vec![])),
            Type::Node(head, children) => Type::Node(
                head,
                children
                    .into_iter()
                    .map(|body| Type::Apply {
                        body: Box::new(body),
                        arguments: arguments.clone(),
                    })
                    .collect(),
            ),
            _ => ty.clone(),
        }
    }

    pub fn shape_hint(&self, ty: &Type) -> Type {
        let head = self.head(ty);
        if let Type::Node(Head::Value, parts) = &head {
            // Shape queries may inspect an unfinished error set or expression,
            // without equating the expression's eventual reference and value types.
            let shape = self.shape_hint(&parts[0]);
            return match shape {
                Type::Variable(_) | Type::Apply { .. } => shape,
                _ => self.head(&Type::value(shape)),
            };
        }
        if let Type::Apply { body, arguments } = &head {
            let hint = self.shape_hint(body);
            if &hint != body.as_ref() {
                return self.head(&Type::Apply {
                    body: Box::new(hint),
                    arguments: arguments.clone(),
                });
            }
        }
        if let Type::Variable(id) = head
            && self.variables[id].class == Class::Errors
            && !self.variables[id].variants.is_empty()
        {
            return self.union(self.variables[id].variants.clone());
        }
        head
    }

    pub fn resolve(&self, ty: &Type) -> Option<Ty> {
        match self.head(ty) {
            Type::Invalid | Type::Variable(_) | Type::Apply { .. } => None,
            Type::Node(head, args) => head.concrete(
                args.iter()
                    .map(|t| self.resolve(t))
                    .collect::<Option<_>>()?,
            ),
        }
    }

    pub fn complete(&self, ty: &Type) -> Option<crate::Type> {
        match self.head(ty) {
            Type::Invalid | Type::Variable(_) | Type::Apply { .. } => None,
            Type::Node(head, children) => Some(
                head.completed(
                    children
                        .iter()
                        .map(|child| self.complete(child))
                        .collect::<Option<_>>()?,
                ),
            ),
        }
    }

    fn dependent(&self, ty: &Type) -> bool {
        match self.head(ty) {
            Type::Apply { .. } => true,
            Type::Node(head, _) if head.determining() => true,
            Type::Node(_, children) => children.iter().any(|ty| self.dependent(ty)),
            _ => false,
        }
    }

    pub fn require_complete(&self, ty: &Type, span: Span) -> Result<crate::Type> {
        self.complete(ty)
            .ok_or_else(|| error(span, "cannot infer this type; add an explicit annotation"))
    }

    /// Alias substitution may duplicate structure without requesting a function.
    /// Bound that work independently of the LIR monomorph allowance.
    pub fn require_bounded(&self, ty: &Type, span: Span) -> Result<crate::Type> {
        self.complete_at(ty, span, 0, &mut 65536)
    }

    fn complete_at(
        &self,
        ty: &Type,
        span: Span,
        depth: usize,
        remaining: &mut usize,
    ) -> Result<crate::Type> {
        if depth >= 256 {
            return Err(error(
                span,
                "type expansion exceeds the HIR depth limit of 256",
            ));
        }
        if *remaining == 0 {
            return Err(error(
                span,
                "type expansion exceeds the HIR size limit of 65536",
            ));
        }
        *remaining -= 1;
        let Type::Node(head, children) = self.head(ty) else {
            return Err(error(
                span,
                "cannot infer this type; add an explicit annotation",
            ));
        };
        Ok(head.completed(
            children
                .iter()
                .map(|child| self.complete_at(child, span, depth + 1, remaining))
                .collect::<Result<_>>()?,
        ))
    }

    pub fn invalid(&self, ty: &Type) -> bool {
        match self.head(ty) {
            Type::Invalid => true,
            Type::Node(_, children) => children.iter().any(|ty| self.invalid(ty)),
            Type::Variable(_) => false,
            Type::Apply { body, arguments } => {
                self.invalid(&body) || arguments.iter().any(|(_, ty)| self.invalid(ty))
            }
        }
    }

    pub fn invalidate(&mut self, variable: VariableId) {
        self.variables[variable.0].value = Some(Type::Invalid);
    }

    pub fn require(&self, ty: &Type, span: Span) -> Result<Ty> {
        self.resolve(ty)
            .ok_or_else(|| error(span, "cannot infer this type; add an explicit annotation"))
    }

    pub fn unify(&mut self, left: &Type, right: &Type, span: Span) -> Result<bool> {
        let left = self.head(left);
        let right = self.head(right);
        if left == right {
            return Ok(true);
        }
        match (&left, &right) {
            (Type::Variable(id), _) => self.bind(*id, right, span),
            (_, Type::Variable(id)) => self.bind(*id, left, span),
            _ if [&left, &right].iter().any(|ty| matches!(ty, Type::Node(head, _) if head.projection() || matches!(head, Head::Union))) => {
                // Projections and unions are not injective. Consumers retain
                // ground relations without determining receivers or constituents.
                Ok(self.complete(&left).is_some() && self.complete(&right).is_some())
            }
            (Type::Node(a, aa), Type::Node(b, bb)) if a == b && aa.len() == bb.len() => {
                let mut complete = true;
                for (a, b) in aa.iter().zip(bb) {
                    complete &= self.unify(a, b, span)?;
                }
                Ok(complete)
            }
            (Type::Apply { .. }, _) | (_, Type::Apply { .. }) => Ok(false),
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
            (_, Type::Variable(_)) => self.unify(to, from, span).map(|_| ()),
            (Type::Node(a, aa), Type::Node(b, bb)) if a == b => {
                for (a, b) in aa.iter().zip(bb.iter()) {
                    self.cast(a, b, span)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn bind(&mut self, id: usize, ty: Type, span: Span) -> Result<bool> {
        if self.occurs(id, &ty) {
            if matches!(ty, Type::Apply { .. } | Type::Node(Head::Value, _)) {
                // T = Value<T> does not determine T. Recursive result holes
                // must wait for evidence from their own definition's body.
                return Ok(false);
            }
            return Err(error(span, "inference would create an infinite type"));
        }
        let class = self.variables[id].class;
        if let Type::Variable(other) = ty {
            let other_class = self.variables[other].class;
            if (class == Class::Errors
                && !matches!(other_class, Class::Any | Class::Expression | Class::Errors))
                || (other_class == Class::Errors
                    && !matches!(class, Class::Any | Class::Expression | Class::Errors))
            {
                return Err(error(span, "an error set cannot be a numeric type"));
            }
            self.variables[other].class = match (class, other_class) {
                (Class::Errors, _) | (_, Class::Errors) => Class::Errors,
                (Class::Float, _) | (_, Class::Float) => Class::Float,
                (Class::Number, _) | (_, Class::Number) => Class::Number,
                (Class::Expression, Class::Expression) => Class::Expression,
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
            for variant in self.variables[id].variants.clone() {
                self.include(&variant, &ty, span)?;
            }
        } else if matches!(class, Class::Number | Class::Float) && !self.dependent(&ty) {
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
        Ok(true)
    }

    fn occurs(&self, id: usize, ty: &Type) -> bool {
        match self.head(ty) {
            Type::Invalid => false,
            Type::Variable(other) => id == other,
            Type::Node(_, args) => args.iter().any(|arg| self.occurs(id, arg)),
            Type::Apply { body, arguments } => {
                self.occurs(id, &body) || arguments.iter().any(|(_, ty)| self.occurs(id, ty))
            }
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
                Class::Any | Class::Expression | Class::Errors => return,
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
        "size_of" | "sizeof" | "align_of" | "absurd" | "iota"
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Rule(usize);

#[derive(Clone)]
pub(crate) struct Equation {
    span: Span,
    owner: Rule,
    relation: Constraint,
}

#[derive(Clone, Debug)]
pub(crate) struct OverloadCandidate {
    pub function: FunctionId,
    pub declaration: DeclarationId,
    pub parameters: Vec<crate::TypeParameter>,
    pub signature: Type,
}

#[derive(Clone)]
pub(crate) struct AppliedMethod {
    pub declaration: DeclarationId,
    pub type_args: Vec<Type>,
    pub params: Vec<Type>,
    pub result: Type,
}

#[derive(Clone)]
pub(crate) enum ResolvedMethod {
    Operation {
        signature: Type,
    },
    Intrinsic {
        signature: super::context::IntrinsicMethod,
        receiver_conversion: Option<crate::ReceiverConversion>,
    },
    GpuPipeline {
        method: super::gpu::PipelineMethod,
    },
    Dependent {
        signature: Type,
    },
    Source {
        declaration: DeclarationId,
        type_args: Vec<Type>,
        params: Vec<Type>,
        result: Type,
        receiver_conversion: Option<crate::ReceiverConversion>,
    },
    Compiler {
        declaration: FunctionDecl,
    },
}

impl AppliedMethod {
    fn resolved(self, receiver_conversion: Option<crate::ReceiverConversion>) -> ResolvedMethod {
        ResolvedMethod::Source {
            declaration: self.declaration,
            type_args: self.type_args,
            params: self.params,
            result: self.result,
            receiver_conversion,
        }
    }
}

/// Constraint services injected into source checking; this layer never traverses expressions.
pub(crate) struct Inference<'a> {
    pub typer: &'a mut Context,
    pub solver: Solver,
    pub constraints: Vec<Equation>,
    owners: Vec<Vec<VariableId>>,
    // Choices belong to typing operations, not source spans; retries restore them with the solver.
    pub methods: BTreeMap<Rule, ResolvedMethod>,
    applications: BTreeMap<Rule, AppliedMethod>,
}
impl<'a> Inference<'a> {
    pub fn new(typer: &'a mut Context) -> Self {
        Self {
            typer,
            solver: Solver::default(),
            constraints: vec![],
            owners: vec![],
            methods: BTreeMap::new(),
            applications: BTreeMap::new(),
        }
    }
    fn rule(&mut self, variables: Vec<VariableId>) -> Rule {
        let rule = Rule(self.owners.len());
        self.owners.push(variables);
        rule
    }
    pub fn expression(&mut self) -> (Rule, Type) {
        self.expression_with_class(Class::Any)
    }
    pub fn reference_expression(&mut self) -> (Rule, Type) {
        self.expression_with_class(Class::Expression)
    }
    fn expression_with_class(&mut self, class: Class) -> (Rule, Type) {
        let variable = self.solver.variable(class);
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
}

//
// Constraint solving
//

#[derive(Clone)]
pub(crate) struct Overload {
    pub name: Arc<str>,
    pub candidates: Vec<OverloadCandidate>,
    pub primitive: Option<Arc<str>>,
    pub expected: Option<Type>,
    pub type_args: Option<Vec<Type>>,
    pub args: Option<Vec<Type>>,
    pub out: Type,
}

#[derive(Clone)]
pub(crate) enum Constraint {
    Overload {
        lookup: Overload,
    },
    Equal(Type, Type),
    Depends(Type),
    Coerce(Type, Type),
    ExcludeNone(Type, Type),
    Try(Type, Type, Type),
    Layout(Type),
    SizeOf(Type),
    Variant(Type, Pattern, Type),
    Boolean(Type),
    Deref(Type, Type),
    Address(Type, Type),
    Field(Type, Arc<str>, Type),
    Call(Type, Vec<Type>, Type),
    Method {
        receiver: Type,
        name: Arc<str>,
        type_args: Option<Vec<Type>>,
        args: Vec<Type>,
        out: Type,
        associated: bool,
    },
    MethodReference {
        receiver: Type,
        name: Arc<str>,
        type_args: Option<Vec<Type>>,
        out: Type,
    },
    Ascribe(Type, Type, bool),
    Record(Vec<(Arc<str>, Type)>, Type),
    Builtin(Arc<str>, Vec<Type>, Type),
}

#[derive(Clone)]
pub(crate) enum Pattern {
    Error,
    Type(Type),
}

impl Inference<'_> {
    /// Retry equations from a clean SCC snapshot after discarding a failed
    /// producer. Pending constraints may have mutated the solver on earlier
    /// attempts, so rolling back only the final attempt is insufficient.
    pub fn solve(&mut self, roots: &[Type]) -> Vec<GenerateError> {
        let baseline = self.solver.clone();
        let methods = self.methods.clone();
        let applications = self.applications.clone();
        let equations = std::mem::take(&mut self.constraints);
        let mut failed = Vec::new();
        let mut errors = Vec::new();
        'retry: loop {
            self.solver = baseline.clone();
            self.methods = methods.clone();
            self.applications = applications.clone();
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
                    match self.constraint(owner, &constraint, span) {
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
                    if let Constraint::Method { receiver, .. }
                    | Constraint::MethodReference { receiver, .. } = constraint
                    {
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
        if let Some(body) = self.typer.nominal_body(&ty, &self.solver) {
            ty = self.solver.shape_hint(&body);
        } else if let Type::Node(Head::Atom(t @ Ty::Defined { .. }), _) = &ty {
            ty = self
                .typer
                .body(t)
                .map_err(|e| GenerateError::typing(span, e))?
                .into();
        }
        Ok(ty)
    }

    fn nominal_ascription(&mut self, from: &Type, to: &Type, span: Span) -> Result<Option<bool>> {
        let source = self.solver.head(from);
        let target = self.solver.head(to);
        let source_body = self.typer.nominal_body(from, &self.solver);
        let target_body = self.typer.nominal_body(to, &self.solver);
        let application = matches!(source, Type::Node(Head::Nominal { .. }, _))
            || matches!(target, Type::Node(Head::Nominal { .. }, _));
        let symbolic_body = [source_body.as_ref(), target_body.as_ref()]
            .into_iter()
            .flatten()
            .any(|body| self.solver.resolve(body).is_none());
        let incomplete_view = [&source, &target].into_iter().any(|ty| {
            matches!(ty, Type::Node(Head::Atom(Ty::Defined { definition }), _)
                if self.typer.definition(*definition).is_ok_and(|definition| definition.body().is_none()))
        });
        if !application && !symbolic_body && !incomplete_view {
            return Ok(None);
        }
        let nominal = |ty: &Type| {
            matches!(
                ty,
                Type::Node(Head::Nominal { .. } | Head::Atom(Ty::Defined { .. }), _)
            )
        };
        let representation = match (&source, &target) {
            _ if nominal(&source) && nominal(&target) => {
                return self.solver.unify(from, to, span).map(Some);
            }
            (Type::Variable(_) | Type::Node(Head::Record(_) | Head::Atom(Ty::Unit), _), _)
                if nominal(&target) =>
            {
                target_body.map(|body| (from, body))
            }
            (_, Type::Node(Head::Record(_), _)) if nominal(&source) => {
                source_body.map(|body| (to, body))
            }
            _ => return Ok(None),
        };
        let Some((value, representation)) = representation else {
            return Ok(Some(false));
        };
        if self.solver.head(value) == Type::from(Ty::Unit)
            && matches!(self.solver.head(&representation), Type::Node(Head::Record(names), _) if names.is_empty())
        {
            return Ok(Some(true));
        }
        self.solver.unify(value, &representation, span).map(Some)
    }

    fn has_layout_bodies(&self, ty: &Ty) -> bool {
        let mut pending = vec![ty];
        let mut visited = HashSet::new();
        while let Some(ty) = pending.pop() {
            match ty {
                Ty::Defined { definition } if visited.insert(*definition) => {
                    let Some(body) = self
                        .typer
                        .definitions()
                        .get(definition.index())
                        .and_then(TypeDef::body)
                    else {
                        return false;
                    };
                    pending.push(body);
                }
                Ty::Record { fields } => pending.extend(fields.iter().map(|field| &field.ty)),
                Ty::Array { element, length } if *length != 0 => pending.push(element),
                _ => {}
            }
        }
        true
    }

    fn nominal_drop(&self, ty: &Type) -> Option<TypeId> {
        let Type::Node(Head::Nominal { definition } | Head::Atom(Ty::Defined { definition }), _) =
            self.solver.head(ty)
        else {
            return None;
        };
        self.typer
            .definition(definition)
            .ok()?
            .drop_hook()
            .map(|_| definition)
    }

    fn method_receiver(&self, receiver: &Type) -> Type {
        let mut receiver = self.solver.head(receiver);
        while let Some(pointee) = receiver.deref_target() {
            receiver = self.solver.head(pointee);
        }
        receiver
    }

    fn dependent_method(
        &self,
        receiver: &Type,
        name: &crate::MethodName,
        type_args: &Option<Vec<Type>>,
        associated: bool,
    ) -> Option<Type> {
        let Type::Node(head, _) = self.method_receiver(receiver) else {
            return None;
        };
        (head.determining() && self.solver.complete(receiver).is_some()).then(|| {
            Type::Node(
                Head::Method {
                    name: name.clone(),
                    associated,
                },
                std::iter::once(receiver.clone())
                    .chain(type_args.iter().flatten().cloned())
                    .collect(),
            )
        })
    }

    fn dependent_call(
        &mut self,
        function: &Type,
        args: &[Type],
        result: &Type,
        span: Span,
    ) -> Result<bool> {
        let mut arguments = true;
        for (index, arg) in args.iter().enumerate() {
            arguments &= self.solver.coerce(
                arg,
                &Type::function_parameter(function.clone(), index),
                span,
            )?;
        }
        let result = self
            .solver
            .unify(&Type::function_result(function.clone()), result, span)?;
        Ok(arguments && result)
    }

    fn method_application(
        &mut self,
        owner: Rule,
        receiver: &Type,
        name: &crate::MethodName,
        explicit: &Option<Vec<Type>>,
        span: Span,
    ) -> Result<Option<AppliedMethod>> {
        if let Some(application) = self.applications.get(&owner) {
            return Ok(Some(application.clone()));
        }
        let receiver = if matches!(name, crate::MethodName::Operator { .. }) {
            self.solver.head(receiver)
        } else {
            self.method_receiver(receiver)
        };
        let (definition, mut arguments) = match receiver {
            Type::Node(Head::Nominal { definition }, arguments) => (definition, arguments),
            Type::Node(Head::Atom(Ty::Defined { definition }), _) => (definition, vec![]),
            _ => return Ok(None),
        };
        let Some(method) = self.typer.source_method(definition, name).cloned() else {
            return Ok(None);
        };
        if arguments.len() != method.owner_params.len() {
            return Err(error(
                span,
                "method receiver has the wrong number of owner type arguments",
            ));
        }
        if let Some(explicit) = explicit {
            if explicit.len() != method.type_params.len() {
                return Err(error(
                    span,
                    format!(
                        "expected {} method type arguments, found {}",
                        method.type_params.len(),
                        explicit.len()
                    ),
                ));
            }
            arguments.extend(explicit.iter().cloned());
        } else {
            arguments.extend(method.type_params.iter().map(|_| self.solver.fresh()));
        }
        let parameters = method
            .owner_params
            .into_iter()
            .chain(method.type_params)
            .collect::<Vec<_>>();
        let signature = Type::function(method.params, method.result);
        let (signature, type_args) =
            self.solver
                .apply(signature, &parameters, Some(arguments), span)?;
        let Type::Node(Head::Function, parts) = self.solver.head(&signature) else {
            unreachable!("method signature");
        };
        let application = AppliedMethod {
            declaration: method.declaration,
            type_args,
            params: parts[1..].to_vec(),
            result: parts[0].clone(),
        };
        self.applications.insert(owner, application.clone());
        Ok(Some(application))
    }

    fn check_method_call(
        &mut self,
        constraint: &Constraint,
        params: &[Type],
        result: &Type,
        span: Span,
    ) -> Result<(bool, Option<crate::ReceiverConversion>)> {
        let Constraint::Method {
            receiver,
            args,
            out,
            associated,
            ..
        } = constraint
        else {
            unreachable!("method call constraint");
        };
        let offset = usize::from(!associated);
        let arguments = params.get(offset..).ok_or_else(|| {
            error(
                span,
                "method receiver does not match: this function has no receiver parameter",
            )
        })?;
        let arguments = self.arguments(args, arguments, span)?;
        let result = self.solver.unify(result, out, span)?;
        let conversion = if *associated {
            None
        } else {
            let Some(conversion) = self.source_receiver(receiver, &params[0], span)? else {
                return Ok((false, None));
            };
            Some(conversion)
        };
        Ok((arguments && result, conversion))
    }

    fn source_receiver(
        &mut self,
        from: &Type,
        to: &Type,
        span: Span,
    ) -> Result<Option<crate::ReceiverConversion>> {
        use crate::ReceiverConversion;
        let source = self.solver.head(from);
        let target = self.solver.head(to);
        if matches!(source, Type::Variable(_) | Type::Apply { .. })
            || matches!(target, Type::Apply { .. })
        {
            return Ok(None);
        }
        let (conversion, adapted) = match (&source, &target) {
            (_, Type::Node(Head::Reference, _)) => {
                (ReceiverConversion::Address, Type::reference(from.clone()))
            }
            (_, Type::Variable(_)) => (ReceiverConversion::Value, from.clone()),
            (Type::Node(a, _), Type::Node(b, _)) if a == b => {
                (ReceiverConversion::Value, from.clone())
            }
            (Type::Node(Head::Pointer, parts), _) => (ReceiverConversion::Load, parts[0].clone()),
            (_, Type::Node(Head::Pointer, _)) => {
                (ReceiverConversion::Address, Type::pointer(from.clone()))
            }
            _ => {
                return Err(error(
                    span,
                    "method receiver does not match the first parameter",
                ));
            }
        };
        if !self.solver.unify(&adapted, to, span)? {
            return Ok(None);
        }
        Ok(Some(conversion))
    }

    fn arguments(&mut self, args: &[Type], params: &[Type], span: Span) -> Result<bool> {
        argument_count(params.len(), args.len(), span)?;
        let mut complete = true;
        for (arg, param) in args.iter().zip(params) {
            complete &= self.solver.coerce(arg, param, span)?;
        }
        Ok(complete)
    }

    /// Select a source operator before applying primitive operand relationships.
    /// Literal variables remain on the primitive path so expected numeric types
    /// still flow into expressions such as `1 + 2`.
    fn operator(
        &mut self,
        owner: Rule,
        symbol: &Arc<str>,
        args: &[Type],
        out: &Type,
        span: Span,
    ) -> Result<Option<bool>> {
        let Some(name) = crate::MethodName::operator(symbol, args.len()) else {
            return Ok(None);
        };
        let receiver = self.solver.head(&args[0]);
        match &receiver {
            Type::Variable(id)
                if !matches!(
                    self.solver.variables[*id].class,
                    Class::Number | Class::Float
                ) =>
            {
                return Ok(Some(false));
            }
            Type::Apply { .. } => return Ok(Some(false)),
            Type::Node(Head::Nominal { .. } | Head::Atom(Ty::Defined { .. }), _) => {
                let method = self
                    .method_application(owner, &receiver, &name, &Some(vec![]), span)?
                    .ok_or_else(|| error(span, format!("type has no {name}")))?;
                let a = self.arguments(args, &method.params, span)?;
                let b = self.solver.unify(&method.result, out, span)?;
                if a && b {
                    self.methods.insert(owner, method.resolved(None));
                }
                return Ok(Some(a && b));
            }
            Type::Node(head, _) if head.determining() => {
                if self.solver.complete(&receiver).is_none() {
                    return Ok(Some(false));
                }
                let signature = Type::Node(
                    Head::Method {
                        name,
                        associated: true,
                    },
                    vec![receiver],
                );
                let complete = self.dependent_call(&signature, args, out, span)?;
                if complete {
                    self.methods
                        .insert(owner, ResolvedMethod::Dependent { signature });
                }
                return Ok(Some(complete));
            }
            _ => {}
        }
        Ok(None)
    }

    // GPU projection reads fields from a source aggregate without converting or
    // discarding its nominal identity. Only this boundary accepts matching shapes.
    fn projection_argument(
        &mut self,
        from: &Type,
        to: &Type,
        span: Span,
        depth: usize,
    ) -> Result<bool> {
        if depth >= 128 {
            return Err(error(
                span,
                "GPU argument projection exceeds the depth limit",
            ));
        }
        match (self.shape(from, false, span)?, self.solver.head(to)) {
            (
                Type::Node(Head::Record(names), fields),
                Type::Node(Head::Record(expected), targets),
            ) => {
                if names != expected {
                    return Err(error(
                        span,
                        "GPU argument fields must match the shader root's field names and order",
                    ));
                }
                let mut complete = true;
                for (field, target) in fields.iter().zip(targets) {
                    complete &= self.projection_argument(field, &target, span, depth + 1)?;
                }
                Ok(complete)
            }
            (
                Type::Node(Head::Array(length), fields),
                Type::Node(Head::Array(expected), targets),
            ) if length == expected => {
                self.projection_argument(&fields[0], &targets[0], span, depth + 1)
            }
            _ => self.solver.coerce(from, to, span),
        }
    }

    fn overload(&mut self, owner: Rule, lookup: &Overload, span: Span) -> Result<bool> {
        let Overload {
            name,
            candidates,
            primitive,
            expected,
            type_args: explicit,
            args,
            out,
        } = lookup;
        let args = args.as_deref();
        if candidates.is_empty()
            && let (Some(symbol), Some(args)) = (primitive, args)
            && !super::context::is_primitive_operation(symbol)
        {
            return self.constraint(
                owner,
                &Constraint::Builtin(symbol.clone(), args.to_vec(), out.clone()),
                span,
            );
        }
        if let Some(application) = self.applications.get(&owner).cloned() {
            return self.complete_overload(owner, application, args, out, span);
        }
        if primitive
            .as_ref()
            .is_some_and(|name| super::context::is_primitive_operation(name))
            && args.and_then(|args| args.first()).is_some_and(|arg| {
                matches!(
                    self.solver.shape_hint(arg),
                    Type::Variable(_) | Type::Apply { .. }
                )
            })
        {
            // A source candidate cannot win while a primitive candidate's
            // receiver shape is still unknown (for example record.values).
            return Ok(false);
        }
        if let [candidate] = candidates.as_slice()
            && primitive.is_none()
            && !self
                .typer
                .functions
                .get(&candidate.function)
                .is_some_and(|function| {
                    matches!(
                        function.body,
                        FunctionBody::GpuPipelineFactory { .. }
                            | FunctionBody::GpuPipelineRecord { .. }
                    )
                })
        {
            // A single compatible generic signature already determines its
            // result's shape. Retain that information (for example Ptr<T>)
            // instead of hiding it behind a dependent operation projection.
            let baseline = self.solver.clone();
            if let Ok(application) = self.match_overload(candidate, lookup, span) {
                self.applications.insert(owner, application.clone());
                return self.complete_overload(owner, application, args, out, span);
            }
            self.solver = baseline;
        }
        if let Some(args) = args
            && args.iter().any(|arg| self.solver.dependent(arg))
        {
            if args
                .iter()
                .chain(explicit.iter().flatten())
                .any(|arg| self.solver.complete(arg).is_none())
            {
                return Ok(false);
            }
            let signature = Type::Node(
                Head::Operation {
                    primitive: primitive.clone(),
                    name: name.clone(),
                    candidates: candidates
                        .iter()
                        .map(|candidate| candidate.function)
                        .collect(),
                    explicit: explicit.as_ref().map(Vec::len),
                },
                explicit.iter().flatten().chain(args).cloned().collect(),
            );
            let complete = self.dependent_call(&signature, args, out, span)?;
            if complete {
                self.methods
                    .insert(owner, ResolvedMethod::Operation { signature });
            }
            return Ok(complete);
        }
        let baseline = self.solver.clone();
        let mut viable = Vec::new();
        let mut rejected = Vec::new();
        for candidate in candidates {
            self.solver = baseline.clone();
            if let Some(method) = self.typer.functions.get(&candidate.function).cloned()
                && matches!(
                    method.body,
                    FunctionBody::GpuPipelineFactory { .. }
                        | FunctionBody::GpuPipelineRecord { .. }
                )
            {
                match self.pipeline_overload(&method, lookup, span) {
                    Ok(Some((mut method, complete))) => {
                        method.declaration = Some(candidate.declaration);
                        viable.push((
                            None,
                            Some(ResolvedMethod::GpuPipeline { method }),
                            self.solver.clone(),
                            complete,
                        ));
                    }
                    Ok(None) => {}
                    Err(error) => rejected.push(error),
                }
                continue;
            }
            let matched = self.match_overload(candidate, lookup, span);
            match matched {
                Ok(application) => {
                    viable.push((Some(application), None, self.solver.clone(), true))
                }
                Err(error) => rejected.push(error),
            }
        }
        self.solver = baseline.clone();
        if primitive.is_some()
            && args.is_some()
            && let Ok((operation, complete)) = self.primitive_overload(owner, lookup, span)
        {
            let context = expected
                .as_ref()
                .map_or(Ok(true), |expected| self.solver.coerce(out, expected, span));
            if let Ok(context) = context {
                viable.push((None, operation, self.solver.clone(), complete && context));
            }
        }
        self.solver = baseline;
        if viable.len() == 1 {
            let (application, operation, solver, complete) = viable.pop().unwrap();
            self.solver = solver;
            let Some(application) = application else {
                if complete && let Some(operation) = operation {
                    self.methods.insert(owner, operation);
                }
                return Ok(complete);
            };
            self.applications.insert(owner, application.clone());
            return self.complete_overload(owner, application, args, out, span);
        }
        let known = args.map_or_else(
            || self.solver.complete(out).is_some(),
            |args| args.iter().all(|arg| self.solver.complete(arg).is_some()),
        );
        if viable.is_empty()
            && candidates.len() == 1
            && primitive.is_none()
            && let Some(error) = rejected.pop()
        {
            return Err(error);
        }
        if !known || !viable.is_empty() {
            return Ok(false);
        }
        Err(error(
            span,
            if viable.is_empty() {
                format!("no overload of `{name}` matches this signature")
            } else {
                format!(
                    "ambiguous overload of `{name}`: {} signatures match",
                    viable.len()
                )
            },
        ))
    }

    fn match_overload(
        &mut self,
        candidate: &OverloadCandidate,
        lookup: &Overload,
        span: Span,
    ) -> Result<AppliedMethod> {
        let (signature, type_args) = self.solver.apply(
            candidate.signature.clone(),
            &candidate.parameters,
            lookup.type_args.clone(),
            span,
        )?;
        let Type::Node(Head::Function, parts) = self.solver.head(&signature) else {
            return Err(error(span, "overload candidate is not a function"));
        };
        let application = AppliedMethod {
            declaration: candidate.declaration,
            type_args,
            params: parts[1..].to_vec(),
            result: parts[0].clone(),
        };
        if let Some(args) = &lookup.args {
            self.arguments(args, &application.params, span)?;
            self.solver.unify(&application.result, &lookup.out, span)?;
            if let Some(expected) = &lookup.expected {
                self.solver.coerce(&application.result, expected, span)?;
            }
        } else {
            self.solver.coerce(&signature, &lookup.out, span)?;
        }
        Ok(application)
    }

    fn pipeline_overload(
        &mut self,
        bridge: &FunctionDecl,
        lookup: &Overload,
        span: Span,
    ) -> Result<Option<(super::gpu::PipelineMethod, bool)>> {
        let args = lookup
            .args
            .as_ref()
            .ok_or_else(|| error(span, "GPU bridges require a direct call"))?;
        if lookup.type_args.is_some() {
            return Err(error(
                span,
                "GPU bridge arguments are inferred from shader declarations",
            ));
        }
        argument_count(bridge.params.len(), args.len(), span)?;
        let needed = if matches!(bridge.body, FunctionBody::GpuPipelineFactory { .. }) {
            args.len() - 1
        } else {
            1
        };
        let Some(inputs) = args[1..]
            .iter()
            .take(needed)
            .map(|ty| self.solver.complete(&Type::value(ty.clone())))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let mut method = self
            .typer
            .source_pipeline_method(bridge, &inputs)
            .map_err(|message| error(span, message))?;
        let projection =
            matches!(method.body, FunctionBody::GpuPipelineDispatch { .. }).then_some(2);
        let mut complete = true;
        for (index, (arg, param)) in args.iter().zip(&method.params).enumerate() {
            let param = Type::from_hir(param);
            complete &= if projection == Some(index) {
                self.projection_argument(arg, &param, span, 0)?
            } else {
                self.solver.coerce(arg, &param, span)?
            };
        }
        complete &= self
            .solver
            .unify(&Type::from_hir(&method.result), &lookup.out, span)?;
        if let Some(expected) = &lookup.expected {
            complete &= self
                .solver
                .coerce(&Type::from_hir(&method.result), expected, span)?;
        }
        if complete && let Some(index) = projection {
            method.params[index] = self
                .solver
                .require_complete(&Type::value(args[index].clone()), span)?;
        }
        Ok(Some((method, complete)))
    }

    fn primitive_overload(
        &mut self,
        owner: Rule,
        lookup: &Overload,
        span: Span,
    ) -> Result<(Option<ResolvedMethod>, bool)> {
        let name = lookup.primitive.as_ref().expect("primitive candidate");
        let args = lookup.args.as_ref().expect("primitive operands");
        if !super::context::is_primitive_operation(name) {
            let complete = self.constraint(
                owner,
                &Constraint::Builtin(name.clone(), args.clone(), lookup.out.clone()),
                span,
            )?;
            return Ok((None, complete));
        }
        if lookup.type_args.is_some() {
            return Err(error(
                span,
                "primitive operations infer their type arguments",
            ));
        }
        let signature = super::context::primitive_operation(name, args, &self.solver)
            .ok_or_else(|| error(span, "no primitive operation matches these operands"))?;
        let arguments = self.arguments(args, &signature.params, span)?;
        let result = self.solver.unify(&signature.result, &lookup.out, span)?;
        Ok((
            Some(ResolvedMethod::Intrinsic {
                signature,
                receiver_conversion: None,
            }),
            arguments && result,
        ))
    }

    fn complete_overload(
        &mut self,
        owner: Rule,
        application: AppliedMethod,
        args: Option<&[Type]>,
        out: &Type,
        span: Span,
    ) -> Result<bool> {
        for argument in &application.type_args {
            super::eval::reference_type(&self.solver, argument, false, span)?;
        }
        let complete = if let Some(args) = args {
            let arguments = self.arguments(args, &application.params, span)?;
            self.solver.unify(&application.result, out, span)? && arguments
        } else {
            self.solver.coerce(
                &Type::function(application.params.clone(), application.result.clone()),
                out,
                span,
            )?
        };
        if complete {
            self.methods.insert(owner, application.resolved(None));
        }
        Ok(complete)
    }

    fn constraint(&mut self, owner: Rule, constraint: &Constraint, span: Span) -> Result<bool> {
        match constraint {
            Constraint::Overload { lookup } => return self.overload(owner, lookup, span),
            Constraint::Method {
                receiver: receiver_type,
                name,
                type_args,
                args,
                out,
                associated,
            } => {
                if matches!(
                    self.method_receiver(receiver_type),
                    Type::Variable(_) | Type::Apply { .. }
                ) {
                    return Ok(false);
                }
                if let Some(receiver) = self.solver.resolve(receiver_type)
                    && let Some(method) = self.typer.method(&receiver, name)
                    && matches!(
                        method.body,
                        FunctionBody::GpuPipelineFactory { .. }
                            | FunctionBody::GpuPipelineRecord { .. }
                    )
                {
                    if !associated
                        && crate::ReceiverConversion::between(&receiver, &method.params[0])
                            .is_none()
                    {
                        return Err(error(
                            span,
                            "pipeline method receiver does not match its native bridge",
                        ));
                    }
                    argument_count(
                        method.params.len() - usize::from(!associated),
                        args.len(),
                        span,
                    )?;
                    let inputs = &args[usize::from(*associated)..];
                    let needed = if matches!(method.body, FunctionBody::GpuPipelineFactory { .. }) {
                        inputs.len()
                    } else {
                        1
                    };
                    let Some(inputs) = inputs
                        .iter()
                        .take(needed)
                        .map(|ty| self.solver.complete(ty))
                        .collect::<Option<Vec<_>>>()
                    else {
                        return Ok(false);
                    };
                    let mut method = self
                        .typer
                        .source_pipeline_method(&method, &inputs)
                        .map_err(|message| error(span, message))?;
                    let params = method.params[usize::from(!associated)..]
                        .iter()
                        .map(Type::from_hir)
                        .collect::<Vec<_>>();
                    let projection =
                        matches!(method.body, FunctionBody::GpuPipelineDispatch { .. })
                            .then_some(2 - usize::from(!associated));
                    let mut a = true;
                    for (index, (arg, param)) in args.iter().zip(&params).enumerate() {
                        a &= if projection == Some(index) {
                            self.projection_argument(arg, param, span, 0)?
                        } else {
                            self.solver.coerce(arg, param, span)?
                        };
                    }
                    let b = self
                        .solver
                        .coerce(&Type::from_hir(&method.result), out, span)?;
                    if a && b {
                        if let Some(index) = projection {
                            method.params[2] = self
                                .solver
                                .require_complete(&Type::value(args[index].clone()), span)?;
                        }
                        self.methods
                            .insert(owner, ResolvedMethod::GpuPipeline { method });
                    }
                    return Ok(a && b);
                }
                if let Some(application) = self.method_application(
                    owner,
                    receiver_type,
                    &name.clone().into(),
                    type_args,
                    span,
                )? {
                    let (complete, conversion) = self.check_method_call(
                        constraint,
                        &application.params,
                        &application.result,
                        span,
                    )?;
                    if complete {
                        self.methods.insert(owner, application.resolved(conversion));
                    }
                    return Ok(complete);
                }
                if let Some((_, signature)) =
                    super::context::intrinsic_methods(receiver_type, &self.solver)
                        .into_iter()
                        .find(|(candidate, _)| *candidate == name.as_ref())
                {
                    if type_args.is_some() {
                        return Err(error(span, "compiler methods do not accept type arguments"));
                    }
                    let (complete, receiver_conversion) = self.check_method_call(
                        constraint,
                        &signature.params,
                        &signature.result,
                        span,
                    )?;
                    if complete {
                        self.methods.insert(
                            owner,
                            ResolvedMethod::Intrinsic {
                                signature,
                                receiver_conversion,
                            },
                        );
                    }
                    return Ok(complete);
                }
                if let Some(signature) = self.dependent_method(
                    receiver_type,
                    &name.clone().into(),
                    type_args,
                    *associated,
                ) {
                    let complete = self.dependent_call(&signature, args, out, span)?;
                    if complete {
                        self.methods
                            .insert(owner, ResolvedMethod::Dependent { signature });
                    }
                    return Ok(complete);
                }
                if type_args.is_some() {
                    return Err(error(span, "compiler methods do not accept type arguments"));
                }
                let Some(receiver_type) = self.solver.resolve(receiver_type) else {
                    if matches!(
                        self.solver.head(receiver_type),
                        Type::Node(Head::Nominal { .. }, _)
                    ) {
                        return Err(error(span, format!("unknown method `{name}`")));
                    }
                    return Ok(false);
                };
                let method = self
                    .typer
                    .method(&receiver_type, name)
                    .ok_or_else(|| error(span, format!("unknown method `{name}`")))?;
                let params = method
                    .arguments(&receiver_type, *associated)
                    .ok_or_else(|| {
                        error(span, "method receiver does not match the first parameter")
                    })?;
                let params = params.iter().cloned().map(Type::from).collect::<Vec<_>>();
                let a = self.arguments(args, &params, span)?;
                let b = self
                    .solver
                    .coerce(&method.result.clone().into(), out, span)?;
                if a && b {
                    self.methods.insert(
                        owner,
                        ResolvedMethod::Compiler {
                            declaration: method,
                        },
                    );
                }
                return Ok(a && b);
            }
            Constraint::MethodReference {
                receiver,
                name,
                type_args,
                out,
            } => {
                if matches!(
                    self.method_receiver(receiver),
                    Type::Variable(_) | Type::Apply { .. }
                ) {
                    return Ok(false);
                }
                if let Some(signature) =
                    self.dependent_method(receiver, &name.clone().into(), type_args, true)
                {
                    let complete = self.solver.coerce(&signature, out, span)?;
                    if complete {
                        self.methods
                            .insert(owner, ResolvedMethod::Dependent { signature });
                    }
                    return Ok(complete);
                }
                let Some(application) = self.method_application(
                    owner,
                    receiver,
                    &name.clone().into(),
                    type_args,
                    span,
                )?
                else {
                    return Err(error(
                        span,
                        "only source methods can be referenced as function values",
                    ));
                };
                let signature =
                    Type::function(application.params.clone(), application.result.clone());
                let complete = self.solver.coerce(&signature, out, span)?;
                if complete {
                    self.methods.insert(owner, application.resolved(None));
                }
                return Ok(complete);
            }
            Constraint::Depends(_) => {}
            Constraint::Equal(from, to) => {
                if !self.solver.invalid(to) {
                    return self.solver.unify(from, to, span);
                }
            }
            Constraint::Layout(ty) | Constraint::SizeOf(ty) => {
                let Some(ty) = self.solver.resolve(ty) else {
                    return Ok(self.solver.complete(ty).is_some());
                };
                if !self.has_layout_bodies(&ty) {
                    return Ok(true);
                }
                let layout = if matches!(constraint, Constraint::SizeOf(_)) {
                    resin_types::layout::value(self.typer.definitions(), &ty)
                } else {
                    resin_types::layout::layout(self.typer.definitions(), &ty)
                };
                layout.map_err(|e| error(span, e.to_string()))?;
            }
            Constraint::Try(input, out, result) => {
                let Some(members) = self.solver.known_union_members(input) else {
                    return Ok(false);
                };
                let mut values = vec![];
                let mut errors = vec![];
                for member in members {
                    if matches!(self.solver.head(&member), Type::Node(Head::Error, _)) {
                        errors.push(member);
                    } else {
                        values.push(member);
                    }
                }
                if errors.is_empty() {
                    return Err(error(span, "postfix ? requires a type containing Err"));
                }
                if self.solver.invalid(result) {
                    return self.solver.unify(out, &self.solver.union(values), span);
                }
                // Propagation determines an error union, but not its success type.
                // Keep that type open for the function's ordinary return values.
                if matches!(self.solver.head(result), Type::Variable(_)) {
                    let success = self.solver.fresh();
                    let payload = self.solver.fresh();
                    self.solver.errors(&payload, span)?;
                    let result_shape = self
                        .solver
                        .union(vec![success, Type::Node(Head::Error, vec![payload])]);
                    self.solver.unify(result, &result_shape, span)?;
                }
                if let Some(target) = self.solver.known_union_members(result)
                    && !target
                        .iter()
                        .any(|ty| matches!(self.solver.head(ty), Type::Node(Head::Error, _)))
                {
                    return Err(error(
                        span,
                        "postfix ? requires a return type containing Err",
                    ));
                }
                let mut complete = self.solver.unify(out, &self.solver.union(values), span)?;
                for propagated in errors {
                    if let (Some(source), Some(target)) = (
                        self.solver.resolve(&propagated),
                        self.solver.resolve(result),
                    ) && !source.widens_to(&target)
                    {
                        return Err(error(
                            span,
                            "the return type does not include every propagated error",
                        ));
                    }
                    complete &= self.solver.coerce(&propagated, result, span)?;
                }
                return Ok(complete);
            }
            Constraint::ExcludeNone(input, out) => {
                let Some(members) = self.solver.known_union_members(input) else {
                    return Ok(false);
                };
                let none = Type::from(Ty::None);
                if !members.contains(&none) {
                    return Err(error(span, "postfix ! requires a type containing None"));
                }
                let remaining = self.solver.union(
                    members
                        .into_iter()
                        .filter(|member| member != &none)
                        .collect(),
                );
                if !self.solver.unify(out, &remaining, span)? {
                    return Ok(false);
                }
            }
            Constraint::Coerce(from, to) => {
                if !self.solver.invalid(to) {
                    return self.solver.coerce(from, to, span);
                }
            }
            Constraint::Variant(input, variant, out) => match (self.solver.head(input), variant) {
                (Type::Variable(_) | Type::Apply { .. }, _) => return Ok(false),
                (_, Pattern::Error) => {
                    let Some(members) = self.solver.known_union_members(input) else {
                        return Ok(false);
                    };
                    let payloads: Vec<_> = members
                        .into_iter()
                        .filter_map(|member| match self.solver.head(&member) {
                            Type::Node(Head::Error, parts) => Some(parts[0].clone()),
                            _ => None,
                        })
                        .collect();
                    if payloads.is_empty() {
                        return Err(error(span, "Err pattern requires an error member"));
                    }
                    return self.solver.unify(out, &self.solver.union(payloads), span);
                }
                (_, Pattern::Type(ty)) => return self.solver.unify(out, ty, span),
            },
            Constraint::Boolean(input) => {
                let Some(ty) = self.solver.resolve(input) else {
                    return Ok(self.solver.complete(input).is_some());
                };
                self.typer
                    .as_bool(&ty)
                    .map_err(|e| GenerateError::typing(span, e))?;
            }

            Constraint::Address(pointee, out) => {
                let pointer = Type::pointer(pointee.clone());
                if !self.solver.unify(out, &pointer, span)? {
                    return Ok(false);
                }
            }
            Constraint::Deref(input, out) => {
                let shape = self.shape(input, false, span)?;
                if matches!(shape, Type::Variable(_) | Type::Apply { .. }) {
                    return Ok(false);
                }
                let pointee = shape
                    .deref_target()
                    .ok_or_else(|| error(span, "dereference requires a pointer"))?;
                if !self.solver.unify(out, pointee, span)? {
                    return Ok(false);
                }
            }
            Constraint::Field(input, name, out) => {
                let shape = self.shape(input, true, span)?;
                if let Some(element) = shape.view_element() {
                    let ty = match name.as_ref() {
                        "data" => Type::pointer(element),
                        "length" => Ty::UInt64.into(),
                        _ => return Err(error(span, "unknown string or span field")),
                    };
                    if !self.solver.unify(out, &ty, span)? {
                        return Ok(false);
                    }
                    return Ok(true);
                }

                match shape {
                    Type::Variable(_) | Type::Apply { .. } => return Ok(false),
                    Type::Node(head, _) if head.determining() => {
                        let member =
                            Type::Node(Head::Member { name: name.clone() }, vec![input.clone()]);
                        return self.solver.unify(out, &member, span);
                    }
                    Type::Node(Head::Record(names), children) => {
                        let index = names
                            .iter()
                            .position(|n| n == name)
                            .ok_or_else(|| error(span, format!("unknown field `{name}`")))?;
                        if !self.solver.unify(out, &children[index], span)? {
                            return Ok(false);
                        }
                    }
                    _ => return Err(error(span, "field access requires a record")),
                }
            }
            Constraint::Call(func, args, out) => {
                let shape = self.shape(func, false, span)?;
                if let Some(element) = shape.index_element() {
                    let reference = Type::reference(element);
                    if !self.solver.unify(out, &reference, span)? {
                        return Ok(false);
                    }
                    argument_count(1, args.len(), span)?;
                    let Some(index) = self.solver.resolve(&args[0]) else {
                        return Ok(false);
                    };
                    if !index.is_integer() {
                        return Err(error(span, "index must be an integer"));
                    }
                    return Ok(true);
                }

                match shape {
                    Type::Variable(_) | Type::Apply { .. } => return Ok(false),
                    Type::Node(head, _) if head.determining() => {
                        return self.dependent_call(func, args, out, span);
                    }
                    Type::Node(Head::Function, children) => {
                        let a = self.arguments(args, &children[1..], span)?;
                        let b = self.solver.unify(&children[0], out, span)?;
                        return Ok(a && b);
                    }
                    _ => return Err(error(span, "call requires a function")),
                }
            }
            Constraint::Record(fields, out) => {
                let unique: HashSet<_> = fields.iter().map(|(name, _)| name).collect();
                if unique.len() != fields.len() {
                    return Err(error(span, "record fields do not match the expected type"));
                }
                if matches!(self.solver.head(out), Type::Node(head, _) if head.determining()) {
                    let mut complete = true;
                    for (name, ty) in fields {
                        let member =
                            Type::Node(Head::Member { name: name.clone() }, vec![out.clone()]);
                        complete &= self.solver.coerce(ty, &member, span)?;
                    }
                    return Ok(complete);
                }
                let Type::Node(Head::Record(names), types) = self.solver.head(out) else {
                    if matches!(self.solver.head(out), Type::Variable(_)) {
                        return Ok(false);
                    }
                    let record = Type::record(fields.clone());
                    if self.solver.complete(&record).is_none() {
                        return Ok(false);
                    }
                    if !self.solver.unify(&record, out, span)? {
                        return Ok(false);
                    }
                    unreachable!("a record cannot equal a non-record");
                };
                if fields.len() != names.len() {
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
                if matches!(
                    self.solver.head(to),
                    Type::Node(Head::Union | Head::Atom(Ty::Union { .. }), _)
                ) {
                    return self.solver.coerce(from, to, span);
                }
                if let Type::Node(Head::Error, parts) = self.solver.head(to)
                    && !matches!(self.solver.head(from), Type::Node(Head::Error, _))
                {
                    return self.solver.coerce(from, &parts[0], span);
                }
                if let Type::Node(Head::Error, parts) = self.solver.head(from)
                    && !matches!(self.solver.head(to), Type::Node(Head::Error, _))
                {
                    return self.solver.unify(&parts[0], to, span);
                }
                if matches!(self.solver.head(to), Type::Node(Head::Record(_), _))
                    && let Some(definition) = self.nominal_drop(from)
                {
                    return Err(GenerateError::typing(
                        span,
                        TypeError {
                            kind: TypeErrorKind::UnwrapManaged { definition },
                        },
                    ));
                }
                if let Some(complete) = self.nominal_ascription(from, to, span)? {
                    return Ok(complete);
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
                        (_, Type::Variable(_)) => {
                            self.solver.unify(to, from, span)?;
                        }
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
                            if !self.solver.unify(from, &context, span)? {
                                return Ok(false);
                            }
                        }
                        (Type::Node(Head::Pointer, _), Type::Node(Head::Pointer, _)) => {
                            self.solver.cast(from, to, span)?
                        }
                        _ => {}
                    }
                    return Ok(
                        self.solver.complete(from).is_some() && self.solver.complete(to).is_some()
                    );
                }
            }
            Constraint::Builtin(name, args, out) => {
                if let Some(complete) = self.operator(owner, name, args, out, span)? {
                    return Ok(complete);
                }
                use resin_types::BuiltinRule;
                let rule = BuiltinRule::lookup(name, args.len())
                    .map_err(|e| GenerateError::typing(span, e))?;
                let complete = match rule {
                    BuiltinRule::Repr | BuiltinRule::Format | BuiltinRule::StringFromBytes => {
                        self.solver.unify(out, &Ty::StrongOwner.into(), span)?
                    }
                    BuiltinRule::Assert => {
                        let complete = self.solver.unify(out, &Ty::Unit.into(), span)?;
                        if !self.constraint(owner, &Constraint::Boolean(args[0].clone()), span)? {
                            return Ok(false);
                        }
                        complete
                    }
                    BuiltinRule::Float => self.solver.unify(out, &args[0], span)?,
                    BuiltinRule::Boolean => {
                        let complete = self.solver.unify(out, &Ty::Bool.into(), span)?;
                        for arg in args {
                            if !self.constraint(owner, &Constraint::Boolean(arg.clone()), span)? {
                                return Ok(false);
                            }
                        }
                        complete
                    }
                    BuiltinRule::Arithmetic | BuiltinRule::Comparison => {
                        let mut complete = if rule == BuiltinRule::Comparison {
                            self.solver.unify(out, &Ty::Bool.into(), span)?
                        } else {
                            self.solver.unify(out, &args[0], span)?
                        };
                        for arg in &args[1..] {
                            complete &= self.solver.unify(&args[0], arg, span)?;
                        }
                        complete
                    }
                };
                if !complete {
                    return Ok(false);
                }
                let Some(args) = args
                    .iter()
                    .map(|t| self.solver.resolve(t))
                    .collect::<Option<Vec<_>>>()
                else {
                    return Ok(args.iter().all(|ty| self.solver.complete(ty).is_some()));
                };
                let call = self
                    .typer
                    .type_builtin_call(name, &args)
                    .map_err(|e| GenerateError::typing(span, e))?;
                if !self.solver.unify(out, &call.result.into(), span)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

impl Constraint {
    fn inputs(&self) -> Vec<&Type> {
        match self {
            Self::Overload {
                lookup: Overload {
                    args, type_args, ..
                },
            } => args
                .iter()
                .flatten()
                .chain(type_args.iter().flatten())
                .collect(),
            Self::Depends(from)
            | Self::ExcludeNone(from, _)
            | Self::Equal(from, _)
            | Self::Coerce(from, _)
            | Self::Try(from, _, _)
            | Self::Layout(from)
            | Self::SizeOf(from)
            | Self::Boolean(from)
            | Self::Deref(from, _)
            | Self::Field(from, _, _)
            | Self::Ascribe(from, _, _)
            | Self::Variant(from, _, _) => vec![from],
            Self::Call(func, args, _) => std::iter::once(func).chain(args).collect(),
            Self::Method {
                receiver,
                args,
                type_args,
                ..
            } => std::iter::once(receiver)
                .chain(args)
                .chain(type_args.iter().flatten())
                .collect(),
            Self::MethodReference {
                receiver,
                type_args,
                ..
            } => std::iter::once(receiver)
                .chain(type_args.iter().flatten())
                .collect(),
            Self::Address(pointee, _) => vec![pointee],
            Self::Record(fields, _) => fields.iter().map(|(_, ty)| ty).collect(),
            Self::Builtin(_, args, _) => args.iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests;

fn argument_count(expected: usize, found: usize, span: Span) -> Result<()> {
    if expected == found {
        Ok(())
    } else {
        Err(error(
            span,
            format!(
                "expected {expected} argument{}, found {found}",
                if expected == 1 { "" } else { "s" }
            ),
        ))
    }
}
