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
    Atom(Ty),
    Parameter { id: crate::TypeParameterId },
    Member { name: Arc<str> },
    Method { name: Arc<str>, associated: bool },
    FunctionParameter { index: usize },
    FunctionResult,
    Nominal { definition: TypeId },
    Pointer,
    GpuPointer,
    GpuSpan,
    GpuComputePipeline,
    GpuGraphicsPipeline,
    Array(usize),
    Record(Vec<Arc<str>>),
    // Result first, followed by the parameter types in declaration order.
    Function,
    Result,
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

    /// The inference representation of Ty::deref_target; Weak is not dereferenceable.
    pub fn deref_target(&self) -> Option<&Type> {
        match self {
            Self::Node(Head::Pointer | Head::GpuPointer, children) => children.first(),
            _ => None,
        }
    }

    pub fn view_element(&self) -> Option<Type> {
        match self {
            Self::Node(Head::Atom(Ty::Str), _) => Some(Ty::UInt8.into()),
            Self::Node(Head::GpuSpan, children) => children.first().cloned(),
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
            crate::Type::Pointer { pointee } => {
                Self::Node(Head::Pointer, vec![Self::from_hir(pointee)])
            }
            crate::Type::GpuPointer { pointee } => {
                Self::Node(Head::GpuPointer, vec![Self::from_hir(pointee)])
            }
            crate::Type::GpuSpan { element } => {
                Self::Node(Head::GpuSpan, vec![Self::from_hir(element)])
            }
            crate::Type::GpuComputePipeline { root, owner } => Self::Node(
                Head::GpuComputePipeline,
                vec![Self::from_hir(root), Self::from_hir(owner)],
            ),
            crate::Type::GpuGraphicsPipeline { root, owner } => Self::Node(
                Head::GpuGraphicsPipeline,
                vec![Self::from_hir(root), Self::from_hir(owner)],
            ),
            crate::Type::Function { params, result } => Self::function(
                params.iter().map(Self::from_hir).collect(),
                Self::from_hir(result),
            ),
            crate::Type::Result { value, error } => Self::Node(
                Head::Result,
                vec![Self::from_hir(value), Self::from_hir(error)],
            ),
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
            Ty::GpuPointer { pointee } => Self::gpu_pointer((*pointee).into()),
            Ty::GpuSpan { element } => Self::Node(Head::GpuSpan, vec![(*element).into()]),
            Ty::GpuComputePipeline { root, owner } => Self::Node(
                Head::GpuComputePipeline,
                vec![(*root).into(), (*owner).into()],
            ),
            Ty::GpuGraphicsPipeline { root, owner } => Self::Node(
                Head::GpuGraphicsPipeline,
                vec![(*root).into(), (*owner).into()],
            ),
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
            Ty::Result { value, error } => Self::result((*value).into(), (*error).into()),
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
                | Self::FunctionParameter { .. }
                | Self::FunctionResult
        )
    }

    pub fn concrete(&self, children: Vec<Ty>) -> Option<Ty> {
        let mut children = children.into_iter();
        Some(match self {
            Self::Parameter { .. }
            | Self::Member { .. }
            | Self::Method { .. }
            | Self::FunctionParameter { .. }
            | Self::FunctionResult
            | Self::Nominal { .. } => return None,
            Self::Atom(ty) => ty.clone(),
            Self::Union => Ty::union_of(children),
            Self::Pointer => Ty::Pointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuPointer => Ty::GpuPointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuSpan => Ty::GpuSpan {
                element: Box::new(children.next().unwrap()),
            },
            Self::GpuComputePipeline => Ty::GpuComputePipeline {
                root: Box::new(children.next().unwrap()),
                owner: Box::new(children.next().unwrap()),
            },
            Self::GpuGraphicsPipeline => Ty::GpuGraphicsPipeline {
                root: Box::new(children.next().unwrap()),
                owner: Box::new(children.next().unwrap()),
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
            Self::Result => Ty::Result {
                value: Box::new(children.next().unwrap()),
                error: Box::new(children.next().unwrap()),
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

impl Head {
    fn completed(&self, children: Vec<crate::Type>) -> crate::Type {
        let mut children = children.into_iter();
        match self {
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
            Self::Pointer => crate::Type::Pointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuPointer => crate::Type::GpuPointer {
                pointee: Box::new(children.next().unwrap()),
            },
            Self::GpuSpan => crate::Type::GpuSpan {
                element: Box::new(children.next().unwrap()),
            },
            Self::GpuComputePipeline => crate::Type::GpuComputePipeline {
                root: Box::new(children.next().unwrap()),
                owner: Box::new(children.next().unwrap()),
            },
            Self::GpuGraphicsPipeline => crate::Type::GpuGraphicsPipeline {
                root: Box::new(children.next().unwrap()),
                owner: Box::new(children.next().unwrap()),
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
            Self::Result => crate::Type::Result {
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
                if matches!(self.variables[id].class, Class::Any | Class::Errors) =>
            {
                self.variables[id].class = Class::Errors;
                Ok(())
            }
            Type::Node(Head::Atom(ty), _) if ty.variants().is_some() => Ok(()),
            Type::Node(head, _) if head.determining() || matches!(head, Head::Nominal { .. }) => {
                Ok(())
            }
            Type::Apply { .. } => Ok(()),
            Type::Node(Head::Union, members) => {
                for member in members {
                    self.errors(&member, span)?;
                }
                Ok(())
            }
            _ => Err(error(
                span,
                "Result errors must be structs or unions of structs",
            )),
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
            Type::Node(Head::Atom(ty), _) => ty
                .variants()
                .map(|ids| {
                    Some(
                        ids.into_iter()
                            .map(|definition| Ty::Defined { definition }.into())
                            .collect(),
                    )
                })
                .ok_or_else(|| error(span, "error and union payloads must be nominal structs")),
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
            Type::Node(head, children)
                if head.determining() || matches!(head, Head::Nominal { .. }) =>
            {
                Ok(Some(vec![Type::Node(head, children)]))
            }
            _ => Err(error(
                span,
                "error and union payloads must be nominal structs",
            )),
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
        // Propagation retains this ground inclusion in the Result operations. It
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

    fn known_union_members(&self, ty: &Type) -> Option<Vec<Type>> {
        let mut pending = vec![ty.clone()];
        let mut members = vec![];
        while let Some(ty) = pending.pop() {
            match self.head(&ty) {
                Type::Node(Head::Union, children) => pending.extend(children.into_iter().rev()),
                Type::Node(Head::Atom(ty @ Ty::Union { .. }), _) => {
                    pending.extend(ty.members().into_iter().rev().map(Type::from));
                }
                Type::Invalid | Type::Variable(_) | Type::Apply { .. } => return None,
                Type::Node(head, _) if head.determining() => return None,
                ty => members.push(ty),
            }
        }
        Some(members)
    }

    pub fn coerce(&mut self, from: &Type, to: &Type, span: Span) -> Result<bool> {
        if matches!(self.head(to), Type::Node(Head::Union, _)) {
            if let Some(target) = self.resolve(to) {
                return self.coerce(from, &target.into(), span);
            }
            // A union does not determine its constituent arguments. Keep the
            // ground widening operation for specialization once both sides finish.
            return Ok(self.complete(from).is_some() && self.complete(to).is_some());
        }
        match (self.head(from), self.head(to)) {
            (Type::Node(Head::Record(a), aa), Type::Node(Head::Record(b), bb)) if a == b => {
                let mut complete = true;
                for (from, to) in aa.iter().zip(&bb) {
                    complete &= self.coerce(from, to, span)?;
                }
                Ok(complete)
            }
            (Type::Node(Head::Result, a), Type::Node(Head::Result, b)) => {
                let value = self.unify(&a[0], &b[0], span)?;
                Ok(self.include(&a[1], &b[1], span)? && value)
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
            if matches!(ty, Type::Apply { .. }) {
                return Ok(false);
            }
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
            for variant in self.variables[id].variants.clone() {
                self.include(&variant, &ty, span)?;
            }
        } else if class != Class::Any && !self.dependent(&ty) {
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
        "ok" | "err" | "size_of" | "align_of" | "absurd"
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

#[derive(Clone)]
pub(crate) struct AppliedMethod {
    pub declaration: DeclarationId,
    pub type_args: Vec<Type>,
    pub params: Vec<Type>,
    pub result: Type,
}

#[derive(Clone)]
pub(crate) enum ResolvedMethod {
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
    Call(Type, Vec<Type>, Type),
    Method {
        receiver: Type,
        name: Arc<str>,
        type_args: Option<Vec<Type>>,
        args: Vec<Type>,
        out: Type,
        associated: bool,
        origins: Vec<AddressOrigin>,
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

    fn require_gpu_element(&self, element: &Ty, span: Span) -> Result<()> {
        if element.gpu_element(self.typer.definitions()) {
            return Ok(());
        }
        Err(error(
            span,
            "GPU elements require a shared host/device layout without pointers, owners, or drop hooks",
        ))
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

    fn pipeline_method(
        &self,
        method: FunctionDecl,
        args: &[Type],
        associated: bool,
        span: Span,
    ) -> Result<Option<FunctionDecl>> {
        let factory = match method.body {
            FunctionBody::GpuPipelineFactory { .. } => true,
            FunctionBody::GpuPipelineRecord { .. } => false,
            _ => return Ok(Some(method)),
        };
        let count = method.params.len() - usize::from(!associated);
        argument_count(count, args.len(), span)?;
        let inputs = args;
        let inputs = &inputs[usize::from(associated)..];
        let needed = if factory { inputs.len() } else { 1 };
        let Some(arguments) = inputs
            .iter()
            .take(needed)
            .map(|ty| self.solver.resolve(ty))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        self.typer
            .specialize_gpu_method(method, &arguments)
            .map(Some)
            .map_err(|message| error(span, message))
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
        name: &Arc<str>,
        type_args: &Option<Vec<Type>>,
        associated: bool,
    ) -> Option<Type> {
        let Type::Node(head, _) = self.method_receiver(receiver) else {
            return None;
        };
        head.determining().then(|| {
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
            .coerce(&Type::function_result(function.clone()), result, span)?;
        Ok(arguments && result)
    }

    fn method_application(
        &mut self,
        owner: Rule,
        receiver: &Type,
        name: &str,
        explicit: &Option<Vec<Type>>,
        span: Span,
    ) -> Result<Option<AppliedMethod>> {
        if let Some(application) = self.applications.get(&owner) {
            return Ok(Some(application.clone()));
        }
        let (definition, mut arguments) = match self.method_receiver(receiver) {
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

    fn source_method_call(
        &mut self,
        owner: Rule,
        application: AppliedMethod,
        constraint: &Constraint,
        span: Span,
    ) -> Result<bool> {
        let Constraint::Method {
            receiver,
            args,
            out,
            associated,
            origins,
            ..
        } = constraint
        else {
            unreachable!("source method call constraint");
        };
        let offset = usize::from(!associated);
        let params = application.params.get(offset..).ok_or_else(|| {
            error(
                span,
                "method receiver does not match: this function has no receiver parameter",
            )
        })?;
        let arguments = self.arguments(args, params, span)?;
        let result = self.solver.coerce(&application.result, out, span)?;
        let conversion = if *associated {
            None
        } else {
            let Some(conversion) =
                self.source_receiver(receiver, &application.params[0], origins, span)?
            else {
                return Ok(false);
            };
            Some(conversion)
        };
        if arguments && result {
            self.methods.insert(owner, application.resolved(conversion));
        }
        Ok(arguments && result)
    }

    fn source_receiver(
        &mut self,
        from: &Type,
        to: &Type,
        origins: &[AddressOrigin],
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
            (_, Type::Variable(_)) => (ReceiverConversion::Value, from.clone()),
            (Type::Node(a, _), Type::Node(b, _)) if a == b => {
                (ReceiverConversion::Value, from.clone())
            }
            (Type::Node(Head::Pointer | Head::GpuPointer, parts), _) => {
                (ReceiverConversion::Load, parts[0].clone())
            }
            (_, Type::Node(Head::Pointer | Head::GpuPointer, _)) => {
                let Some(gpu) = self.gpu_address(origins) else {
                    return Ok(None);
                };
                if gpu != matches!(target, Type::Node(Head::GpuPointer, _)) {
                    return Err(error(
                        span,
                        if gpu {
                            "GPU storage requires a GpuPtr receiver; it cannot be borrowed as a raw Ptr"
                        } else {
                            "a GpuPtr receiver requires an address in GPU storage"
                        },
                    ));
                }
                (
                    ReceiverConversion::Address,
                    if gpu {
                        Type::gpu_pointer(from.clone())
                    } else {
                        Type::pointer(from.clone())
                    },
                )
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

    fn constraint(&mut self, owner: Rule, constraint: &Constraint, span: Span) -> Result<bool> {
        match constraint {
            Constraint::Method {
                receiver: receiver_type,
                name,
                type_args,
                args,
                out,
                associated,
                origins,
            } => {
                if matches!(
                    self.method_receiver(receiver_type),
                    Type::Variable(_) | Type::Apply { .. }
                ) {
                    return Ok(false);
                }
                let allocation = !associated
                    && name.as_ref() == "new"
                    && type_args.is_none()
                    && self
                        .solver
                        .resolve(receiver_type)
                        .and_then(|receiver| self.typer.gpu_allocator(&receiver))
                        .is_some();
                if !allocation
                    && let Some(application) =
                        self.method_application(owner, receiver_type, name, type_args, span)?
                {
                    return self.source_method_call(owner, application, constraint, span);
                }
                if let Some(signature) =
                    self.dependent_method(receiver_type, name, type_args, *associated)
                {
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
                if !associated
                    && name.as_ref() == "new"
                    && let Some(allocator) = self.typer.gpu_allocator(&receiver_type)
                {
                    argument_count(1, args.len(), span)?;
                    let arg = &args[0];
                    let result = Type::result(
                        Type::gpu_pointer(arg.clone()),
                        self.typer.gpu_error(allocator).into(),
                    );
                    self.solver.coerce(&result, out, span)?;
                    if self.nominal_drop(arg).is_some() {
                        return Err(error(span, "GPU elements cannot have drop hooks"));
                    }
                    let Some(element) = self.solver.resolve(arg) else {
                        return Ok(false);
                    };
                    self.require_gpu_element(&element, span)?;
                    let method = self
                        .typer
                        .method_call(&receiver_type, name, &[element], false)
                        .expect("registered GPU allocator");
                    self.methods.insert(
                        owner,
                        ResolvedMethod::Compiler {
                            declaration: method,
                        },
                    );
                    return Ok(true);
                }
                let method = if *associated
                    && matches!(
                        (&receiver_type, name.as_ref()),
                        (Ty::GpuPointer { .. }, "new") | (Ty::GpuSpan { .. }, "allocate")
                    ) {
                    argument_count(2, args.len(), span)?;
                    let parts = args;
                    let Some(gpu) = parts.first().and_then(|ty| self.solver.resolve(ty)) else {
                        return Ok(false);
                    };
                    let element = match &receiver_type {
                        Ty::GpuPointer { pointee } => &**pointee,
                        Ty::GpuSpan { element } => &**element,
                        _ => unreachable!(),
                    };
                    self.require_gpu_element(element, span)?;
                    let argument = vec![gpu, element.clone()];
                    self.typer
                        .method_call(&receiver_type, name, &argument, true)
                } else if *associated
                    && name.as_ref() == "allocate_native"
                    && receiver_type
                        == (Ty::GpuPointer {
                            pointee: Box::new(Ty::UInt8),
                        })
                {
                    argument_count(5, args.len(), span)?;
                    let parts = args;
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
                    let argument = vec![
                        prefix[0].clone(),
                        prefix[1].clone(),
                        Ty::UInt64,
                        Ty::UInt64,
                        Ty::Int32,
                    ];
                    self.typer
                        .method_call(&receiver_type, name, &argument, true)
                } else {
                    self.typer.method(&receiver_type, name)
                }
                .ok_or_else(|| error(span, format!("unknown method `{name}`")))?;
                let Some(method) = self.pipeline_method(method, args, *associated, span)? else {
                    return Ok(false);
                };
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
                if let Some(signature) = self.dependent_method(receiver, name, type_args, true) {
                    let complete = self.solver.coerce(&signature, out, span)?;
                    if complete {
                        self.methods
                            .insert(owner, ResolvedMethod::Dependent { signature });
                    }
                    return Ok(complete);
                }
                let Some(application) =
                    self.method_application(owner, receiver, name, type_args, span)?
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
            Constraint::Layout(ty) => {
                let Some(ty) = self.solver.resolve(ty) else {
                    return Ok(self.solver.complete(ty).is_some());
                };
                if !self.has_layout_bodies(&ty) {
                    return Ok(true);
                }
                resin_types::layout::layout(self.typer.definitions(), &ty)
                    .map_err(|e| error(span, e.to_string()))?;
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
            Constraint::Errors(from, to) => {
                if !self.solver.invalid(to) {
                    return self.solver.include(from, to, span);
                }
            }
            Constraint::Variant(input, variant, out) => match (self.solver.head(input), variant) {
                (Type::Variable(_) | Type::Apply { .. }, _) => return Ok(false),
                (Type::Node(Head::Result, parts), Pattern::Ok) => {
                    return self.solver.unify(out, &parts[0], span);
                }
                (Type::Node(Head::Result, parts), Pattern::Err) => {
                    return self.solver.unify(out, &parts[1], span);
                }
                (_, Pattern::Type(ty)) => return self.solver.unify(out, ty, span),
                _ => return Err(error(span, "match pattern does not belong to this type")),
            },
            Constraint::Boolean(input) => {
                let Some(ty) = self.solver.resolve(input) else {
                    return Ok(self.solver.complete(input).is_some());
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
                if let Type::Node(Head::Function, _) = &shape
                    && name.as_ref() == "spirv"
                {
                    if !self.solver.unify(out, &Ty::shader().into(), span)? {
                        return Ok(false);
                    }
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
                    let pointer =
                        if matches!(shape, Type::Node(Head::GpuPointer | Head::GpuSpan, _)) {
                            Type::gpu_pointer(element)
                        } else {
                            Type::pointer(element)
                        };
                    if !self.solver.unify(out, &pointer, span)? {
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
                        let b = self.solver.coerce(&children[0], out, span)?;
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
                if matches!(self.solver.head(to), Type::Node(Head::Result, _)) {
                    return self.solver.coerce(from, to, span);
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
                use resin_types::BuiltinRule;
                let rule = BuiltinRule::lookup(name, args.len())
                    .map_err(|e| GenerateError::typing(span, e))?;
                let complete = match rule {
                    BuiltinRule::Format | BuiltinRule::StringFromBytes => {
                        self.solver.unify(out, &Ty::StrongOwner.into(), span)?
                    }
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
            Self::Call(func, args, _) => std::iter::once(func).chain(args).collect(),
            Self::Method {
                receiver,
                args,
                type_args,
                origins,
                ..
            } => origins
                .iter()
                .map(AddressOrigin::input)
                .chain(std::iter::once(receiver))
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
