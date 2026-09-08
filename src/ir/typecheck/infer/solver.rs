use crate::{
    ast::Span,
    ir::{Ty, TypeError, TypeErrorKind, TypeId},
};

use super::{
    GenerateError, Result, error,
    types::{Head, Type},
};

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
pub(in crate::ir) struct VariableId(usize);

impl VariableId {
    pub fn ty(self) -> Type {
        Type::Variable(self.0)
    }
}

#[derive(Clone, Default)]
pub(in crate::ir) struct Solver {
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
        if let (_, Some(ty)) = crate::ir::literal::split(text) {
            return ty.into();
        }
        self.variable(if crate::ir::literal::unsuffixed_type(text).is_integer() {
            Class::Number
        } else {
            Class::Float
        })
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
                        Ty::Int32
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
                Class::Number => Ty::Int32,
                Class::Float => Ty::Float64,
            };
            variable.value = Some(ty.into());
            self.revision += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const SPAN: Span = Span { start: 0, end: 1 };

    #[test]
    fn nested_variables_are_independent() {
        let mut solver = Solver::default();
        let a = solver.fresh();
        let b = solver.fresh();
        let ty = Type::function(Type::pointer(a.clone()), Type::pointer(b.clone()));
        let concrete = Type::function(
            Type::pointer(Ty::Int32.into()),
            Type::pointer(Ty::Bool.into()),
        );
        solver.unify(&ty, &concrete, SPAN).unwrap();
        assert_eq!(solver.resolve(&a), Some(Ty::Int32));
        assert_eq!(solver.resolve(&b), Some(Ty::Bool));
    }

    #[test]
    fn cycles_and_ambiguity_are_errors_not_unit() {
        let mut solver = Solver::default();
        let a = solver.fresh();
        let b = solver.fresh();
        solver.unify(&a, &Type::pointer(b.clone()), SPAN).unwrap();
        assert!(solver.unify(&a, &b, SPAN).is_err());
        assert!(solver.require(&a, SPAN).is_err());
    }

    #[test]
    fn numeric_defaults_wait_for_context() {
        let mut solver = Solver::default();
        let literal = solver.number("1");
        let other = solver.number("1.0");
        let hex = solver.number("-0xdead");
        solver.unify(&literal, &Ty::UInt64.into(), SPAN).unwrap();
        solver.default_numbers(&[literal.clone(), other.clone(), hex.clone()]);
        assert_eq!(solver.resolve(&literal), Some(Ty::UInt64));
        assert_eq!(solver.resolve(&other), Some(Ty::Float64));
        assert_eq!(solver.resolve(&hex), Some(Ty::Int32));
    }
}
