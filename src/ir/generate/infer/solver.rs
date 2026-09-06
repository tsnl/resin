use crate::{ast::Span, ir::Ty};

use super::{
    Result, error,
    types::{Head, Type},
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Any,
    Number,
    Float,
}

struct Variable {
    value: Option<Type>,
    class: Class,
}

#[derive(Default)]
pub(super) struct Solver {
    variables: Vec<Variable>,
    pub revision: usize,
}

impl Solver {
    pub fn fresh(&mut self) -> Type {
        self.variable(Class::Any)
    }

    pub fn number(&mut self, text: &str) -> Type {
        let hex = text.starts_with("0x") || text.starts_with("0X");
        let float = !hex && (text.contains('.') || text.contains(['e', 'E']));
        self.variable(if float { Class::Float } else { Class::Number })
    }

    fn variable(&mut self, class: Class) -> Type {
        let id = self.variables.len();
        self.variables.push(Variable { value: None, class });
        Type::Variable(id)
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

    pub fn resolve(&self, ty: &Type) -> Option<Ty> {
        match self.head(ty) {
            Type::Variable(_) => None,
            Type::Node(head, args) => Some(
                head.concrete(
                    args.iter()
                        .map(|t| self.resolve(t))
                        .collect::<Option<_>>()?,
                ),
            ),
        }
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
            _ => Err(error(
                span,
                format!("incompatible inferred types: {left:?} and {right:?}"),
            )),
        }
    }

    fn bind(&mut self, id: usize, ty: Type, span: Span) -> Result<()> {
        if self.occurs(id, &ty) {
            return Err(error(span, "inference would create an infinite type"));
        }
        let class = self.variables[id].class;
        if let Type::Variable(other) = ty {
            let other_class = self.variables[other].class;
            self.variables[other].class = match (class, other_class) {
                (Class::Float, _) | (_, Class::Float) => Class::Float,
                (Class::Number, _) | (_, Class::Number) => Class::Number,
                _ => Class::Any,
            };
        } else if class != Class::Any {
            let numeric = matches!(&ty, Type::Node(Head::Atom(t), _) if t.is_numeric());
            let float = matches!(&ty, Type::Node(Head::Atom(Ty::Float32 | Ty::Float64), _));
            if !numeric || (class == Class::Float && !float) {
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
            Type::Variable(other) => id == other,
            Type::Node(_, args) => args.iter().any(|arg| self.occurs(id, arg)),
        }
    }

    pub fn default_numbers(&mut self) -> bool {
        let before = self.revision;
        for variable in &mut self.variables {
            if variable.value.is_none() {
                let ty = match variable.class {
                    Class::Any => continue,
                    Class::Number => Ty::Int32,
                    Class::Float => Ty::Float64,
                };
                variable.value = Some(ty.into());
                self.revision += 1;
            }
        }
        self.revision != before
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
        solver.unify(&literal, &Ty::UInt64.into(), SPAN).unwrap();
        solver.default_numbers();
        assert_eq!(solver.resolve(&literal), Some(Ty::UInt64));
        assert_eq!(solver.resolve(&other), Some(Ty::Float64));
    }
}
