use std::sync::Arc;

use crate::ir::{RecordField, Ty};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ir) enum Head {
    Atom(Ty),
    Pointer,
    Arc,
    Weak,
    Span,
    Array(usize),
    Record(Vec<Arc<str>>),
    Function,
    Result,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(in crate::ir) enum Type {
    Invalid,
    Variable(usize),
    Node(Head, Vec<Type>),
}

impl Type {
    /// The inference representation of Ty::deref_target; Weak is not dereferenceable.
    pub fn deref_target(&self) -> Option<&Type> {
        match self {
            Self::Node(Head::Pointer | Head::Arc, children) => children.first(),
            _ => None,
        }
    }

    pub fn result(value: Type, error: Type) -> Self {
        Self::Node(Head::Result, vec![value, error])
    }
    pub fn pointer(pointee: Type) -> Self {
        Self::Node(Head::Pointer, vec![pointee])
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
