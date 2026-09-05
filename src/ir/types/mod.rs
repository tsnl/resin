//! Resolved monomorphic types used by the initial IR.

use std::sync::Arc;

use crate::util::define_id;

pub(crate) mod definitions;

define_id! {
    pub struct TypeId(usize);
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeDef {
    pub name: Arc<str>,
    pub(super) body: Option<Ty>,
}

impl TypeDef {
    pub fn new(name: impl Into<Arc<str>>, body: Ty) -> Self {
        Self {
            name: name.into(),
            body: Some(body),
        }
    }
    pub fn body(&self) -> Option<&Ty> {
        self.body.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecordField {
    pub name: Arc<str>,
    pub ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Type,
    Unit,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Foreign { name: Arc<str> },
    Defined { definition: TypeId },
    Pointer { pointee: Box<Ty> },
    Span { element: Box<Ty> },
    Array { element: Box<Ty>, length: usize },
    Record { fields: Vec<RecordField> },
    Function { param: Box<Ty>, result: Box<Ty> },
}

impl Ty {
    pub fn shader() -> Self {
        Self::Record { fields: vec![
            RecordField { name: "data".into(), ty: Self::Pointer { pointee: Box::new(Self::UInt8) } },
            RecordField { name: "length".into(), ty: Self::UInt64 },
        ] }
    }

    pub fn pointer_cast(&self, to: &Self) -> bool {
        matches!((self, to),
            (Self::Pointer { .. }, Self::Pointer { .. } | Self::UInt64)
            | (Self::UInt64, Self::Pointer { .. }))
    }

    pub fn foreign_value(&self) -> bool {
        self.is_numeric() || matches!(self, Self::Bool | Self::Pointer { .. })
    }

    pub const fn is_numeric(&self) -> bool {
        self.is_integer() || matches!(self, Self::Float32 | Self::Float64)
    }

    pub fn parameter(types: &[Ty]) -> Self {
        match types {
            [] => Self::Unit,
            [ty] => ty.clone(),
            _ => Self::Record {
                fields: types
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| RecordField {
                        name: format!("_{i}").into(),
                        ty: ty.clone(),
                    })
                    .collect(),
            },
        }
    }

    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::UInt8
                | Self::UInt16
                | Self::UInt32
                | Self::UInt64
        )
    }
}
