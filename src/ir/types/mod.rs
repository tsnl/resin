//! Resolved monomorphic types used by the initial IR.

use std::sync::Arc;

use crate::util::define_id;

pub(crate) mod definitions;
mod table;
pub use table::TypeTable;

define_id! {
    pub struct TypeId(usize);
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An entry in the module's canonical type table. Only nominal records can have
/// incomplete bodies while resolving recursive fields.
pub enum TypeDef {
    Nominal { name: Arc<str>, body: Option<Ty> },
    Structural(Ty),
}

impl TypeDef {
    pub fn new(name: impl Into<Arc<str>>, body: Ty) -> Self {
        Self::Nominal {
            name: name.into(),
            body: Some(body),
            methods: Default::default(),
        }
    }
    pub fn name(&self) -> Option<&Arc<str>> {
        match self {
            Self::Nominal { name, .. } => Some(name),
            Self::Structural(_) => None,
        }
    }
    pub fn body(&self) -> Option<&Ty> {
        match self {
            Self::Nominal { body, .. } => body.as_ref(),
            Self::Structural(ty) => Some(ty),
        }
    }
    pub fn ty(&self, id: TypeId) -> Ty {
        match self {
            Self::Nominal { .. } => Ty::Defined { definition: id },
            Self::Structural(ty) => ty.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Method {
    pub function: super::FunctionId,
    pub params: Vec<Ty>,
    pub result: Ty,
    pub receiver: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordField {
    pub name: Arc<str>,
    pub ty: Ty,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Ty {
    Type,
    Unit,
    None,
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
    Union { variants: Vec<Ty> },
    Result { value: Box<Ty>, error: Box<Ty> },
}

/// Result cases are tagged independently of their payload type. Ordinary union
/// cases use the index of their payload type in the module's type table.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Case {
    Ok,
    Err,
    Type(Ty),
}

impl Case {
    pub(crate) fn tag(&self, types: &TypeTable) -> u32 {
        match self {
            Self::Ok => 0,
            Self::Err => 1,
            Self::Type(ty) => types.id(ty).expect("verified union member").tag(),
        }
    }
}

impl Ty {
    pub fn payloads(&self) -> Option<Vec<(Case, Ty)>> {
        match self {
            Self::Result { value, error } => Some(vec![
                (Case::Ok, *value.clone()),
                (Case::Err, *error.clone()),
            ]),
            Self::Union { variants } => Some(
                variants
                    .iter()
                    .map(|ty| (Case::Type(ty.clone()), ty.clone()))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn payload(&self, case: &Case) -> Option<Ty> {
        match (self, case) {
            (Self::Result { value, .. }, Case::Ok) => Some(*value.clone()),
            (Self::Result { error, .. }, Case::Err) => Some(*error.clone()),
            (_, Case::Type(ty)) if self.members().contains(ty) => Some(ty.clone()),
            _ => None,
        }
    }

    /// Nominal variants used by Result error-set inference.
    pub fn variants(&self) -> Option<Vec<TypeId>> {
        self.members()
            .into_iter()
            .map(|ty| match ty {
                Self::Defined { definition } => Some(definition),
                _ => None,
            })
            .collect()
    }

    pub fn members(&self) -> Vec<Ty> {
        match self {
            Self::Union { variants } => variants.clone(),
            _ => vec![self.clone()],
        }
    }

    pub fn union(variants: impl IntoIterator<Item = TypeId>) -> Self {
        Self::union_of(
            variants
                .into_iter()
                .map(|definition| Self::Defined { definition }),
        )
    }

    pub fn union_of(variants: impl IntoIterator<Item = Ty>) -> Self {
        let mut variants: Vec<_> = variants.into_iter().flat_map(|ty| ty.members()).collect();
        variants.sort();
        variants.dedup();
        match variants.as_slice() {
            [ty] => ty.clone(),
            _ => Self::Union { variants },
        }
    }

    pub fn without_none(&self) -> Option<Self> {
        let members = self.members();
        members
            .contains(&Self::None)
            .then(|| Self::union_of(members.into_iter().filter(|ty| ty != &Self::None)))
    }

    pub fn widens_to(&self, to: &Self) -> bool {
        if self == to {
            return true;
        }
        match (self, to) {
            (
                Self::Result {
                    value: av,
                    error: ae,
                },
                Self::Result {
                    value: bv,
                    error: be,
                },
            ) => av == bv && ae.widens_to(be),
            _ => self.members().iter().all(|ty| to.members().contains(ty)),
        }
    }

    pub fn shader() -> Self {
        Self::Span {
            element: Box::new(Self::UInt8),
        }
    }

    pub(crate) fn shader_properties() -> Self {
        Self::Record {
            fields: vec![RecordField {
                name: "spirv".into(),
                ty: Self::shader(),
            }],
        }
    }

    pub fn span_record(&self) -> Option<Self> {
        let Self::Span { element } = self else {
            return None;
        };
        Some(Self::Record {
            fields: vec![
                RecordField {
                    name: "data".into(),
                    ty: Self::Pointer {
                        pointee: element.clone(),
                    },
                },
                RecordField {
                    name: "length".into(),
                    ty: Self::UInt64,
                },
            ],
        })
    }

    pub fn pointer_cast(&self, to: &Self) -> bool {
        matches!(
            (self, to),
            (Self::Pointer { .. }, Self::Pointer { .. } | Self::UInt64)
                | (Self::UInt64, Self::Pointer { .. })
        )
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

impl TypeId {
    pub fn tag(self) -> u32 {
        u32::try_from(self.index()).expect("too many types")
    }
}
