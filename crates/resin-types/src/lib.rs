//! Resolved types, values, and representation rules shared by compiler phases.
//!
//! This crate owns concrete types, never source text, syntax, or inference variables.
//! Type IDs belong to a program's canonical table; that same table drives host and
//! device layout, conversions, union tags, and diagnostics.
//!
//! Definition validation is an implementation detail:
//! ```compile_fail
//! use resin_types::definitions::DefinitionError;
//! ```

use std::{collections::HashMap, fmt, ops::Deref, sync::Arc};

mod definitions;
use definitions::DefinitionError;

/// Import shared type vocabulary privately; keep operations qualified.
pub mod prelude {
    pub use crate::shader::{ShaderEntry, Stage};
    pub use crate::{
        ArrayValue, BuiltinCall, BuiltinRule, Case, Conv, Converted, ExplicitConversion,
        FieldAccess, Foreign, FunctionId, Intrinsic, LocalId, RecordField, RecordFieldValue,
        RecordValue, StaticAddressValue, Ty, TypeDef, TypeError, TypeErrorKind, TypeId, TypeTable,
        TyperContext, Value, define_id,
    };
}

//
// Canonical types and values
//

/// Define distinct index types without giving them interchangeable integer identities.
#[macro_export]
macro_rules! define_id {
    (
        $(
            $(#[$attr:meta])*
            $visibility:vis struct $name:ident(usize);
        )+
    ) => {
        $(
            $(#[$attr])*
            #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
            $visibility struct $name(usize);

            impl $name {
                pub const fn from_index(index: usize) -> Self {
                    Self(index)
                }

                pub const fn index(self) -> usize {
                    self.0
                }
            }
        )+
    };
}

define_id! {
    pub struct TypeId(usize);
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// An entry in the module's canonical type table. Only nominal records can have
/// incomplete bodies while resolving recursive fields.
pub enum TypeDef {
    Nominal {
        name: Arc<str>,
        body: Option<Ty>,
        /// Builtin destruction hook; ordinary method namespaces remain in the frontend.
        drop: Option<FunctionId>,
    },
    Structural(Ty),
}

impl TypeDef {
    pub fn new(name: impl Into<Arc<str>>, body: Ty) -> Self {
        Self::Nominal {
            name: name.into(),
            body: Some(body),
            drop: None,
        }
    }
    pub fn name(&self) -> Option<&Arc<str>> {
        match self {
            Self::Nominal { name, .. } => Some(name),
            Self::Structural(_) => None,
        }
    }
    pub fn drop_hook(&self) -> Option<FunctionId> {
        match self {
            Self::Nominal { drop, .. } => *drop,
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
    Arc { pointee: Box<Ty> },
    Weak { pointee: Box<Ty> },
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
    pub fn tag(&self, types: &TypeTable) -> u32 {
        match self {
            Self::Ok => 0,
            Self::Err => 1,
            Self::Type(ty) => types.id(ty).expect("verified union member").tag(),
        }
    }
}

impl Ty {
    /// Types that permit direct pointee access. Weak handles must first upgrade
    /// successfully; their payload may already have been destroyed.
    pub fn deref_target(&self) -> Option<&Ty> {
        match self {
            Self::Pointer { pointee } | Self::Arc { pointee } => Some(pointee),
            _ => None,
        }
    }

    pub fn needs_drop(&self, definitions: &[TypeDef]) -> bool {
        match self {
            Self::Arc { .. } | Self::Weak { .. } => true,
            Self::Defined { definition } => {
                let d = &definitions[definition.index()];
                d.drop_hook().is_some() || d.body().is_some_and(|t| t.needs_drop(definitions))
            }
            Self::Record { fields } => fields.iter().any(|f| f.ty.needs_drop(definitions)),
            Self::Array { element, .. } => element.needs_drop(definitions),
            Self::Result { value, error } => {
                value.needs_drop(definitions) || error.needs_drop(definitions)
            }
            Self::Union { variants } => {
                variants.iter().any(|member| member.needs_drop(definitions))
            }
            _ => false,
        }
    }

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

impl Ty {
    /// An owned span whose byte storage lives in the same Arc allocation.
    pub fn formatted_bytes() -> Self {
        Self::Arc {
            pointee: Box::new(Self::Span {
                element: Box::new(Self::UInt8),
            }),
        }
    }

    pub fn byte_span() -> Self {
        Self::Span {
            element: Box::new(Self::UInt8),
        }
    }
}

/// Primitive operations exposed through compiler-provided methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intrinsic {
    StringFromStr,
    Replace,
    Index,
    ArcGet,
    Downgrade,
    Upgrade,
}

/// Validate references in a concrete type against a program's canonical table.
pub fn check_references(table: &[TypeDef], ty: &Ty) -> Result<(), TypeError> {
    Ok(definitions::check_references(table, ty)?)
}
/// Validate that a nominal body's layout terminates without following indirection.
pub fn check_layout(table: &[TypeDef], id: TypeId, body: &Ty) -> Result<(), TypeError> {
    Ok(definitions::check_layout(table, id, body)?)
}
/// Resolve one nominal representation, rejecting incomplete or invalid definitions.
pub fn definition_body(table: &[TypeDef], id: TypeId) -> Result<&Ty, TypeError> {
    Ok(definitions::body(table, id)?)
}

define_id! {
    pub struct LocalId(usize);
    pub struct FunctionId(usize);
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Type {
        ty: Ty,
    },
    /// A non-owning byte span backed by static literal storage.
    Bytes {
        value: Arc<[u8]>,
    },
    Unit,
    None,
    Bool {
        value: bool,
    },
    Int8 {
        value: i8,
    },
    Int16 {
        value: i16,
    },
    Int32 {
        value: i32,
    },
    Int64 {
        value: i64,
    },
    UInt8 {
        value: u8,
    },
    UInt16 {
        value: u16,
    },
    UInt32 {
        value: u32,
    },
    UInt64 {
        value: u64,
    },
    Float32 {
        value: f32,
    },
    Float64 {
        value: f64,
    },
    StaticAddress {
        address: StaticAddressValue,
    },
    DynamicAddress {
        address: usize,
    },
    Array {
        value: ArrayValue,
    },
    Record {
        value: RecordValue,
    },
}

/// A storage root plus type-directed child indices.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaticAddressValue {
    Local { local: LocalId, path: Vec<usize> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordValue {
    pub fields: Vec<RecordFieldValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RecordFieldValue {
    pub name: Arc<str>,
    pub value: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArrayValue {
    pub element_ty: Ty,
    pub elements: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Foreign {
    pub header: Arc<str>,
    pub params: Vec<Ty>,
}

impl Foreign {
    pub fn valid(&self, result: &Ty) -> bool {
        self.params.iter().all(Ty::foreign_value)
            && (*result == Ty::Unit || result.foreign_value())
            && !self.header.is_empty()
            && self
                .header
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_./- :~()".contains(&c))
    }
}

/// The canonical definitions of every type in one compiled program. IDs are
/// indices into `definitions`; the map only accelerates structural lookup.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeTable {
    definitions: Vec<TypeDef>,
    ids: HashMap<Ty, TypeId>,
}

impl Deref for TypeTable {
    type Target = [TypeDef];

    fn deref(&self) -> &Self::Target {
        &self.definitions
    }
}

impl From<Vec<TypeDef>> for TypeTable {
    fn from(definitions: Vec<TypeDef>) -> Self {
        let ids = definitions
            .iter()
            .enumerate()
            .map(|(index, definition)| {
                let id = TypeId::from_index(index);
                (definition.ty(id), id)
            })
            .collect();
        Self { definitions, ids }
    }
}

impl TypeTable {
    pub fn id(&self, ty: &Ty) -> Option<TypeId> {
        self.ids.get(ty).copied()
    }

    pub fn types(&self) -> impl Iterator<Item = Ty> + '_ {
        self.iter()
            .enumerate()
            .map(|(index, definition)| definition.ty(TypeId::from_index(index)))
    }

    fn reserve(&mut self, name: Arc<str>) -> TypeId {
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Nominal {
            name,
            body: None,
            drop: None,
        });
        self.ids.insert(Ty::Defined { definition: id }, id);
        id
    }

    fn define(&mut self, id: TypeId, body: Ty) {
        let TypeDef::Nominal { body: slot, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have pending bodies")
        };
        *slot = Some(body);
    }

    fn set_drop(&mut self, id: TypeId, function: FunctionId) {
        let TypeDef::Nominal { drop, .. } = &mut self.definitions[id.index()] else {
            unreachable!("only nominal definitions have destruction hooks")
        };
        *drop = Some(function);
    }

    fn pop_nominal(&mut self) {
        let id = TypeId::from_index(self.len() - 1);
        assert!(matches!(
            self.definitions.pop(),
            Some(TypeDef::Nominal { .. })
        ));
        self.ids.remove(&Ty::Defined { definition: id });
    }

    /// Intern a resolved type without changing nominal identity. Nominal slots
    /// are reserved before their bodies, so recursive pointers terminate here.
    pub fn intern(&mut self, ty: &Ty) -> TypeId {
        if let Some(id) = self.id(ty) {
            if !matches!(ty, Ty::Defined { .. }) {
                self.components(ty);
            }
            return id;
        }
        assert!(!matches!(ty, Ty::Defined { .. }), "unreserved nominal type");
        let id = TypeId::from_index(self.len());
        self.definitions.push(TypeDef::Structural(ty.clone()));
        self.ids.insert(ty.clone(), id);
        self.components(ty);
        id
    }

    fn components(&mut self, ty: &Ty) {
        match ty {
            Ty::Union { variants } => {
                for member in variants {
                    self.intern(member);
                }
                self.intern(&Ty::UInt32);
            }
            Ty::Result { value, error } => {
                self.intern(value);
                self.intern(error);
                self.intern(&Ty::UInt32);
            }
            Ty::Pointer { pointee } | Ty::Arc { pointee } | Ty::Weak { pointee } => {
                self.intern(pointee);
            }
            Ty::Span { element } | Ty::Array { element, .. } => {
                self.intern(element);
            }
            Ty::Record { fields } => {
                for field in fields {
                    self.intern(&field.ty);
                }
            }
            Ty::Function { param, result } => {
                self.intern(param);
                self.intern(result);
            }
            _ => {}
        }
    }
}

//
// Concrete type checking and conversions
//

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeErrorKind {
    InvalidUnion,
    PointerArithmetic,
    EmptyArrayNeedsElementType,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedInteger { found: Ty },
    ExpectedBoolean { found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedRecord { found: Ty },
    ExpectedFunction { found: Ty },
    InvalidBuiltinArgumentCount { name: Arc<str>, found: usize },
    UnknownBuiltin { name: Arc<str> },
    InvalidPrintArguments { found: Ty },
    InvalidFormatArguments { found: Ty },
    UnformattableType { found: Ty },
    UnknownField { name: Arc<str> },
    DuplicateField { name: Arc<str> },
    InvalidTypeDefinition { definition: TypeId },
    IncompleteTypeDefinition { definition: TypeId },
    NominalTypeMustBeRecord { definition: TypeId },
    TypeAlreadyDefined { definition: TypeId },
    RecursiveTypeWithoutIndirection { definition: TypeId },
}

impl TypeError {
    const fn new(kind: TypeErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind == TypeErrorKind::PointerArithmetic {
            return f.write_str("pointer arithmetic is not allowed; index a Span or explicitly convert the pointer to ulong for byte arithmetic");
        }
        write!(f, "type error: {:?}", self.kind)
    }
}

impl std::error::Error for TypeError {}

impl From<DefinitionError> for TypeError {
    fn from(error: DefinitionError) -> Self {
        let kind = match error {
            DefinitionError::NonRecord(definition) => {
                TypeErrorKind::NominalTypeMustBeRecord { definition }
            }
            DefinitionError::InvalidUnion => TypeErrorKind::InvalidUnion,
            DefinitionError::Invalid(definition) => {
                TypeErrorKind::InvalidTypeDefinition { definition }
            }
            DefinitionError::Incomplete(definition) => {
                TypeErrorKind::IncompleteTypeDefinition { definition }
            }
            DefinitionError::Recursive(definition) => {
                TypeErrorKind::RecursiveTypeWithoutIndirection { definition }
            }
        };
        Self::new(kind)
    }
}

/// Operand relationships shared by concrete checking and inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinRule {
    Arithmetic,
    Comparison,
    Boolean,
    Print,
    Format,
    StringFromStr,
}

impl BuiltinRule {
    pub fn lookup(name: &str, arity: usize) -> Result<Self, TypeError> {
        let (rule, valid_arity) = match name {
            "+" | "-" => (Self::Arithmetic, matches!(arity, 1 | 2)),
            "~" => (Self::Arithmetic, arity == 1),
            "*" | "/" | "%" | "<<" | ">>" | "&" | "|" | "^" => (Self::Arithmetic, arity == 2),
            "==" | "!=" | "<" | "<=" | ">" | ">=" => (Self::Comparison, arity == 2),
            "!" => (Self::Boolean, arity == 1),
            "&&" | "||" => (Self::Boolean, arity == 2),
            "print" => (Self::Print, arity == 1),
            "fmt" => (Self::Format, arity == 1),
            "string_from_str" => (Self::StringFromStr, arity == 1),
            _ => {
                return Err(TypeError::new(TypeErrorKind::UnknownBuiltin {
                    name: name.into(),
                }));
            }
        };
        if !valid_arity {
            return Err(TypeError::new(TypeErrorKind::InvalidBuiltinArgumentCount {
                name: name.into(),
                found: arity,
            }));
        }
        Ok(rule)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv {
    Unwrap { definition: TypeId },
    Wrap { definition: TypeId },
    Deref,
    SpanRecord,
    MakeSpan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplicitConversion {
    Ascribe(Vec<Conv>),
    Widen,
    NumericCast,
    PointerCast,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Converted {
    pub ty: Ty,
    pub steps: Vec<Conv>,
}

impl TyperContext {
    pub fn same(&self, expected: &Ty, found: &Ty) -> Result<(), TypeError> {
        if expected == found {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
            }))
        }
    }

    /// Identity, a Span representation change, or exactly one nominal wrap or unwrap.
    pub fn ascribe(&self, from: &Ty, to: &Ty) -> Result<Vec<Conv>, TypeError> {
        ascription(self.definitions(), from, to)?.ok_or_else(|| {
            TypeError::new(TypeErrorKind::TypeMismatch {
                expected: to.clone(),
                found: from.clone(),
            })
        })
    }

    /// Select the operation for an explicit source conversion. Unlike IR
    /// ascription, this also permits widening and numeric or pointer casts.
    pub fn explicit_conversion(&self, from: &Ty, to: &Ty) -> Result<ExplicitConversion, TypeError> {
        if from != to {
            if from.widens_to(to) {
                return Ok(ExplicitConversion::Widen);
            }
            if from.is_numeric() && to.is_numeric() {
                return Ok(ExplicitConversion::NumericCast);
            }
            if from.pointer_cast(to) {
                return Ok(ExplicitConversion::PointerCast);
            }
        }
        self.ascribe(from, to).map(ExplicitConversion::Ascribe)
    }

    pub fn as_bool(&self, ty: &Ty) -> Result<(), TypeError> {
        if ty == &Ty::Bool {
            Ok(())
        } else {
            Err(TypeError::new(TypeErrorKind::ExpectedBoolean {
                found: ty.clone(),
            }))
        }
    }

    pub fn as_record(&self, ty: &Ty) -> Result<Converted, TypeError> {
        let mut current = ty.clone();
        let mut steps = Vec::new();
        while let Some(pointee) = current.deref_target() {
            steps.push(Conv::Deref);
            current = pointee.clone();
        }
        if let Ty::Defined { definition } = current {
            steps.push(Conv::Unwrap { definition });
            current = self.definition_body(definition)?.clone();
        }
        if matches!(current, Ty::Record { .. } | Ty::Span { .. }) {
            Ok(Converted { ty: current, steps })
        } else {
            Err(TypeError::new(TypeErrorKind::ExpectedRecord {
                found: ty.clone(),
            }))
        }
    }
}

/// Shared rule for `Instr::Ascribe`; `None` means the types are incompatible.
/// Borrow the definition table so verification can use the same nominal rules.
pub fn ascription(table: &[TypeDef], from: &Ty, to: &Ty) -> Result<Option<Vec<Conv>>, TypeError> {
    let step = if from == to {
        return Ok(Some(Vec::new()));
    } else if to.span_record().as_ref() == Some(from) {
        Conv::MakeSpan
    } else if from.span_record().as_ref() == Some(to) {
        Conv::SpanRecord
    } else if let Ty::Defined { definition } = to
        && from == definitions::body(table, *definition)?
    {
        Conv::Wrap {
            definition: *definition,
        }
    } else if let Ty::Defined { definition } = from
        && to == definitions::body(table, *definition)?
    {
        Conv::Unwrap {
            definition: *definition,
        }
    } else {
        return Ok(None);
    };
    Ok(Some(vec![step]))
}

/// Type IDs are local to this context's definition table.
#[derive(Debug, Clone, Default)]
pub struct TyperContext {
    definitions: TypeTable,
    string_type: Option<Ty>,
}

impl TyperContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_definitions(definitions: impl Into<TypeTable>) -> Self {
        let definitions = definitions.into();
        let string_type = definitions
            .iter()
            .enumerate()
            .find_map(|(index, definition)| {
                (definition
                    .name()
                    .is_some_and(|name| name.as_ref() == "String"))
                .then_some(Ty::Defined {
                    definition: TypeId::from_index(index),
                })
            });
        Self {
            definitions,
            string_type,
        }
    }

    pub fn definitions(&self) -> &[TypeDef] {
        &self.definitions
    }

    pub fn into_definitions(self) -> Result<TypeTable, TypeError> {
        for index in 0..self.definitions.len() {
            if self.definitions[index].name().is_some() {
                self.definition_body(TypeId::from_index(index))?;
            }
        }
        Ok(self.definitions)
    }

    pub fn create_type(
        &mut self,
        name: impl Into<Arc<str>>,
        body: Ty,
    ) -> Result<TypeId, TypeError> {
        let definition = self.reserve_type(name);
        if let Err(err) = self.define_type(definition, body) {
            self.definitions.pop_nominal();
            return Err(err);
        }
        Ok(definition)
    }

    pub fn reserve_type(&mut self, name: impl Into<Arc<str>>) -> TypeId {
        self.definitions.reserve(name.into())
    }

    pub fn define_type(&mut self, definition: TypeId, body: Ty) -> Result<(), TypeError> {
        if self.definition(definition)?.body().is_some() {
            return Err(TypeError::new(TypeErrorKind::TypeAlreadyDefined {
                definition,
            }));
        }
        definitions::check_references(&self.definitions, &body)?;
        definitions::check_layout(&self.definitions, definition, &body)?;
        self.definitions.define(definition, body);
        Ok(())
    }

    pub fn definition(&self, definition: TypeId) -> Result<&TypeDef, TypeError> {
        Ok(definitions::get(&self.definitions, definition)?)
    }

    fn definition_body(&self, definition: TypeId) -> Result<&Ty, TypeError> {
        Ok(definitions::body(&self.definitions, definition)?)
    }

    pub fn body(&self, ty: &Ty) -> Result<Ty, TypeError> {
        match ty {
            Ty::Defined { definition } => Ok(self.definition_body(*definition)?.clone()),
            other => Ok(other.clone()),
        }
    }
}

impl TyperContext {
    pub fn define_drop(&mut self, ty: TypeId, function: FunctionId) {
        self.definitions.set_drop(ty, function);
    }
}

impl TyperContext {
    pub fn string_type(&self) -> Option<&Ty> {
        self.string_type.as_ref()
    }
    pub fn set_string_type(&mut self, ty: Ty) {
        self.string_type = Some(ty);
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldAccess {
    pub ty: Ty,
    pub index: usize,
    pub steps: Vec<Conv>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinCall {
    pub params: Vec<Ty>,
    pub result: Ty,
}

impl TyperContext {
    pub fn type_num(&self, value: &str) -> Ty {
        literal::split(value)
            .1
            .unwrap_or_else(|| literal::unsuffixed_type(value))
    }

    pub fn type_field(&self, base: &Ty, name: &str) -> Result<FieldAccess, TypeError> {
        let converted = self.as_record(base)?;
        let shape = converted.ty.span_record().unwrap_or(converted.ty);
        let Ty::Record { fields } = shape else {
            return Err(TypeError::new(TypeErrorKind::ExpectedRecord {
                found: base.clone(),
            }));
        };
        for (index, field) in fields.into_iter().enumerate() {
            if field.name.as_ref() == name {
                return Ok(FieldAccess {
                    ty: field.ty,
                    index,
                    steps: converted.steps,
                });
            }
        }
        Err(TypeError::new(TypeErrorKind::UnknownField {
            name: Arc::from(name),
        }))
    }

    pub fn type_builtin_call(&self, name: &str, args: &[Ty]) -> Result<BuiltinCall, TypeError> {
        let rule = BuiltinRule::lookup(name, args.len())?;
        let result = match rule {
            BuiltinRule::Print if self.is_string(&args[0]) => Ty::Unit,
            BuiltinRule::Print => {
                return Err(TypeError::new(TypeErrorKind::InvalidPrintArguments {
                    found: args[0].clone(),
                }));
            }
            BuiltinRule::Format => self.type_format(&args[0])?,
            BuiltinRule::StringFromStr => {
                self.same(&Ty::byte_span(), &args[0])?;
                self.string_type.clone().ok_or_else(|| {
                    TypeError::new(TypeErrorKind::UnknownBuiltin { name: name.into() })
                })?
            }
            BuiltinRule::Boolean => {
                for arg in args {
                    self.as_bool(arg)?;
                }
                Ty::Bool
            }
            BuiltinRule::Arithmetic | BuiltinRule::Comparison => {
                if rule == BuiltinRule::Arithmetic
                    && args.iter().any(|ty| matches!(ty, Ty::Pointer { .. }))
                {
                    return Err(TypeError::new(TypeErrorKind::PointerArithmetic));
                }
                for arg in &args[1..] {
                    self.same(&args[0], arg)?;
                }
                if rule == BuiltinRule::Comparison {
                    Ty::Bool
                } else {
                    args[0].clone()
                }
            }
        };
        Ok(BuiltinCall {
            params: args.to_vec(),
            result,
        })
    }

    fn is_string(&self, ty: &Ty) -> bool {
        ty == &Ty::byte_span() || self.string_type.as_ref() == Some(ty)
    }

    fn type_format(&self, arg: &Ty) -> Result<Ty, TypeError> {
        let invalid =
            || TypeError::new(TypeErrorKind::InvalidFormatArguments { found: arg.clone() });
        let Ty::Record { fields } = arg else {
            return Err(invalid());
        };
        let [format, values] = fields.as_slice() else {
            return Err(invalid());
        };
        if format.name.as_ref() != "_0"
            || values.name.as_ref() != "_1"
            || !self.is_string(&format.ty)
        {
            return Err(invalid());
        }
        let fields = match &values.ty {
            Ty::Unit => &[][..],
            Ty::Record { fields } => fields,
            _ => return Err(invalid()),
        };
        for (i, field) in fields.iter().enumerate() {
            if field.name.as_ref() != format!("_{i}") {
                return Err(invalid());
            }
            let ty = &field.ty;
            if !(ty.is_numeric()
                || matches!(ty, Ty::Bool | Ty::Unit | Ty::Pointer { .. })
                || self.is_string(ty))
            {
                return Err(TypeError::new(TypeErrorKind::UnformattableType {
                    found: ty.clone(),
                }));
            }
        }
        self.string_type.clone().ok_or_else(invalid)
    }
}

//
// Type rendering
//

pub fn format_type(ty: &Ty, definitions: &[TypeDef]) -> String {
    match ty {
        Ty::Union { variants } => {
            if variants.is_empty() {
                return "Never".into();
            }
            variants
                .iter()
                .map(|member| format_type(member, definitions))
                .collect::<Vec<_>>()
                .join(" | ")
        }
        Ty::Result { value, error } => format!(
            "Result<{}, {}>",
            format_type(value, definitions),
            format_type(error, definitions)
        ),
        Ty::Type => "type".into(),
        Ty::Unit => "()".into(),
        Ty::None => "None".into(),
        Ty::Bool => "bool".into(),
        Ty::Int8 => "sbyte".into(),
        Ty::Int16 => "short".into(),
        Ty::Int32 => "int".into(),
        Ty::Int64 => "long".into(),
        Ty::UInt8 => "ubyte".into(),
        Ty::UInt16 => "ushort".into(),
        Ty::UInt32 => "uint".into(),
        Ty::UInt64 => "ulong".into(),
        Ty::Float32 => "float32".into(),
        Ty::Float64 => "float64".into(),
        Ty::Foreign { name } => name.to_string(),
        Ty::Defined { definition } => definitions
            .get(definition.index())
            .and_then(|d| d.name())
            .map(ToString::to_string)
            .unwrap_or_else(|| "?".into()),
        Ty::Pointer { pointee } => format!("Ptr<{}>", format_type(pointee, definitions)),
        Ty::Arc { pointee } => format!("Arc<{}>", format_type(pointee, definitions)),
        Ty::Weak { pointee } => format!("Weak<{}>", format_type(pointee, definitions)),
        Ty::Span { element } => format!("Span<{}>", format_type(element, definitions)),
        Ty::Array { element, length } => {
            format!("[{}; {length}]", format_type(element, definitions))
        }
        Ty::Record { fields } => format!(
            "{{ {} }}",
            fields
                .iter()
                .map(|f| format!("{}: {}", f.name, format_type(&f.ty, definitions)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Ty::Function { param, result } => format!(
            "({}) -> {}",
            format_type(param, definitions),
            format_type(result, definitions)
        ),
    }
}

//
// Host/device storage layout
//

pub mod layout {
    use crate::{Ty, TypeDef};

    #[derive(Debug)]
    pub struct Error(pub String);
    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(&self.0)
        }
    }
    impl std::error::Error for Error {}

    pub struct Layout {
        pub size: usize,
        pub align: usize,
        pub offsets: Vec<usize>,
    }

    pub fn layout(definitions: &[TypeDef], ty: &Ty) -> Result<Layout, Error> {
        let scalar = match ty {
            Ty::UInt8 => Some(1),
            Ty::Int32 | Ty::UInt32 | Ty::Float32 => Some(4),
            Ty::UInt64 | Ty::Pointer { .. } | Ty::Arc { .. } | Ty::Weak { .. } => Some(8),
            _ => None,
        };
        if let Some(size) = scalar {
            return Ok(Layout {
                size,
                align: size,
                offsets: Vec::new(),
            });
        }
        if let Ty::Span { .. } = ty {
            return Ok(Layout {
                size: 16,
                align: 8,
                offsets: vec![0, 8],
            });
        }
        if let Ty::Array { element, length } = ty {
            if **element == Ty::UInt8 {
                return Err(Error("byte arrays have a host sentinel and no shared host/device layout; use Span<ubyte> for packed storage".into()));
            }
            if *length == 0 {
                return Err(Error(
                    "empty arrays have no shared host/device layout".into(),
                ));
            }
            let element = layout(definitions, element)?;
            return Ok(Layout {
                size: element.size.checked_mul(*length).ok_or_else(overflow)?,
                align: element.align,
                offsets: vec![],
            });
        }
        if let Ty::Defined { definition } = ty {
            return layout(definitions, definitions[definition.index()].body().unwrap());
        }
        if let Ty::Record { fields } = ty
            && !fields.is_empty()
        {
            let mut size = 0;
            let mut align = 1;
            let mut offsets = Vec::new();
            for field in fields {
                let field = layout(definitions, &field.ty)?;
                size = round_up(size, field.align)?;
                offsets.push(size);
                size = size.checked_add(field.size).ok_or_else(overflow)?;
                align = align.max(field.align);
            }
            return Ok(Layout {
                size: round_up(size, align)?,
                align,
                offsets,
            });
        }
        Err(Error(format!(
            "type {ty:?} has no shared host/device layout"
        )))
    }

    fn round_up(size: usize, align: usize) -> Result<usize, Error> {
        size.checked_add(align - 1)
            .map(|n| n & !(align - 1))
            .ok_or_else(overflow)
    }

    fn overflow() -> Error {
        Error("shared host/device layout is too large".into())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::{RecordField, TypeDef, TypeId};

        fn record(types: Vec<Ty>) -> Ty {
            Ty::Record {
                fields: types
                    .into_iter()
                    .enumerate()
                    .map(|(i, ty)| RecordField {
                        name: format!("f{i}").into(),
                        ty,
                    })
                    .collect(),
            }
        }

        #[test]
        fn scalar_records_match_c_and_std430_padding() {
            let table = Vec::new();
            let inner = record(vec![Ty::UInt32, Ty::UInt64, Ty::Float32]);
            let outer = record(vec![Ty::UInt32, inner, Ty::Float32]);
            let l = layout(&table, &outer).unwrap();
            assert_eq!((l.size, l.align, l.offsets), (40, 8, vec![0, 8, 32]));
        }

        #[test]
        fn recursive_pointers_do_not_recurse_through_memory_layout() {
            let node = Ty::Defined {
                definition: TypeId::from_index(0),
            };
            let table = vec![TypeDef::new(
                "Node",
                record(vec![
                    Ty::UInt32,
                    Ty::Pointer {
                        pointee: Box::new(node.clone()),
                    },
                ]),
            )];
            let l = layout(&table, &node).unwrap();
            assert_eq!((l.size, l.align, l.offsets), (16, 8, vec![0, 8]));
        }

        #[test]
        fn incompatible_storage_types_are_rejected() {
            for ty in [
                Ty::Bool,
                Ty::Unit,
                Ty::Int64,
                record(vec![]),
                record(vec![Ty::Bool]),
            ] {
                assert!(layout(&[], &ty).is_err());
            }
        }
    }
}

//
// Numeric literal spelling and default types
//

pub mod literal {
    //! Numeric suffixes are case insensitive and select an exact primitive type.
    use super::Ty;

    pub fn split(text: &str) -> (&str, Option<Ty>) {
        let hex = is_hex(text);
        let Some(width) = text.as_bytes().last().map(u8::to_ascii_lowercase) else {
            return (text, None);
        };
        let mut end = text.len() - 1;
        let unsigned = end > 0 && text.as_bytes()[end - 1].eq_ignore_ascii_case(&b'u');
        if unsigned {
            end -= 1;
        }
        // Hex b/B is a digit unless an underscore or unsigned qualifier separates it.
        // Hex d/D/f/F always remain digits; hexadecimal floats are unsupported.
        let separated = end > 0 && text.as_bytes()[end - 1] == b'_';
        let ty = match (width, unsigned) {
            (b'b', false) if !hex || separated => Ty::Int8,
            (b'b', true) => Ty::UInt8,
            (b'h', false) => Ty::Int16,
            (b'h', true) => Ty::UInt16,
            (b'i', false) => Ty::Int32,
            (b'i', true) => Ty::UInt32,
            (b'l', false) => Ty::Int64,
            (b'l', true) => Ty::UInt64,
            (b'f', false) if !hex => Ty::Float32,
            (b'd', false) if !hex => Ty::Float64,
            _ => return (text, None),
        };
        (text[..end].trim_end_matches('_'), Some(ty))
    }

    /// Normalize only the suffix; keep digit grouping and hexadecimal digit case.
    pub fn format(text: &str) -> String {
        let (digits, ty) = split(text);
        let suffix = match ty {
            Some(Ty::Int8) => "b",
            Some(Ty::UInt8) => "ub",
            Some(Ty::Int16) => "h",
            Some(Ty::UInt16) => "uh",
            Some(Ty::Int32) => "i",
            Some(Ty::UInt32) => "ui",
            Some(Ty::Int64) => "l",
            Some(Ty::UInt64) => "ul",
            Some(Ty::Float32) => "f",
            Some(Ty::Float64) => "d",
            _ => return text.into(),
        };
        format!("{digits}_{suffix}")
    }

    /// The fallback type for an unsuffixed literal, before any contextual inference.
    /// Hexadecimal e/E digits are not decimal exponents.
    pub fn unsuffixed_type(text: &str) -> Ty {
        if text.contains('.') || (!is_hex(text) && text.contains(['e', 'E'])) {
            Ty::Float64
        } else {
            Ty::Int64
        }
    }

    fn is_hex(text: &str) -> bool {
        let magnitude = text.strip_prefix('-').unwrap_or(text);
        magnitude.starts_with("0x") || magnitude.starts_with("0X")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn hex_digits_and_signs_do_not_change_suffix_or_exponent_rules() {
            for (text, body, suffix, default) in [
                ("-0xdead", "-0xdead", None, Ty::Int64),
                ("0XAB", "0XAB", None, Ty::Int64),
                ("-0xFF_ul", "-0xFF", Some(Ty::UInt64), Ty::Int64),
                ("-12_b", "-12", Some(Ty::Int8), Ty::Int64),
                ("12_ub", "12", Some(Ty::UInt8), Ty::Int64),
                ("-1e2", "-1e2", None, Ty::Float64),
                ("1E2_f", "1E2", Some(Ty::Float32), Ty::Float64),
                ("-1.5_d", "-1.5", Some(Ty::Float64), Ty::Float64),
            ] {
                assert_eq!(split(text), (body, suffix), "{text}");
                assert_eq!(unsuffixed_type(body), default, "{text}");
            }
        }
    }
}

//
// Declared shader stages and their interfaces
//

pub mod shader {
    use super::{Ty, TyperContext};
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct ShaderEntry {
        pub stage: Arc<str>,
        pub embedded: bool,
    }

    /// A checked stage interface, shared by declaration checking and GLSL emission.
    #[derive(Debug, Clone)]
    pub enum Interface {
        Compute {
            index: Ty,
        },
        Vertex {
            index: Ty,
            root: bool,
            position: Ty,
            color: Ty,
        },
        Fragment {
            color: Ty,
            root: bool,
        },
    }

    pub fn validate(
        typer: &TyperContext,
        parameter: &Ty,
        result: &Ty,
        foreign: bool,
        stage: &str,
    ) -> Result<Interface, String> {
        fn shape(typer: &TyperContext, ty: &Ty) -> Result<Ty, String> {
            let mut ty = ty.clone();
            while matches!(ty, Ty::Defined { .. }) {
                ty = typer.body(&ty).map_err(|e| e.to_string())?;
            }
            Ok(ty)
        }
        fn vector(typer: &TyperContext, ty: &Ty, names: &[&str]) -> bool {
            matches!(shape(typer, ty), Ok(Ty::Record { fields }) if fields.len() == names.len()
                && fields.iter().zip(names).all(|(field, name)| field.name.as_ref() == *name && field.ty == Ty::Float32))
        }
        if foreign {
            return Err("foreign functions cannot be shader entries".into());
        }
        let param = shape(typer, parameter)?;
        let (input, root) = match &param {
            Ty::Record { fields }
                if fields.len() == 2
                    && fields[0].name.as_ref() == "_0"
                    && fields[1].name.as_ref() == "_1"
                    && matches!(fields[1].ty, Ty::Pointer { .. }) =>
            {
                (&fields[0].ty, true)
            }
            _ => (parameter, false),
        };
        let input_shape = shape(typer, input)?;
        let result = shape(typer, result)?;
        let interface = match stage {
            "compute" if root && input_shape == Ty::UInt64 && result == Ty::Unit => {
                Some(Interface::Compute {
                    index: input.clone(),
                })
            }
            "vertex" if input_shape == Ty::Int32 => match &result {
                Ty::Record { fields }
                    if fields.len() == 2
                        && fields[0].name.as_ref() == "position"
                        && fields[1].name.as_ref() == "color"
                        && vector(typer, &fields[0].ty, &["x", "y", "z", "w"])
                        && vector(typer, &fields[1].ty, &["r", "g", "b", "a"]) =>
                {
                    Some(Interface::Vertex {
                        index: input.clone(),
                        root,
                        position: fields[0].ty.clone(),
                        color: fields[1].ty.clone(),
                    })
                }
                _ => None,
            },
            "fragment"
                if vector(typer, input, &["r", "g", "b", "a"])
                    && vector(typer, &result, &["r", "g", "b", "a"]) =>
            {
                Some(Interface::Fragment {
                    color: input.clone(),
                    root,
                })
            }
            _ => None,
        };
        if let Some(interface) = interface {
            Ok(interface)
        } else {
            Err(format!(
                "invalid @{stage}_shader signature: {}",
                match stage {
                    "compute" => "expected (ulong, Ptr<T>) -> ()",
                    "vertex" => "expected int or (int, Ptr<T>) returning a position/color record",
                    "fragment" =>
                        "expected Color or (Color, Ptr<T>) returning Color with float32 r/g/b/a fields",
                    _ => "unknown shader stage",
                }
            ))
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Stage {
        Compute,
        Vertex,
        Fragment,
    }

    impl std::str::FromStr for Stage {
        type Err = String;

        fn from_str(name: &str) -> Result<Self, Self::Err> {
            match name {
                "compute" => Ok(Self::Compute),
                "vertex" => Ok(Self::Vertex),
                "fragment" => Ok(Self::Fragment),
                _ => Err("stage must be compute, vertex, or fragment".into()),
            }
        }
    }

    impl Stage {
        pub fn name(self) -> &'static str {
            match self {
                Self::Compute => "compute",
                Self::Vertex => "vertex",
                Self::Fragment => "fragment",
            }
        }
        pub fn entry(self) -> &'static str {
            match self {
                Self::Compute => "kernel",
                Self::Vertex => "vertex",
                Self::Fragment => "fragment",
            }
        }
    }
}

#[cfg(test)]
mod tests;
