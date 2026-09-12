//! Resolved types, values, and representation rules shared by compiler phases.
//!
//! This crate owns concrete types, never source text, syntax, or inference variables.
//! Type IDs belong to a program's canonical table; that same table drives host and
//! device layout, conversions, union tags, and diagnostics.
//!
//! Representation and checking implementations stay private:
//! ```compile_fail
//! use resin_types::types;
//! ```
//! ```compile_fail
//! use resin_types::typer;
//! ```

use resin_common::define_id;
use std::{collections::HashMap, fmt, ops::Deref, sync::Arc};

mod typer;
mod types;
use types::DefinitionError;

/// Import shared type vocabulary privately; keep operations qualified.
pub mod prelude {
    pub use crate::shader::{ShaderEntry, Stage};
    pub use crate::{
        ArrayValue, BuiltinCall, BuiltinRule, Case, Conv, Converted, ExplicitConversion,
        FieldAccess, Foreign, FunctionId, Intrinsic, LocalId, RecordField, RecordFieldValue,
        RecordValue, StaticAddressValue, Ty, TypeDef, TypeError, TypeErrorKind, TypeId, TypeTable,
        TyperContext, Value,
    };
}

//
// Canonical types and values
//

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
    /// A non-owning string view. Literal storage has a NUL beyond its byte length.
    Str,
    Foreign {
        name: Arc<str>,
    },
    Defined {
        definition: TypeId,
    },
    Pointer {
        pointee: Box<Ty>,
    },
    Span {
        element: Box<Ty>,
    },
    /// An owning GPU allocation view with a byte offset and CPU access permissions.
    /// Host storage occupies 24 bytes aligned to 8; shader projection produces Ptr<T>.
    GpuPointer {
        pointee: Box<Ty>,
    },
    /// An owning GPU pointer and element count: 32 host bytes aligned to 8.
    /// Shader projection produces Span<T>; owners never reside in device storage.
    GpuSpan {
        element: Box<Ty>,
    },
    /// Opaque projected shader arguments, retained by a host Arc handle.
    GpuArguments,
    /// A compute pipeline whose shader root and shared host owner stay in its type.
    /// Storage is the owner's Arc handle; neither type parameter is device storage.
    GpuComputePipeline {
        root: Box<Ty>,
        owner: Box<Ty>,
    },
    /// A graphics pipeline with one root shared by its vertex and fragment stages.
    /// A None root denotes shaders without a root argument. Storage is the owner's Arc.
    GpuGraphicsPipeline {
        root: Box<Ty>,
        owner: Box<Ty>,
    },
    Arc {
        pointee: Box<Ty>,
    },
    Weak {
        pointee: Box<Ty>,
    },
    Array {
        element: Box<Ty>,
        length: usize,
    },
    Record {
        fields: Vec<RecordField>,
    },
    Function {
        param: Box<Ty>,
        result: Box<Ty>,
    },
    Union {
        variants: Vec<Ty>,
    },
    Result {
        value: Box<Ty>,
        error: Box<Ty>,
    },
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
            Self::Pointer { pointee } | Self::GpuPointer { pointee } | Self::Arc { pointee } => {
                Some(pointee)
            }
            _ => None,
        }
    }

    pub fn needs_drop(&self, definitions: &[TypeDef]) -> bool {
        types::needs_drop(self, definitions)
    }

    /// Values that can live directly in GPU storage: shared scalar/aggregate layout,
    /// with no pointers, spans, managed owners, or custom destruction hooks.
    pub fn gpu_element(&self, definitions: &[TypeDef]) -> bool {
        types::gpu_element(self, definitions)
    }

    /// Host argument shape whose GPU views project into this shader root type.
    /// Nominal records expose structural host fields; raw pointer graphs are rejected.
    pub fn gpu_projection(&self, definitions: &[TypeDef]) -> Option<Ty> {
        types::gpu_projection(self, definitions)
    }

    /// Shader root and shared owner carried by an opaque pipeline value.
    pub fn gpu_pipeline(&self) -> Option<(&Ty, &Ty)> {
        match self {
            Self::GpuComputePipeline { root, owner }
            | Self::GpuGraphicsPipeline { root, owner } => Some((root, owner)),
            _ => None,
        }
    }

    /// Host arguments accepted by dispatch or draw, including None for rootless draw.
    /// Invalid pipeline owners and roots have no argument contract.
    pub fn gpu_pipeline_argument(&self, definitions: &[TypeDef]) -> Option<Ty> {
        let (root, owner) = self.gpu_pipeline()?;
        if !matches!(owner, Self::Arc { .. }) {
            return None;
        }
        if *root == Self::None {
            return matches!(self, Self::GpuGraphicsPipeline { .. }).then_some(Self::None);
        }
        root.gpu_projection(definitions)
    }

    pub fn payloads(&self) -> Option<Vec<(Case, Ty)>> {
        types::payloads(self)
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
        types::variants(self)
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
        types::union_of(variants)
    }

    pub fn without_none(&self) -> Option<Self> {
        let members = self.members();
        members
            .contains(&Self::None)
            .then(|| Self::union_of(members.into_iter().filter(|ty| ty != &Self::None)))
    }

    pub fn widens_to(&self, to: &Self) -> bool {
        types::widens_to(self, to)
    }

    pub fn shader() -> Self {
        Self::Span {
            element: Box::new(Self::UInt8),
        }
    }

    pub fn view_record(&self) -> Option<Self> {
        types::view_record(self)
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
        types::parameter(types)
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
    StringFromBytes,
    Replace,
    Index,
    ArcGet,
    Downgrade,
    Upgrade,
    GpuIndex,
    GpuSlice,
    GpuReadOnly,
    GpuWriteOnly,
    GpuAllocateNative,
    GpuArgumentsDispatch,
    GpuArgumentsDraw,
    GpuCopyTo,
    GpuCopyImage,
}

/// Validate references in a concrete type against a program's canonical table.
pub fn check_references(table: &[TypeDef], ty: &Ty) -> Result<(), TypeError> {
    Ok(types::check_references(table, ty)?)
}
/// Validate that a nominal body's layout terminates without following indirection.
pub fn check_layout(table: &[TypeDef], id: TypeId, body: &Ty) -> Result<(), TypeError> {
    Ok(types::check_layout(table, id, body)?)
}
/// Resolve one nominal representation, rejecting incomplete or invalid definitions.
pub fn definition_body(table: &[TypeDef], id: TypeId) -> Result<&Ty, TypeError> {
    Ok(types::body(table, id)?)
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
    /// Literal bytes, excluding the trailing NUL added to their static storage.
    Str {
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

    /// Intern a resolved type without changing nominal identity. Nominal slots
    /// are reserved before their bodies, so recursive pointers terminate here.
    pub fn intern(&mut self, ty: &Ty) -> TypeId {
        types::intern(self, ty)
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
    StringFromBytes,
}

impl BuiltinRule {
    pub fn lookup(name: &str, arity: usize) -> Result<Self, TypeError> {
        typer::lookup(name, arity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv {
    Unwrap { definition: TypeId },
    Wrap { definition: TypeId },
    Deref,
    ViewRecord,
    MakeSpan,
    StrSpan,
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
        typer::same(expected, found)
    }

    /// Identity, a view representation change, or exactly one nominal wrap or unwrap.
    pub fn ascribe(&self, from: &Ty, to: &Ty) -> Result<Vec<Conv>, TypeError> {
        typer::ascribe(self, from, to)
    }

    /// Select the operation for an explicit source conversion. Unlike IR
    /// ascription, this also permits widening and numeric or pointer casts.
    pub fn explicit_conversion(&self, from: &Ty, to: &Ty) -> Result<ExplicitConversion, TypeError> {
        typer::explicit_conversion(self, from, to)
    }

    pub fn as_bool(&self, ty: &Ty) -> Result<(), TypeError> {
        typer::as_bool(ty)
    }

    pub fn as_record(&self, ty: &Ty) -> Result<Converted, TypeError> {
        typer::as_record(self, ty)
    }
}

/// Shared rule for `Instr::Ascribe`; `None` means the types are incompatible.
/// Borrow the definition table so verification can use the same nominal rules.
pub fn ascription(table: &[TypeDef], from: &Ty, to: &Ty) -> Result<Option<Vec<Conv>>, TypeError> {
    typer::ascription(table, from, to)
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
        typer::from_definitions(definitions)
    }

    pub fn definitions(&self) -> &[TypeDef] {
        &self.definitions
    }

    pub fn into_definitions(self) -> Result<TypeTable, TypeError> {
        typer::into_definitions(self)
    }

    pub fn create_type(
        &mut self,
        name: impl Into<Arc<str>>,
        body: Ty,
    ) -> Result<TypeId, TypeError> {
        typer::create_type(self, name, body)
    }

    pub fn reserve_type(&mut self, name: impl Into<Arc<str>>) -> TypeId {
        self.definitions.reserve(name.into())
    }

    pub fn define_type(&mut self, definition: TypeId, body: Ty) -> Result<(), TypeError> {
        typer::define_type(self, definition, body)
    }

    pub fn definition(&self, definition: TypeId) -> Result<&TypeDef, TypeError> {
        Ok(types::get(&self.definitions, definition)?)
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
        typer::type_field(self, base, name)
    }

    pub fn type_builtin_call(&self, name: &str, args: &[Ty]) -> Result<BuiltinCall, TypeError> {
        typer::type_builtin_call(self, name, args)
    }
}

//
// Type rendering
//

pub fn format_type(ty: &Ty, definitions: &[TypeDef]) -> String {
    types::format_type(ty, definitions)
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
        super::types::storage_layout(definitions, ty)
    }
}

//
// Numeric literal spelling and default types
//

pub mod literal {
    //! Numeric suffixes are case insensitive and select an exact primitive type.
    use super::Ty;

    pub fn split(text: &str) -> (&str, Option<Ty>) {
        super::types::split_literal(text)
    }

    /// Normalize only the suffix; keep digit grouping and hexadecimal digit case.
    pub fn format(text: &str) -> String {
        super::types::format_literal(text)
    }

    /// The fallback type for an unsuffixed literal, before any contextual inference.
    /// Hexadecimal e/E digits are not decimal exponents.
    pub fn unsuffixed_type(text: &str) -> Ty {
        super::types::unsuffixed_literal_type(text)
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

    /// A checked stage interface, shared by declaration checking and shader emission.
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
        super::typer::validate_shader(typer, parameter, result, foreign, stage)
    }

    /// Validate a compute stage or an ordered vertex/fragment pair for pipeline creation.
    /// Graphics stages must agree on their color type and any declared root; every root
    /// must support host GPU-view projection. Rootless graphics returns None.
    pub fn pipeline_root(typer: &TyperContext, stages: &[(&Ty, &Ty, &str)]) -> Result<Ty, String> {
        super::typer::pipeline_root(typer, stages)
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
