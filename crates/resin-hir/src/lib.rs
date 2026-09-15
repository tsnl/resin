//! High-level language: resolved, typed expressions with structured control flow.
//!
//! The public tree is the complete pass contract: resolved bindings, type schemes,
//! and completed applications, with no source scopes, inference variables, or stack instructions.
//! Construction establishes definite initialization for every source body. Editor
//! recovery stays behind the lowering and query operations.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_hir::lower;
//! ```
//! ```compile_fail
//! use resin_hir::snapshot;
//! ```

use resin_source::prelude::*;
use resin_types::prelude::*;
mod lower;
mod print;
mod snapshot;

pub use snapshot::{Diagnostic, Hir};

use resin_common::define_id;
use std::{collections::BTreeMap, fmt, sync::Arc};

//
// HIR language
//

define_id! {
    /// A named binder in a definition's type scheme.
    pub struct TypeParameterId(usize);
}

#[derive(Debug, Clone)]
pub struct TypeParameter {
    pub id: TypeParameterId,
    pub name: Ident,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RecordField {
    pub name: Arc<str>,
    pub ty: Type,
}

/// A method namespace and application determined by substituting the receiver type.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MethodLookup {
    pub receiver: Type,
    pub name: Arc<str>,
    /// Completed method-local arguments; owner arguments come from the receiver.
    pub type_args: Vec<Type>,
    /// Associated lookup retains every parameter; instance lookup omits the receiver.
    pub associated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Type {
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
    Str,
    Foreign {
        name: Arc<str>,
    },
    Parameter {
        parameter: TypeParameterId,
    },
    /// Determined by a field of the substituted base, without inverse inference.
    Member {
        base: Box<Type>,
        name: Arc<str>,
    },
    /// The selected method's caller-facing function type.
    Method {
        lookup: Box<MethodLookup>,
    },
    FunctionParameter {
        function: Box<Type>,
        index: usize,
    },
    FunctionResult {
        function: Box<Type>,
    },
    Defined {
        definition: TypeId,
        /// Arguments to this nominal declaration, retaining its original identity.
        arguments: Vec<Type>,
    },
    Pointer {
        pointee: Box<Type>,
    },
    /// An opaque shared allocation view with checked byte offsets and host permissions.
    GpuView,
    GpuPipelineContract,
    GpuArguments,
    /// Opaque shared allocation handle; copies retain and destruction releases.
    StrongOwner,
    /// Opaque weak allocation handle; it does not keep payloads alive.
    WeakOwner,
    Array {
        element: Box<Type>,
        length: usize,
    },
    Record {
        fields: Vec<RecordField>,
    },
    Function {
        params: Vec<Type>,
        result: Box<Type>,
    },
    Union {
        variants: Vec<Type>,
    },
    Result {
        value: Box<Type>,
        error: Box<Type>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Constant {
    Type {
        ty: Type,
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
}

#[derive(Debug, Clone)]
pub struct TypeDefinition {
    pub type_params: Vec<TypeParameter>,
    pub name: Arc<str>,
    pub body: Type,
    pub methods: BTreeMap<Arc<str>, FunctionId>,
    /// A hook whose type parameters are supplied by this nominal application.
    pub drop: Option<FunctionId>,
    pub gpu_projection: Option<GpuProjection>,
    pub gpu_pipeline: Option<GpuPipeline>,
}

/// A source declaration binds the nominal pipeline's root and owner parameters.
#[derive(Debug, Clone)]
pub struct GpuPipeline {
    pub declaration: FunctionId,
    pub kind: resin_types::GpuPipelineKind,
    pub root: Type,
    pub owner: Type,
}

/// A source declaration explicitly registers a wrapper's shader projection.
#[derive(Debug, Clone)]
pub struct GpuProjection {
    pub declaration: FunctionId,
    pub kind: resin_types::GpuProjectionKind,
    pub target: Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Case {
    Ok,
    Err,
    Type { ty: Type },
}

/// A lexical declaration's identity. Names survive only for diagnostics.
pub type BindingId = usize;

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    pub types: Vec<TypeDefinition>,
    pub functions: Vec<Function>,
    pub shaders: BTreeMap<FunctionId, ShaderEntry>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub location: Option<SourceLocation>,
    pub name: Arc<str>,
    pub signature: Signature,
    /// External declarations have no function body.
    pub foreign_header: Option<Arc<str>>,
    pub body: Option<Term>,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub type_params: Vec<TypeParameter>,
    pub params: Vec<Parameter>,
    pub result: Annotation,
}

#[derive(Debug, Clone)]
pub struct Parameter {
    /// Required for every parameter of a function with a body, including unused ones.
    pub binding: Option<BindingId>,
    pub name: Ident,
    pub annotation: Annotation,
}

#[derive(Debug, Clone)]
pub struct Annotation {
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub span: Span,
    pub ty: Type,
    pub kind: TermKind,
}

#[derive(Debug, Clone)]
pub enum TermKind {
    Constant {
        value: Constant,
    },
    /// Its type is already determined; specialization checks the concrete range.
    Numeric {
        text: Arc<str>,
    },
    /// A type-only query: the source operand is never evaluated.
    Layout {
        of: Type,
        size: bool,
    },
    Local {
        binding: BindingId,
        name: Ident,
    },
    Function {
        function: FunctionId,
        type_args: Vec<Type>,
    },
    /// An associated method reference whose declaration depends on substitution.
    DependentMethod {
        lookup: MethodLookup,
    },
    /// Select and adapt the receiver while lowering a concrete function instance.
    DependentMethodCall {
        lookup: MethodLookup,
        receiver: Option<Box<Term>>,
        args: Vec<Term>,
    },
    Unwrap {
        value: Box<Term>,
    },
    Try {
        value: Box<Term>,
    },
    Match {
        value: Box<Term>,
        arms: Vec<MatchArm>,
    },
    If {
        cond: Box<Term>,
        then: Box<Term>,
        els: Box<Term>,
    },
    While {
        cond: Box<Term>,
        body: Box<Term>,
    },
    Block {
        stmts: Vec<Statement>,
        tail: Box<Term>,
    },
    Record {
        fields: Vec<(Ident, Term)>,
    },
    Array {
        elems: Vec<Term>,
    },
    Builtin {
        name: Arc<str>,
        args: Vec<Term>,
    },
    Call {
        func: Box<Term>,
        args: Vec<Term>,
    },
    Intrinsic {
        op: Intrinsic,
        type_args: Vec<Type>,
        args: Arguments,
    },
    Adapt {
        conversion: ReceiverConversion,
        arg: Box<Term>,
    },
    Convert {
        arg: Box<Term>,
    },
    /// Create an owning pipeline whose root type comes from its shader declarations.
    GpuPipelineCreate {
        factory: FunctionId,
        shaders: Vec<FunctionId>,
        args: Arguments,
    },
    /// Project the checked host arguments and record a dispatch or draw.
    GpuPipelineDispatch {
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
        args: Arguments,
    },
    Result {
        failure: bool,
        arg: Box<Term>,
    },
    Absurd {
        arg: Box<Term>,
    },
    Assign {
        place: Box<Term>,
        value: Box<Term>,
    },
    Address {
        place: Box<Term>,
    },
    Deref {
        pointer: Box<Term>,
    },
    /// The member name is resolved; its concrete index and representation belong to LIR.
    Field {
        base: Box<Term>,
        name: Arc<str>,
    },
}

/// Intrinsic operands in evaluation order, paired with their expected parameter types.
/// An adapted method receiver occupies the first position; both lists have equal length.
#[derive(Debug, Clone)]
pub struct Arguments {
    pub values: Vec<Term>,
    pub params: Vec<Type>,
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub tag: Case,
    pub binding: Option<BindingId>,
    pub body: Term,
}

#[derive(Debug, Clone)]
pub enum Statement {
    Define {
        binding: BindingId,
        name: Ident,
        init: Term,
    },
    Declare {
        binding: BindingId,
        name: Ident,
        ty: Annotation,
    },
    Expr {
        term: Term,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReceiverConversion {
    Value,
    Address,
    Load,
}

//
// HIR construction and printing
//

/// Build HIR from declarations in import order.
/// Editor facts remain even when a complete tree cannot be produced.
pub fn build_hir(program: &resin_ast::Program) -> CheckedProgram {
    lower::analyze_program(program)
}

pub fn format_module(module: &Module) -> String {
    print::format_module(module)
}

/// Render a type using nominal declaration names and their parameter names.
pub fn format_type(ty: &Type, definitions: &[TypeDefinition]) -> String {
    print::format_type(ty, definitions)
}

//
// Editor analysis
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DefinitionKind {
    Function,
    Variable,
    Parameter,
    Type,
    Keyword,
    Field,
}
#[derive(Debug, Clone, Default)]
pub struct Analysis {
    imports: BTreeMap<SourceLocation, Source>,
    contexts: crate::lower::scope::Contexts,
    fields: BTreeMap<SourceLocation, Vec<Member>>,
    field_origins: BTreeMap<(TypeId, String), SourceLocation>,
    method_origins: BTreeMap<(TypeId, String), lower::scope::DeclarationId>,
    typer: lower::context::Context,
}

/// Diagnostics and editor facts, with a complete module only when checking succeeds.
pub struct CheckedProgram {
    pub module: Option<crate::Module>,
    pub diagnostics: Vec<SourceError>,
    pub semantics: Analysis,
}

impl CheckedProgram {
    /// Take the completed module, or the first diagnostic.
    pub fn into_module(self) -> Result<Module, SourceError> {
        match self.module {
            Some(module) if self.diagnostics.is_empty() => Ok(module),
            _ => Err(self
                .diagnostics
                .into_iter()
                .next()
                .expect("failed HIR has a diagnostic")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hover {
    pub span: Span,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Completion {
    pub name: String,
    pub detail: String,
    pub kind: DefinitionKind,
    pub replace: Span,
}

impl Analysis {
    pub fn definition(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        source: &Source,
        offset: usize,
    ) -> Option<SourceLocation> {
        let document = documents.get(source)?;
        let token = document.token(offset)?;
        self.import_definition(source, offset)
            .or_else(|| self.token_definition(source, offset, document, token))
    }

    pub fn hover(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        source: &Source,
        offset: usize,
    ) -> Option<Hover> {
        let document = documents.get(source)?;
        let token = document.token(offset)?;
        let text = builtin_hover(document, token)
            .or_else(|| self.resolved_member_hover(source, document, token))
            .or_else(|| self.definition_hover(documents, source, offset))
            .or_else(|| self.member_hover(source, document, token))?;
        Some(Hover {
            span: resin_cst::span(token),
            text,
        })
    }

    pub fn completions(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        source: &Source,
        offset: usize,
    ) -> Vec<Completion> {
        let Some(document) = documents.get(source) else {
            return Vec::new();
        };
        let Some(replace) = completion_range(document, offset) else {
            return Vec::new();
        };
        if document.source()[..replace.start].trim_end().ends_with('.') {
            return self.field_completions(source, document, replace, offset);
        }
        let items = self.visible_completions(documents, source, offset, replace);
        matching_completions(items, &document.source()[replace.start..offset])
    }

    fn import_definition(&self, source: &Source, offset: usize) -> Option<SourceLocation> {
        self.imports.iter().find_map(|(location, target)| {
            (&location.source == source && resin_cst::contains(location.span, offset)).then(|| {
                SourceLocation {
                    source: target.clone(),
                    span: Span { start: 0, end: 0 },
                }
            })
        })
    }

    fn token_definition(
        &self,
        source: &Source,
        offset: usize,
        document: &resin_cst::Document,
        token: resin_cst::Node<'_>,
    ) -> Option<SourceLocation> {
        let location = SourceLocation {
            source: source.clone(),
            span: resin_cst::span(token),
        };
        if self
            .contexts
            .definitions
            .iter()
            .any(|definition| definition.location == location)
        {
            return Some(location);
        }
        if let Some(origin) = self
            .member(&location, document.node_text(token))
            .and_then(|member| member.origin.clone())
        {
            return Some(origin);
        }
        if !document.reference(token) {
            return None;
        }
        self.contexts
            .definition(
                source,
                offset,
                document.node_text(token),
                token.kind() == "uid",
            )
            .map(|definition| definition.location.clone())
    }

    fn member(&self, location: &SourceLocation, name: &str) -> Option<&Member> {
        self.fields
            .get(location)?
            .iter()
            .find(|member| member.name == name)
    }

    fn definition_hover(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        source: &Source,
        offset: usize,
    ) -> Option<String> {
        let origin = self.definition(documents, source, offset)?;
        let definition = self
            .contexts
            .definitions
            .iter()
            .find(|definition| definition.location == origin)?;
        Some(self.definition_label(documents, definition))
    }

    fn member_hover(
        &self,
        source: &Source,
        document: &resin_cst::Document,
        token: resin_cst::Node<'_>,
    ) -> Option<String> {
        let location = SourceLocation {
            source: source.clone(),
            span: resin_cst::span(token),
        };
        let member = self.member(&location, document.node_text(token))?;
        Some(format!("{}: {}", member.name, member.ty))
    }

    fn resolved_member_hover(
        &self,
        source: &Source,
        document: &resin_cst::Document,
        token: resin_cst::Node<'_>,
    ) -> Option<String> {
        let location = SourceLocation {
            source: source.clone(),
            span: resin_cst::span(token),
        };
        let member = self.member(&location, document.node_text(token))?;
        (member.compiler_signature || member.kind == DefinitionKind::Field)
            .then(|| format!("{}: {}", member.name, member.ty))
    }

    fn type_names(&self) -> print::TypeNames {
        self.type_names_with(&self.typer)
    }

    fn type_names_with(&self, typer: &lower::context::Context) -> print::TypeNames {
        print::TypeNames {
            definitions: typer
                .definitions()
                .iter()
                .map(|definition| definition.name().cloned().unwrap_or_else(|| "?".into()))
                .collect(),
            parameters: self
                .contexts
                .definitions
                .iter()
                .filter_map(|definition| {
                    if let Some(Type::Parameter { parameter }) = &definition.ty {
                        (definition.kind == DefinitionKind::Type)
                            .then(|| (*parameter, definition.name.clone().into()))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    fn definition_label(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        definition: &Definition,
    ) -> String {
        if matches!(
            definition.kind,
            DefinitionKind::Variable | DefinitionKind::Parameter
        ) {
            let ty = definition
                .ty
                .as_ref()
                .map(|ty| self.type_names().format(ty))
                .unwrap_or_else(|| "?".into());
            return format!("{}: {ty}", definition.name);
        }
        documents
            .get(&definition.location.source)
            .and_then(|document| declaration_label(document, definition))
            .unwrap_or_else(|| definition.label.clone())
    }

    fn visible_completions(
        &self,
        documents: &BTreeMap<Source, Arc<resin_cst::Document>>,
        source: &Source,
        offset: usize,
        replace: Span,
    ) -> Vec<Completion> {
        let types = documents[source].type_context(offset);
        let mut items = self
            .contexts
            .visible(source, offset)
            .into_iter()
            .filter(|definition| !types || definition.kind == DefinitionKind::Type)
            .map(|definition| Completion {
                detail: self.definition_label(documents, &definition),
                name: definition.name,
                kind: definition.kind,
                replace,
            })
            .collect::<Vec<_>>();
        items.extend(builtin_completions(types, replace));
        items
    }

    fn field_completions(
        &self,
        source: &Source,
        document: &resin_cst::Document,
        replace: Span,
        offset: usize,
    ) -> Vec<Completion> {
        let prefix = &document.source()[replace.start..offset];
        let dot = document.source()[..replace.start].trim_end().len() - 1;
        // While typing after a dot, the parser may use an identifier on the
        // following line as the member name. Its receiver still applies here.
        let fields = self.fields.iter().find_map(|(location, fields)| {
            (&location.source == source
                && location.span.start > dot
                && document.source()[dot + 1..location.span.start]
                    .trim()
                    .is_empty())
            .then_some(fields)
        });
        let mut items = fields
            .into_iter()
            .flatten()
            .filter(|member| member.name.starts_with(prefix))
            .map(|member| Completion {
                name: member.name.clone(),
                detail: format!("{}: {}", member.name, member.ty),
                kind: member.kind,
                replace,
            })
            .collect::<Vec<_>>();
        items.sort_by(|a, b| {
            (a.kind != DefinitionKind::Field, &a.name)
                .cmp(&(b.kind != DefinitionKind::Field, &b.name))
        });
        items
    }
}

fn builtin_hover(document: &resin_cst::Document, token: resin_cst::Node<'_>) -> Option<String> {
    if !document.reference(token)
        && !matches!(
            token.kind(),
            "builtin_type"
                | "Ptr"
                | "GpuPipelineContract"
                | "GpuView"
                | "GpuArguments"
                | "Result"
                | "None"
        )
    {
        return None;
    }
    BUILTINS
        .iter()
        .find(|(name, _, _)| *name == document.node_text(token))
        .map(|(_, help, _)| help.to_string())
}

fn declaration_label(document: &resin_cst::Document, definition: &Definition) -> Option<String> {
    let node = document.token(definition.location.span.start)?.parent()?;
    match definition.kind {
        DefinitionKind::Function => function_label(document, node),
        DefinitionKind::Type => node
            .child_by_field_name("init")
            .map(|init| format!("{} = {}", definition.name, document.node_text(init))),
        _ => None,
    }
}

fn function_label(document: &resin_cst::Document, node: resin_cst::Node<'_>) -> Option<String> {
    if let Some(result) = node.child_by_field_name("result") {
        return Some(document.source()[node.start_byte()..result.end_byte()].to_owned());
    }
    node.children(&mut node.walk())
        .find(|child| child.kind() == ")" && !child.is_missing())
        .map(|end| {
            format!(
                "{} -> ()",
                &document.source()[node.start_byte()..end.end_byte()]
            )
        })
}

fn completion_range(document: &resin_cst::Document, offset: usize) -> Option<Span> {
    if !document.source().is_char_boundary(offset) {
        return None;
    }
    if document
        .token(offset)
        .is_some_and(|token| match token.kind() {
            "string" => offset < token.end_byte(),
            "comment" => offset < token.end_byte() || document.node_text(token).starts_with("//"),
            _ => false,
        })
    {
        return None;
    }
    let bytes = document.source().as_bytes();
    let identifier = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut start = offset;
    let mut end = offset;
    while start > 0 && identifier(bytes[start - 1]) {
        start -= 1;
    }
    while end < bytes.len() && identifier(bytes[end]) {
        end += 1;
    }
    Some(Span { start, end })
}

fn builtin_completions(types: bool, replace: Span) -> impl Iterator<Item = Completion> {
    BUILTINS
        .iter()
        .filter(move |(_, _, kind)| !types || *kind == DefinitionKind::Type)
        .map(move |(name, help, kind)| Completion {
            name: name.to_string(),
            detail: help.to_string(),
            kind: *kind,
            replace,
        })
}

fn matching_completions(mut items: Vec<Completion>, prefix: &str) -> Vec<Completion> {
    items.retain(|item| item.name.starts_with(prefix));
    items.sort_by(|a, b| a.name.cmp(&b.name));
    items.dedup_by(|a, b| a.name == b.name);
    items
}

const BUILTINS: &[(&str, &str, DefinitionKind)] = &[
    (
        "str",
        "str\n\nA string literal view with data: Ptr<ubyte> and length: ulong. Static storage has a trailing NUL excluded from length. Import $/span.resin and use bytes(text) to borrow its bytes. Host-only.",
        DefinitionKind::Type,
    ),
    (
        "Ptr",
        "Ptr<T>\n\nAn unchecked pointer to T.",
        DefinitionKind::Type,
    ),
    (
        "StrongOwner",
        "StrongOwner\n\nAn opaque shared host allocation handle. Copies retain its initialized payload; the final release destroys it.",
        DefinitionKind::Type,
    ),
    (
        "WeakOwner",
        "WeakOwner\n\nAn opaque weak allocation handle. It retains bookkeeping without keeping the payload alive.",
        DefinitionKind::Type,
    ),
    (
        "GpuView",
        "GpuView\n\nAn opaque GPU allocation view retaining its owner, byte offset, and host access permissions.",
        DefinitionKind::Type,
    ),
    (
        "GpuPipelineContract",
        "GpuPipelineContract\n\nAn opaque pipeline token retaining its native owner and binding its shader root, owner type, and stage.",
        DefinitionKind::Type,
    ),
    (
        "GpuArguments",
        "GpuArguments\n\nAn internal dispatch projection retaining referenced GPU allocations.",
        DefinitionKind::Type,
    ),
    (
        "Result",
        "Result<T, E> — success or a typed error; postfix ? propagates errors.",
        DefinitionKind::Type,
    ),
    (
        "Never",
        "Never — the empty union, with no possible values.",
        DefinitionKind::Type,
    ),
    (
        "ok",
        "ok(value) — construct a successful Result.",
        DefinitionKind::Function,
    ),
    (
        "err",
        "err(error) — construct a failed Result.",
        DefinitionKind::Function,
    ),
    (
        "struct",
        "struct Name { field: Type }; — a nominal record type.",
        DefinitionKind::Keyword,
    ),
    (
        "match",
        "match (value) { Variant(name) => { body }, ... }",
        DefinitionKind::Keyword,
    ),
    (
        "None",
        "None — singleton value and type; T | None permits absence, postfix ! excludes it or traps.",
        DefinitionKind::Type,
    ),
    ("bool", "bool", DefinitionKind::Type),
    (
        "sbyte",
        "sbyte — signed 8-bit integer",
        DefinitionKind::Type,
    ),
    (
        "short",
        "short — signed 16-bit integer",
        DefinitionKind::Type,
    ),
    ("int", "int — signed 32-bit integer", DefinitionKind::Type),
    ("long", "long — signed 64-bit integer", DefinitionKind::Type),
    (
        "ubyte",
        "ubyte — unsigned 8-bit integer",
        DefinitionKind::Type,
    ),
    (
        "ushort",
        "ushort — unsigned 16-bit integer",
        DefinitionKind::Type,
    ),
    (
        "uint",
        "uint — unsigned 32-bit integer",
        DefinitionKind::Type,
    ),
    (
        "ulong",
        "ulong — unsigned 64-bit integer",
        DefinitionKind::Type,
    ),
    ("float32", "float32", DefinitionKind::Type),
    ("float64", "float64", DefinitionKind::Type),
    ("export", "export { name };", DefinitionKind::Keyword),
    (
        "import",
        "import { \"path.resin\" };",
        DefinitionKind::Keyword,
    ),
    (
        "extern",
        "extern — foreign function or opaque type declaration",
        DefinitionKind::Keyword,
    ),
    (
        "intrinsic",
        "intrinsic \"operation\" def name<T>(parameters) -> Type;",
        DefinitionKind::Keyword,
    ),
    (
        "def",
        "def name(parameters) -> Type = { body };\n\nOmitted result annotations default to ().",
        DefinitionKind::Keyword,
    ),
    (
        "var",
        "var name = value;\nvar name: Type;",
        DefinitionKind::Keyword,
    ),
    (
        "type",
        "type Name = Type;\nextern type Name;",
        DefinitionKind::Keyword,
    ),
    (
        "if",
        "if (condition) { value } else { value }",
        DefinitionKind::Keyword,
    ),
    ("else", "else { value }", DefinitionKind::Keyword),
    (
        "while",
        "while (condition) { body };",
        DefinitionKind::Keyword,
    ),
];

#[derive(Debug, Clone)]
struct Definition {
    name: String,
    location: SourceLocation,
    kind: DefinitionKind,
    label: String,
    ty: Option<Type>,
    member: bool,
}
#[derive(Debug, Clone)]
struct Member {
    name: String,
    ty: String,
    kind: DefinitionKind,
    origin: Option<SourceLocation>,
    compiler_signature: bool,
}
impl Analysis {
    fn record_method_call(
        &mut self,
        location: &SourceLocation,
        receiver: &Ty,
        name: &str,
        arguments: &[Ty],
        associated: bool,
        typer: &lower::context::Context,
    ) {
        let Some(method) = typer.method(receiver, name) else {
            return;
        };
        if !matches!(
            method.body,
            lower::context::FunctionBody::GpuPipelineFactory { .. }
                | lower::context::FunctionBody::GpuPipelineRecord { .. }
        ) {
            return;
        }
        let count = method.params.len() - usize::from(!associated);
        if arguments.len() != count {
            return;
        }
        let arguments = arguments[usize::from(associated)..]
            .iter()
            .map(|ty| {
                lower::infer::Solver::default()
                    .complete(&ty.clone().into())
                    .expect("concrete bridge argument")
            })
            .collect::<Vec<_>>();
        let Ok(method) = typer.source_pipeline_method(&method, &arguments) else {
            return;
        };
        let signature = Type::Function {
            params: method.params[usize::from(!associated)..].to_vec(),
            result: Box::new(method.result),
        };
        let label = self.type_names_with(typer).format(&signature);
        if let Some(member) = self
            .fields
            .get_mut(location)
            .and_then(|members| members.iter_mut().find(|member| member.name == name))
        {
            member.ty = label;
        }
    }

    fn record_members(
        &mut self,
        location: SourceLocation,
        ty: &Ty,
        associated: bool,
        typer: &lower::context::Context,
    ) {
        let mut members = Vec::new();
        if !associated
            && let Ok(converted) = typer.as_record(ty)
            && let Ty::Record { fields } = converted.ty.view_record().unwrap_or(converted.ty)
        {
            members.extend(fields.into_iter().map(|field| {
                Member {
                    name: field
                        .name
                        .strip_prefix('_')
                        .filter(|name| name.chars().all(|c| c.is_ascii_digit()))
                        .unwrap_or(&field.name)
                        .to_string(),
                    ty: format_concrete_type(&field.ty, typer),
                    kind: DefinitionKind::Field,
                    origin: typer.receiver_definition(ty).and_then(|receiver| {
                        self.field_origins
                            .get(&(receiver, field.name.to_string()))
                            .cloned()
                    }),
                    compiler_signature: false,
                }
            }));
        }
        for (name, method) in typer.methods(ty) {
            let Some(params) = method.arguments(ty, associated) else {
                continue;
            };
            let signature = Ty::Function {
                params: params.to_vec(),
                result: Box::new(method.result.clone()),
            };
            let origin = if matches!(
                method.body,
                crate::lower::context::FunctionBody::Defined(_)
                    | crate::lower::context::FunctionBody::GpuPipelineFactory { .. }
                    | crate::lower::context::FunctionBody::GpuPipelineRecord { .. }
            ) {
                typer.receiver_definition(ty).and_then(|receiver| {
                    self.method_origins
                        .get(&(receiver, name.to_string()))
                        .map(|origin| self.contexts.definitions[*origin].location.clone())
                })
            } else {
                None
            };
            members.retain(|member| member.name != name.as_ref());
            members.push(Member {
                name: name.to_string(),
                ty: typer
                    .gpu_method_label(&method, associated)
                    .unwrap_or_else(|| format_concrete_type(&signature, typer)),
                kind: DefinitionKind::Function,
                origin,
                compiler_signature: typer.gpu_method_label(&method, associated).is_some(),
            });
        }
        self.fields.insert(location, members);
    }

    fn record_symbolic_members(
        &mut self,
        location: SourceLocation,
        ty: &Type,
        associated: bool,
        typer: &lower::context::Context,
        solver: &lower::infer::Solver,
    ) {
        if associated {
            return;
        }
        let mut receiver = ty;
        while let Type::Pointer { pointee } = receiver {
            receiver = pointee;
        }
        let body = match receiver {
            Type::Defined { .. } => typer
                .nominal_body(&lower::infer::Type::from_hir(receiver), solver)
                .and_then(|body| solver.complete(&body)),
            Type::Record { .. } => Some(receiver.clone()),
            _ => None,
        };
        let Some(Type::Record { fields }) = body else {
            return;
        };
        let names = self.type_names_with(typer);
        let members = fields
            .into_iter()
            .map(|field| Member {
                name: field
                    .name
                    .strip_prefix('_')
                    .filter(|name| name.chars().all(|c| c.is_ascii_digit()))
                    .unwrap_or(&field.name)
                    .to_string(),
                ty: names.format(&field.ty),
                kind: DefinitionKind::Field,
                origin: match receiver {
                    Type::Defined { definition, .. } => self
                        .field_origins
                        .get(&(*definition, field.name.to_string()))
                        .cloned(),
                    _ => None,
                },
                compiler_signature: false,
            })
            .collect::<Vec<_>>();
        let existing = self.fields.entry(location).or_default();
        for member in members {
            if !existing.iter().any(|existing| existing.name == member.name) {
                existing.push(member);
            }
        }
    }

    fn record_intrinsic_methods(
        &mut self,
        location: SourceLocation,
        ty: &lower::infer::Type,
        associated: bool,
        typer: &lower::context::Context,
        solver: &lower::infer::Solver,
    ) {
        let names = self.type_names_with(typer);
        let members = lower::context::intrinsic_methods(ty, solver)
            .into_iter()
            .filter_map(|(name, method)| {
                let signature = lower::infer::Type::function(
                    method.params[usize::from(!associated)..].to_vec(),
                    method.result,
                );
                Some(Member {
                    name: name.into(),
                    ty: names.format(&solver.complete(&signature)?),
                    kind: DefinitionKind::Function,
                    origin: None,
                    compiler_signature: true,
                })
            })
            .collect::<Vec<_>>();
        let existing = self.fields.entry(location).or_default();
        for member in members {
            existing.retain(|existing| existing.name != member.name);
            existing.push(member);
        }
    }

    fn record_source_methods(
        &mut self,
        location: SourceLocation,
        ty: &Type,
        associated: bool,
        typer: &lower::context::Context,
        solver: &lower::infer::Solver,
    ) {
        let mut owner = ty;
        while let Type::Pointer { pointee } = owner {
            owner = pointee;
        }
        let Type::Defined {
            definition,
            arguments,
        } = owner
        else {
            return;
        };
        let names = self.type_names_with(typer);
        let mut members = Vec::new();
        for (name, method) in typer.source_methods_for(*definition) {
            let substitute = |body: &lower::infer::Type| lower::infer::Type::Apply {
                body: Box::new(body.clone()),
                arguments: method
                    .owner_params
                    .iter()
                    .zip(arguments)
                    .map(|(parameter, argument)| {
                        (parameter.id, lower::infer::Type::from_hir(argument))
                    })
                    .collect(),
            };
            let params = method.params.iter().map(substitute).collect::<Vec<_>>();
            if !associated
                && !source_method_receiver(ty, params.first(), method, solver, location.span)
            {
                continue;
            }
            let signature = lower::infer::Type::function(
                params[usize::from(!associated)..].to_vec(),
                substitute(&method.result),
            );
            let Some(signature) = solver.complete(&signature) else {
                continue;
            };
            let binders = method
                .type_params
                .iter()
                .map(|parameter| parameter.name.val.as_ref())
                .collect::<Vec<_>>();
            let signature = if binders.is_empty() {
                names.format(&signature)
            } else {
                format!("<{}> {}", binders.join(", "), names.format(&signature))
            };
            members.push(Member {
                name: name.to_string(),
                ty: signature,
                kind: DefinitionKind::Function,
                origin: Some(
                    self.contexts.definitions[method.declaration]
                        .location
                        .clone(),
                ),
                compiler_signature: !method.owner_params.is_empty()
                    || !method.type_params.is_empty(),
            });
        }
        let existing = self.fields.entry(location).or_default();
        for member in members {
            existing.retain(|existing| existing.name != member.name);
            existing.push(member);
        }
    }

    fn record_resolved_method_call(
        &mut self,
        location: &SourceLocation,
        name: &str,
        method: &lower::infer::ResolvedMethod,
        associated: bool,
        typer: &lower::context::Context,
        solver: &lower::infer::Solver,
    ) {
        if let lower::infer::ResolvedMethod::GpuPipeline { method } = method {
            let signature = Type::Function {
                params: method.params[usize::from(!associated)..].to_vec(),
                result: Box::new(method.result.clone()),
            };
            let label = self.type_names_with(typer).format(&signature);
            let members = self.fields.entry(location.clone()).or_default();
            if let Some(member) = members.iter_mut().find(|member| member.name == name) {
                member.ty = label;
            }
            return;
        }
        let (params, result, origin, compiler_signature) = match method {
            lower::infer::ResolvedMethod::Source {
                declaration,
                type_args,
                params,
                result,
                ..
            } => (
                params,
                result,
                Some(self.contexts.definitions[*declaration].location.clone()),
                !type_args.is_empty(),
            ),
            lower::infer::ResolvedMethod::Intrinsic { signature, .. } => {
                (&signature.params, &signature.result, None, true)
            }
            _ => return,
        };
        let signature = lower::infer::Type::function(
            params[usize::from(!associated)..].to_vec(),
            result.clone(),
        );
        let Some(signature) = solver.complete(&signature) else {
            return;
        };
        let member = Member {
            name: name.to_owned(),
            ty: self.type_names_with(typer).format(&signature),
            kind: DefinitionKind::Function,
            origin,
            compiler_signature,
        };
        let existing = self.fields.entry(location.clone()).or_default();
        existing.retain(|existing| existing.name != name);
        existing.push(member);
    }
}

fn source_method_receiver(
    receiver: &Type,
    first: Option<&lower::infer::Type>,
    method: &lower::context::SourceMethod,
    solver: &lower::infer::Solver,
    span: Span,
) -> bool {
    let Some(first) = first else {
        return false;
    };
    let mut solver = solver.clone();
    let Ok((first, _)) = solver.apply(first.clone(), &method.type_params, None, span) else {
        return false;
    };
    let source = lower::infer::Type::from_hir(receiver);
    let mut candidates = vec![source.clone(), lower::infer::Type::pointer(source)];
    if let Type::Pointer { pointee } = receiver {
        candidates.push(lower::infer::Type::from_hir(pointee));
    }
    candidates.into_iter().any(|candidate| {
        solver
            .clone()
            .unify(&candidate, &first, span)
            .is_ok_and(|complete| complete)
    })
}
fn format_concrete_type(ty: &Ty, typer: &lower::context::Context) -> String {
    resin_types::format_type(ty, typer.definitions())
}

//
// HIR construction errors
//

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateError {
    pub span: Span,
    pub kind: GenerateErrorKind,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateErrorKind {
    IncompleteSyntax,
    Inference { message: Arc<str> },
    InvalidModuleItem,
    InvalidForeignSignature,
    InvalidShader { message: Arc<str> },
    UnresolvedImport { path: Arc<str> },
    UnknownExport { name: Arc<str> },
    DuplicateExport { name: Arc<str> },
    ReservedBuiltin { name: Arc<str> },
    Type { kind: TypeErrorKind },
    UnboundValue { name: Arc<str> },
    UnboundType { name: Arc<str> },
    UnknownTypeFormer { name: Arc<str> },
    EagerRecursion { name: Arc<str> },
    UninitializedValue { name: Arc<str> },
    DuplicateValue { name: Arc<str> },
    DuplicateType { name: Arc<str> },
    NeedsTypeAnnotation { name: Arc<str> },
    MissingField { name: Arc<str> },
    ExtraField { name: Arc<str> },
    NotAPlace,
    InvalidLiteral { message: Arc<str> },
}
impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if matches!(
            self.kind,
            GenerateErrorKind::Type {
                kind: TypeErrorKind::PointerArithmetic
            }
        ) {
            return write!(
                f,
                "compile error at {}..{}: {}",
                self.span.start,
                self.span.end,
                TypeError {
                    kind: TypeErrorKind::PointerArithmetic
                }
            );
        }
        if let GenerateErrorKind::Inference { message } = &self.kind {
            return write!(
                f,
                "compile error at {}..{}: {message}",
                self.span.start, self.span.end
            );
        }
        write!(
            f,
            "compile error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}
impl std::error::Error for GenerateError {}

impl GenerateError {
    pub fn inference(span: Span, message: impl Into<Arc<str>>) -> Self {
        Self {
            span,
            kind: GenerateErrorKind::Inference {
                message: message.into(),
            },
        }
    }

    pub fn typing(span: Span, error: TypeError) -> Self {
        Self {
            span,
            kind: GenerateErrorKind::Type { kind: error.kind },
        }
    }
}
