//! High-level language: resolved, typed expressions with structured control flow.
//!
//! The public tree is the complete pass contract: resolved bindings and concrete
//! types, with no source scopes, inference variables, or stack instructions.
//! Construction and editor recovery stay behind the lowering and query operations.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_hir::lower;
//! ```

use resin_source::prelude::*;
use resin_types::prelude::*;
mod lower;
mod print;

use std::{collections::BTreeMap, fmt, sync::Arc};

//
// HIR language
//

/// A lexical declaration's identity. Names survive only for diagnostics.
pub type BindingId = usize;

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    pub types: TypeTable,
    pub functions: Vec<Function>,
    pub shaders: BTreeMap<FunctionId, ShaderEntry>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub location: Option<SourceLocation>,
    pub name: Arc<str>,
    pub signature: Signature,
    /// External declarations have no function body.
    pub foreign: Option<Foreign>,
    pub body: Option<Term>,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub params: Vec<Parameter>,
    pub result: Annotation,
}

impl Signature {
    pub fn parameter_type(&self) -> Ty {
        Ty::parameter(
            &self
                .params
                .iter()
                .map(|parameter| parameter.annotation.ty.clone())
                .collect::<Vec<_>>(),
        )
    }
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
    pub ty: Ty,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Term {
    pub span: Span,
    pub ty: Ty,
    pub kind: TermKind,
}

#[derive(Debug, Clone)]
pub enum TermKind {
    Constant {
        value: Value,
    },
    Local {
        binding: BindingId,
        name: Ident,
    },
    Function {
        function: FunctionId,
    },
    Shader {
        function: FunctionId,
        stage: Arc<str>,
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
        arg: Box<Term>,
    },
    /// Evaluate a receiver, then unpack the remaining arguments in source order.
    Pack {
        args: Arguments,
    },
    Intrinsic {
        op: Intrinsic,
        args: Arguments,
    },
    Adapt {
        conversion: ReceiverConversion,
        arg: Box<Term>,
    },
    Convert {
        conversion: resin_types::ExplicitConversion,
        arg: Box<Term>,
    },
    ArcNew {
        value: Box<Term>,
    },
    /// Allocate and initialize a GPU element through the registered allocator.
    GpuNew {
        allocator: FunctionId,
        args: Arguments,
    },
    /// Allocate uninitialized GPU elements through the registered allocator.
    GpuAllocate {
        allocator: FunctionId,
        args: Arguments,
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
    WeakEmpty {
        pointee: Ty,
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
    /// The projection is resolved during HIR construction; names are not looked up again.
    Field {
        base: Box<Term>,
        access: FieldAccess,
    },
}

#[derive(Debug, Clone)]
pub struct Arguments {
    pub receiver: Option<Box<Term>>,
    pub argument: Box<Term>,
    pub params: Vec<Ty>,
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
    ArcAddress,
    ArcLoad,
}

//
// HIR construction and printing
//

/// Check a standalone source file; imports require a resolved Program.
pub fn generate(file: &resin_ast::SourceFile) -> Result<Module, GenerateError> {
    lower::generate(file)
}

/// Check declarations in import order, requiring a completely typed tree.
pub fn generate_program(program: &resin_ast::Program) -> Result<Module, SourceError> {
    lower::generate_program(program)
}

pub fn format_module(module: &Module) -> String {
    print::format_module(module)
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
    method_origins: BTreeMap<(TypeId, String), lower::scope::DeclarationId>,
    typer: lower::context::Context,
}

/// Diagnostics and editor facts, with a complete module only when checking succeeds.
pub struct CheckedProgram {
    pub module: Option<crate::Module>,
    pub diagnostics: Vec<SourceError>,
    pub semantics: Analysis,
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

/// Retain diagnostics and editor facts even when a complete tree cannot be produced.
pub fn analyze_program(program: &resin_ast::Program) -> CheckedProgram {
    lower::analyze_program(program)
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
            .or_else(|| self.compiler_member_hover(source, document, token))
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
            .and_then(|member| member.origin)
        {
            return Some(self.contexts.definitions[origin].location.clone());
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

    fn compiler_member_hover(
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
        member
            .compiler_signature
            .then(|| format!("{}: {}", member.name, member.ty))
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
                .map(|ty| format_type(ty, &self.typer))
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
        let fields = self.fields.iter().find_map(|(location, fields)| {
            (&location.source == source
                && location.span.start > dot
                && location.span.start <= replace.start)
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
                | "Span"
                | "GpuPtr"
                | "GpuSpan"
                | "GpuArguments"
                | "GpuComputePipeline"
                | "GpuGraphicsPipeline"
                | "Result"
                | "Arc"
                | "Weak"
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
        "fmt",
        "fmt(format, arguments) -> String\n\nFormat a tuple using numbered placeholders {0}, {1}, … into an owned String. Host-only.",
        DefinitionKind::Function,
    ),
    (
        "str",
        "str\n\nA string literal view with data: Ptr<ubyte> and length: ulong. Static storage has a trailing NUL excluded from length. Span<ubyte>(text) exposes its bytes. Host-only.",
        DefinitionKind::Type,
    ),
    (
        "String",
        "String\n\nOwned bytes, wrapping Arc<Span<ubyte>>. Copies retain the allocation. String literals have type str.",
        DefinitionKind::Type,
    ),
    (
        "print",
        "print(text) -> ()\n\nWrite a str, String, or Span<ubyte> to stdout verbatim and flush.",
        DefinitionKind::Function,
    ),
    (
        "Ptr",
        "Ptr<T>\n\nAn unchecked pointer to T.",
        DefinitionKind::Type,
    ),
    (
        "Span",
        "Span<T>\n\nA pointer and length describing elements of T. Calling span.at(index: ulong) returns Ptr<T>; shader indexing is unchecked.",
        DefinitionKind::Type,
    ),
    (
        "GpuPtr",
        "GpuPtr<T>\n\nAn owning GPU allocation view with checked host access. Indexing and field addresses retain its allocation.",
        DefinitionKind::Type,
    ),
    (
        "GpuSpan",
        "GpuSpan<T>\n\nAn owning GPU range. Indexing and slicing preserve its owner and access permissions.",
        DefinitionKind::Type,
    ),
    (
        "GpuArguments",
        "GpuArguments\n\nAn internal dispatch projection retaining referenced GPU allocations.",
        DefinitionKind::Type,
    ),
    (
        "GpuComputePipeline",
        "GpuComputePipeline<T, Owner>\n\nAn owning compute pipeline retaining its shader root type T. Dispatch checks and projects its host arguments.",
        DefinitionKind::Type,
    ),
    (
        "GpuGraphicsPipeline",
        "GpuGraphicsPipeline<T, Owner>\n\nAn owning graphics pipeline retaining the shared shader root T. Rootless shaders use None.",
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
        "impl",
        "impl T { def method(self: Ptr<T>) = {}; } — inherent methods.",
        DefinitionKind::Keyword,
    ),
    (
        "None",
        "None — singleton value and type; T | None permits absence, postfix ! excludes it or traps.",
        DefinitionKind::Type,
    ),
    (
        "Arc",
        "Arc<T> — a copyable shared owner; copying retains the allocation.",
        DefinitionKind::Type,
    ),
    (
        "Weak",
        "Weak<T> — a weak handle; upgrade() returns Arc<T> | None.",
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
    ty: Option<Ty>,
    member: bool,
}
#[derive(Debug, Clone)]
struct Member {
    name: String,
    ty: String,
    kind: DefinitionKind,
    origin: Option<lower::scope::DeclarationId>,
    compiler_signature: bool,
}
impl Analysis {
    fn record_method_call(
        &mut self,
        location: &SourceLocation,
        receiver: &Ty,
        name: &str,
        argument: &Ty,
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
        let arguments = if count == 1 {
            vec![argument.clone()]
        } else {
            let Ty::Record { fields } = argument else {
                return;
            };
            if fields.len() != count {
                return;
            }
            fields.iter().map(|field| field.ty.clone()).collect()
        };
        let Ok(method) = typer.specialize_gpu_method(method, &arguments[usize::from(associated)..])
        else {
            return;
        };
        let Some(params) = method.arguments(receiver, associated) else {
            return;
        };
        let signature = Ty::Function {
            param: Box::new(Ty::parameter(params)),
            result: Box::new(method.result),
        };
        if let Some(member) = self
            .fields
            .get_mut(location)
            .and_then(|members| members.iter_mut().find(|member| member.name == name))
        {
            member.ty = format_type(&signature, typer);
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
            members.extend(fields.into_iter().map(|field| Member {
                name: field.name.to_string(),
                ty: format_type(&field.ty, typer),
                kind: DefinitionKind::Field,
                origin: None,
                compiler_signature: false,
            }));
        }
        for (name, method) in typer.methods(ty) {
            let Some(params) = method.arguments(ty, associated) else {
                continue;
            };
            let signature = Ty::Function {
                param: Box::new(Ty::parameter(params)),
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
                        .copied()
                })
            } else {
                None
            };
            members.retain(|member| member.name != name.as_ref());
            members.push(Member {
                name: name.to_string(),
                ty: typer
                    .gpu_method_label(&method, associated)
                    .unwrap_or_else(|| format_type(&signature, typer)),
                kind: DefinitionKind::Function,
                origin,
                compiler_signature: typer.gpu_method_label(&method, associated).is_some(),
            });
        }
        if let Some((name, signature)) = typer.generic_method_label(ty, associated) {
            members.retain(|member| member.name != name);
            members.push(Member {
                name: name.into(),
                ty: signature,
                kind: DefinitionKind::Function,
                origin: None,
                compiler_signature: true,
            });
        }
        self.fields.insert(location, members);
    }
}
fn format_type(ty: &Ty, typer: &lower::context::Context) -> String {
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
