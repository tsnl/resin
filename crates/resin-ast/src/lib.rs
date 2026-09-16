//! Abstract syntax tree for Resin.
//!
//! Implementation modules are deliberately private:
//! ```compile_fail
//! use resin_ast::lower;
//! ```
//! ```compile_fail
//! use resin_ast::load;
//! ```

use resin_source::prelude::*;
mod load;
mod lower;
mod print;

use std::{fmt, sync::Arc};

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub exports: Vec<Ident>,
    /// Declared native dependencies, including header groups with no functions.
    pub foreign_headers: Vec<Spanned<Arc<str>>>,
    pub imports: Vec<Spanned<Arc<str>>>,
    pub stmts: Vec<Stmt>,
}

impl SourceFile {
    /// Visit module declarations and the methods owned by their structs.
    pub fn declarations(&self) -> impl Iterator<Item = &Stmt> {
        self.stmts.iter().flat_map(|stmt| {
            let methods = match &stmt.val {
                StmtKind::Struct { methods, .. } => methods.as_slice(),
                _ => &[],
            };
            std::iter::once(stmt).chain(methods)
        })
    }
}

pub type Type = Spanned<TypeKind>;

#[derive(Debug, Clone)]
pub enum TypeKind {
    /// Missing or malformed type syntax; never an executable type.
    Hole,
    /// An explicit inference request, distinct from malformed editor syntax.
    Infer,
    Unit,
    Atom {
        name: Ident,
    },
    App {
        head: Ident,
        args: Vec<Type>,
    },
    Result {
        value: Box<Type>,
        error: Box<Type>,
    },
    Union {
        left: Box<Type>,
        right: Box<Type>,
    },
    Func {
        params: Vec<Type>,
        to: Box<Type>,
    },
    Record {
        fields: Vec<(Ident, Type)>,
    },
}

pub type Term = Spanned<TermKind>;

#[derive(Debug, Clone)]
pub enum TermKind {
    Break,
    Continue,
    Return {
        value: Box<Term>,
    },
    SizeOf {
        ty: Type,
    },
    /// Unknown expression, retaining recognizable children for editor analysis.
    Hole {
        children: Vec<Term>,
    },
    /// A receiver followed by a dot with no field name.
    FieldHole {
        base: Box<Term>,
    },
    Var {
        name: Ident,
    },
    Num {
        value: Arc<str>,
    },
    String {
        value: Arc<str>,
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
    Array {
        elems: Vec<Term>,
    },
    Record {
        fields: Vec<(Ident, Term)>,
    },
    Block {
        stmts: Vec<Stmt>,
        tail: Box<Term>,
    },
    Unit,
    None,
    Bool {
        value: bool,
    },
    MethodCall {
        receiver: Box<Term>,
        name: Ident,
        type_args: Vec<Type>,
        args: Vec<Term>,
    },
    TypeApply {
        function: Box<Term>,
        args: Vec<Type>,
    },
    Call {
        func: Box<Term>,
        args: Vec<Term>,
    },
    /// Privileged operator syntax; operands are evaluated in source order.
    Builtin {
        name: Arc<str>,
        name_span: Span,
        args: Vec<Term>,
    },
    Assign {
        place: Box<Term>,
        value: Box<Term>,
    },
    Deref {
        pointer: Box<Term>,
    },
    Address {
        place: Box<Term>,
    },
    Field {
        base: Box<Term>,
        name: Ident,
    },
    Type {
        ty: Type,
    },
}

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub variant: MatchVariant,
    pub name: Option<Ident>,
    pub body: Term,
}

#[derive(Debug, Clone)]
pub enum MatchVariant {
    Error,
    Wildcard,
    Ok,
    Err,
    Type(Type),
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// A single declaration or a parenthesized group. Each specification has
    /// explicit initializers and its own zero-based `iota` value.
    Const {
        specs: Vec<ConstSpec>,
    },
    ForeignType {
        name: Ident,
    },
    IntrinsicFunction {
        operation: Arc<str>,
        type_params: Vec<Ident>,
        name: Ident,
        params: Vec<(Ident, Type)>,
        result: Type,
    },
    ForeignFunction {
        header: Arc<str>,
        name: Ident,
        params: Vec<(Ident, Type)>,
        result: Type,
    },
    Function {
        type_params: Vec<Ident>,
        decorators: Vec<Ident>,
        name: Ident,
        params: Vec<(Ident, Type)>,
        result: Type,
        body: Term,
    },
    /// `var name[: Type] = init;` The name is in scope; eager recursive reads are invalid.
    Define {
        name: Ident,
        ann: Option<Type>,
        init: Term,
    },
    /// `type Name = init;` A transparent alias.
    DefineType {
        type_params: Vec<Ident>,
        name: Ident,
        init: Type,
    },
    Struct {
        type_params: Vec<Ident>,
        name: Ident,
        body: Type,
        methods: Vec<Stmt>,
    },
    /// `var name: ann;`
    Declare {
        name: Ident,
        ann: Type,
    },
    /// `term;`
    Expr {
        term: Term,
    },
}

#[derive(Debug, Clone)]
pub struct ConstSpec {
    pub names: Vec<Ident>,
    pub ann: Option<Type>,
    pub init: Vec<Term>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Program {
    /// Dependencies precede their consumers; the entry file is last.
    pub modules: Vec<SourceModule>,
}

#[derive(Debug, Clone)]
pub struct SourceModule {
    pub source: Source,
    pub file: Arc<SourceFile>,
    pub imports: Vec<(Span, usize)>,
}

/// One completed file translation, shared by every graph selecting this source.
pub struct ModuleDocument {
    pub source: Source,
    pub syntax: Arc<resin_cst::Document>,
    pub file: Arc<SourceFile>,
    pub errors: Vec<(Span, String)>,
}

/// A resolved AST graph and its exact per-file inputs, including recovery diagnostics.
pub struct BuiltProgram {
    pub source: Source,
    pub inputs: resin_source::SourceGraph,
    pub program: Program,
    pub documents: std::collections::BTreeMap<Source, Arc<ModuleDocument>>,
    pub diagnostics: Vec<SourceError>,
}

impl BuiltProgram {
    pub fn graph(&self) -> Vec<(Source, Vec<(Span, usize)>)> {
        self.program
            .modules
            .iter()
            .map(|module| (module.source.clone(), module.imports.clone()))
            .collect()
    }

    pub fn syntax(&self) -> std::collections::BTreeMap<Source, Arc<resin_cst::Document>> {
        self.documents
            .iter()
            .map(|(source, document)| (source.clone(), document.syntax.clone()))
            .collect()
    }
}

/// Assemble already-parsed files using explicit immutable import bindings.
/// Missing inputs and cycles become source diagnostics; this operation reads no files.
pub async fn build_program(
    inputs: resin_source::SourceGraph,
    documents: std::collections::BTreeMap<Source, Arc<ModuleDocument>>,
    execution: &resin_executor::Execution,
    cancellation: &resin_executor::Cancellation,
) -> Result<BuiltProgram, resin_executor::Error> {
    execution
        .run(cancellation, move |cancellation| {
            load::assemble(inputs, documents, cancellation)
        })
        .await
}

/// An AST and diagnostics recovered from one concrete syntax document.
pub struct Parsed {
    pub file: SourceFile,
    pub errors: Vec<(Span, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstError {
    pub span: Span,
    pub kind: AstErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AstErrorKind {
    Unexpected { found: Arc<str> },
    Missing { expected: Arc<str> },
}

impl fmt::Display for AstError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}

impl std::error::Error for AstError {}

/// Build an AST, inserting holes for incomplete syntax and retaining every diagnostic.
pub async fn build_ast(
    source: Arc<resin_cst::Document>,
    execution: &resin_executor::Execution,
    cancellation: &resin_executor::Cancellation,
) -> Result<Parsed, resin_executor::Error> {
    execution
        .run(cancellation, move |_| lower::document(&source))
        .await
}

/// Render the AST for inspection.
pub fn format_source(file: &SourceFile) -> String {
    print::format_source(file)
}

/// Render a resolved set of source modules as an AST listing.
pub fn format_program(program: &Program) -> String {
    print::format_program(program)
}

impl SourceModule {
    pub fn location(&self, span: Span) -> String {
        let text = self.source.text();
        let mut start = span.start.min(text.len());
        while !text.is_char_boundary(start) {
            start -= 1;
        }
        let prefix = &text[..start];
        let line = prefix.bytes().filter(|&byte| byte == b'\n').count() + 1;
        let column = prefix.rsplit('\n').next().unwrap().chars().count() + 1;
        format!("{}:{line}:{column}", self.source.name())
    }

    pub fn error(&self, span: Span, message: impl fmt::Display) -> SourceError {
        let mut error = SourceError::new(self.source.clone(), Some(span), message.to_string());
        error.message = format!("{}: {}", self.location(span), error.diagnostic).into();
        error
    }
}
