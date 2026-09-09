//! Source declarations, terms, and type syntax. Names and sugar are preserved.
pub use crate::source::{Ident, Span, Spanned};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub exports: Vec<Ident>,
    pub imports: Vec<Spanned<Arc<str>>>,
    pub stmts: Vec<Stmt>,
}

impl SourceFile {
    /// Visit declarations inside impl blocks without erasing their AST structure.
    pub fn declarations(&self) -> impl Iterator<Item = &Stmt> {
        self.stmts.iter().flat_map(|stmt| match &stmt.val {
            StmtKind::Impl { methods, .. } => methods.as_slice(),
            _ => std::slice::from_ref(stmt),
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
        arg: Box<Type>,
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
        from: Box<Type>,
        to: Box<Type>,
    },
    Record {
        fields: Vec<(Ident, Type)>,
    },
}

pub type Term = Spanned<TermKind>;

#[derive(Debug, Clone)]
pub enum TermKind {
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
    MethodCall {
        receiver: Box<Term>,
        name: Ident,
        arg: Box<Term>,
    },
    Call {
        func: Box<Term>,
        arg: Box<Term>,
    },
    /// Privileged operator syntax; operands are evaluated in source order.
    Builtin {
        name: Arc<str>,
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
    Ok,
    Err,
    Type(Type),
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    ForeignType {
        name: Ident,
    },
    ForeignFunction {
        header: Arc<str>,
        name: Ident,
        params: Vec<(Ident, Type)>,
        result: Type,
    },
    Impl {
        receiver: Ident,
        methods: Vec<Stmt>,
    },
    Function {
        receiver: Option<Ident>,
        decorators: Vec<Ident>,
        name: Ident,
        params: Vec<(Ident, Type)>,
        result: Type,
        body: Term,
    },
    /// `var name = init;` The name is in scope, but eager recursive reads are invalid.
    Define {
        name: Ident,
        init: Term,
    },
    /// `type Name = init;` A transparent alias.
    DefineType {
        name: Ident,
        init: Type,
    },
    Struct {
        name: Ident,
        body: Type,
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
pub struct Program {
    /// Dependencies precede their consumers; the entry file is last.
    pub modules: Vec<SourceModule>,
}

#[derive(Debug, Clone)]
pub struct SourceModule {
    pub path: std::path::PathBuf,
    pub source: String,
    pub file: SourceFile,
    pub imports: Vec<(Span, usize)>,
}
