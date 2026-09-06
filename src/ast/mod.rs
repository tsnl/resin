//! Abstract syntax tree for Resin.

#![allow(dead_code)]

use std::sync::Arc;

pub mod generate;
mod load;
pub mod print;
pub use load::{
    FileSystem, Program, SourceError, SourceLocation, SourceModule, SourceNote, SourceProvider,
    load, load_with, resolve_import, stdlib_path,
};

pub use generate::{AstError, AstErrorKind, AstGen};

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub exports: Vec<Ident>,
    pub imports: Vec<Spanned<Arc<str>>>,
    pub stmts: Vec<Stmt>,
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
    pub name: Ident,
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
    Function {
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
    Defer {
        // Shared so repeated cleanup lowering preserves inference's node identities.
        body: Arc<Term>,
    },
    /// `term;`
    Expr {
        term: Term,
    },
}

pub type Ident = Spanned<Arc<str>>;

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub val: T,
    pub span: Span,
}
impl<T> Spanned<T> {
    pub fn new(val: T, span: Span) -> Self {
        Self { val, span }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl<T> std::ops::Deref for Spanned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.val
    }
}
