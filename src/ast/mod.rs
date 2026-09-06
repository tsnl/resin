//! Abstract syntax tree for Resin.

#![allow(dead_code)]

use std::sync::Arc;

pub mod generate;
mod load;
pub mod print;
pub use load::{Program, SourceError, SourceModule, load};

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
    Unit,
    Atom { name: Ident },
    App { head: Ident, arg: Box<Term> },
    Func { from: Box<Type>, to: Box<Type> },
    Record { fields: Vec<(Ident, Type)> },
}

pub type Term = Spanned<TermKind>;

#[derive(Debug, Clone)]
pub enum TermKind {
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
    /// `name = init;` The name is in scope, but eager recursive reads are invalid.
    Define {
        name: Ident,
        init: Term,
    },
    /// `Name = init;` A fresh nominal identity, in scope within its own RHS.
    DefineType {
        name: Ident,
        init: Type,
    },
    /// `name: ann;`
    Declare {
        name: Ident,
        ann: Type,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
