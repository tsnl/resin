//! Abstract syntax tree for Resin.

#![allow(dead_code)]

use std::sync::Arc;

pub mod generate;
pub mod sexpfmt;

//
// SourceFile
//

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub stmts: Vec<Stmt>,
}

//
// Type
//

pub type Type = Spanned<TypeKind>;

#[derive(Debug, Clone)]
pub enum TypeKind {
    Atom { name: Ident },
    App { head: Ident, arg: Box<Term> },
    Func { from: Box<Type>, to: Box<Type> },
    Record { fields: Vec<(Ident, Type)> },
}

//
// Term
//

pub type Term = Spanned<TermKind>;

#[derive(Debug, Clone)]
pub enum TermKind {
    Var {
        name: Ident,
    },
    Num {
        value: Arc<str>,
    },
    Lambda {
        params: Vec<(Ident, Type)>,
        body: Box<Term>,
    },
    If {
        cond: Box<Term>,
        then: Box<Term>,
        els: Box<Term>,
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
        args: Vec<Term>,
    },
    Assign {
        place: Box<Term>,
        value: Box<Term>,
    },
    Deref {
        pointer: Box<Term>,
    },
    Field {
        base: Box<Term>,
        name: Ident,
    },
    Type {
        ty: Type,
    },
}

//
// Statements
//

pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, Clone)]
pub enum StmtKind {
    /// `name = init;`
    Define { name: Ident, init: Term },
    /// `name: ann;`
    Declare { name: Ident, ann: Type },
    /// `term;`
    Expr { term: Term },
}

//
// Ident
//

pub type Ident = Spanned<Arc<str>>;

//
// Spanned<T>
//

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
