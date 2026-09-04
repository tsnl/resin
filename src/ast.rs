//! Abstract syntax tree for Resin.

#![allow(dead_code)]

use std::sync::Arc;

//
// Ident, Term, Stmt
//

pub type Ident = Spanned<Arc<str>>;
pub type Term = Spanned<TermKind>;
pub type Stmt = Spanned<StmtKind>;

#[derive(Debug, Clone)]
pub struct SourceFile {
    pub stmts: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum TermKind {
    Var(Ident),
    Num(Arc<str>),
    Lambda {
        params: Vec<(Ident, Box<Term>)>,
        body: Box<Term>,
    },
    If {
        cond: Box<Term>,
        then: Box<Term>,
        els: Box<Term>,
    },
    Array(Vec<Term>),
    Record(Vec<(Ident, Term)>),
    RecordType(Vec<(Ident, Term)>),
    Block {
        stmts: Vec<Stmt>,
        tail: Box<Term>,
    },
    Unit,
    Call {
        func: Box<Term>,
        args: Vec<Term>,
    },
}

#[derive(Debug, Clone)]
pub struct StmtKind {
    pub name: Ident,
    pub init: Term,
}

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
