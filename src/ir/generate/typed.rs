//! The tree passed from source checking to IR lowering.
//!
//! Checking builds the same shape with private inference types, then resolves it
//! to `Term<Ty>`. Lowering receives only concrete types and ordinary data: no
//! solver, typing rules, or deferred emission callbacks travel with the tree.
use super::{
    GenerateError,
    scope::{Cursor, DeclarationId},
};
use crate::{
    ast::{Ident, Span},
    ir::Ty,
};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct Term<T = Ty> {
    pub span: Span,
    pub context: Cursor,
    pub ty: T,
    pub kind: TermKind<T>,
}

#[derive(Debug)]
pub(super) enum TermKind<T = Ty> {
    Error(GenerateError),
    Unit,
    None,
    Num {
        value: Arc<str>,
    },
    String {
        value: Arc<str>,
    },
    Var {
        name: Ident,
    },
    Type {
        ty: Annotation<T>,
    },
    Unwrap {
        value: Box<Term<T>>,
    },
    Try {
        value: Box<Term<T>>,
    },
    Match {
        value: Box<Term<T>>,
        arms: Vec<MatchArm<T>>,
    },
    If {
        cond: Box<Term<T>>,
        then: Box<Term<T>>,
        els: Box<Term<T>>,
    },
    While {
        cond: Box<Term<T>>,
        body: Box<Term<T>>,
    },
    Block {
        stmts: Vec<Statement<T>>,
        tail: Box<Term<T>>,
    },
    Record {
        fields: Vec<(Ident, Term<T>)>,
    },
    Array {
        elems: Vec<Term<T>>,
    },
    Builtin {
        name: Arc<str>,
        args: Vec<Term<T>>,
    },
    MethodCall {
        receiver: Option<Box<Term<T>>>,
        receiver_type: Annotation<T>,
        name: Ident,
        arg: Box<Term<T>>,
    },
    Call {
        func: Box<Term<T>>,
        arg: Box<Term<T>>,
    },
    Ascribe {
        ty: Annotation<T>,
        arg: Box<Term<T>>,
    },
    Result {
        failure: bool,
        arg: Box<Term<T>>,
    },
    Absurd {
        arg: Box<Term<T>>,
    },
    // Only the operand's type survives checking; querying layout cannot run it.
    Layout {
        ty: Annotation<T>,
        size: bool,
    },
    Assign {
        place: Box<Term<T>>,
        value: Box<Term<T>>,
    },
    Address {
        place: Box<Term<T>>,
    },
    Deref {
        pointer: Box<Term<T>>,
    },
    Field {
        base: Box<Term<T>>,
        name: Ident,
    },
}

#[derive(Debug)]
pub(super) struct Annotation<T = Ty> {
    pub ty: T,
    pub span: Span,
}

#[derive(Debug)]
pub(super) struct MatchArm<T = Ty> {
    pub variant: Option<Annotation<T>>,
    pub failure: bool,
    pub binding: Option<DeclarationId>,
    pub body: Term<T>,
}

#[derive(Debug)]
pub(super) struct Statement<T = Ty> {
    pub context: Cursor,
    pub kind: StatementKind<T>,
}

#[derive(Debug)]
pub(super) enum StatementKind<T = Ty> {
    Error(GenerateError),
    Define {
        binding: Option<DeclarationId>,
        name: Ident,
        init: Term<T>,
    },
    Declare {
        binding: DeclarationId,
        name: Ident,
        ty: Annotation<T>,
    },
    TypeDefinition,
    Expr {
        term: Term<T>,
    },
}

pub(super) struct Signature {
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation)>,
    pub result: Annotation,
}
