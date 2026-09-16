//! Private checking tree, retaining source forms and resolved declarations for elaboration.
//! Inference fills this tree with `Type`; completion constructs public HIR directly.
use super::infer::{Rule, Type};
use super::scope::DeclarationId;
use crate::GenerateError;
use resin_source::prelude::*;
use resin_types::prelude::*;

use std::sync::Arc;

#[derive(Debug, Clone)]
pub(super) struct Term {
    pub span: Span,
    /// Type requested by this expression's consumer.
    pub ty: Type,
    /// Expression result before reference binding, reading, or value widening.
    pub actual: Type,
    pub kind: TermKind,
}

#[derive(Debug, Clone)]
pub(super) enum TermKind {
    Break,
    Continue,
    Return {
        value: Box<Term>,
    },
    SizeOf {
        ty: Annotation<Type>,
    },
    Constant {
        value: crate::Term,
    },
    Error(GenerateError),
    Unit,
    None,
    Bool {
        value: bool,
    },
    Num {
        value: Arc<str>,
    },
    String {
        value: Arc<str>,
    },
    Var {
        declaration: DeclarationId,
        name: Ident,
        type_args: Vec<Type>,
    },
    Type {
        ty: Annotation<Type>,
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
        rule: Rule,
        name_span: Span,
        name: Arc<str>,
        args: Vec<Term>,
    },
    MethodCall {
        rule: Rule,
        receiver: Option<Box<Term>>,
        receiver_type: Annotation<Type>,
        name: Ident,
        args: Vec<Term>,
    },
    MethodReference {
        rule: Rule,
        name: Ident,
    },
    Call {
        func: Box<Term>,
        args: Vec<Term>,
    },
    Ascribe {
        ty: Annotation<Type>,
        arg: Box<Term>,
    },
    Result {
        failure: bool,
        arg: Box<Term>,
    },
    Absurd {
        arg: Box<Term>,
    },
    // Only the operand's type survives checking; querying layout cannot run it.
    Layout {
        ty: Annotation<Type>,
        size: bool,
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
    Field {
        base: Box<Term>,
        name: Ident,
    },
}

#[derive(Debug, Clone)]
pub(super) struct Annotation<T = Ty> {
    pub ty: T,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub(super) struct MatchArm {
    pub error: bool,
    pub wildcard: bool,
    pub variant: Option<Annotation<Type>>,
    pub failure: bool,
    pub binding: Option<DeclarationId>,
    pub body: Term,
}

#[derive(Debug, Clone)]
pub(super) struct Statement {
    pub kind: StatementKind,
}

#[derive(Debug, Clone)]
pub(super) enum StatementKind {
    Error(GenerateError),
    Define {
        binding: Option<DeclarationId>,
        name: Ident,
        init: Term,
    },
    Declare {
        binding: DeclarationId,
        name: Ident,
        ty: Annotation<Type>,
    },
    CompileTimeDefinition,
    Expr {
        term: Term,
    },
}

#[derive(Debug, Clone)]
pub(super) struct Signature {
    pub type_params: Vec<crate::TypeParameter>,
    pub declaration: Option<DeclarationId>,
    pub parameters: Vec<Option<DeclarationId>>,
    pub params: Vec<(Ident, Annotation<crate::Type>)>,
    pub result: Annotation<crate::Type>,
}

/// Source-order function metadata retained when its declaration is reserved.
pub(super) struct Declaration {
    pub id: DeclarationId,
    pub name: Ident,
    pub kind: DeclarationKind,
}

pub(super) enum DeclarationKind {
    Function { decorators: Vec<Ident> },
    Foreign { header: Arc<str> },
    Intrinsic { operation: Arc<str> },
}
