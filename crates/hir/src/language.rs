use crate::source::{Ident, SourceLocation, Span};
use crate::types::{Case, Foreign, FunctionId, Intrinsic, Ty, TypeTable, Value};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};

/// A lexical declaration's identity. Names survive only for diagnostics.
pub type BindingId = usize;

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    pub types: TypeTable,
    pub functions: Vec<Function>,
    pub shaders: BTreeMap<FunctionId, crate::types::shader::ShaderEntry>,
    pub origins: SourceMap,
}

#[derive(Debug, Clone, Default)]
pub struct SourceMap {
    pub sources: BTreeMap<PathBuf, Arc<str>>,
    pub functions: BTreeMap<FunctionId, SourceLocation>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: Arc<str>,
    pub signature: Signature,
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
    /// Foreign declarations have no body and therefore no lexical binding.
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
    Constant(Value),
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
    Pack(Arguments),
    Intrinsic {
        op: Intrinsic,
        args: Arguments,
    },
    Adapt {
        conversion: ReceiverConversion,
        arg: Box<Term>,
    },
    Convert {
        conversion: crate::types::check::ExplicitConversion,
        arg: Box<Term>,
    },
    ArcNew {
        value: Box<Term>,
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
        access: crate::types::FieldAccess,
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

impl ReceiverConversion {
    pub fn between(from: &Ty, to: &Ty) -> Option<Self> {
        if from == to {
            Some(Self::Value)
        } else if matches!(to, Ty::Pointer { pointee } if pointee.as_ref() == from) {
            Some(Self::Address)
        } else if matches!(from, Ty::Pointer { pointee } if pointee.as_ref() == to) {
            Some(Self::Load)
        } else if matches!(from, Ty::Arc { pointee } if to == &Ty::Pointer { pointee: pointee.clone() })
        {
            Some(Self::ArcAddress)
        } else if matches!(from, Ty::Arc { pointee } if pointee.as_ref() == to) {
            Some(Self::ArcLoad)
        } else {
            None
        }
    }
}
