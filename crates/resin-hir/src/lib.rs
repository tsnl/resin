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

mod analysis;
mod lower;
mod print;

use resin_common::source::{Ident, SourceLocation, Span};
use resin_common::types::{Case, Foreign, FunctionId, Intrinsic, Ty, TypeTable, Value};
use resin_cst::Document;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
};

/// A lexical declaration's identity. Names survive only for diagnostics.
pub type BindingId = usize;

#[derive(Debug, Clone, Default)]
pub struct Module {
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    pub types: TypeTable,
    pub functions: Vec<Function>,
    pub shaders: BTreeMap<FunctionId, resin_common::types::shader::ShaderEntry>,
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
        conversion: resin_common::types::check::ExplicitConversion,
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
        access: resin_common::types::FieldAccess,
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
pub struct Analysis(pub(crate) lower::semantic::SemanticData);

/// Read-only syntax access for editor queries, supplied by the compiler.
pub trait Documents {
    fn get(&self, path: &Path) -> Option<&Document>;
}

/// Diagnostics and editor facts, with a complete module only when checking succeeds.
pub struct CheckedProgram {
    pub module: Option<crate::Module>,
    pub diagnostics: Vec<resin_common::source::SourceError>,
    pub semantics: Analysis,
}

impl Analysis {
    pub fn definition(
        &self,
        documents: &dyn Documents,
        path: &Path,
        offset: usize,
    ) -> Option<SourceLocation> {
        analysis::Query {
            semantics: &self.0,
            documents,
        }
        .definition(path, offset)
    }
    pub fn hover(&self, documents: &dyn Documents, path: &Path, offset: usize) -> Option<Hover> {
        analysis::Query {
            semantics: &self.0,
            documents,
        }
        .hover(path, offset)
    }
    pub fn completions(
        &self,
        documents: &dyn Documents,
        path: &Path,
        offset: usize,
    ) -> Vec<Completion> {
        analysis::Query {
            semantics: &self.0,
            documents,
        }
        .completions(path, offset)
    }
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

/// Check a standalone source file; imports require a resolved Program.
pub fn generate(
    file: &resin_ast::SourceFile,
) -> Result<Module, resin_common::diagnostic::GenerateError> {
    lower::generate(file)
}

/// Check declarations in import order, requiring a completely typed tree.
pub fn generate_program(
    program: &resin_ast::Program,
) -> Result<Module, resin_common::source::SourceError> {
    lower::generate_program(program)
}

/// Retain diagnostics and editor facts even when a complete tree cannot be produced.
pub fn analyze_program(program: &resin_ast::Program) -> CheckedProgram {
    lower::analyze_program(program)
}

pub fn format_module(module: &Module) -> String {
    print::format_module(module)
}
