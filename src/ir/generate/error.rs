use std::{fmt, sync::Arc};

use crate::ast::Span;
use crate::ir::{TypeError, TypeErrorKind, VerifyError};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateError {
    pub span: Span,
    pub kind: GenerateErrorKind,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateErrorKind {
    InvalidForeignSignature,
    InvalidShader { message: Arc<str> },
    UnresolvedImport { path: Arc<str> },
    UnknownExport { name: Arc<str> },
    DuplicateExport { name: Arc<str> },
    ReservedBuiltin { name: Arc<str> },
    Type(TypeErrorKind),
    UnboundValue { name: Arc<str> },
    UnboundType { name: Arc<str> },
    UnknownTypeFormer { name: Arc<str> },
    EagerRecursion { name: Arc<str> },
    UninitializedValue { name: Arc<str> },
    DuplicateValue { name: Arc<str> },
    DuplicateType { name: Arc<str> },
    NeedsTypeAnnotation { name: Arc<str> },
    MissingField { name: Arc<str> },
    ExtraField { name: Arc<str> },
    NotAPlace,
    InvalidLiteral { message: Arc<str> },
    InvalidIr(VerifyError),
}
impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "compile error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}
impl std::error::Error for GenerateError {}

impl GenerateError {
    pub(super) fn typing(span: Span, error: TypeError) -> Self {
        Self {
            span,
            kind: GenerateErrorKind::Type(error.kind),
        }
    }
}
