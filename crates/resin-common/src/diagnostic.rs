use std::{fmt, sync::Arc};

use crate::source::Span;
use crate::types::{TypeError, TypeErrorKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateError {
    pub span: Span,
    pub kind: GenerateErrorKind,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerateErrorKind {
    IncompleteSyntax,
    Inference { message: Arc<str> },
    InvalidModuleItem,
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
    InvalidIr(Arc<str>),
}
impl fmt::Display for GenerateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind == GenerateErrorKind::Type(TypeErrorKind::PointerArithmetic) {
            return write!(
                f,
                "compile error at {}..{}: {}",
                self.span.start,
                self.span.end,
                TypeError {
                    kind: TypeErrorKind::PointerArithmetic
                }
            );
        }
        if let GenerateErrorKind::Inference { message } = &self.kind {
            return write!(
                f,
                "compile error at {}..{}: {message}",
                self.span.start, self.span.end
            );
        }
        write!(
            f,
            "compile error at {}..{}: {:?}",
            self.span.start, self.span.end, self.kind
        )
    }
}
impl std::error::Error for GenerateError {}

impl GenerateError {
    pub fn inference(span: Span, message: impl Into<Arc<str>>) -> Self {
        Self {
            span,
            kind: GenerateErrorKind::Inference {
                message: message.into(),
            },
        }
    }

    pub fn typing(span: Span, error: TypeError) -> Self {
        Self {
            span,
            kind: GenerateErrorKind::Type(error.kind),
        }
    }
}
