use std::{fmt, sync::Arc};

use crate::types::definitions::DefinitionError;
use crate::types::{Ty, TypeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeErrorKind {
    InvalidUnion,
    PointerArithmetic,
    EmptyArrayNeedsElementType,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedInteger { found: Ty },
    ExpectedBoolean { found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedRecord { found: Ty },
    ExpectedFunction { found: Ty },
    InvalidBuiltinArgumentCount { name: Arc<str>, found: usize },
    UnknownBuiltin { name: Arc<str> },
    InvalidPrintArguments { found: Ty },
    InvalidFormatArguments { found: Ty },
    UnformattableType { found: Ty },
    UnknownField { name: Arc<str> },
    DuplicateField { name: Arc<str> },
    InvalidTypeDefinition { definition: TypeId },
    IncompleteTypeDefinition { definition: TypeId },
    NominalTypeMustBeRecord { definition: TypeId },
    TypeAlreadyDefined { definition: TypeId },
    RecursiveTypeWithoutIndirection { definition: TypeId },
}

impl TypeError {
    pub(super) const fn new(kind: TypeErrorKind) -> Self {
        Self { kind }
    }
}

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.kind == TypeErrorKind::PointerArithmetic {
            return f.write_str("pointer arithmetic is not allowed; index a Span or explicitly convert the pointer to ulong for byte arithmetic");
        }
        write!(f, "type error: {:?}", self.kind)
    }
}

impl std::error::Error for TypeError {}

impl From<DefinitionError> for TypeError {
    fn from(error: DefinitionError) -> Self {
        let kind = match error {
            DefinitionError::NonRecord(definition) => {
                TypeErrorKind::NominalTypeMustBeRecord { definition }
            }
            DefinitionError::InvalidUnion => TypeErrorKind::InvalidUnion,
            DefinitionError::Invalid(definition) => {
                TypeErrorKind::InvalidTypeDefinition { definition }
            }
            DefinitionError::Incomplete(definition) => {
                TypeErrorKind::IncompleteTypeDefinition { definition }
            }
            DefinitionError::Recursive(definition) => {
                TypeErrorKind::RecursiveTypeWithoutIndirection { definition }
            }
        };
        Self::new(kind)
    }
}
