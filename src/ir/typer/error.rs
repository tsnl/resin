use std::{fmt, sync::Arc};

use crate::ir::types::definitions::DefinitionError;
use crate::ir::{Ty, TypeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeError {
    pub kind: TypeErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeErrorKind {
    InvalidUnion,
    EmptyArrayNeedsElementType,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedBoolean { found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedRecord { found: Ty },
    ExpectedFunction { found: Ty },
    InvalidBuiltinArgumentCount { name: Arc<str>, found: usize },
    UnknownBuiltin { name: Arc<str> },
    InvalidPrintArguments { found: Ty },
    UnprintableType { found: Ty },
    UnknownField { name: Arc<str> },
    DuplicateField { name: Arc<str> },
    InvalidTypeDefinition { definition: TypeId },
    IncompleteTypeDefinition { definition: TypeId },
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
        write!(f, "type error: {:?}", self.kind)
    }
}

impl std::error::Error for TypeError {}

impl From<DefinitionError> for TypeError {
    fn from(error: DefinitionError) -> Self {
        let kind = match error {
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
