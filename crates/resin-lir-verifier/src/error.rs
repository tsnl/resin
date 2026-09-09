use std::fmt;

use resin_common::types::{TypeError, TypeErrorKind};
use resin_lir::{BlockId, FunctionId, TypeId};

use crate::{VerifyError, VerifyErrorKind, VerifyLocation};

impl fmt::Display for VerifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid IR in ")?;
        match self.location {
            VerifyLocation::TypeDefinition { definition } => {
                write!(f, "type definition {}", definition.index())?;
            }
            VerifyLocation::Function { function } => {
                write!(f, "function {}", function.index())?;
            }
            VerifyLocation::BasicBlock {
                function,
                basic_block,
            } => write!(
                f,
                "function {}, basic block {}",
                function.index(),
                basic_block.index()
            )?,
            VerifyLocation::Instruction {
                function,
                basic_block,
                instruction,
            } => write!(
                f,
                "function {}, basic block {}, instruction {instruction}",
                function.index(),
                basic_block.index()
            )?,
        }
        write!(f, ": {:?}", self.kind)
    }
}

impl std::error::Error for VerifyError {}

#[derive(Debug, Clone, Copy)]
pub(super) struct Location(VerifyLocation);

impl Location {
    pub(super) const fn type_definition(definition: TypeId) -> Self {
        Self(VerifyLocation::TypeDefinition { definition })
    }

    pub(super) const fn function(function: FunctionId) -> Self {
        Self(VerifyLocation::Function { function })
    }

    pub(super) const fn instruction(
        function: FunctionId,
        basic_block: BlockId,
        instruction: usize,
    ) -> Self {
        Self(VerifyLocation::Instruction {
            function,
            basic_block,
            instruction,
        })
    }

    pub(super) const fn basic_block(function: FunctionId, basic_block: BlockId) -> Self {
        Self(VerifyLocation::BasicBlock {
            function,
            basic_block,
        })
    }

    pub(super) fn error(self, kind: VerifyErrorKind) -> VerifyError {
        VerifyError {
            location: self.0,
            kind,
        }
    }
}

impl From<TypeError> for VerifyErrorKind {
    fn from(error: TypeError) -> Self {
        match error.kind {
            TypeErrorKind::NominalTypeMustBeRecord { definition } => {
                Self::NominalTypeMustBeRecord { definition }
            }
            TypeErrorKind::InvalidUnion => Self::InvalidVariant,
            TypeErrorKind::InvalidTypeDefinition { definition } => Self::InvalidTypeDefinition {
                definition: definition.index(),
            },
            TypeErrorKind::IncompleteTypeDefinition { definition } => {
                Self::IncompleteTypeDefinition { definition }
            }
            TypeErrorKind::RecursiveTypeWithoutIndirection { definition } => {
                Self::RecursiveTypeWithoutIndirection { definition }
            }
            _ => Self::InvalidBuiltin(error),
        }
    }
}
