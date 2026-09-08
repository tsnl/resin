use std::fmt;

use crate::ir::types::definitions::DefinitionError;
use crate::ir::{BlockId, FunctionId, Ty, TypeId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyError {
    pub location: VerifyLocation,
    pub kind: VerifyErrorKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyLocation {
    TypeDefinition {
        definition: TypeId,
    },
    Function {
        function: FunctionId,
    },
    BasicBlock {
        function: FunctionId,
        basic_block: BlockId,
    },
    Instruction {
        function: FunctionId,
        basic_block: BlockId,
        instruction: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyErrorKind {
    InvalidMethod,
    InvalidVariant,
    InvalidForeignSignature,
    OpaqueValue { ty: Ty },
    InvalidShader,
    PointerArithmetic,
    InvalidPointerCast { from: Ty, to: Ty },
    InvalidTypeDefinition { definition: usize },
    IncompleteTypeDefinition { definition: TypeId },
    NominalTypeMustBeRecord { definition: TypeId },
    RecursiveTypeWithoutIndirection { definition: TypeId },
    InvalidLocal { local: usize },
    InvalidFunction { function: usize },
    InvalidBasicBlock { basic_block: usize },
    UnreachableBasicBlock,
    StackUnderflow { needed: usize, available: usize },
    InvalidImmediate,
    TypeMismatch { expected: Ty, found: Ty },
    ExpectedPointer { found: Ty },
    ExpectedAggregate { found: Ty },
    ExpectedArray { found: Ty },
    ExpectedFunction { found: Ty },
    ExpectedInteger { found: Ty },
    StaticIndexOutOfBounds { index: usize, length: usize },
    ArgumentCount { expected: usize, found: usize },
    ConflictingBasicBlockStack { expected: Vec<Ty>, found: Vec<Ty> },
    InvalidReturnStack { expected: Ty, found: Vec<Ty> },
}

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

impl From<DefinitionError> for VerifyErrorKind {
    fn from(error: DefinitionError) -> Self {
        match error {
            DefinitionError::NonRecord(definition) => Self::NominalTypeMustBeRecord { definition },
            DefinitionError::InvalidUnion => Self::InvalidVariant,
            DefinitionError::Invalid(definition) => Self::InvalidTypeDefinition {
                definition: definition.index(),
            },
            DefinitionError::Incomplete(definition) => {
                Self::IncompleteTypeDefinition { definition }
            }
            DefinitionError::Recursive(definition) => {
                Self::RecursiveTypeWithoutIndirection { definition }
            }
        }
    }
}
