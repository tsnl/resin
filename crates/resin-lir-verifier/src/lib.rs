//! Stack and control-flow verification for typed IR.

use resin_common::prelude::*;
use resin_lir::{BlockId, Module};

mod error;
mod flow;
mod instructions;
mod module;
mod rules;
mod table;

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
    InvalidVariant,
    InvalidDropHook,
    InvalidForeignSignature,
    OpaqueValue { ty: Ty },
    InvalidShader,
    PointerArithmetic,
    InvalidBuiltin(TypeError),
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

use flow::check_function;

/// Check block and edge types; incoming edges must agree on the entry stack.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    analyze(module).map(|_| ())
}

/// A borrow of LIR and the analysis certifying that exact immutable module.
/// Only the verifier can construct a certificate.
///
/// ```compile_fail,E0451
/// use resin_lir_verifier::{ModuleTypes, Verified};
/// let module = resin_lir::Module::default();
/// let analysis = ModuleTypes { functions: vec![], types: Default::default() };
/// let forged = Verified { module: &module, analysis: &analysis };
/// ```
#[derive(Clone, Copy)]
pub struct Verified<'a> {
    module: &'a Module,
    analysis: &'a ModuleTypes,
}

impl<'a> Verified<'a> {
    pub fn module(self) -> &'a Module {
        self.module
    }

    pub fn analysis(self) -> &'a ModuleTypes {
        self.analysis
    }
}

/// The checked module and the analysis that certifies this exact immutable IR.
/// Editing requires [`Self::into_module`], which consumes and discards the certificate.
///
/// ```compile_fail,E0596
/// let checked = resin_lir_verifier::VerifiedModule::new(Default::default()).unwrap();
/// checked.view().module().functions.clear();
/// ```
///
/// ```compile_fail,E0505
/// let checked = resin_lir_verifier::VerifiedModule::new(Default::default()).unwrap();
/// let borrowed = checked.view();
/// let editable = checked.into_module();
/// let _ = borrowed.module();
/// ```
pub struct VerifiedModule {
    module: Module,
    analysis: ModuleTypes,
}

impl VerifiedModule {
    pub fn new(mut module: Module) -> Result<Self, VerifyError> {
        let analysis = analyze(&module)?;
        module.types = analysis.types.clone();
        Ok(Self { module, analysis })
    }

    pub fn view(&self) -> Verified<'_> {
        Verified {
            module: &self.module,
            analysis: &self.analysis,
        }
    }

    /// Discard the certificate and recover editable LIR.
    pub fn into_module(self) -> Module {
        self.module
    }
}

/// Direct IR clients borrow their module only for the duration of validation/use.
pub fn with_verified<T, E: From<VerifyError>>(
    module: &Module,
    use_module: impl FnOnce(Verified<'_>) -> Result<T, E>,
) -> Result<T, E> {
    let analysis = analyze(module)?;
    use_module(Verified {
        module,
        analysis: &analysis,
    })
}

pub struct ModuleTypes {
    pub functions: Vec<FunctionTypes>,
    pub types: TypeTable,
}

pub struct FunctionTypes {
    pub inputs: Vec<Vec<Ty>>,
    pub results: Vec<Vec<Option<Ty>>>,
}

fn analyze(module: &Module) -> Result<ModuleTypes, VerifyError> {
    module::check(module)?;
    let functions = module
        .functions
        .iter()
        .enumerate()
        .map(|(index, function)| check_function(module, FunctionId::from_index(index), function))
        .collect::<Result<Vec<_>, _>>()?;
    let types = table::collect(module, &functions);
    Ok(ModuleTypes { functions, types })
}

#[cfg(test)]
mod tests;
