//! Low-level language: typed stack instructions and structured control flow.
//! [`generate`] translates HIR into this representation.
//! [`verify`] checks storage and control flow before target lowering.
//! Construction, verification, and printing internals are private.
//!
//! ```compile_fail,E0603
//! use resin_lir::lower;
//! ```
//!
//! ```compile_fail,E0603
//! use resin_lir::verify::instructions;
//! ```

use resin_common::define_id;
use resin_source::prelude::*;
use resin_types::prelude::*;
mod lower;
mod print;
mod verify;

use std::{collections::BTreeMap, sync::Arc};

define_id! {
    pub struct BlockId(usize);
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Arc<str>>,
    pub foreign: Option<Foreign>,
    pub result: Ty,
    /// Local zero holds the function's single argument value and starts initialized.
    /// HIR's zero parameters become unit, one keeps its type, and multiple become
    /// a positional record (tuple). Body lowering binds or unpacks this slot.
    /// Foreign declarations also reserve it; the C wrapper unpacks it into C arguments.
    pub locals: Vec<Local>,
    /// Root of a structured block tree. Every block is owned exactly once by this
    /// root or by an If/Loop terminator; block IDs identify storage, not jump labels.
    pub entry: BlockId,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicBlock {
    pub name: Option<Arc<str>>,
    pub instrs: Vec<Instr>,
    pub terminator: Terminator,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Local {
    pub name: Option<Arc<str>>,
    pub ty: Ty,
}

impl Function {
    pub fn ty(&self) -> Option<Ty> {
        Some(Ty::Function {
            param: Box::new(self.locals.first()?.ty.clone()),
            result: Box::new(self.result.clone()),
        })
    }
}

/// Typed operand-stack operations. Contracts show the stack top on the right;
/// an unchanged prefix is omitted. Popped values transfer ownership unless an
/// instruction specifies copying, borrowing, or destruction. Addresses borrow storage.
/// The verifier checks types and stack shape; lowering establishes ownership and
/// initialization, with runtime initialization flags protecting managed locals.
///
/// ```compile_fail,E0432
/// use resin_lir::StackEffect;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// `[payload] -> [Arc<payload>]`: transfer the payload into a new shared allocation.
    ArcNew,
    /// `[address] -> [value]`: transfer a pointee without copying or clearing storage.
    /// Lowering must disarm its previous owner, typically with `ForgetLocal`.
    TransferLoad,
    /// `[] -> []`: clear a local's initialization flag without destroying its value.
    ForgetLocal { local: LocalId },
    /// `[Arc<T>] -> [Ptr<T>]`: release this owner and borrow its payload address.
    /// Another owner must keep the allocation alive for the entire access.
    ArcData,
    /// `[Arc<T>] -> [Weak<T>]`: create a weak reference and release this strong owner.
    Downgrade,
    /// `[Weak<T>] -> [Arc<T> | None]`: acquire a live owner if possible; release the weak reference.
    Upgrade,
    /// `[] -> [Weak<T>]`: create an empty weak reference of the given pointee type.
    WeakEmpty { pointee: Ty },
    /// `[] -> [value]`: transfer an initialized local and clear its initialization flag.
    /// Used for compiler temporaries; source reads still copy.
    TakeLocal { local: LocalId },
    /// `[] -> []`: destroy a local if initialized, then clear its initialization flag.
    DropLocal { local: LocalId },
    /// `[value] -> []`: destroy a local's previous initialized value, then transfer
    /// the new value into the local and mark it initialized.
    SetLocal { local: LocalId },
    /// `[payload] -> [variant]`: transfer a payload into case `tag` of type `ty`.
    MakeVariant { ty: Ty, tag: Case },
    /// `[variant or address] -> [bool]`: test the active case; destroy a value operand
    /// after testing, or borrow the pointee when given an address.
    IsVariant { tag: Case },
    /// `[T | None] -> [T]`: transfer the remaining value, trapping if its case is `None`.
    ExcludeNone,
    /// `[variant] -> [payload]`: transfer the active payload, trapping on a different tag.
    VariantPayload { tag: Case },
    /// `[value] -> [widened value]`: transfer union/Result payloads into the wider type.
    Widen { ty: Ty },
    /// `[] -> [Span<ubyte>]`: borrow the decorated function's embedded SPIR-V bytes.
    Shader {
        function: FunctionId,
        stage: Arc<str>,
    },
    /// `[pointer or ulong] -> [cast value]`: reinterpret a pointer or its integer address.
    PointerCast { ty: Ty },
    /// `[] -> [value]`: materialize an immediate; byte literals borrow static storage.
    Push { value: Value },
    /// `[] -> [address]`: borrow a local's storage without reading or initializing it.
    LocalAddress { local: LocalId },
    /// `[aggregate or address] -> [child or address]`: project by declaration index.
    /// A value operand copies the child and destroys the aggregate; an address borrows.
    AccessStatic { index: usize },
    /// `[base, index] -> [element or address]`: index an array, array address, or span.
    /// Array addresses and spans produce borrowed element addresses.
    AccessDynamic,
    /// `[address] -> [value]`: copy an initialized pointee, retaining managed owners.
    Load,
    /// `[address, value] -> [value]`: copy into storage, destroying its previous live
    /// value and marking a tracked local initialized; preserve the input value as result.
    Store,
    /// `[address, replacement] -> [previous value]`: exchange an initialized pointee
    /// with an owned replacement, transferring both values without copying or destruction.
    Replace,
    /// `[value] -> [ascribed value]`: preserve ownership while changing its type view.
    /// Types must match, differ by one nominal layer, or bridge a span and its record layout.
    Ascribe { ty: Ty },
    /// `[number] -> [converted number]`: convert explicitly, trapping on integer overflow.
    NumericCast { ty: Ty },
    /// `[Never] -> [result]` for type checking only: consume Never and terminate execution.
    /// `result` describes the unreachable continuation; no runtime value is constructed.
    Eliminate { result: Ty },
    /// `[value] -> []`: destroy the value.
    Discard,
    /// `[field_0, ..., field_n] -> [record]`: transfer fields in declaration order.
    MakeRecord { fields: Vec<Arc<str>> },
    /// `[element_0, ..., element_n] -> [array]`: transfer elements in order.
    /// The explicit element type permits an empty array.
    MakeArray { elements: usize, element: Ty },
    /// `[] -> [function]`: materialize a callable reference to the given function.
    Function { function: FunctionId },
    /// `[function, argument] -> [result]`: transfer one unit, single, or tuple argument
    /// into local zero of the callee; transfer its returned value to this stack.
    Call,
    /// `[argument_0, ..., argument_n] -> [result]`: invoke the checked builtin signature.
    /// Unlike `Call`, operands are separate; the builtin borrows them, then they are destroyed.
    CallBuiltin {
        name: Arc<str>,
        params: Vec<Ty>,
        result: Ty,
    },
}

/// Complete a structured region. Stack tops are on the right, as in [`Instr`].
/// Child IDs describe nesting, never arbitrary jumps. `next` runs after a region
/// yields; without `next`, its yielded operands pass to the enclosing region.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Yield all operands to the enclosing If or Loop. Invalid at function scope.
    Yield,
    /// Consume a bool, then execute one child with the remaining operands.
    /// Yielding arms must agree on their output stack; returning arms do not join.
    If {
        then: BlockId,
        els: BlockId,
        next: Option<BlockId>,
    },
    /// Repeatedly evaluate `condition` with the carried operands. It must yield
    /// the same operand types followed by a bool. False exits; true runs `body`.
    /// A yielding body must restore the condition's input types for the next iteration.
    /// Both children can return early, but the condition needs a yielding path.
    /// The loop yields its last condition operands when false, then runs `next`.
    Loop {
        condition: BlockId,
        body: BlockId,
        next: Option<BlockId>,
    },
    /// Transfer the sole operand to the caller. Cleanup must already be explicit.
    Return,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Module {
    /// Functions exported by the entry source file.
    pub entries: BTreeMap<Arc<str>, FunctionId>,
    /// Canonical nominal and structural definitions, indexed by [`TypeId`].
    pub types: TypeTable,
    pub functions: Vec<Function>,
    /// Decorated shader candidates and whether their static artifact is requested.
    pub shaders: BTreeMap<FunctionId, ShaderEntry>,
    /// Optional source origins; direct IR clients may leave this empty.
    pub origins: SourceMap,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SourceMap {
    pub functions: BTreeMap<FunctionId, SourceLocation>,
    pub instructions: BTreeMap<(FunctionId, BlockId, usize), SourceLocation>,
}

/// A rejected HIR operation, with its function and source span.
#[derive(Debug)]
pub struct Error {
    pub function: FunctionId,
    pub span: Span,
    pub kind: ErrorKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidHir { message: std::sync::Arc<str> },
    Type { kind: TypeErrorKind },
    UnboundValue { name: std::sync::Arc<str> },
    EagerRecursion { name: std::sync::Arc<str> },
    UninitializedValue { name: std::sync::Arc<str> },
    NotAPlace,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lowering error at {}..{}: ",
            self.span.start, self.span.end
        )?;
        match &self.kind {
            ErrorKind::InvalidHir { message } => f.write_str(message),
            ErrorKind::Type { kind } => TypeError { kind: kind.clone() }.fmt(f),
            kind => write!(f, "{kind:?}"),
        }
    }
}
impl std::error::Error for Error {}

/// Lower a typed tree into storage and control flow; verification is a separate pass.
pub fn generate(source: &resin_hir::Module) -> Result<Module, Error> {
    lower::generate(source)
}

/// Collect independent lowering errors across functions.
pub fn analyze(source: &resin_hir::Module) -> Result<Module, Vec<Error>> {
    lower::analyze(source)
}

pub fn format_module(module: &Module) -> String {
    print::format_module(module)
}

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
    ReusedBasicBlock,
    UnexpectedYield,
    MissingYield,
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

/// Check structured ownership, region yields, loop invariants, and instruction types.
pub fn verify(module: &Module) -> Result<(), VerifyError> {
    verify::analyze(module).map(|_| ())
}

/// A borrow of LIR and the analysis certifying that exact immutable module.
/// Construction requires successful verification.
///
/// ```compile_fail,E0451
/// use resin_lir::{ModuleTypes, Verified};
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
/// let checked = resin_lir::VerifiedModule::new(Default::default()).unwrap();
/// checked.view().module().functions.clear();
/// ```
///
/// ```compile_fail,E0505
/// let checked = resin_lir::VerifiedModule::new(Default::default()).unwrap();
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
        let analysis = verify::analyze(&module)?;
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
    let analysis = verify::analyze(module)?;
    use_module(Verified {
        module,
        analysis: &analysis,
    })
}

pub struct ModuleTypes {
    pub functions: Vec<FunctionTypes>,
    pub types: TypeTable,
}

/// Verified stack information, indexed by block and then instruction.
///
/// ```compile_fail,E0451
/// let forged = resin_lir::FunctionTypes {
///     inputs: vec![], results: vec![], operand_counts: vec![],
/// };
/// ```
pub struct FunctionTypes {
    /// Operand types at each block's entry, with the stack top last.
    pub inputs: Vec<Vec<Ty>>,
    /// The value produced by each instruction, or `None` for instructions without a result.
    pub results: Vec<Vec<Option<Ty>>>,
    operand_counts: Vec<Vec<usize>>,
}

impl FunctionTypes {
    /// Operand count for an instruction in the verified function.
    /// Indices must refer to a block and instruction in that same function.
    pub fn operand_count(&self, block: BlockId, instruction: usize) -> usize {
        self.operand_counts[block.index()][instruction]
    }
}
