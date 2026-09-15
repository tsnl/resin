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
mod profile;
mod verify;

use std::{collections::BTreeMap, num::NonZeroUsize, sync::Arc};

define_id! {
    pub struct BlockId(usize);
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: Option<Arc<str>>,
    pub profile: Profile,
    pub foreign: Option<Foreign>,
    pub result: Ty,
    /// The first `parameter_count` locals are parameters, in declaration order.
    /// They start initialized; a zero-argument function needs no parameter locals.
    pub parameter_count: usize,
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
            params: self
                .locals
                .get(..self.parameter_count)?
                .iter()
                .map(|local| local.ty.clone())
                .collect(),
            result: Box::new(self.result.clone()),
        })
    }
}

/// Typed operand-stack operations. Contracts show the stack top on the right;
/// an unchanged prefix is omitted. Popped values transfer ownership unless an
/// instruction specifies copying, borrowing, or destruction. Addresses borrow storage.
/// HIR construction establishes definite initialization. Storage lowering makes ownership
/// explicit and emits runtime initialization flags for managed locals. The verifier
/// checks instruction types and stack shape.
///
/// ```compile_fail,E0432
/// use resin_lir::StackEffect;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    /// `[native_gpu, strong_owner, bytes, alignment, memory] -> [{value: GpuView | None, status: int}]`.
    /// Allocation retains the device owner. All operands are consumed.
    GpuViewAllocate,
    /// `[view, byte_offset, bytes, alignment] -> [view]`: validate range/alignment and transfer its owner.
    GpuViewOffset,
    /// `[view, capacity, start, length] -> [view]`: check an element range and transfer its owner.
    GpuViewRange { element: Ty },
    /// `[view, access_mask] -> [view]`: remove permissions and transfer its owner.
    GpuViewRestrict,
    /// `[view] -> [element]`: copy plain storage after checking read access.
    GpuViewLoad { element: Ty },
    /// `[view, element] -> [unit]`: copy plain storage after checking write access.
    GpuViewStore,
    /// `[view, element] -> [element]`: exchange plain storage after checking read/write access.
    GpuViewReplace,
    /// `[view, count, destination: Ptr<T>, destination_length] -> [unit]`: check and copy readable elements.
    GpuViewCopyTo,
    /// `[view, length, commands, image] -> [int]`: record an image copy retaining its allocation.
    GpuViewCopyImage,

    /// `[gpu] -> [Result<Pipeline, E>]`: create the registered source pipeline
    /// from the declared compute shader, retaining its root type and factory owner.
    GpuComputePipeline {
        pipeline: Ty,
        factory: FunctionId,
        shader: FunctionId,
    },
    /// `[gpu] -> [Result<Pipeline, E>]`: create the registered source pipeline
    /// from compatible vertex and fragment declarations. Rootless stages use None.
    GpuGraphicsPipeline {
        pipeline: Ty,
        factory: FunctionId,
        vertex: FunctionId,
        fragment: FunctionId,
    },
    /// `[commands, pipeline, host root, x, y, z] -> [Result<(), E>]`: project
    /// checked arguments and pass them with the pipeline owner to the recording function.
    GpuDispatch {
        projection: resin_types::GpuProjectionPlan,
        context: FunctionId,
        allocator: FunctionId,
        record: FunctionId,
    },
    /// `[commands, pipeline, host root or None, count] -> [Result<(), E>]`.
    /// Rootless graphics performs no allocation or projection.
    GpuDraw {
        projection: Option<resin_types::GpuProjectionPlan>,
        context: FunctionId,
        allocator: Option<FunctionId>,
        record: FunctionId,
    },
    /// `[arguments, commands, x, y, z] -> [int]`: record a dispatch with retained arguments.
    GpuArgumentsDispatch,
    /// `[arguments, commands, count] -> [int]`: record a draw with retained arguments.
    GpuArgumentsDraw,
    /// `[count, initial] -> [StrongOwner | None]`: allocate initialized element storage.
    /// Installs the concrete element destructor; failed allocations publish no owner.
    OwnerAllocate { element: Ty },
    /// `[Ptr<StrongOwner>] -> [Ptr<T>]`: borrow live payload storage.
    OwnerData { pointee: Ty },
    /// `[Ptr<StrongOwner>] -> [ulong]`: read the immutable element count.
    OwnerLength,
    /// `[Ptr<StrongOwner>] -> [WeakOwner]`: acquire a weak reference.
    OwnerDowngrade,
    /// `[Ptr<WeakOwner>] -> [StrongOwner | None]`: acquire a strong reference if live.
    OwnerUpgrade,
    /// `[] -> [WeakOwner]`: construct an empty weak reference.
    WeakEmpty,
    /// `[address] -> [value]`: transfer a pointee without copying or clearing storage.
    /// Lowering must disarm its previous owner, typically with `ForgetLocal`.
    TransferLoad,
    /// `[] -> []`: clear a local's initialization flag without destroying its value.
    ForgetLocal { local: LocalId },
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
    /// `[] -> [{data: Ptr<ubyte>, length: ulong}]`: borrow the decorated
    /// function's embedded SPIR-V bytes as structural byte transport.
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
    /// A value operand copies the child and destroys the aggregate; raw addresses borrow.
    AccessStatic { index: usize },
    /// `[base, index] -> [element or address]`: index an array, array address, or `str`.
    /// Array addresses and string literals produce borrowed element addresses;
    /// array values copy the element and destroy the consumed array.
    AccessDynamic,
    /// `[Ptr<T>, length, index] -> [Ptr<T>]`: typed element addressing; the host
    /// diagnoses an index outside length, while shaders require a valid index.
    PointerIndex,
    /// `[Ptr<T>, capacity, start, count] -> [Ptr<T>]`: check and address a range.
    /// An empty range may start one past the end. Host-only.
    PointerRange,
    /// `[Ptr<numeric>, count] -> [{data: Ptr<ubyte>, length: ulong}]`.
    /// Checks byte-count overflow. Host-only.
    PointerBytes,
    /// `[address] -> [value]`: copy an initialized pointee, retaining managed owners.
    Load,
    /// `[address, value] -> [value]`: copy into storage, destroying its previous live
    /// value and marking a tracked local initialized; preserve the input value as result.
    Store,
    /// `[address, replacement] -> [previous value]`: exchange an initialized pointee
    /// with an owned replacement, transferring both values without copying or destruction.
    Replace,
    /// `[value] -> [ascribed value]`: preserve ownership while changing its type view.
    /// Types must match, differ by one nominal layer, or expose a `str` byte view.
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
    /// `[function, argument_0, ..., argument_n] -> [result]`: transfer arguments
    /// into the callee's parameter locals, then transfer its returned value.
    Call { arguments: usize },
    /// `[argument_0, ..., argument_n] -> [result]`: invoke the checked builtin signature.
    /// The builtin borrows its operands, then they are destroyed.
    CallBuiltin {
        name: Arc<str>,
        params: Vec<Ty>,
        result: Ty,
    },
}

/// Complete a structured region. Stack tops are on the right, as in [`Instr`].
/// Child IDs describe nesting, never arbitrary jumps. `next` runs after a region
/// completes. A result-producing region without `next` must be the tail of a
/// selection arm; its operands then pass to that selection's merge. Function,
/// loop-condition, and loop-body continuations need their own explicit terminator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Terminator {
    /// Complete a selection arm, transferring all operands to its If's merge.
    /// Nested tail selections may forward to the same merge. Invalid at function
    /// scope or as the completion of a loop condition or body.
    Merge,
    /// `[carried..., bool] -> [carried...]`: test the loop condition. True enters
    /// the body; false exits the Loop with the carried operands. Only valid as
    /// the completion of a loop's condition region.
    LoopTest,
    /// Complete a loop body, transferring all operands to its condition region
    /// for the next iteration. Only valid as the completion of a loop body.
    Continue,
    /// Consume a bool, then execute one child with the remaining operands.
    /// Merging arms must agree on their output stack; returning arms do not join.
    If {
        then: BlockId,
        els: BlockId,
        next: Option<BlockId>,
    },
    /// Repeatedly evaluate `condition` with the carried operands. Its LoopTest
    /// needs the same operand types followed by a bool. False exits; true runs `body`.
    /// The body's Continue must restore the condition's input types.
    /// Both children can return early, but the condition needs a LoopTest path.
    /// The final condition's carried operands enter `next` on the false exit.
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

/// A rejected HIR operation, with its span and optional immutable source.
/// Constructed HIR can omit sources; its spans are still preserved.
#[derive(Debug)]
pub struct Error {
    pub source: Option<Source>,
    pub span: Span,
    pub kind: ErrorKind,
    /// Bounded application trace, from the immediate requester toward its root.
    pub applications: Vec<ApplicationNote>,
}

#[derive(Debug)]
pub struct ApplicationNote {
    pub function: Arc<str>,
    /// Rendered with source nominal names, independent of a partial concrete catalog.
    pub arguments: Vec<Arc<str>>,
    pub profile: Profile,
    pub location: Option<SourceLocation>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    InvalidInstance {
        message: Arc<str>,
    },
    UnsupportedProfile {
        profile: Profile,
        message: Arc<str>,
    },
    MonomorphLimit {
        function: Arc<str>,
        limit: usize,
        arguments: Vec<Arc<str>>,
        profile: Profile,
    },
    TypeExpansionLimit {
        limit: usize,
    },
    TypeSizeLimit {
        limit: usize,
    },
    InvalidHir {
        message: std::sync::Arc<str>,
    },
    Type {
        kind: TypeErrorKind,
    },
    UnboundValue {
        name: std::sync::Arc<str>,
    },
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
            ErrorKind::InvalidHir { message }
            | ErrorKind::InvalidInstance { message }
            | ErrorKind::UnsupportedProfile { message, .. } => f.write_str(message),
            ErrorKind::MonomorphLimit {
                function,
                limit,
                arguments,
                profile,
            } => write!(
                f,
                "function {function} exceeds its limit of {limit} monomorphs while requesting {arguments:?} for {profile:?}"
            ),
            ErrorKind::TypeExpansionLimit { limit } => write!(
                f,
                "type expression exceeds the expansion depth limit of {limit}"
            ),
            ErrorKind::TypeSizeLimit { limit } => write!(
                f,
                "type expression exceeds the expansion size limit of {limit} nodes"
            ),
            ErrorKind::Type { kind } => TypeError { kind: kind.clone() }.fmt(f),
            kind => write!(f, "{kind:?}"),
        }
    }
}
impl std::error::Error for Error {}

/// Semantic target of a concrete function instance. Shader stages share helper rules;
/// their entry conventions are retained separately in `Module::shaders`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Profile {
    Host,
    Shader,
}

/// One externally requested application. Arguments are closed HIR type expressions;
/// nominal origins become concrete identities only when an operation demands their types.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Entry {
    pub name: Arc<str>,
    pub function: FunctionId,
    pub arguments: Vec<resin_hir::Type>,
    pub profile: Profile,
}

/// Build LIR for only requested entries and their transitive function/type dependencies.
/// The input still contains all structurally completed source declarations.
pub fn build_lir(
    source: &resin_hir::Module,
    entries: &[Entry],
    options: &LoweringOptions,
) -> Result<Module, Vec<Error>> {
    lower::instantiate(source, entries, options)
}

/// Resource limits for one HIR-to-LIR construction run.
#[derive(Debug, Clone)]
pub struct LoweringOptions {
    pub max_monomorphs_per_function: NonZeroUsize,
}

impl Default for LoweringOptions {
    fn default() -> Self {
        Self {
            max_monomorphs_per_function: NonZeroUsize::new(16 * 1024).unwrap(),
        }
    }
}

/// Build LIR for every concrete body; verification is a separate pass.
pub fn build_lir_all(source: &resin_hir::Module) -> Result<Module, Error> {
    build_lir_all_with_options(source, &LoweringOptions::default())
        .map_err(|mut errors| errors.remove(0))
}

/// Build LIR for every concrete body, collecting independent errors.
pub fn build_lir_all_with_options(
    source: &resin_hir::Module,
    options: &LoweringOptions,
) -> Result<Module, Vec<Error>> {
    lower::analyze(source, options)
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
    InvalidGpuOperation,
    UnsupportedGpuElement { ty: Ty },
    InvalidVariant,
    InvalidDropHook,
    InvalidForeignSignature,
    OpaqueValue { ty: Ty },
    InvalidShader,
    UnsupportedProfile { profile: Profile, message: Arc<str> },
    InvalidProfile { expected: Profile, found: Profile },
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
    UnexpectedMerge,
    UnexpectedLoopTest,
    UnexpectedContinue,
    MissingLoopTest,
    MissingRegionResult,
    MissingRegionContinuation,
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

/// Check structured ownership, explicit region exits, loop invariants, and instruction types.
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

    /// Dependency order for a requested shader entry, established during verification.
    pub fn shader_functions(self, entry: FunctionId) -> Option<&'a [FunctionId]> {
        self.analysis.shaders.get(&entry).map(Vec::as_slice)
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
    shaders: BTreeMap<FunctionId, Vec<FunctionId>>,
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
