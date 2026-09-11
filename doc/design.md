# Resin design

The [ownership specification](lifetimes.md) describes `Arc<T>`, `Weak<T>`,
automatic destruction, and inherent methods through `impl`.

Resin is a deliberately small systems programming language in the spirit of C and Go.
Data layout, mutation, pointers, control flow, and cost stay visible. Reading existing values performs compiler-defined copying; function and type
applications consume the resulting arguments. Constructors consume field initializers.
Shared handles make everyday resource copying safe. Raw `Ptr<T>` and `Span<T>` values remain
non-owning and there is no borrow checker or tracing collector. Separate value and
type namespaces keep definitions simple, including recursion through pointers.

Every function takes exactly one argument. `()` supplies unit, and `(a, b)` supplies a tuple;
`def f(a: A, b: B) -> R = { body };` destructures that tuple into local bindings. Function types
follow the same rule: `() -> R`, `(A) -> R`, and `(A, B) -> R`. Tuples use positional record fields in IR.
Calls and assignments preserve nominal identity; explicit `T(value)` ascriptions
wrap or unwrap one nominal record layer. Union and Result values may widen their variant sets,
but mutable pointers remain invariant. Record initializers evaluate fields in source order before
assembling them in the type's layout order.

Functions are top-level, immutable definitions without captured environments. Signatures are
available before bodies are checked, so mutually recursive functions need no forward declarations.
An omitted result annotation means unit; explicit `_` holes enable inference in locals and function
results, including nested positions. Value bindings use `var name = value;`, nominal records use
`struct Name { field: Type };`, and `type Name = Type;` creates transparent aliases. Declarations
such as `var name: Type;` reserve uninitialized local storage: reads require prior initialization
on every control-flow path. An aggregate must be initialized as a whole before its fields can be
accessed. Record initializers keep bare `name = value` fields; parameters and record type fields
keep bare `name: Type` declarations. Files have no runtime globals or initialization phase.

Unions are canonical sets of nominal structs, with program-local u32 tags independent of union
membership. `Result<T, E>` is first-class, including nested Results. `ok` and `err` construct its
branches; exhaustive `match` expressions bind payloads, and postfix `?` returns errors early.
An inferred error set is the least union of errors propagated by a dependency group, or `Never`
when empty.

Initialized locals receive automatic destruction in reverse scope order, including
loop iterations and early returns through `?`. Return values are preserved before
cleanup. `impl` defines inherent methods and `drop(self: Ptr<T>)` hooks; the compiler
runs the hook before releasing fields. Statement-only chain blocks yield unit.
Process termination and traps do not unwind scopes.

## Host and GPU

Host code uses builtin `GpuPtr<T>` and `GpuSpan<T>` views, carrying an allocation
owner, byte offset, and access permissions. Copies and interior views retain the
owner. Checked host operations enforce bounds, alignment, mapping state, permissions,
and exclusion while a command recording can use the allocation. Views cannot be
cast to ordinary pointers or constructed from raw addresses.

Shader entries retain ordinary `Ptr<T>` parameters. The compiler builtin
`shader.project(gpu, arguments)` derives a host launch-record shape from the shader
root: shader pointers and spans become owning GPU views on the host. Projection
builds separate root storage, converts the views internally, and retains their
allocations. GPU buffer elements use one shared host/device layout and cannot
contain pointers or managed owners. This boundary does not traverse pointer graphs
or modify host records into device representations.

Shader-local places remain distinct from device addresses in backend metadata.
They may be read, written, and indexed but cannot escape through calls, return
values, stored pointers, or reinterpretation. Shared helpers may take device pointers
when compiled for a shader and host pointers when compiled for the CPU. Numeric
pointer casts preserve bits and do not translate addresses or confer ownership.

Command recordings accept projected `GpuArguments`, retain the root and every
referenced allocation, and exclude CPU access until synchronous submission or
cancellation. Device writes require completion and visibility before host access.
Raw host pointers still borrow memory without retaining it, and shader consumption
of managed host values remains rejected.

The runtime draws on Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), behind a small
C ABI. Vulkan buffer device addresses implement shader access internally.
See [GPU buffers](gpu-buffers.md) for typed allocation, projection, and command lifetime.

## Compiler architecture

The compiler is a workspace of unpublished phase crates. The data flow is source text →
CST → AST → HIR → LIR → verified LIR → C/SPIR-V. Each phase declares its public
language and operations in `lib.rs`, with private incoming `lower` and `print` modules.
LIR's private verifier certifies its language before target lowering; the certificate
and verification operations belong to `resin_lir`'s public interface. See
[compiler architecture](architecture.md) for the crate graph, pass contracts, public
entry points, and a reading path.

HIR is a resolved, typed tree. Its construction declares names, checks expressions,
solves dependency groups, and elaborates source forms: methods become ordinary calls,
short-circuit operators become conditionals, field projections are resolved, and layout
queries become constants. Inference variables and lexical scopes remain private to HIR
construction and editor analysis. Failed constraints preserve healthy editor facts;
source errors prevent publishing a complete HIR module.

LIR lowering consumes HIR alone and makes storage, definite initialization, evaluation
order, cleanup, and control flow explicit. Its typed stack machine has one parameter
local per function and a tree of blocks with explicit `If`, `Loop`, `Merge`,
`LoopTest`, `Continue`, and `Return` terminators. Block IDs locate storage in an
arena; child references express unique ownership, never arbitrary jumps.
Verification checks selection merges, distinct loop tests and continuations, and loop
stack invariants, and both backends preserve the nesting in native control flow.
Lowering records owned locals per lexical scope and emits conditional destruction
at normal and error exits. Copy operations retain shared fields; compiler temporary transfers disarm the source's cleanup. Named
values remain usable after reads; this is not source-level move checking.

Shared concrete types and layout rules live in `resin-types`. The source checker and
LIR verifier reuse these rules without sharing source scopes or inference state. Target
lowering chooses the ABI and device representation. The C printer consumes its private
source tree; the SPIR-V backend assembles binary instructions directly.

Compiler inputs are immutable `Source` handles from `resin-source`. A clone shares
one text version; a replacement keeps the logical source ID and leaves the old
version usable. Names serve diagnostics and need not be paths or unique. Locations
retain the source alongside a byte span, so their meaning survives later edits.

`resin_compiler::Compiler::compile(entry, loader)` resolves imports and returns an
`Arc<Compilation>` containing completed phase products and editor facts. The concrete
`resin_source::Loader` supplies files, registered buffer text, and explicit import
bindings. It resolves relative and `$/` imports and reuses unchanged source
instances. Source loading has no dependency on compiler phases or concrete types.
Editors register changes and remove closed buffers through the loader.

`Compiler` owns its cache fields directly; `Compilation` owns its retained products.
Their definitions, queries, and private implementation stay together in `lib.rs`.
A small public interface can have a substantial, cohesive implementation. The
compiler has no native-build, file-notification, or protocol-version API.

Codegen consumes a compilation's verified LIR and writes a complete C/SPIR-V/Ninja
project. Its target ASTs stay private. `resin-toolchain` builds the directory through
Ninja and retains output files; it depends on no compiler or type crate. The root `resin` package provides one CLI for compilation, execution, formatting,
and the language server; its request and destination validation stay private to the CLI.

The Rust runtime methods are an unsafe convenience interface with the same lifetime and
synchronization contracts as the C ABI. Command recordings keep pending image layouts separate
from committed layouts. Submission rejects stale layout assumptions; cancellation discards pending
changes. The single queue conservatively orders buffer-device-address accesses between compute,
rendering, and copies with global memory barriers. This favors correctness until resource access
information permits narrower barriers.

Target lowering produces C, SPIR-V binaries, and Ninja dependency edges in one operation.
The toolchain supplies native command rules and explicit settings: Ninja runs `spirv-opt -O`,
invokes the current Resin executable to embed the optimized SPIR-V in C headers, and compiles
and links the host against the runtime. It performs no tool preflight. Install Ninja,
SPIR-V Tools, and a C compiler, choosing the optimizer with `SPIRV_OPT` or `--spirv-opt`
and the C compiler with `CC` or `--cc` when needed.
Pipeline construction passes the embedded SPIR-V pointer and byte length to the runtime. This keeps
the compiler/runtime seam small while leaving room for multiple host compilers, graphics APIs, and
device code generators.

## Immediate path

Start with headless Vulkan compute: runtime creation, host-visible and device-local allocation,
host-to-device pointer translation, compute-pipeline creation from SPIR-V, command recording,
dispatch with one root buffer handle, synchronous submission, and automatic owner cleanup. Add
proper asynchronous queues and synchronization before transfers, textures and their descriptor
heap, raster pipelines, acceleration structures, and ray queries.
