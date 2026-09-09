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

The initial model uses explicit addresses and shared layouts. There is no automatic
value or allocation projection: compiling a shader preserves shared scalar/record
representations and the numeric bits of pointers. It neither walks nor copies a
pointer graph. A host allocation address is not translated by a cast or by calling
a helper. Applications obtain mapped host addresses and device virtual addresses
from the runtime and put the appropriate addresses in each representation.

For example, a record uploaded for device use may contain a device address obtained
from `gpu_host_to_device_pointer`; writing a mapped host address into that field is invalid, even
though both have source type `Ptr<T>`. Scalar/record fields must satisfy the shared
layout profile. Host-only types such as function values and unsupported storage
layouts are rejected from shader code. Aliases, nulls, cycles, and interior addresses
are preserved as bits; they do not trigger traversal, relocation, or allocation.
Null must not be dereferenced. An interior address must remain inside live storage
with the pointee's alignment and enough bytes for the accessed value.

Address spaces initially live in backend metadata: shader-local places are distinct
from device addresses. They may be read, written, and indexed in the shader, but may
not escape through calls, return values, stored pointer values, or reinterpretation.
Host and device addresses both use `Ptr<T>` in source; the compiler does not prove
which kind an arbitrary numeric address contains. Shared helpers may accept device
pointers when compiled for a shader and host pointers when compiled for the CPU.
A pointer/`ulong` cast preserves bits, never changes their address space, and confers
no ownership or lifetime guarantee. Shader-local address rejection remains explicit.

The caller owns allocation lifetime, upload/readback, synchronization, and visibility.
Host writes become device inputs only through the runtime's documented synchronization;
device writes require completion and visibility before host access. Copying a raw pointer
copies its address without retaining its allocation. Host copies of Arc fields retain
shared ownership; shader consumption of those managed fields is rejected.
The portable contract is deliberately limited to backends supporting this explicit
address/shared-layout profile. A future backend must implement it or reject the
program; recursive projection or address-space source types would be a separate
language change, not an implicit reinterpretation of existing programs.

The runtime follows the pointer-oriented direction of Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api), behind a small
C ABI. Vulkan buffer device addresses implement the current device address profile.

## Compiler architecture

The compiler is a workspace of unpublished phase crates. The data flow is source text →
CST → AST → HIR → LIR → verified LIR → C/GLSL. Each phase declares its public
language and operations in `lib.rs`, with private incoming `lower` and `print` modules.
The separate `resin-lir-verifier` crate certifies LIR before target lowering. See [compiler architecture](architecture.md) for the
crate graph, pass contracts, public entry points, and a reading path.

HIR is a resolved, typed tree. Its construction declares names, checks expressions,
solves dependency groups, and elaborates source forms: methods become ordinary calls,
short-circuit operators become conditionals, field projections are resolved, and layout
queries become constants. Inference variables and lexical scopes remain private to HIR
construction and editor analysis. Failed constraints preserve healthy editor facts;
source errors prevent publishing a complete HIR module.

LIR lowering consumes HIR alone and makes storage, definite initialization, evaluation
order, cleanup, and control flow explicit. Its typed stack machine has one parameter
local per function and a flat list of basic blocks. Lowering records owned locals per
lexical scope and emits conditional destruction at normal and error exits. Copy operations
retain shared fields; compiler temporary transfers disarm the source's cleanup. Named
values remain usable after reads; this is not source-level move checking.

Shared concrete types and layout rules live in `resin-common`. The source checker and
LIR verifier reuse these rules without sharing source scopes or inference state. Target
lowering chooses the ABI and device representation; target printers consume only their
own C/GLSL source trees. `resin-compiler` sequences these passes and retains immutable `Compilation` results.
`resin-toolchain` owns native tool resolution, process invocation, and artifact
caches. The root `resin` package provides one CLI for compilation, execution, formatting,
and the language server.

The Rust runtime methods are an unsafe convenience interface with the same lifetime and
synchronization contracts as the C ABI. Command recordings keep pending image layouts separate
from committed layouts. Submission rejects stale layout assumptions; cancellation discards pending
changes. The single queue conservatively orders buffer-device-address accesses between compute,
rendering, and copies with global memory barriers. This favors correctness until resource access
information permits narrower barriers.

Target lowering produces C and GLSL source. The toolchain compiles GLSL with `glslc` and
embeds the resulting SPIR-V words in C, then compiles and links C against the runtime.
Pipeline construction passes the embedded SPIR-V pointer and byte length to the runtime. This keeps
the compiler/runtime seam small while leaving room for multiple host compilers, graphics APIs, and
device code generators.

## Immediate path

Start with headless Vulkan compute: runtime creation, host-visible and device-local allocation,
host-to-device pointer translation, compute-pipeline creation from SPIR-V, command recording,
dispatch with one root-data GPU address, synchronous submission, and explicit destruction. Add
proper asynchronous queues and synchronization before transfers, textures and their descriptor
heap, raster pipelines, acceleration structures, and ray queries.
