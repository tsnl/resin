# Resin design

Resin is a deliberately small systems programming language in the spirit of C and Go. It should
make data layout, mutation, pointers, control flow, and cost easy to see while removing incidental
complexity from the toolchain. The language is monomorphic and manually managed: `Ptr<T>`
and `Span<T>` are the fundamental ways to share memory, and there is no implicit tracing, reference
counting, or exception machinery. Separate value and type namespaces keep ordinary definitions and
nominal type definitions simple, including productive recursive definitions through pointers.

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

`defer { ... };` registers a unit-valued block for lexical scope exit, including early
returns through `?`. Cleanup runs in reverse registration order, only for registrations
reached on that path. Names bind at registration; local values and initialization facts
are read at exit. Return values are saved before cleanup. Deferred blocks may nest, but
cannot themselves propagate errors. This is explicit cleanup, not an ownership system;
process termination and traps do not unwind scopes.

## Host and GPU

The long-term goal is CUDA-like heterogeneous programming for graphics. An ordinary Resin function
may be compiled for the host or specialized into a compute, raster, or ray-tracing pipeline. Device
authority is expressed with ordinary capability arguments—such as a `RayQuery`—instead of a language
effect system. Compiling a device entry point recursively projects its parameter and result types
from their host representations to device representations. Projection is both a type operation and
the recipe for constructing the device value; types with no valid projection are compile-time
errors, even if a host-only execution path happens to work.

The runtime follows the pointer-oriented direction of Sebastian Aaltonen's
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api). CPU-visible GPU allocations
have both a mapped host address and a GPU virtual address, and the runtime translates between them;
device code uses ordinary pointers and spans rather than buffer-plus-offset bindings. Textures and
acceleration structures remain opaque, bindless resources whose compact handles can live inside the
same projected data structures. The runtime hides operating-system and graphics-API details behind
a small C ABI. Vulkan and buffer device address are the first implementation, not part of the
language contract; other native backends can follow.

## Compiler architecture

The tree-sitter AST remains an untyped source representation. The generator's lexical scopes
resolve value and type names. `TyperContext` owns a `Vec<TypeDef>` and the bottom-up rules that
accept already-resolved child types and return the enclosing type; it does not walk syntax or emit
code. It reserves nominal identities before recursive RHS evaluation. Bodies start as `None`;
completion validates the body before setting `Some(body)` through `&mut self`. Redefinition and
exporting unfinished tables are errors; failed validation remains retryable. The generator reuses
one context and moves its completed definition table into the IR module without cloning.
Standalone clients can start with `TyperContext::new()` or take ownership of an existing table
with `from_definitions`.
The evaluator resolves type expressions and literals. Functions needing inference first collect
constraints, unify ordinary type holes, and solve error-set inclusion to a fixed point. Concrete
node-keyed results guide lowering; inference variables never enter the IR. A generator can then
interleave scope resolution, typing, evaluation, and emission without coupling the reusable typing
rules to a particular backend. Errors propagate
immediately and compilation stops after the first useful diagnostic.

Deferred bodies retain their AST identities and lexical environments. Lowering emits them
at normal and error exits, using current initialization facts keyed by local ID rather than
name. This keeps shadowing correct and introduces no new IR instructions or backend machinery.

The IR data model lives in `types`, `value`, and `instr`. Nominal reference and layout checks
belong to `types::definitions`, shared by the typer and verifier. The typer separates definition
ownership, typing rules, and conversions. Generation keeps syntax lowering and scopes together,
but its evaluator borrows only scopes and types, and its function builder
depends only on IR data. Verification separates control-flow traversal, instruction checks, and
type checks; printing separates name allocation from formatting. Each pass owns its diagnostics.

The first IR is a typed stack machine: each function has one parameter local and owns a flat list
of basic blocks. Instructions make
evaluation order explicit, and terminators provide control flow. Locals provide stable storage;
stack values include literals, aggregates, function references, and addresses,
with field and array access resolved from type information. A separate verifier checks stack effects
and block edges after generation. Privileged operators retain their checked monomorphic signatures
in IR, while the backend delays selecting or synthesizing their concrete implementations until it
must emit the target.

The Rust runtime methods are an unsafe convenience interface with the same lifetime and
synchronization contracts as the C ABI. Command recordings keep pending image layouts separate
from committed layouts. Submission rejects stale layout assumptions; cancellation discards pending
changes. The single queue conservatively orders buffer-device-address accesses between compute,
rendering, and copies with global memory barriers. This favors correctness until resource access
information permits narrower barriers.

Backends scan this IR to produce the artifacts for one program. The host path emits C which links
against the runtime. Device paths emit SPIR-V and embed its words directly in that generated C; the
first implementation may emit GLSL and invoke `glslc`, while later backends can lower IR directly.
Pipeline construction passes the embedded SPIR-V pointer and byte length to the runtime. This keeps
the compiler/runtime seam small while leaving room for multiple host compilers, graphics APIs, and
device code generators.

## Immediate path

Start with headless Vulkan compute: runtime creation, host-visible and device-local allocation,
host-to-device pointer translation, compute-pipeline creation from SPIR-V, command recording,
dispatch with one root-data GPU address, synchronous submission, and explicit destruction. Add
proper asynchronous queues and synchronization before transfers, textures and their descriptor
heap, raster pipelines, acceleration structures, and ray queries.
