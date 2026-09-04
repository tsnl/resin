# Resin design

Resin is a deliberately small systems programming language in the spirit of C and Go. It should
make data layout, mutation, pointers, control flow, and cost easy to see while removing incidental
complexity from the toolchain. The MVP is monomorphic, fail-fast, and manually managed: `Ptr<T>`
and `Span<T>` are the fundamental ways to share memory, and there is no implicit tracing, reference
counting, or exception machinery. Separate value and type namespaces keep ordinary definitions and
nominal type definitions simple, including productive recursive definitions through pointers.

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

The tree-sitter AST remains an untyped source representation. Lexical scopes stored alongside the
AST carry resolved value and type definitions. A small `typer` module contains bottom-up rules that
accept already-resolved child types and return the enclosing type; it does not walk syntax or emit
code. Compile-time instantiation uses a deliberately restricted `evaluate()` operation, initially
limited to literals. An IR generator can therefore interleave scope resolution, typing, evaluation,
and emission without coupling the reusable typing rules to a particular backend. Errors propagate
immediately and compilation stops after the first useful diagnostic.

The first IR is a typed stack machine: functions own flat lists of basic blocks, instructions make
evaluation order explicit, and terminators provide control flow. Locals, globals, and closure
nonlocals provide stable storage; stack values include literals, aggregates, closures, and addresses,
with field and array access resolved from type information. A separate verifier checks stack effects
and block edges after generation. Privileged operators retain their checked monomorphic signatures
in IR, while the backend delays selecting or synthesizing their concrete implementations until it
must emit the target.

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
