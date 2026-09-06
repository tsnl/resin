# A tour of Resin

Resin is a systems language for host CPUs and Vulkan GPUs. This is a reading
path through its implementation, not a language reference; keep the
[README](README.md) nearby for syntax and command-line options.

The workspace has three Rust crates: the compiler at the root, the native
runtime in [resin-runtime/](resin-runtime/), and the parser in
[tree-sitter-resin/](tree-sitter-resin/). The [standard library](stdlib/) is
written in Resin and wraps the runtime's C API.

## 1. Start with a program

Read [examples/eg001.resin](examples/eg001.resin), then run these commands
from the repository root:

```sh
nix-shell
cargo run -- examples/eg001.resin
cargo run -- examples/eg001.resin --output ast
cargo run -- examples/eg001.resin --output ir
cargo run -- examples/eg001.resin --output c
```

These show the same Fibonacci program as source, an abstract syntax tree,
typed intermediate representation, and generated C. Running it compiles
and executes a native program; there is no bytecode interpreter behind the CLI.

A few language choices explain much of the implementation:

- Functions use `def` and are top-level declarations with explicit signatures. Their names
  are available before their bodies are checked, allowing mutual recursion.
- Every function is unary. An empty argument list is unit `()`; multiple
  arguments form a tuple.
- Value binding statements use `var`, including uninitialized locals; nominal
  types use `type`. Record fields remain `name = value`; parameters remain `name: Type`.
- Files have private scopes and explicit exports. Imports expose only exported
  names, and never execute code. There are no runtime global variables.
- Entry points are ordinary exported functions. `main` is only the default
  name; host entries take unit and return unit or `int`.

[examples/eg009_imports.resin](examples/eg009_imports.resin) and
[its counter module](examples/lib/counter.resin) demonstrate modules and
explicitly passed mutable state. Try its other entry point:

```sh
cargo run -- examples/eg009_imports.resin:independent
```

## 2. Follow the host compilation path

The main path is short enough to keep in mind:

```text
source files -> Tree-sitter -> AST -> typed stack IR -> C -> cc -> executable
                                          |
                                          +-> GLSL -> glslc -> SPIR-V
```

The shader branch compiles selected functions. Its SPIR-V is embedded in the
host executable, which uses the runtime to create Vulkan pipelines and run them.

### The CLI connects the stages

Start at `run` in [src/bin/resin.rs](src/bin/resin.rs). It loads the source,
generates IR, and selects an output path. `host` handles C generation and
native execution; `shader` handles standalone GLSL or SPIR-V output.
[source.rs](src/bin/resin/source.rs) parses the `FILE[:ENTRY]` selector.

The compiler stages are library modules exposed by [src/lib.rs](src/lib.rs),
so tests can exercise them without invoking the CLI. Note that `--output check`
currently checks parsing and import loading, not typing; use `--output ir` to
exercise the typed frontend without building an executable.

### The grammar and AST describe source, not execution

[grammar.js](tree-sitter-resin/grammar.js) is the source of truth for concrete
syntax. Skip the generated `src/parser.c` on a first read.

[src/ast/mod.rs](src/ast/mod.rs) defines the small set of source constructs:
files, statements, terms, and types, annotated with byte spans for diagnostics.
[generate.rs](src/ast/generate.rs) translates Tree-sitter nodes into these
structures, decodes literals, and lowers syntax such as operators into builtin
applications.

[load.rs](src/ast/load.rs) builds a `Program` of source modules in dependency
order, with the entry file last. It resolves relative imports and `std/` paths,
deduplicates canonical paths, and rejects import cycles. Loading files and
deciding which names are visible are separate jobs: visibility is handled later
by [IR module lowering](src/ir/generate/modules.rs).

### Learn the IR's vocabulary before its generator

Read these small definitions first:

- [ir/mod.rs](src/ir/mod.rs): a `Module` owns types, functions, and a map of
  function names exported by the entry file.
- [ir/instr.rs](src/ir/instr.rs): functions contain locals and basic blocks;
  blocks contain instructions followed by a terminator.
- [ir/value.rs](src/ir/value.rs) and [ir/types/mod.rs](src/ir/types/mod.rs):
  values, identifiers, and types used by those instructions.

This is a typed operand-stack IR, not SSA. Instructions consume and produce
stack values; terminators connect blocks or return from a function. Local
storage is explicit, with separate address, load, and store instructions.
The C and GLSL backends translate this stack model into target-language
variables and control flow; they do not execute the IR.

### Typing and lowering work together

[TyperContext](src/ir/typer/mod.rs) owns the nominal type-definition table.
A type's identity can be reserved before its body is defined; the completed
table moves into the IR module. Type IDs belong to that table, not to a global
registry. [rules.rs](src/ir/typer/rules.rs) describes operations on types, while
[convert.rs](src/ir/typer/convert.rs) describes allowed conversions.

[ir/generate/mod.rs](src/ir/generate/mod.rs) orchestrates source-to-IR lowering.
It establishes types and function signatures before lowering function bodies.
The neighboring files separate the questions asked during that process:

| Question | Start reading here |
| --- | --- |
| Which imported or exported declaration does a name mean? | [modules.rs](src/ir/generate/modules.rs) |
| Which local names exist, and are they initialized? | [scope.rs](src/ir/generate/scope.rs), [bindings.rs](src/ir/generate/bindings.rs) |
| What value does an expression produce? | [terms.rs](src/ir/generate/terms.rs) |
| Which storage location does an assignment or address refer to? | [places.rs](src/ir/generate/places.rs) |
| How do branches, loops, and short-circuit operators join? | [flow.rs](src/ir/generate/flow.rs) |
| How are blocks, locals, and instructions assembled? | [builder.rs](src/ir/generate/builder.rs) |

The distinction between a value and a place is worth following through one
pointer example: reading `x`, taking `&x`, and assigning to `x` share a name but
require different instructions.

### Verification is an independent boundary

[ir/verify/](src/ir/verify/) checks instruction operands, block-edge stack
types, returns, and type definitions. It must also reject malformed IR built
directly by a caller, without trusting the AST generator. Its type analysis is
reused by the backends.

[ir/types/definitions.rs](src/ir/types/definitions.rs) holds definition checks
shared by the typer and verifier, including invalid references and recursive
inline layouts. This is why those checks live beside the type representation,
not exclusively inside the source-language typer.

### C emission and native builds are separate

[backend/c/mod.rs](src/backend/c/mod.rs) assembles a C translation unit:
types, declarations, function bodies, embedded shaders, and a C `main` wrapper
calling the selected Resin entry. [function.rs](src/backend/c/function.rs)
lowers instructions and block edges; [foreign.rs](src/backend/c/foreign.rs)
bridges Resin's unary calls to conventional C argument lists.

[toolchain/c.rs](src/toolchain/c.rs) invokes the C compiler and links
`libresin_runtime.a`. It also owns the native build cache, dependency tracking,
and locks that keep concurrent builds and runs from interfering.

There are two artifact directories with different owners: Cargo builds the
compiler and runtime under `target/`; Resin builds user programs under `build/`
in the caller's working directory. Default runs reuse an unoptimized native
build. Requesting an executable with `-o` selects the optimized cache. This
does not change Cargo's Rust profile.

## 3. Follow a shader into the runtime

Read [examples/gradient.resin](examples/gradient.resin) for compute, then
[examples/triangle.resin](examples/triangle.resin) and
[its shader functions](examples/lib/triangle.resin) for graphics. To inspect
a shader without running a Vulkan program:

```sh
cargo run -- examples/gradient.resin:kernel --output glsl
```

`shader(function, "stage")` is the compiler-mediated step. It becomes an IR
instruction that [toolchain/shaders.rs](src/toolchain/shaders.rs) discovers.
That code emits GLSL, invokes `glslc`, caches the result, and supplies SPIR-V
for embedding in the generated C. The runtime receives bytes, not a host
function pointer or a source-file path.

[backend/glsl/mod.rs](src/backend/glsl/mod.rs) collects reachable shader
functions. [entry.rs](src/backend/glsl/entry.rs) adapts regular Resin function
signatures to compute, vertex, or fragment interfaces, and
[function.rs](src/backend/glsl/function.rs) lowers their bodies. The GLSL
backend supports a subset of the host language; it rejects operations such as
foreign calls and recursion rather than making them work on the device.

The remaining GPU operations are ordinary standard-library calls. Follow one
from [stdlib/gpu.resin](stdlib/gpu.resin), through
[resin_runtime.h](resin-runtime/include/resin_runtime.h) and its included
headers, to [the runtime](resin-runtime/src/lib.rs). The same pattern applies
to images and windows. [stdlib/status.resin](stdlib/status.resin) is a small
example of adding a Resin convenience wrapper over foreign declarations.

Inside the runtime, the useful landmarks are:

- [gpu/device.rs](resin-runtime/src/gpu/device.rs): Vulkan device discovery and
  required features.
- [gpu/mod.rs](resin-runtime/src/gpu/mod.rs): allocations, images, command
  recording, synchronization between operations, and submission.
- [gpu/pipeline.rs](resin-runtime/src/gpu/pipeline.rs): conventional Vulkan
  compute and graphics pipelines.
- [gpu/present.rs](resin-runtime/src/gpu/present.rs) and
  [window/](resin-runtime/src/window/): swapchains, presentation, and statically
  linked GLFW.
- [allocator/range.rs](resin-runtime/src/allocator/range.rs): aligned
  suballocation using a sorted list of free ranges.

Buffers make the host/device boundary concrete. The host allocates memory,
writes root data, and passes its device address when dispatching or drawing.
Shader entry wrappers interpret that root according to their supported
interface. [backend/layout.rs](src/backend/layout.rs) keeps supported buffer
layouts consistent between C and GLSL; start there when investigating a field
offset or alignment mismatch.

These APIs expose resource lifetimes explicitly. The C API and its unsafe Rust
convenience API are not ownership-safe GPU abstractions: resources must remain
alive while commands use them. Shader bodies describe individual invocations;
the compiler does not synthesize workgroup-local storage or barriers.

Finally, [examples/particles.resin](examples/particles.resin) combines compute
and graphics over a shared buffer, and [examples/window.resin](examples/window.resin)
adds presentation. Actual GPU runs need a supported Linux Vulkan device; window
runs also need a display. The headless image demos write PNGs in cwd.

## 4. Find the test closest to your change

Tests are executable descriptions of the boundaries above:

| Change | Useful tests |
| --- | --- |
| Syntax or AST shape | [parser corpus](tree-sitter-resin/test/corpus/), [mutation_ast.rs](tests/mutation_ast.rs) |
| Imports, exports, or entry visibility | [modules.rs](tests/modules.rs), [cli.rs](tests/cli.rs) |
| Typing, conversions, or IR invariants | [nominal_types.rs](tests/nominal_types.rs), [typer tests](src/ir/typer/tests.rs), [verifier tests](src/ir/verify/tests.rs) |
| Host code generation or C interop | [c_backend.rs](tests/c_backend.rs), [foreign.rs](tests/foreign.rs), [printing.rs](tests/printing.rs) |
| Compilation and artifact reuse | [build_cache.rs](tests/build_cache.rs) |
| Shader generation or execution | [glsl_backend.rs](tests/glsl_backend.rs), [gpu_backend.rs](tests/gpu_backend.rs), [window_backend.rs](tests/window_backend.rs) |

[resin-runtime/tests/](resin-runtime/tests/) also exercises the native runtime
with GLSL fixtures, independently of the Resin compiler. This is useful for
separating a Vulkan runtime bug from a language or code-generation bug.

For a focused host-only starting point, inside `nix-shell`:

```sh
cargo test -p resin --test modules --test c_backend
```

For the full path, with Vulkan and a desktop display or Xvfb available:

```sh
RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_WINDOW=1 \
  cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The requirement variables prevent missing GPU tools or facilities from turning
coverage into skipped tests. See [Development in the README](README.md#development)
for environment setup and parser regeneration. Grammar changes must include
regenerated parser files; runtime API changes generally span the Rust
implementation, C headers, and standard-library declarations. Most new runtime
operations do not need a new compiler builtin.
