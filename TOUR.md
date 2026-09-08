# A tour of Resin

Resin is a systems language for host CPUs and Vulkan GPUs. This is a reading
path through its implementation, not a language reference; keep the
[README](README.md) nearby for syntax and command-line options.

The workspace has four Rust crates: the compiler at the root, the native
runtime in [resin-runtime/](resin-runtime/), the parser in
[tree-sitter-resin/](tree-sitter-resin/), and the language server in
[resin-lsp/](resin-lsp/). The [Zed extension](zed-resin/) is a separate Cargo
workspace. The [standard library](stdlib/) is written in Resin and wraps the
runtime's C API.

## 1. Start with a program

Read [examples/eg001.resin](examples/eg001.resin). On Linux/macOS, enter
`nix-shell` from the repository root first; [.envrc](.envrc) also loads that
environment if you use direnv. On Windows, use the Visual Studio developer
PowerShell and LLVM Clang setup in [Development](README.md#development).
Then run these commands from the repository root:

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg001.resin -o dist/
```

The first command builds and runs the Fibonacci program; the second builds an optimized
executable without running it. Inspect the generated `program.c` beside each cached executable
under `build/`. There is no bytecode interpreter behind the CLI.

A few language choices explain much of the implementation:

- Functions use `def` and are top-level declarations with typed parameters;
  an omitted result type means unit. Their names are available before their
  bodies are checked, allowing mutual recursion. Explicit `_` holes opt into
  inference in local annotations and function results; omission still means unit.
- Every function is unary. An empty argument list is unit `()`; multiple
  arguments form a tuple.
- Value binding statements use `var`, including uninitialized locals; nominal
  records use `struct`, and `type` creates transparent aliases. Record fields remain
  `name = value`; parameters remain `name: Type`.
- `A | B` is a structural union of nominal structs. `Result<T, E>` is first-class;
  `ok` and `err` construct it, `match` handles variants, and postfix `?` propagates errors.
- `Arc<T>` and `Weak<T>` provide shared ownership. Value reads copy; fresh results
  transfer into consumers. `impl` defines inherent methods and destruction hooks.
  Initialized owners are released in reverse scope order, including through `?`.
  There is no static move checking. Statement-only chain blocks yield unit.
- Files have private scopes and explicit exports. Imports expose only exported
  names, and never execute code. There are no runtime global variables.
- Entry points are ordinary exported functions. `main` is only the default
  name; host entries take unit and return unit, `int`, or a Result with either success type.

[examples/eg009_imports.resin](examples/eg009_imports.resin) and
[its counter module](examples/lib/counter.resin) demonstrate modules and
explicitly passed mutable state. Try its other entry point:

```sh
cargo run -- examples/eg009_imports.resin:independent
```

For a host-only standard-library example, read
[examples/input.resin](examples/input.resin) alongside
[stdlib/console.resin](stdlib/console.resin). The module builds a growing line
buffer on top of C's `getchar`, returns typed errors, and wraps successful lines
in shared owners. Scope cleanup releases them automatically.

## 2. Follow the host compilation path

The main path is short enough to keep in mind:

```text
source files -> Tree-sitter -> AST -> typed stack IR
                                            |
                                            v
                                 requested shaders' GLSL
                                            |
                                          glslc
                                            |
                                          SPIR-V
                                            |
                              host C with embedded SPIR-V
                                            |
                                       C compiler
                                            |
                                       executable
```

The checked IR supplies both the shader functions and the host code. Generated SPIR-V is
embedded in C before building the executable, which uses the runtime to create Vulkan
pipelines and run them. Host-only programs follow the same recipe with an empty shader list.

### The CLI connects the stages

[src/bin/resin.rs](src/bin/resin.rs) only calls `cli::main`.
[cli/mod.rs](src/cli/mod.rs) dispatches modes, and [args.rs](src/cli/args.rs)
parses flags and chooses `Mode::Interpreter`, `Compiler`, or `Formatter`;
[source.rs](src/cli/source.rs) parses the `FILE[:ENTRY]` selector.
Interpreter mode builds a debug native executable and runs it. Compiler mode builds
an optimized executable and copies it to the destination selected with `-o`.

[cli/environment.rs](src/cli/environment.rs) captures the environment, working directory,
and executable/temp paths once. CLI arguments override `CC` and `GLSLC`, which override
platform defaults. It resolves compiler paths, `RESIN_STDLIB`, runtime headers/archive,
and cache settings before compilation. The CLI also chooses the explicit `CProfile`.
Tool discovery errors are reported only if that tool is needed, so host-only programs
remain independent of `glslc`. Compiler subprocesses and cache fingerprints use the
same captured environment.

[compiler::Request::new](src/compiler.rs) validates the input/output combination and
resolves native directory destinations, rejecting outputs that would overwrite the
source. `Session::compile(&request)` analyzes through the caller's session, then passes
verified IR to [backend/build.rs](src/backend/build.rs). Every compilation follows the
same recipe: generate GLSL for all requested shaders, compile it to SPIR-V, embed the bytes
in generated C, then compile and link the executable. `Session::compile` returns an
`Executable` that keeps the build-cache lock while the caller runs it. Execution remains
a separate step. Generated C, GLSL, and SPIR-V remain in the build cache for inspection.

The session owns source overlays, cached parses, import dependencies, and
immutable [compilation snapshots](src/compiler/snapshot.rs). A snapshot retains
the AST, declaration contexts, type facts, diagnostics, and verified IR when
compilation succeeds. The CLI uses one session for its invocation; the language
server retains one across edits.

The compiler stages are library modules exposed by [src/lib.rs](src/lib.rs),
so tests and editor adapters can inspect the AST and verified IR through
`Session::analyze` without building an executable.

Formatting takes a separate path from `main` through
[format.rs](src/cli/format.rs) to the shared
[formatting.rs](src/formatting.rs) library module. `--format` (or `-f`) formats
files in place and searches directories recursively for `.resin` files;
`--format --check` reports differences without writing and exits with status 1
on differences or file/syntax errors. In the development environment, try:

```sh
cargo run -- --format --check examples
```

The formatter uses Tree-sitter syntax, changes only whitespace outside comments
and literals, and indents with hard tabs. It rejects invalid syntax without
modifying the file and needs no semantic analysis or entry point.

### The grammar and AST describe source, not execution

[grammar.js](tree-sitter-resin/grammar.js) is the source of truth for concrete
syntax. Skip the generated `src/parser.c` on a first read.

[src/ast/mod.rs](src/ast/mod.rs) defines the small set of source constructs:
files, statements, terms, and types, annotated with byte spans for diagnostics.
[generate.rs](src/ast/generate.rs) translates Tree-sitter nodes into these
structures, decodes literals, inserts the unit branch for one-armed `if`, and lowers operators into builtin
applications. Malformed expressions become holes alongside syntax diagnostics;
strict parsing checks those diagnostics before returning the same AST.

[load.rs](src/ast/load.rs) builds a `Program` of source modules in dependency
order, with the entry file last. It resolves relative imports and `std/` paths,
deduplicates canonical paths, and reports import cycles. The shared loading
traversal retains accessible modules and failed dependencies after errors.
Loading files and deciding which names are visible are separate jobs: visibility is handled later
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
storage is explicit, with separate address, load, and store instructions. Local zero
is always the function parameter, including unit and tuple parameters and foreign
declarations; there is no configurable parameter index.
The C and GLSL backends translate this stack model into target-language
variables and control flow; they do not execute the IR.

### Generation composes typing and emission

[ir/generate/plan/expressions.rs](src/ir/generate/plan/expressions.rs) walks each
expression once. Its `Expression` builder plans children through `child` and
type annotations through `annotation`, automatically registering dependencies
on both. The builder passes their type handles to the injected
[inference services](src/ir/typecheck/infer/) and selects an emission operation.
Each equation has an explicit `Rule` owner, which identifies the inference
variables invalidated if that operation fails. Adding an expression composes
these methods; the typing layer does not traverse syntax or own lexical scopes.

[ir/typecheck/](src/ir/typecheck/) contains the shared operation rules.
[TyperContext](src/ir/typecheck/mod.rs) owns type definitions and method namespaces;
[builtin.rs](src/ir/typecheck/builtin.rs) classifies builtin names and arities,
[rules.rs](src/ir/typecheck/rules.rs) checks their signatures, and
[convert.rs](src/ir/typecheck/convert.rs) classifies explicit conversions.
Inference, emission, and the independent IR verifier use these same rules.
[literal.rs](src/ir/literal.rs) shares numeric suffixes and literal classification.
[annotation.rs](src/ir/generate/annotation.rs) evaluates type syntax with an
injected name resolver. Its decoder returns the type and its explicit hole
handles together, publishing neither after a failed decode. Function result
inference owns those holes, so a failed body preserves concrete annotations.

[plan/mod.rs](src/ir/generate/plan/mod.rs) resolves function dependency groups
before executing their emission operations. Errors retain valid declarations
and independent expression facts. Failed relations are removed and surviving
constraints are retried from a solver checkpoint, without repeating the source
traversal. The private
[solver](src/ir/typecheck/infer/solver.rs) delays numeric choices and accumulates
error sets to a fixed point, including mutually recursive functions. Final IR
contains only concrete types. There is no AST-node-keyed type table or second
expression AST walk for instruction generation.

In [scope.rs](src/ir/generate/scope.rs), `Scopes` constructs declarations and
parent links and owns pending type facts until they resolve. Its retained
`ContextView` provides lookup at a captured declaration prefix. Lowering uses
an `Environment` that maps declaration IDs to storage; it cannot declare names
or rebuild scopes. Planned terms and statements carry their context cursors,
so an earlier expression cannot see later declarations. Inherent methods keep
canonical declaration IDs in their receiver namespace; they do not become
ordinary lexical bindings. Instance and associated calls use the same method
signatures, with explicit receiver conversions during lowering.

Local structs reserve their nominal identity while planning. Ownership cleanup
tracks initialized locals and destroys them in reverse scope order at each exit,
preserving returned values first. Layout queries plan their operands for typing
but never execute their runtime operations.
The [builder](src/ir/generate/builder.rs) assembles concrete stack IR, which the
independent [verifier](src/ir/verify/) checks before any backend consumes it.

The neighboring files separate the questions asked during that process:

| Question | Start reading here |
| --- | --- |
| Which imported or exported declaration does a name mean? | [modules.rs](src/ir/generate/modules.rs) |
| Which declarations are visible, and where are their types and origins? | [scope.rs](src/ir/generate/scope.rs), [semantic.rs](src/ir/generate/semantic.rs) |
| Where is a binding stored, and is it initialized? | [bindings.rs](src/ir/generate/bindings.rs), [builder.rs](src/ir/generate/builder.rs) |
| How are typing and emission composed for an expression? | [plan/expressions.rs](src/ir/generate/plan/expressions.rs), [terms.rs](src/ir/generate/terms.rs) |
| Which storage location does an assignment or address refer to? | [places.rs](src/ir/generate/places.rs) |
| How do branches, loops, and short-circuit operators join? | [flow.rs](src/ir/generate/flow.rs) |
| How do Results, exhaustive matches, and early error returns lower? | [sums.rs](src/ir/generate/sums.rs) |
| How are inherent and builtin methods declared and called? | [methods.rs](src/ir/generate/methods.rs), [builtin_methods.rs](src/ir/generate/builtin_methods.rs), [typecheck/methods.rs](src/ir/typecheck/methods.rs) |
| How are owned locals destroyed at scope exits? | [cleanup.rs](src/ir/generate/cleanup.rs) |
| How are blocks, locals, and instructions assembled? | [builder.rs](src/ir/generate/builder.rs) |

The distinction between a value and a place is worth following through one
pointer example: reading `x`, taking `&x`, and assigning to `x` share a name but
require different instructions.

### Verification is an independent boundary

[ir/verify/](src/ir/verify/) checks instruction operands, block-edge stack
types, returns, and type definitions. Its `VerifiedModule` owns the checked
module, function typing results, and canonical type table behind private fields. Snapshots retain
this product, and native builds borrow a `Verified` view of it. Public backend
entry points accepting arbitrary IR use `with_verified` to validate their input
and borrow the resulting analysis for emission. Consuming `into_module` returns
ordinary IR and explicitly drops its verification proof.

[verify/flow.rs](src/ir/verify/flow.rs) propagates stack types through existing
blocks and checks that incoming edges agree. [generate/flow.rs](src/ir/generate/flow.rs)
creates those blocks and branches from source expressions.

[ir/types/definitions.rs](src/ir/types/definitions.rs) holds definition checks
shared by the type checker and verifier, including invalid references and recursive
inline layouts. This is why those checks live beside the type representation,
not exclusively inside the source-language type checker.

### C emission and native builds are separate

[backend/c/mod.rs](src/backend/c/mod.rs) assembles a C translation unit:
types, declarations, function bodies, embedded shaders, and a C `main` wrapper
calling the selected Resin entry. [function.rs](src/backend/c/function.rs)
lowers instructions and block edges; [foreign.rs](src/backend/c/foreign.rs)
bridges Resin's unary calls to conventional C argument lists.

[toolchain/c.rs](src/toolchain/c.rs) invokes the C compiler and statically links
the runtime. [platform.rs](src/toolchain/platform.rs) selects the default
compiler, archive name, flags, and system libraries: `cc` and
`libresin_runtime.a` on Unix; GNU-style LLVM `clang` and `resin_runtime.lib`
on Windows MSVC. Windows builds must keep Rust, GLFW, and emitted C on the
same C runtime. Host-only programs need neither a Vulkan SDK nor a GPU.

The C toolchain also owns the native build cache and locks that keep concurrent
builds and runs from interfering. [dependencies.rs](src/toolchain/dependencies.rs)
reads C compiler dependency files to track included headers.

There are two artifact directories with different owners: Cargo builds the
compiler and runtime under `target/`; Resin builds user programs under `build/`
in the caller's working directory. Default runs reuse an unoptimized native
build and execute it. Requesting an executable with `-o` selects the optimized cache
and copies the output without running it. This does not change Cargo's Rust profile.

## 3. Follow an editor change through the compiler

Start again at [compiler::Session](src/compiler.rs). An editor supplies unsaved
text through `set_overlay`, removes it with `remove_overlay`, and reports disk
changes with `file_changed`. Overlays take precedence over disk. Changes
invalidate dependent entries; retained snapshots remain valid for their readers.

[compiler/source.rs](src/compiler/source.rs) handles source lookup and path
normalization; [compiler/syntax.rs](src/compiler/syntax.rs) caches ASTs and
incrementally reparses Tree-sitter trees. [snapshot.rs](src/compiler/snapshot.rs)
builds the common result for compilation and editor queries. Parsing is
incremental per file; semantic checking reruns an affected entry's import
closure. This is separate from the native artifact cache.

Incomplete code goes through the same compiler traversal. Expression, type,
and missing-field holes preserve useful children; bounded delimiter repair
recovers unfinished scopes without clearing the original syntax diagnostics.
Invalid declarations still shadow outer names, and healthy siblings retain
their types. Unknown types display as `?`; errors prevent executable generation.

[analysis.rs](src/analysis.rs) implements editor queries on the compiler's
snapshot, also exposed under the compatibility name `Analysis`. Queries select
the source context view and look up declarations on demand. Member observations
retain available fields, signatures, and canonical method origins even when later code fails.
Hover and member completion use these facts and shared formatting from
[generate/semantic.rs](src/ir/generate/semantic.rs). There is no separate
recovery compiler or fallback declaration index.

The [language server](resin-lsp/README.md) adapts that compiler state to the
Language Server Protocol over stdio. [server.rs](resin-lsp/src/server.rs)
handles requests, document versions, and file notifications;
[text.rs](resin-lsp/src/text.rs) converts byte offsets to UTF-16 positions.
[worker.rs](resin-lsp/src/worker.rs) runs the session in
the background, coalesces edits, and discards obsolete results. Editor analysis
never compiles C/GLSL, initializes a GPU, or executes Resin programs.

`textDocument/formatting` uses the same [formatter](src/formatting.rs) as the
CLI. The server formats the open document's current text and returns a text edit
for the changed region. See the [formatting rules](resin-lsp/README.md#formatting)
for layout conventions.

Finally, [zed-resin/src/lib.rs](zed-resin/src/lib.rs) locates and launches the
native server from Zed's WASI extension. Its [language queries](zed-resin/languages/resin/)
provide highlighting, outlines, and other syntax features. See the
[extension README](zed-resin/README.md) for installation and configuration;
the extension builds separately from the main Cargo workspace.

## 4. Follow a shader into the runtime

Read [examples/gradient.resin](examples/gradient.resin) for compute, then
[examples/triangle.resin](examples/triangle.resin) for graphics. Each example keeps
its decorated shader entries and ordinary helpers alongside its host code. To inspect
the generated shaders without running a Vulkan program:

```sh
cargo run -- examples/gradient.resin -o dist/
```

Read `build/shaders/<hash>/shader.glsl` and `shader.spv` after the build.

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register and validate
shader entry declarations. [ir/shader.rs](src/ir/shader.rs) defines their metadata
and signature contracts. Decorated functions and their unannotated helpers remain
host-callable. Accessing `function.spirv` requests a static `Span<ubyte>` artifact;
[backend/shaders.rs](src/backend/shaders.rs) enumerates those declaration
requests and emits GLSL. [toolchain/shaders.rs](src/toolchain/shaders.rs) caches the
external compiler output, supplying SPIR-V for embedding in C.
No runtime function-value analysis is involved. The runtime receives bytes, not a
host function pointer or source-file path.

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
to images and windows. Wrappers omit the native `resin_` prefix and return
`Result<T, RuntimeError>`, with created handles in the success value.
[stdlib/status.resin](stdlib/status.resin) translates integer status codes into
named error structs; the C ABI remains unchanged.

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

Buffers make the host/device boundary concrete. `Span<T>` pairs an address with
a length; both spans and arrays return a checked element pointer through `buffer(index)`.
Use `buffer(index).*` to read or write it. Raw pointer arithmetic requires an explicit
conversion to `ulong` and operates on byte addresses. The host allocates memory,
writes root data, and passes its device address when dispatching or drawing.
Shader entry wrappers interpret that root according to their supported
interface. [backend/layout.rs](src/backend/layout.rs) keeps supported buffer
layouts consistent between C and GLSL; start there when investigating a field
offset or alignment mismatch.

These APIs expose resource lifetimes explicitly. The C API and its unsafe Rust
convenience API are not ownership-safe GPU abstractions: resources must remain
alive while commands use them. Standard-library wrappers retain shared owners and
propagate failures with `?`. `commands.submit()` and cancellation clear the
shared native handle; automatic destruction cancels unfinished recordings.
Raw shader root addresses do not retain their backing allocations.
Shader bodies describe individual invocations;
the compiler does not synthesize workgroup-local storage or barriers.

Finally, [examples/particles.resin](examples/particles.resin) combines compute
and graphics over a shared buffer, and [examples/window.resin](examples/window.resin)
adds presentation. Builds target 64-bit Linux, macOS, and Windows, but actual
GPU runs also need the Vulkan features checked in `gpu/device.rs`. On macOS,
portability enumeration and MoltenVK discovery do not guarantee those features
or presentation support. Window runs need a display; the headless image demos
write PNGs in cwd.

## 5. Find the test closest to your change

Tests are executable descriptions of the boundaries above:

| Change | Useful tests |
| --- | --- |
| Syntax or AST shape | [parser corpus](tree-sitter-resin/test/corpus/), [mutation_ast.rs](tests/mutation_ast.rs) |
| Grammar JavaScript types, lint, or formatting | Run `npm run check` in [tree-sitter-resin/](tree-sitter-resin/README.md) |
| Imports, exports, or entry visibility | [modules.rs](tests/modules.rs), [cli.rs](tests/cli.rs) |
| Typing, conversions, or IR invariants | [nominal_types.rs](tests/nominal_types.rs), [typing-rule tests](src/ir/typecheck/tests.rs), [verifier tests](src/ir/verify/tests.rs) |
| Explicit type holes and return inference | [inference.rs](tests/inference.rs), [inference example](examples/inference.resin) |
| Structs, aliases, unions, and typed errors | [results.rs](tests/results.rs), [errors example](examples/errors.resin) |
| Automatic destruction, scope exits, and copying | [shared.rs](tests/shared.rs), [ownership example](examples/ownership.resin), [C execution tests](tests/c_backend.rs) |
| Host code generation or C interop | [c_backend.rs](tests/c_backend.rs), [foreign.rs](tests/foreign.rs), [printing.rs](tests/printing.rs) |
| Standard-library Results and native failure cleanup | [stdlib.rs](tests/stdlib.rs) (no GPU or windows required) |
| Console input, byte handling, and allocation failures | [console.rs](tests/console.rs), [input example](examples/input.resin) |
| Compilation and artifact reuse | [build_cache.rs](tests/build_cache.rs), [cli.rs](tests/cli.rs) |
| Source formatting, file traversal, or format checks | [formatting.rs](tests/formatting.rs), [format_cli.rs](tests/format_cli.rs), [LSP formatting tests](resin-lsp/tests/stdio.rs) |
| Session invalidation, editor queries, or recovery | [session tests](src/compiler.rs), [analysis.rs](tests/analysis.rs) |
| LSP protocol, buffer versions, or watched files | [stdio.rs](resin-lsp/tests/stdio.rs) |
| Zed syntax features | [zed_queries.rs](tests/zed_queries.rs) |
| Shader generation or execution | [glsl_backend.rs](tests/glsl_backend.rs), [gpu_backend.rs](tests/gpu_backend.rs), [window_backend.rs](tests/window_backend.rs) |

[resin-runtime/tests/](resin-runtime/tests/) also exercises the native runtime
with GLSL fixtures, independently of the Resin compiler. This is useful for
separating a Vulkan runtime bug from a language or code-generation bug.

For a focused host-only starting point, in your development environment:

```sh
cargo test -p resin --test modules --test c_backend
```

For editor work, without launching an editor or opening windows:

```sh
cargo test -p resin --lib --test analysis --test zed_queries
cargo test -p resin --test formatting --test format_cli
cargo test -p resin-lsp
```

[CI](.github/workflows/build.yml) first checks the grammar's JavaScript types,
lint, and formatting. The native jobs depend on these checks passing, then build
the workspace, run a host example, check example formatting with
`--format --check examples`, test, and lint on Linux, macOS, and Windows.
They explicitly exclude
window-opening tests; optional GPU checks can skip when facilities are absent.
Passing this matrix establishes native build and host coverage, not full GPU
compatibility.

For the full GPU path, in a POSIX shell with a compatible Vulkan device and a
desktop display or Xvfb available (for Xvfb, set `DISPLAY` to its display and
`XDG_SESSION_TYPE=x11` so GLFW does not select a Wayland compositor):

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
