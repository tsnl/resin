# A tour of Resin

Resin is a systems language for host CPUs and Vulkan GPUs. This is a reading
path through its implementation, not a language reference; keep the
[README](README.md) nearby for syntax and command-line options.

The root is a virtual Cargo workspace. [crates/](crates/) contains the compiler
driver in [resin/](crates/resin/), seven compiler phase crates, and the supporting
[runtime](crates/resin-runtime/), [parser](crates/tree-sitter-resin/), and
[language server](crates/resin-lsp/). All are unpublished. The
[Zed extension](editors/zed/) has a separate Cargo workspace under `editors/`.
The [standard library](stdlib/) is written in Resin and wraps the runtime's C API.

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
- `A | B` is a structural union of value types. `Result<T, E>` is first-class;
  `ok` and `err` construct it, `match` handles variants, and postfix `?` propagates errors.
- `Arc<T>` and `Weak<T>` provide shared ownership. Value reads copy; fresh results
  transfer into consumers. `impl` defines inherent methods and destruction hooks.
  Initialized owners are released in reverse scope order, including through `?`.
  There is no static move checking. Statement-only chain blocks yield unit.
- Files have private scopes and explicit exports. Imports expose only exported
  names, and never execute code. There are no runtime global variables.
- Entry points are ordinary exported functions. `main` is only the default
  name; host entries take unit or the argc/argv/envp tuple, and return unit, `int`,
  or a Result with either success type.

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
source files -> CST -> AST -> HIR -> LIR -> verified LIR
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

[crates/resin/src/bin/resin.rs](crates/resin/src/bin/resin.rs) only calls `cli::main`.
[cli/mod.rs](crates/resin/src/cli/mod.rs) dispatches modes, and [args.rs](crates/resin/src/cli/args.rs)
parses flags and chooses `Mode::Interpreter`, `Compiler`, or `Formatter`;
[source.rs](crates/resin/src/cli/source.rs) parses the `FILE[:ENTRY]` selector.
Interpreter mode builds a debug native executable and runs it. Compiler mode builds
an optimized executable and copies it to the destination selected with `-o`.

[cli/environment.rs](crates/resin/src/cli/environment.rs) captures the environment, working directory,
and executable/temp paths once. CLI arguments override `CC` and `GLSLC`, which override
platform defaults. It resolves compiler paths, `RESIN_STDLIB`, runtime headers/archive,
and cache settings before compilation. The CLI also chooses the explicit `CProfile`.
Tool discovery errors are reported only if that tool is needed, so host-only programs
remain independent of `glslc`. Compiler subprocesses and cache fingerprints use the
same captured environment.

[compiler::Request::new](crates/resin/src/compiler.rs) validates the input/output combination and
resolves native directory destinations, rejecting outputs that would overwrite the
source. `Session::compile(&request)` analyzes through the caller's session, then passes
verified LIR to [compiler/build.rs](crates/resin/src/compiler/build.rs). Every compilation follows the
same recipe: generate GLSL for all requested shaders, compile it to SPIR-V, embed the bytes
in generated C, then compile and link the executable. `Session::compile` returns an
`Executable` that keeps the build-cache lock while the caller runs it. Execution remains
a separate step. Generated C, GLSL, and SPIR-V remain in the build cache for inspection.

The session owns source overlays, cached parses, import dependencies, and
immutable [compilation snapshots](crates/resin/src/compiler/snapshot.rs). A snapshot retains
the AST, resolved HIR, opaque editor analysis, diagnostics, and verified LIR when
compilation succeeds. The CLI uses one session for its invocation; the language
server retains one across edits.

The compiler stages are separate crates re-exported by [crates/resin/src/lib.rs](crates/resin/src/lib.rs),
so tests and editor adapters can inspect the AST, HIR, and verified LIR through
`Session::analyze` without building an executable.

Formatting takes a separate path from `main` through
[format.rs](crates/resin/src/cli/format.rs) to the shared
[CST formatter](crates/resin-cst/src/print.rs) library module. `--format` (or `-f`) formats
files in place and searches directories recursively for `.resin` files;
`--format --check` reports differences without writing and exits with status 1
on differences or file/syntax errors. In the development environment, try:

```sh
cargo run -- --format --check examples
```

The formatter uses Tree-sitter syntax, changes only whitespace outside comments
and literals, and indents with hard tabs. It rejects invalid syntax without
modifying the file and needs no semantic analysis or entry point.

### Read each language, then its incoming pass

The [architecture guide](doc/architecture.md) gives the full crate graph and pass
contracts. Every phase follows the same path: `language.rs` describes its data,
`lower` produces it from the preceding language, and `print` renders it.

[grammar.js](crates/tree-sitter-resin/grammar.js) defines concrete syntax. The
[CST document](crates/resin-cst/src/language.rs) pairs a Tree-sitter tree with source
text; [CST lowering](crates/resin-cst/src/lower.rs) reparses it incrementally. Skip the
generated `src/parser.c` inside the grammar package on a first read.

[AST language](crates/resin-ast/src/language.rs) defines source files, declarations,
terms, and type syntax with byte spans. [AST lowering](crates/resin-ast/src/lower.rs)
translates CST nodes, decodes strings, inserts the unit branch of one-armed `if`,
and represents operators as builtin applications. It also preserves incomplete
expressions as holes for editor recovery. [Module loading](crates/resin-ast/src/load.rs)
builds a `Program` in dependency order and resolves relative and `std/` imports.

[HIR language](crates/resin-hir/src/language.rs) is a self-contained, typed tree. Start
at [HIR lowering](crates/resin-hir/src/lower/mod.rs), then follow
[checking a file](crates/resin-hir/src/lower/check/file.rs): declare signatures, check
bodies, solve dependency groups, and replace inference variables with concrete types.
The [solver](crates/resin-hir/src/lower/infer/solver.rs) handles numeric constraints
and recursive error sets. It stays private to this crate.

[Elaboration](crates/resin-hir/src/lower/elaborate.rs) resolves lexical bindings,
method calls, field projections, conversions, and shader references. Short-circuit
operators become conditionals and layout queries become constants. The public HIR
contains neither AST nodes nor scope cursors. The temporary checking tree in
[typed.rs](crates/resin-hir/src/lower/typed.rs) is an internal construction step.

[LIR language](crates/resin-lir/src/language.rs) defines a typed operand stack machine.
Functions own locals and basic blocks; instructions consume and produce stack
values, and terminators connect blocks. Local zero is always the parameter,
including unit, tuples, and foreign declarations. [LIR lowering](crates/resin-lir/src/lower/mod.rs)
consumes only HIR and shared concrete types. It chooses storage, checks definite
initialization, makes evaluation order explicit, and inserts cleanup.

| Question | Start reading here |
| --- | --- |
| Which imported declaration does a name mean? | [HIR modules](crates/resin-hir/src/lower/modules.rs) |
| Which names are visible at this source position? | [HIR scopes](crates/resin-hir/src/lower/scope.rs) |
| How are a function's constraints solved? | [HIR checking](crates/resin-hir/src/lower/check/file.rs) |
| How does a method become an ordinary call? | [HIR elaboration](crates/resin-hir/src/lower/elaborate.rs) |
| Where is a binding stored, and is it initialized? | [LIR bindings](crates/resin-lir/src/lower/bindings.rs) |
| Which storage location does an assignment address? | [LIR places](crates/resin-lir/src/lower/places.rs) |
| How do branches and loops join? | [LIR control flow](crates/resin-lir/src/lower/flow.rs) |
| How are Result propagation and cleanup lowered? | [LIR sums](crates/resin-lir/src/lower/sums.rs), [cleanup](crates/resin-lir/src/lower/cleanup.rs) |
| How are concrete instructions assembled? | [LIR builder](crates/resin-lir/src/lower/builder.rs) |

### Verification is a separate crate

[resin-lir-verifier](crates/resin-lir-verifier/src/lib.rs) checks definitions, instruction
operands, block-edge stack types, and returns. LIR has no dependency on this crate.
Its `VerifiedModule` owns LIR and the analysis that certifies it behind private
fields. Native builds borrow an immutable `Verified` view. Consuming `into_module`
returns mutable LIR and discards the certificate; changes require verification again.

The [concrete type rules](crates/resin-common/src/types/check/mod.rs) and
[layout checks](crates/resin-common/src/types/definitions.rs) live in `resin-common`.
They depend on no compiler phase. HIR adds inference and method namespaces privately;
the verifier applies concrete rules to instructions independently of source checking.

### C emission and native builds are separate

[C lowering](crates/resin-codegen/src/c/lower/mod.rs) produces a
[C source tree](crates/resin-codegen/src/c/language.rs): declarations, embedded shaders,
functions, blocks, and a `main` wrapper. [The printer](crates/resin-codegen/src/c/print.rs)
formats that tree without accessing LIR or typechecking facts. [function.rs](crates/resin-codegen/src/c/lower/function.rs)
lowers instructions and block edges; [foreign.rs](crates/resin-codegen/src/c/lower/foreign.rs)
bridges Resin's unary calls to conventional C argument lists.

[toolchain/c.rs](crates/resin/src/toolchain/c.rs) invokes the C compiler and statically links
the runtime. [platform.rs](crates/resin/src/toolchain/platform.rs) selects the default
compiler, archive name, flags, and system libraries: `cc` and
`libresin_runtime.a` on Unix; GNU-style LLVM `clang` and `resin_runtime.lib`
on Windows MSVC. Windows builds must keep Rust, GLFW, and emitted C on the
same C runtime. Host-only programs need neither a Vulkan SDK nor a GPU.

The C toolchain also owns the native build cache and locks that keep concurrent
builds and runs from interfering. [dependencies.rs](crates/resin/src/toolchain/dependencies.rs)
reads C compiler dependency files to track included headers.

There are two artifact directories with different owners: Cargo builds the
compiler and runtime under `target/`; Resin builds user programs under `build/`
in the caller's working directory. Default runs reuse an unoptimized native
build and execute it. Requesting an executable with `-o` selects the optimized cache
and copies the output without running it. This does not change Cargo's Rust profile.

## 3. Follow an editor change through the compiler

Start again at [compiler::Session](crates/resin/src/compiler.rs). An editor supplies unsaved
text through `set_overlay`, removes it with `remove_overlay`, and reports disk
changes with `file_changed`. Overlays take precedence over disk. Changes
invalidate dependent entries; retained snapshots remain valid for their readers.

[compiler/source.rs](crates/resin/src/compiler/source.rs) handles source lookup and path
normalization; [compiler/syntax.rs](crates/resin/src/compiler/syntax.rs) caches CST and AST
products together. The CST crate performs incremental reparsing. [snapshot.rs](crates/resin/src/compiler/snapshot.rs)
builds the common result for compilation and editor queries. Parsing is
incremental per file; semantic checking reruns an affected entry's import
closure. This is separate from the native artifact cache.

Incomplete code goes through the same compiler traversal. Expression, type,
and missing-field holes preserve useful children; bounded delimiter repair
recovers unfinished scopes without clearing the original syntax diagnostics.
Invalid declarations still shadow outer names, and healthy siblings retain
their types. Unknown types display as `?`; errors prevent executable generation.

[analysis.rs](crates/resin/src/analysis.rs) adapts compiler snapshots to the opaque
[HIR analysis API](crates/resin-hir/src/analysis.rs). Queries select
the source context view and look up declarations on demand. Member observations
retain available fields, signatures, and canonical method origins even when later code fails.
Hover and member completion use these facts and
[shared type formatting](crates/resin-common/src/types/print.rs). There is no separate
recovery compiler or fallback declaration index.

The [language server](crates/resin-lsp/README.md) adapts that compiler state to the
Language Server Protocol over stdio. [server.rs](crates/resin-lsp/src/server.rs)
handles requests, document versions, and file notifications;
[text.rs](crates/resin-lsp/src/text.rs) converts byte offsets to UTF-16 positions.
[worker.rs](crates/resin-lsp/src/worker.rs) runs the session in
the background, coalesces edits, and discards obsolete results. Editor analysis
never compiles C/GLSL, initializes a GPU, or executes Resin programs.

`textDocument/formatting` uses the same [formatter](crates/resin-cst/src/print.rs) as the
CLI. The server formats the open document's current text and returns a text edit
for the changed region. See the [formatting rules](crates/resin-lsp/README.md#formatting)
for layout conventions.

Finally, [editors/zed/src/lib.rs](editors/zed/src/lib.rs) locates and launches the
native server from Zed's WASI extension. Its [language queries](editors/zed/languages/resin/)
provide highlighting, outlines, and other syntax features. See the
[extension README](editors/zed/README.md) for installation and configuration;
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
shader entry declarations. [shader interfaces](crates/resin-common/src/types/shader.rs) defines their metadata
and signature contracts. Decorated functions and their unannotated helpers remain
host-callable. Accessing `function.spirv` requests a static `Span<ubyte>` artifact;
[shader build orchestration](crates/resin/src/compiler/shaders.rs) enumerates those declaration
requests and emits GLSL. [toolchain/shaders.rs](crates/resin/src/toolchain/shaders.rs) caches the
external compiler output, supplying SPIR-V for embedding in C.
No runtime function-value analysis is involved. The runtime receives bytes, not a
host function pointer or source-file path.

[GLSL lowering](crates/resin-codegen/src/glsl/lower/mod.rs) collects reachable shader
functions. [entry.rs](crates/resin-codegen/src/glsl/lower/entry.rs) adapts regular Resin function
signatures to compute, vertex, or fragment interfaces, and
[function.rs](crates/resin-codegen/src/glsl/lower/function.rs) lowers their bodies. The GLSL
backend supports a subset of the host language; it rejects operations such as
foreign calls and recursion rather than making them work on the device.

The remaining GPU operations are ordinary standard-library calls. Follow one
from [stdlib/gpu.resin](stdlib/gpu.resin), through
[resin_runtime.h](crates/resin-runtime/include/resin_runtime.h) and its included
headers, to [the runtime](crates/resin-runtime/src/lib.rs). The same pattern applies
to images and windows. Wrappers omit the native `resin_` prefix and return
`Result<T, RuntimeError>`, with created handles in the success value.
[stdlib/status.resin](stdlib/status.resin) translates integer status codes into
named error structs; the C ABI remains unchanged.

Inside the runtime, the useful landmarks are:

- [gpu/device.rs](crates/resin-runtime/src/gpu/device.rs): Vulkan device discovery and
  required features.
- [gpu/mod.rs](crates/resin-runtime/src/gpu/mod.rs): allocations, images, command
  recording, synchronization between operations, and submission.
- [gpu/pipeline.rs](crates/resin-runtime/src/gpu/pipeline.rs): conventional Vulkan
  compute and graphics pipelines.
- [gpu/present.rs](crates/resin-runtime/src/gpu/present.rs) and
  [window/](crates/resin-runtime/src/window/): swapchains, presentation, and statically
  linked GLFW.
- [allocator/range.rs](crates/resin-runtime/src/allocator/range.rs): aligned
  suballocation using a sorted list of free ranges.

Buffers make the host/device boundary concrete. `Span<T>` pairs an address with
a length; both spans and arrays return a checked element pointer through `buffer(index)`.
Use `buffer(index).*` to read or write it. Raw pointer arithmetic requires an explicit
conversion to `ulong` and operates on byte addresses. The host allocates memory,
writes root data, and passes its device address when dispatching or drawing.
Shader entry wrappers interpret that root according to their supported
interface. [target layout helpers](crates/resin-codegen/src/layout.rs) keeps supported buffer
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
| Syntax or AST shape | [parser corpus](crates/tree-sitter-resin/test/corpus/), [mutation_ast.rs](crates/resin/tests/mutation_ast.rs) |
| Grammar JavaScript types, lint, or formatting | Run `npm run check` in [crates/tree-sitter-resin/](crates/tree-sitter-resin/README.md) |
| Imports, exports, or entry visibility | [modules.rs](crates/resin/tests/modules.rs), [cli.rs](crates/resin/tests/cli.rs) |
| Typing, conversions, or IR invariants | [nominal_types.rs](crates/resin/tests/nominal_types.rs), [typing-rule tests](crates/resin-common/src/types/check/tests.rs), [verifier tests](crates/resin-lir-verifier/src/tests.rs) |
| Explicit type holes and return inference | [inference.rs](crates/resin/tests/inference.rs), [inference example](examples/inference.resin) |
| Structs, aliases, unions, and typed errors | [results.rs](crates/resin/tests/results.rs), [errors example](examples/errors.resin) |
| Automatic destruction, scope exits, and copying | [shared.rs](crates/resin/tests/shared.rs), [ownership example](examples/ownership.resin), [C execution tests](crates/resin/tests/c_backend.rs) |
| Host code generation or C interop | [c_backend.rs](crates/resin/tests/c_backend.rs), [foreign.rs](crates/resin/tests/foreign.rs), [printing.rs](crates/resin/tests/printing.rs) |
| Standard-library Results and native failure cleanup | [stdlib.rs](crates/resin/tests/stdlib.rs) (no GPU or windows required) |
| Console input, byte handling, and allocation failures | [console.rs](crates/resin/tests/console.rs), [input example](examples/input.resin) |
| Compilation and artifact reuse | [build_cache.rs](crates/resin/tests/build_cache.rs), [cli.rs](crates/resin/tests/cli.rs) |
| Source formatting, file traversal, or format checks | [formatting.rs](crates/resin/tests/formatting.rs), [format_cli.rs](crates/resin/tests/format_cli.rs), [LSP formatting tests](crates/resin-lsp/tests/stdio.rs) |
| Session invalidation, editor queries, or recovery | [session tests](crates/resin/src/compiler.rs), [analysis.rs](crates/resin/tests/analysis.rs) |
| LSP protocol, buffer versions, or watched files | [stdio.rs](crates/resin-lsp/tests/stdio.rs) |
| Zed syntax features | [zed_queries.rs](crates/resin/tests/zed_queries.rs) |
| Shader generation or execution | [glsl_backend.rs](crates/resin/tests/glsl_backend.rs), [gpu_backend.rs](crates/resin/tests/gpu_backend.rs), [window_backend.rs](crates/resin/tests/window_backend.rs) |

[crates/resin-runtime/tests/](crates/resin-runtime/tests/) also exercises the native runtime
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
