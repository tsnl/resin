# A tour of Resin

Resin is a systems language for host CPUs and Vulkan GPUs. This is a reading
path through its implementation. Keep the [language reference](language.md) and
[tool guide](tools.md) nearby for syntax and command-line options.

The root is both a Cargo workspace and the `resin` CLI package. [src/](../src)
contains only the wrapper calling `resin_client::main()`. [crates/](../crates) contains the
[sources and loading](../crates/resin-source), [concrete types](../crates/resin-types),
[platform toolchain](../crates/resin-toolchain),
compiler phases, and the supporting [runtime](../crates/resin-runtime),
[parser](../crates/tree-sitter-resin), [HTTP compiler server](../crates/resin-server),
[wire protocol](../crates/resin-protocol), and [CLI/editor client](../crates/resin-client).
All are unpublished. The
[Zed extension](../editors/zed) has a separate Cargo workspace under `editors/`.
The [standard library](../resin) is written in Resin and wraps the runtime's C API.

## 1. Start with a program

Read [examples/eg001.resin](../examples/eg001.resin) after
[setting up the development environment](development.md#development). Nix users
can enter `nix-shell`; [.envrc](../.envrc) also loads it through direnv.
Start a [compiler service](compiler-service.md#local-development), set
`RESIN_SERVER` to its URL, then run these commands from the repository root:

```sh
cargo run -- examples/eg001.resin
cargo run -- examples/eg001.resin -o dist/
```
The first command builds and runs the Fibonacci program; the second builds an optimized
executable without running it. Inspect the generated `main.c` beside each cached executable
under the service working directory's `build/`. Downloads execute locally; there
is no bytecode interpreter behind the CLI.

A few language choices explain much of the implementation:

- Functions use `fn` and are top-level declarations with typed parameters;
  an omitted result type means unit. Their names are available before their
  bodies are checked, allowing mutual recursion. Explicit `_` holes opt into
  inference in local annotations and function results; omission still means unit.
- Function signatures retain separate parameters. An empty argument list supplies
  no values; a tuple is an explicit single argument.
- Value bindings use `let`, with `mut` to permit reassignment; nominal
  records use `struct`, and `type` creates transparent aliases. Record fields remain
  `name = value`; parameters remain `name: Type`.
- `A | B` is a structural union of value types. `T | Err<E>` represents a
  successful `T` or an error wrapper constructed with `Err(value)`; postfix `?`
  propagates errors. `T | None` represents optional values, handled with `match` or `!`.
- Named structs move; primitives and aggregates containing only copyable values
  copy. HIR checks moves and definite initialization. Structs contain fields only;
  ordinary free functions supply operations, called directly or through UFCS colon
  syntax. A free `drop(value: RefMut<T>)` function supplies a destruction hook.
- `ArcPtr<T>` shares single values, and `ArcSpan<T>` shares fixed-length sequences.
  Weak owners observe them without keeping their payloads alive.
  `arc_span_alloc(count, initial)?` allocates initialized owned elements;
  `owner:get()` borrows a span and `owner:clone()` explicitly retains ownership.
  Cleanup releases initialized owners in reverse scope order, including through `?`.
- Files have private scopes and explicit exports. Imports expose only exported
  names, and never execute code. There are no runtime global variables.
- Entry points are ordinary exported functions. `main` is only the default
  name; host entries take no arguments or the argc, argv, and envp parameters, and return unit, `i32`,
  or either success type in a union with `Err<E>`.

[examples/eg009_imports.resin](../examples/eg009_imports.resin) and
[its counter module](../examples/lib/counter.resin) demonstrate modules and
explicitly passed mutable state. Try its other entry point:

```sh
cargo run -- examples/eg009_imports.resin:independent
```

For a host-only standard-library example, read
[examples/input.resin](../examples/input.resin) alongside
[resin/stdio.resin](../resin/stdio.resin). The module builds a growing line
buffer on top of C's `getchar`, returns typed errors, and wraps successful lines
in shared owners. Scope cleanup releases them automatically.

## 2. Follow the host compilation path

The main path is short enough to keep in mind:

```text
source files -> CST -> AST -> HIR -> LIR -> verified LIR
                                            |
                                            v
                                 requested shaders' SPIR-V
                                            |
                                        spirv-opt
                                            |
                                    optimized SPIR-V
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

### Client capture and explicit server passes

[src/main.rs](../src/main.rs) calls `resin_client::main`. The client's
[mode dispatch](../crates/resin-client/src/cli/mod.rs),
[arguments](../crates/resin-client/src/cli/args.rs), and
[source selector](../crates/resin-client/src/cli/source.rs) turn CLI inputs into local
requests. [interp.rs](../crates/resin-client/src/interp.rs) requires an explicit service
URL, negotiates capabilities, captures inputs, and downloads verified executables.
Without `-o` it requests Debug and executes a temporary download locally; with `-o`
it requests Release and atomically publishes the chosen destination. Arguments,
working directory, and environment remain on the client.

The client [input pass](../crates/resin-client/src/inputs.rs) parses full CST documents,
queries preambles, and follows local imports through an async `Loader`. It freezes
one version per physical identity and uploads entry-parent-relative names plus
explicit edges. [Header capture](../crates/resin-client/src/headers.rs) snapshots complete
selected directories and source-scoped extern bindings. `$/` libraries and pinned
packages belong to the server's startup snapshot; clients do not upload replacements.

The server's [public state](../crates/resin-server/src/lib.rs) directly owns its configuration,
managed inputs, execution bound, and cache heads. [HTTP](../crates/resin-server/src/http.rs)
handles admission, cancellation, and response ownership. The private
[analysis](../crates/resin-server/src/analyze.rs) and [build](../crates/resin-server/src/build.rs)
handlers each sequence compiler passes explicitly. They convert strict
[wire data](../crates/resin-protocol/src/lib.rs) to `SourceGraph` and completed AST inputs,
then call `Hir::build(BuiltProgram, execution, cancellation)`. No phase loads files or
accepts a previous HIR. Shared immutable heads reuse equal source versions and graphs
across independent editor and CLI requests.

Build handlers additionally select entries, lower and verify LIR, generate an owned
C/SPIR-V/Ninja project with complete native header bindings, and invoke the toolchain.
Its [Environment](../crates/resin-toolchain/src/lib.rs) captures native settings once;
[resolution](../crates/resin-toolchain/src/environment.rs) applies server flags before
`CC`/`SPIRV_OPT` and platform defaults. Ninja optimizes and embeds shaders using the
captured service executable, then compiles preprocessed `.i` bytes. A staging lock
ends at build completion. Immutable artifacts retain independent generation directories
through downloads, even while later builds update caches. The final owner cleans up.

Each phase remains directly usable without HTTP; see
[Calling the passes](architecture.md#calling-the-passes) for a complete example.
The client depends on source/CST/executor libraries, with no AST/HIR/backend/native
compiler dependency. Tests can exercise phases directly or the full client/service pair.

Formatting follows [format.rs](../crates/resin-client/src/cli/format.rs) to the shared
[CST formatter](../crates/resin-cst/src/print.rs). `--format` edits files in place;
`--format --check` reports differences and exits with status 1 on differences or
errors. It needs no `RESIN_SERVER`, imports, semantic analysis, or entry point:

```sh
cargo run -- --format --check examples
```

The formatter preserves comments and strings, preserves numeric spelling and
whitespace, indents with hard tabs, and refuses to modify invalid syntax.

### Read each language, then its incoming pass

The [architecture guide](architecture.md) gives the full crate graph and pass
contracts. Every phase starts at `lib.rs`, which defines its language and the
operations clients can call. Private `lower` modules produce that language from
the preceding phase, and private `print` modules render it.

[grammar.js](../crates/tree-sitter-resin/grammar.js) defines concrete syntax. The
[CST document](../crates/resin-cst/src/lib.rs) pairs a Tree-sitter tree with source
text; [CST lowering](../crates/resin-cst/src/lower.rs) reparses it incrementally. Skip the
generated `src/parser.c` inside the grammar package on a first read.

[AST language](../crates/resin-ast/src/lib.rs) defines source files, declarations,
terms, and type syntax with byte spans. [AST lowering](../crates/resin-ast/src/lower.rs)
translates CST nodes, decodes strings, inserts the unit branch of one-armed `if`,
and represents operators as builtin applications. It also preserves incomplete
expressions as holes for editor recovery. AST generation performs no filesystem I/O.
[Program assembly](../crates/resin-ast/src/load.rs) consumes the application's frozen
`SourceGraph` and parsed documents, orders dependencies, and diagnoses missing imports
and cycles. The separate [loader](../crates/resin-source/src/lib.rs) accepts explicit
bindings, supplied text, and disk files for application acquisition. Each AST source
module retains its immutable `Source` alongside its syntax.

[HIR language](../crates/resin-hir/src/lib.rs) is a self-contained, typed tree. Start
at [HIR lowering](../crates/resin-hir/src/lower/mod.rs), then follow
[checking a file](../crates/resin-hir/src/lower/check/mod.rs): declare signatures, check
bodies, solve dependency groups, and resolve inference variables while retaining named
type binders. The [solver](../crates/resin-hir/src/lower/infer/mod.rs) handles numeric constraints
and recursive error sets. It stays private to this crate.

[Elaboration](../crates/resin-hir/src/lower/elaborate.rs) resolves lexical bindings,
method calls, field projections, conversions, and shader references. It establishes
definite initialization in runtime evaluation order, intersecting the states of
alternative branches, including in unused definitions. Short-circuit operators become
conditionals; type-dependent layout queries remain explicit until specialization.
The public HIR contains neither AST nodes nor scope cursors. The temporary checking
tree in [typed.rs](../crates/resin-hir/src/lower/typed.rs) is an internal construction step.

[LIR language](../crates/resin-lir/src/lib.rs) defines a typed operand stack machine.
Functions own locals and a tree of basic blocks. Instructions consume and produce
stack values; `If` and `Loop` terminators own nested regions and their continuations.
Selection arms end with `Merge`, loop conditions with `LoopTest`, and loop bodies
with `Continue`; `Return` leaves the function. Parameters occupy the first locals in
declaration order, and zero-argument functions need no parameter local. Block IDs
identify nodes in the tree; the verifier rejects arbitrary jumps and cycles.
[LIR lowering](../crates/resin-lir/src/lower/mod.rs) consumes HIR, specializes function
and type applications, and builds the concrete type catalog. Storage lowering then
chooses locations, makes evaluation order explicit, and inserts cleanup and runtime
initialization flags for managed locals.

| Question | Start reading here |
| --- | --- |
| Which imported declaration does a name mean? | [HIR modules](../crates/resin-hir/src/lower/mod.rs) |
| Which names are visible at this source position? | [HIR scopes](../crates/resin-hir/src/lower/scope.rs) |
| How are a function's constraints solved? | [HIR checking](../crates/resin-hir/src/lower/check/mod.rs) |
| How does a method become an ordinary call? | [HIR elaboration](../crates/resin-hir/src/lower/elaborate.rs) |
| Is a source binding definitely initialized before a read? | [HIR elaboration](../crates/resin-hir/src/lower/elaborate.rs) |
| Where is a binding stored? | [LIR bindings](../crates/resin-lir/src/lower/bindings.rs) |
| Which storage location does an assignment address? | [LIR places](../crates/resin-lir/src/lower/places.rs) |
| How do branches and loops join? | [LIR control flow](../crates/resin-lir/src/lower/flow.rs) |
| How are error propagation and cleanup lowered? | [LIR sums](../crates/resin-lir/src/lower/sums.rs), [cleanup](../crates/resin-lir/src/lower/mod.rs) |
| How are concrete instructions assembled? | [LIR builder](../crates/resin-lir/src/lower/builder.rs) |

### Verification certifies the LIR language

LIR's private [verify](../crates/resin-lir/src/verify/mod.rs) module checks definitions,
instruction operands, unique region ownership, selection merges, loop stack invariants,
and returns. Its public operations and certificate types live in [lib.rs](../crates/resin-lir/src/lib.rs), beside the language
being checked. `resin_lir::VerifiedModule` owns LIR and its verification analysis behind
private fields. Native builds borrow an immutable `Verified` view. Consuming
`into_module` returns ordinary LIR and discards the certificate; edits require
verification again.

The [public type model and operations](../crates/resin-types/src/lib.rs) live in
`resin-types`. Private [types.rs](../crates/resin-types/src/types.rs) implements
representation and layout; [typer.rs](../crates/resin-types/src/typer.rs) implements
concrete checking and conversions.
They depend on no compiler phase. HIR adds inference and lexical overload resolution privately;
the verifier applies concrete rules to instructions independently of source checking.

### C emission and native builds are separate

[C lowering](../crates/resin-codegen/src/c/lower/mod.rs) produces a
[private C tree](../crates/resin-codegen/src/c/mod.rs): declarations, shader-header references,
functions, structured statements, and a `main` wrapper. [The printer](../crates/resin-codegen/src/c/print.rs)
formats that tree without accessing LIR or typechecking facts. [function.rs](../crates/resin-codegen/src/c/lower/function.rs)
lowers instructions, nested control flow, and operand transfers; [foreign.rs](../crates/resin-codegen/src/c/lower/foreign.rs)
bridges typed Resin parameters and results to native C calls.

[Ninja execution](../crates/resin-toolchain/src/ninja.rs) builds the generated dependency
graph. The toolchain's [public interface](../crates/resin-toolchain/src/lib.rs) names
the default compiler; private [platform.rs](../crates/resin-toolchain/src/platform.rs)
supplies the archive name, flags, and system libraries: `cc` and `libresin_runtime.a`
on Unix; GNU-style LLVM `clang` and `resin_runtime.lib` on Windows MSVC. Windows
builds keep Rust, GLFW, and emitted C on the same C runtime. Install Ninja, SPIR-V Tools (`spirv-opt`),
and a C compiler; `CC` or `--cc` overrides the compiler selection. Host-only programs
invoke no shader optimizer and need neither a Vulkan SDK nor a GPU.

The toolchain stages source projects and retains successful outputs under a cache
lock. Ninja tracks dependencies between shaders, embedded headers, C includes, and
the executable. Its configured commands use captured process settings; there is no
compiler-specific interpretation inside the toolchain.

There are two artifact directories with different owners: Cargo builds the
compiler and runtime under `target/`; Resin builds user programs under `build/`
in the caller's working directory. Default runs reuse an unoptimized native
build and execute it. Requesting an executable with `-o` selects the optimized cache
and copies the output without running it. This does not change Cargo's Rust profile.

## 3. Follow an editor change through the compiler

Start with [Source](../crates/resin-source/src/lib.rs): immutable text with a diagnostic
name and a stable logical `SourceId`. Cloning shares the same version. `with_text`
creates a new version with the same logical identity; both versions remain usable.
Logical identity, diagnostic name, and exact text determine equality. Independent
reconstruction of the same named text can reuse results; distinct modules require
distinct logical identities even when their displayed names and contents match.
`SourceLocation` retains a source handle and byte span, keeping diagnostics tied to
exactly the text that produced them.

The editor's [worker](../crates/resin-client/src/lsp/worker.rs) registers authoritative
open text with `Loader::source_from_text`, preserving physical identity through edits.
Current supplied registrations seed each request-owned capture; closing the last
alias restores disk loading, and close/reopen starts a new normalization epoch.
Captured text and local presentation paths stay immutable for each admitted request.

Each analysis resolves imports before remote semantic lookup. Uploaded names and
edges become a frozen graph; the server selects Source/CST/AST/HIR caches shared with
build requests. Equal text in another logical module never borrows its nominal IDs or
origins. An edited dependency changes the HIR key while unchanged per-file layers
remain reusable. Earlier consumers keep their original handles after cache eviction.

Incomplete code goes through the same compiler traversal. Expression, type,
and missing-field holes preserve useful children; bounded delimiter repair
recovers unfinished scopes without clearing the original syntax diagnostics.
Invalid declarations still shadow outer names, and healthy siblings retain
their types. Unknown types display as `?`; errors prevent executable generation.

`Hir` query methods delegate to HIR's opaque `Analysis`, supplying its shared CST map. HIR's
[lib.rs](../crates/resin-hir/src/lib.rs) keeps the private analysis fields and their query
operations together. They select the source context view and look up declarations
on demand. Member observations
retain available fields, signatures, and canonical method origins even when later code fails.
Hover and member completion use these facts and
[shared type formatting](../crates/resin-types/src/lib.rs). There is no separate
recovery compiler or fallback declaration index.

The [client LSP](editor-client.md) adapts remote facts to stdio editor
messages. [server.rs](../crates/resin-client/src/lsp/server.rs) accepts notifications and
checks response freshness; [worker.rs](../crates/resin-client/src/lsp/worker.rs) coalesces
editor snapshots and schedules bounded capture/HTTP tasks.
[text.rs](../crates/resin-client/src/lsp/text.rs) converts byte offsets to UTF-16 positions.
[mirrors.rs](../crates/resin-client/src/lsp/mirrors.rs) materializes managed definitions as
immutable local files for navigation. Old versions and dependency states cannot
replace current diagnostics. Editor analysis never builds or executes a program.

`textDocument/formatting` stays local and uses the same
[formatter](../crates/resin-cst/src/print.rs) as the CLI. It returns an edit for the
current open buffer; see [formatting rules](editor-client.md#formatting).

Finally, [editors/zed/src/lib.rs](../editors/zed/src/lib.rs) locates and launches the
`resin --lsp` from Zed's WASI extension. Its [language queries](../editors/zed/languages/resin)
provide highlighting, outlines, and other syntax features. See the
[extension README](zed.md) for installation and configuration;
the extension builds separately from the main Cargo workspace.

[editors/helix/](../editors/helix) configures Helix to launch the same server directly
and supplies its syntax queries. See the [Helix setup](helix.md)
for installation and grammar maintenance.

## 4. Follow a shader into the runtime

Read [examples/gradient.resin](../examples/gradient.resin) for compute, then
[examples/triangle.resin](../examples/triangle.resin) for graphics. Each example keeps
its decorated shader entries and ordinary helpers alongside its host code. To inspect
the generated shaders without running a Vulkan program:

```sh
cargo run -- examples/gradient.resin -o dist/
```

Inspect `shader_<function-id>.unoptimized.spv` and `.spv` in the generated project’s
release cache after the build, using `spirv-dis` to view their assembly.

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register and validate
shader entry declarations. [shader interfaces](../crates/resin-types/src/lib.rs) defines their metadata
and signature contracts. Decorated functions and their unannotated helpers remain
host-callable. Passing shader declarations to pipeline creation requests their compiled
representations; [project generation](../crates/resin-codegen/src/lib.rs) writes SPIR-V
and their Ninja dependencies for the current Vulkan backend. `spirv-opt -O` optimizes it,
and `resin --embed` writes the headers included by generated C.
No runtime function-value analysis is involved. The runtime receives bytes, not a
host function pointer or source-file path. Shader functions expose no bytecode property.

[SPIR-V lowering](../crates/resin-codegen/src/spirv/mod.rs) collects reachable shader
functions. [entry.rs](../crates/resin-codegen/src/spirv/entry.rs) adapts regular Resin function
signatures to compute, vertex, or fragment interfaces; the private function and operation
lowering emits their bodies as SPIR-V instructions. The backend uses `rspirv` to allocate IDs
and assemble the binary. Structured control flow remains explicit, and physical pointers
use the shared host/device layout. The shader profile rejects operations such as foreign
calls and recursion. Optimization is a separate `spirv-opt` process owned by the toolchain.

GPU allocation and checked access use ordinary generic source functions backed by
opaque compiler primitives. Pipeline creation, dispatch, and draw use explicitly
registered compiler bridges to check shader types and project owning views. Follow
the native resource wrappers from [resin/gpu.resin](../resin/gpu.resin), through
[resin_runtime.h](../crates/resin-runtime/include/resin_runtime.h) and its included
headers, to [the runtime](../crates/resin-runtime/src/lib.rs). The same pattern applies
to images and windows. Wrappers omit the native `resin_` prefix and return
`T | Err<RuntimeError>`, with created handles in the success value.
[resin/status.resin](../resin/status.resin) translates integer status codes into
named error structs; the C ABI remains unchanged.

Inside the runtime, the useful landmarks are:

- [gpu/device.rs](../crates/resin-runtime/src/gpu/device.rs): Vulkan device discovery and
  required features.
- [gpu/mod.rs](../crates/resin-runtime/src/gpu/mod.rs): allocations, images, command
  recording, synchronization between operations, and submission.
- [gpu/pipeline.rs](../crates/resin-runtime/src/gpu/pipeline.rs): conventional Vulkan
  compute and graphics pipelines.
- [gpu/present.rs](../crates/resin-runtime/src/gpu/present.rs) and
  [window/](../crates/resin-runtime/src/window): swapchains, presentation, and statically
  linked GLFW.
- [allocator/range.rs](../crates/resin-runtime/src/allocator/range.rs): aligned
  suballocation using a sorted list of free ranges.

Buffers make the host/device boundary concrete. The source `Span<T>` pairs a
borrowed address with an element count. Arrays and spans return element references
through `:at(index)` for reads or writes. A pointer-backed span or pointer to an
array can provide an element pointer through `:lea(index)`; a local array cannot.
Host indexing checks bounds, while shader indexing is unchecked.
Host `GpuPtr<T>` and `GpuSpan<T>` retain their allocation and expose checked
`load`, `store`, and `replace` operations. Create them with `gpu:create(value)?`
and `gpu:alloc::<T>(count)?`.

Dispatch and draw accept a typed pipeline and a host argument record. Explicitly
registered projection contracts map its owning GPU views to shader pointers and
spans, preserving offsets and retaining the referenced allocations. Shader entry
wrappers receive the generated root as their declared `Ptr<T>`. The public source
API exposes neither a raw host pointer into GPU storage nor a reusable projected
root. The [shared layout contract](../crates/resin-types/src/lib.rs) keeps C and SPIR-V
storage consistent; start there when investigating a field offset or alignment.

The unsafe native C and Rust APIs retain explicit handles and addresses. Source
resource wrappers retain shared owners and propagate failures with `?`.
`commands:submit()` and cancellation clear the shared native handle; destruction
cancels unfinished recordings. Recorded GPU allocations reject host access until
synchronous submission or cancellation. Shader pointers and spans are nonowning
views whose allocations remain pinned by the recording. See
[GPU buffers](gpu-buffers.md) for projection and command lifetimes.
Shader bodies describe individual invocations;
the compiler does not synthesize workgroup-local storage or barriers.

Finally, [examples/particles.resin](../examples/particles.resin) combines compute
and graphics over a shared buffer, and [examples/window.resin](../examples/window.resin)
adds presentation. Builds target 64-bit Linux, macOS, and Windows, but actual
GPU runs also need the Vulkan features checked in `gpu/device.rs`. On macOS,
portability enumeration and MoltenVK discovery do not guarantee those features
or presentation support. Window runs need a display; the headless image demos
write PNGs in cwd.

## 5. Find the test closest to your change

Tests are executable descriptions of the boundaries above.

The [source test helpers](../tests/support/pipeline.rs) resolve only explicit imports.
Use `source_module` for inline source and `file_module` for a file and its imports;
both lower the HIR retained by compiler analysis. Tests that inspect or edit ASTs use
`load` and then explicitly regenerate with `generate_program` after edits.

| Change | Useful tests |
| --- | --- |
| Syntax or AST shape | [parser corpus](../crates/tree-sitter-resin/test/corpus), [mutation_ast.rs](../tests/mutation_ast.rs) |
| Grammar JavaScript types, lint, or formatting | Run `npm run check` in [the parser directory](parser.md) |
| Imports, exports, or entry visibility | [modules.rs](../tests/modules.rs), [cli.rs](../tests/cli.rs) |
| Typing, conversions, or IR invariants | [nominal_types.rs](../tests/nominal_types.rs), [typing-rule tests](../crates/resin-types/src/tests.rs), [verifier tests](../crates/resin-lir/src/verify/tests.rs) |
| Explicit type holes and return inference | [inference.rs](../tests/inference.rs), [inference example](../examples/inference.resin) |
| Structs, aliases, unions, and typed errors | [results.rs](../tests/results.rs), [errors example](../examples/errors.resin) |
| Automatic destruction, scope exits, and copying | [shared.rs](../tests/shared.rs), [ownership example](../examples/ownership.resin), [C execution tests](../tests/c_backend.rs) |
| Host code generation or C interop | [c_backend.rs](../tests/c_backend.rs), [foreign.rs](../tests/foreign.rs), [printing.rs](../tests/printing.rs) |
| Standard-library error unions and native failure cleanup | [stdlib.rs](../tests/stdlib.rs) (no GPU or windows required) |
| Console input, byte handling, and allocation failures | [console.rs](../tests/console.rs), [input example](../examples/input.resin) |
| Compilation and artifact reuse | [build_cache.rs](../tests/build_cache.rs), [cli.rs](../tests/cli.rs) |
| Source formatting, file traversal, or format checks | [formatting.rs](../tests/formatting.rs), [format_cli.rs](../tests/format_cli.rs), [LSP formatting tests](../tests/lsp.rs) |
| Source versions, import caching, editor queries, or recovery | [HIR import tests](../crates/resin-hir/tests/imports.rs), [loader tests](../crates/resin-source/tests/files.rs), [analysis.rs](../tests/analysis.rs) |
| LSP protocol, buffer versions, or watched files | [lsp.rs](../tests/lsp.rs) |
| Zed and Helix syntax features | [editor_queries.rs](../tests/editor_queries.rs) |
| Shader generation or execution | [spirv_backend.rs](../tests/spirv_backend.rs), [gpu_backend.rs](../tests/gpu_backend.rs), [window_backend.rs](../tests/window_backend.rs) |

[crates/resin-runtime/tests/](../crates/resin-runtime/tests) also exercises the native runtime
with GLSL fixtures, independently of the Resin compiler. This is useful for
separating a Vulkan runtime bug from a language or code-generation bug.

For a focused host-only starting point, in your development environment:

```sh
cargo test -p resin --test modules --test c_backend
```

For editor work, without launching an editor or opening windows:

```sh
cargo test -p resin --test analysis --test editor_queries
cargo test -p resin --test formatting --test format_cli
cargo test -p resin-hir -p resin-client -p resin-server
cargo test -p resin --test lsp
```

[CI](../.github/workflows/build.yml) checks the grammar's JavaScript types, lint,
and formatting, and builds, tests, and lints native Rust packages. Automatic pull
request and push checks run on Linux; manually dispatched checks also run macOS
and Windows. The jobs run a host example and check example formatting with
`--format --check examples`. Window-opening tests are excluded; the Linux job requires GPU execution
with its software Vulkan driver. A passing native matrix therefore does not
establish full GPU compatibility.

For the full GPU path, in a POSIX shell with a compatible Vulkan device and a
desktop display or Xvfb available (for Xvfb, set `DISPLAY` to its display and
`XDG_SESSION_TYPE=x11` so GLFW does not select a Wayland compositor):

```sh
RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1 RESIN_REQUIRE_WINDOW=1 \
  cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

The requirement variables prevent missing GPU tools or facilities from turning
coverage into skipped tests. See [Development](development.md#development)
for environment setup and parser regeneration. Grammar changes must include
regenerated parser files; runtime API changes generally span the Rust
implementation, C headers, and standard-library declarations. Most new runtime
operations do not need a new compiler builtin.
