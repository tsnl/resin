# A tour of Resin

Resin is a systems language for host CPUs and Vulkan GPUs. This is a reading
path through its implementation, not a language reference; keep the
[README](README.md) nearby for syntax and command-line options.

The root is both a Cargo workspace and the `resin` CLI package. [src/](src/)
contains one executable's command dispatch. [crates/](crates/) contains the
[compiler driver](crates/resin-compiler/), [sources and loading](crates/resin-source/), [concrete types](crates/resin-types/),
[platform toolchain](crates/resin-toolchain/),
compiler phases, and the supporting [runtime](crates/resin-runtime/),
[parser](crates/tree-sitter-resin/), and [language server library](crates/resin-lsp/).
All are unpublished. The
[Zed extension](editors/zed/) has a separate Cargo workspace under `editors/`.
The [standard library](resin/std/) is written in Resin and wraps the runtime's C API.

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
executable without running it. Inspect the generated `main.c` beside each cached executable
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
[resin/std/console.resin](resin/std/console.resin). The module builds a growing line
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

[src/main.rs](src/main.rs) only calls `resin::cli::main`, the root package's CLI entry.
[cli/mod.rs](src/cli/mod.rs) dispatches modes, and [args.rs](src/cli/args.rs)
parses flags and chooses `Interpreter`, `Compiler`, `Formatter`, or `LanguageServer`;
[source.rs](src/cli/source.rs) parses the `FILE[:ENTRY]` selector.
Interpreter mode builds a debug native executable and runs it. Compiler mode builds
an optimized executable and copies it to the destination selected with `--output`
(or `-o`). `resin --lsp DIR` runs the language server in that project directory.

The platform toolchain's [Environment](crates/resin-toolchain/src/lib.rs)
captures environment variables, the working directory, and executable/temp paths once.
Its private [resolution](crates/resin-toolchain/src/environment.rs) applies CLI
compiler choices before `CC`/`GLSLC` and platform defaults. It finds runtime headers,
the archive, and cache settings, returning an opaque `Toolchain`. The CLI resolves
relative `RESIN_LIBRARY_ROOT` overrides against the captured working directory,
defaults to the bundled library root, and chooses `CProfile`.
Tools are invoked when required, without preflight checks, so host-only
programs remain independent of `glslc`. Compiler subprocesses and cache fingerprints
use the same captured environment.

The CLI's private [Request](src/cli/request.rs) validates the input/output combination,
resolves directory destinations, and rejects outputs that would overwrite the source.
A [resin_source::Loader](crates/resin-source/src/lib.rs) reads the entry into an immutable `Source`.
`Compiler::compile(entry, &mut loader)` resolves imports and returns an
`Arc<Compilation>` containing analysis and phase products.

The CLI passes `compilation.verified()` to
[codegen](crates/resin-codegen/src/lib.rs), which writes C, requested GLSL, and
`build.ninja` in one operation. It then asks `Toolchain::build` to execute that
project. Ninja compiles SPIR-V, invokes this Resin executable with `--embed` to make
C headers, and compiles and links the host program. The executable path comes from
the platform's `current_exe` API, keeping embedding on the same Resin version.
The returned build and executable handles retain a cache lock. Generated sources,
headers, SPIR-V, and the Ninja graph remain available for inspection.
The [compiler facade](crates/resin-compiler/src/lib.rs) exposes a small contract:
immutable sources go in, a loader discovers their imports, and immutable compilations
come out. A result retains successful phase products and editor facts for one entry
and its imports. Its public accessors expose diagnostics, AST, HIR, LIR, and source
queries. `Compiler` owns private parsing and analysis caches directly; `Compilation`
owns its retained products. Their definitions and implementations live together in
`lib.rs`. The concrete `resin_source::Loader` owns import lookup.

Each phase is also a directly usable crate. The root [src/lib.rs](src/lib.rs) exposes
only the CLI module. Tests call phase APIs directly or use `Compiler::compile`
to inspect a program without building an executable.

Formatting takes a separate path from `main` through
[format.rs](src/cli/format.rs) to the shared
[CST formatter](crates/resin-cst/src/print.rs) library module. `--format` (or `-f`) formats
files in place and searches directories recursively for `.resin` files;
`--format --check` reports differences without writing and exits with status 1
on differences or file/syntax errors. In the development environment, try:

```sh
cargo run -- --format --check examples
```

The formatter uses Tree-sitter syntax, normalizes whitespace and numeric suffix
spelling, preserves comments and string contents, and indents with hard tabs. It rejects invalid syntax without
modifying the file and needs no semantic analysis or entry point.

### Read each language, then its incoming pass

The [architecture guide](doc/architecture.md) gives the full crate graph and pass
contracts. Every phase starts at `lib.rs`, which defines its language and the
operations clients can call. Private `lower` modules produce that language from
the preceding phase, and private `print` modules render it.

[grammar.js](crates/tree-sitter-resin/grammar.js) defines concrete syntax. The
[CST document](crates/resin-cst/src/lib.rs) pairs a Tree-sitter tree with source
text; [CST lowering](crates/resin-cst/src/lower.rs) reparses it incrementally. Skip the
generated `src/parser.c` inside the grammar package on a first read.

[AST language](crates/resin-ast/src/lib.rs) defines source files, declarations,
terms, and type syntax with byte spans. [AST lowering](crates/resin-ast/src/lower.rs)
translates CST nodes, decodes strings, inserts the unit branch of one-armed `if`,
and represents operators as builtin applications. It also preserves incomplete
expressions as holes for editor recovery. AST generation performs no filesystem I/O.
The compiler's [import traversal](crates/resin-compiler/src/lib.rs) calls
`resin_source::Loader::load_import` and builds an AST `Program` in dependency order.
The [loader](crates/resin-source/src/lib.rs) accepts explicit source bindings, supplied
file text, and disk files. It interprets relative and `$/` paths. Each AST source
module retains its immutable `Source` alongside its syntax.

[HIR language](crates/resin-hir/src/lib.rs) is a self-contained, typed tree. Start
at [HIR lowering](crates/resin-hir/src/lower/mod.rs), then follow
[checking a file](crates/resin-hir/src/lower/check/mod.rs): declare signatures, check
bodies, solve dependency groups, and replace inference variables with concrete types.
The [solver](crates/resin-hir/src/lower/infer/mod.rs) handles numeric constraints
and recursive error sets. It stays private to this crate.

[Elaboration](crates/resin-hir/src/lower/elaborate.rs) resolves lexical bindings,
method calls, field projections, conversions, and shader references. Short-circuit
operators become conditionals and layout queries become constants. The public HIR
contains neither AST nodes nor scope cursors. The temporary checking tree in
[typed.rs](crates/resin-hir/src/lower/typed.rs) is an internal construction step.

[LIR language](crates/resin-lir/src/lib.rs) defines a typed operand stack machine.
Functions own locals and basic blocks; instructions consume and produce stack
values, and terminators connect blocks. Local zero is always the parameter,
including unit, tuples, and foreign declarations. [LIR lowering](crates/resin-lir/src/lower/mod.rs)
consumes only HIR and shared concrete types. It chooses storage, checks definite
initialization, makes evaluation order explicit, and inserts cleanup.

| Question | Start reading here |
| --- | --- |
| Which imported declaration does a name mean? | [HIR modules](crates/resin-hir/src/lower/mod.rs) |
| Which names are visible at this source position? | [HIR scopes](crates/resin-hir/src/lower/scope.rs) |
| How are a function's constraints solved? | [HIR checking](crates/resin-hir/src/lower/check/mod.rs) |
| How does a method become an ordinary call? | [HIR elaboration](crates/resin-hir/src/lower/elaborate.rs) |
| Where is a binding stored, and is it initialized? | [LIR bindings](crates/resin-lir/src/lower/bindings.rs) |
| Which storage location does an assignment address? | [LIR places](crates/resin-lir/src/lower/places.rs) |
| How do branches and loops join? | [LIR control flow](crates/resin-lir/src/lower/flow.rs) |
| How are Result propagation and cleanup lowered? | [LIR sums](crates/resin-lir/src/lower/sums.rs), [cleanup](crates/resin-lir/src/lower/mod.rs) |
| How are concrete instructions assembled? | [LIR builder](crates/resin-lir/src/lower/builder.rs) |

### Verification certifies the LIR language

LIR's private [verify](crates/resin-lir/src/verify/mod.rs) module checks definitions,
instruction operands, block-edge stack types, and returns. Its public operations and
certificate types live in [lib.rs](crates/resin-lir/src/lib.rs), beside the language
being checked. `resin_lir::VerifiedModule` owns LIR and its verification analysis behind
private fields. Native builds borrow an immutable `Verified` view. Consuming
`into_module` returns ordinary LIR and discards the certificate; edits require
verification again.

The [public type model and operations](crates/resin-types/src/lib.rs) live in
`resin-types`. Private [types.rs](crates/resin-types/src/types.rs) implements
representation and layout; [typer.rs](crates/resin-types/src/typer.rs) implements
concrete checking and conversions.
They depend on no compiler phase. HIR adds inference and method namespaces privately;
the verifier applies concrete rules to instructions independently of source checking.

### C emission and native builds are separate

[C lowering](crates/resin-codegen/src/c/lower/mod.rs) produces a
[private C tree](crates/resin-codegen/src/c/mod.rs): declarations, shader-header references,
functions, blocks, and a `main` wrapper. [The printer](crates/resin-codegen/src/c/print.rs)
formats that tree without accessing LIR or typechecking facts. [function.rs](crates/resin-codegen/src/c/lower/function.rs)
lowers instructions and block edges; [foreign.rs](crates/resin-codegen/src/c/lower/foreign.rs)
bridges Resin's unary calls to conventional C argument lists.

[Ninja execution](crates/resin-toolchain/src/ninja.rs) builds the generated dependency
graph. The toolchain's [public interface](crates/resin-toolchain/src/lib.rs) names
the default compiler; private [platform.rs](crates/resin-toolchain/src/platform.rs)
supplies the archive name, flags, and system libraries: `cc` and `libresin_runtime.a`
on Unix; GNU-style LLVM `clang` and `resin_runtime.lib` on Windows MSVC. Windows
builds keep Rust, GLFW, and emitted C on the same C runtime. Install Ninja, `glslc`,
and a C compiler; `CC` or `--cc` overrides the compiler selection. Host-only programs
invoke no shader compiler and need neither a Vulkan SDK nor a GPU.

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

Start with [Source](crates/resin-source/src/lib.rs): immutable text with a diagnostic
name and a stable logical `SourceId`. Cloning shares the same version. `with_text`
creates a new version with the same logical identity; both versions remain usable.
Names are labels, so two sources with the same name are still distinct.
`SourceLocation` retains a source handle and byte span, keeping diagnostics tied to
exactly the text that produced them.

The editor's [worker](crates/resin-lsp/src/worker.rs) registers open document text
with `resin_source::Loader::source_from_text`. Those sources take precedence over
disk imports. Closing a buffer calls `remove_source` to restore disk loading;
file notifications schedule another compile call. Generated sources can instead
use `set_import` to bind an import directly to a source handle.

The compiler's [lib.rs](crates/resin-compiler/src/lib.rs) keeps this traversal,
its private cache fields, and retained compilation products together. Each compile
call resolves the import graph before deciding whether analysis can be reused.
An unchanged source version reuses its syntax; a changed version can reuse the
previous Tree-sitter tree for incremental parsing. Semantic checking reruns when
the entry's resolved import graph changes. Previously returned compilations own
their original sources and remain usable after subsequent calls.

Incomplete code goes through the same compiler traversal. Expression, type,
and missing-field holes preserve useful children; bounded delimiter repair
recovers unfinished scopes without clearing the original syntax diagnostics.
Invalid declarations still shadow outer names, and healthy siblings retain
their types. Unknown types display as `?`; errors prevent executable generation.

The `Compilation` query methods in [lib.rs](crates/resin-compiler/src/lib.rs) delegate to
HIR's opaque `Analysis`, supplying its shared CST map. HIR's
[lib.rs](crates/resin-hir/src/lib.rs) keeps the private analysis fields and their query
operations together. They select the source context view and look up declarations
on demand. Member observations
retain available fields, signatures, and canonical method origins even when later code fails.
Hover and member completion use these facts and
[shared type formatting](crates/resin-types/src/lib.rs). There is no separate
recovery compiler or fallback declaration index.

The [language server library](crates/resin-lsp/README.md) adapts that compiler state to
the Language Server Protocol over stdio. `resin --lsp DIR` invokes it inside the same
executable that builds programs. [server.rs](crates/resin-lsp/src/server.rs)
handles requests, document versions, and file notifications;
[text.rs](crates/resin-lsp/src/text.rs) converts byte offsets to UTF-16 positions.
[worker.rs](crates/resin-lsp/src/worker.rs) retains a compiler and loader in
the background, coalesces edits, and discards obsolete results. Editor analysis
never compiles C/GLSL, initializes a GPU, or executes Resin programs.

`textDocument/formatting` uses the same [formatter](crates/resin-cst/src/print.rs) as the
CLI. The server formats the open document's current text and returns a text edit
for the changed region. See the [formatting rules](crates/resin-lsp/README.md#formatting)
for layout conventions.

Finally, [editors/zed/src/lib.rs](editors/zed/src/lib.rs) locates and launches the
`resin --lsp` from Zed's WASI extension. Its [language queries](editors/zed/languages/resin/)
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

Read `shader_<function-id>.glsl` and `.spv` in the generated project’s release cache after the build.

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register and validate
shader entry declarations. [shader interfaces](crates/resin-types/src/lib.rs) defines their metadata
and signature contracts. Decorated functions and their unannotated helpers remain
host-callable. Accessing `function.spirv` requests a static `Span<ubyte>` artifact;
[project generation](crates/resin-codegen/src/lib.rs) enumerates those declaration
requests and writes GLSL plus their Ninja dependencies. `glslc` produces SPIR-V,
and `resin --embed` writes the headers included by generated C.
No runtime function-value analysis is involved. The runtime receives bytes, not a
host function pointer or source-file path.

[GLSL lowering](crates/resin-codegen/src/glsl/lower/mod.rs) collects reachable shader
functions. [entry.rs](crates/resin-codegen/src/glsl/lower/entry.rs) adapts regular Resin function
signatures to compute, vertex, or fragment interfaces, and
[function.rs](crates/resin-codegen/src/glsl/lower/function.rs) lowers their bodies. The GLSL
backend supports a subset of the host language; it rejects operations such as
foreign calls and recursion rather than making them work on the device.

The remaining GPU operations are ordinary standard-library calls. Follow one
from [resin/std/gpu.resin](resin/std/gpu.resin), through
[resin_runtime.h](crates/resin-runtime/include/resin_runtime.h) and its included
headers, to [the runtime](crates/resin-runtime/src/lib.rs). The same pattern applies
to images and windows. Wrappers omit the native `resin_` prefix and return
`Result<T, RuntimeError>`, with created handles in the success value.
[resin/std/status.resin](resin/std/status.resin) translates integer status codes into
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
| Syntax or AST shape | [parser corpus](crates/tree-sitter-resin/test/corpus/), [mutation_ast.rs](tests/mutation_ast.rs) |
| Grammar JavaScript types, lint, or formatting | Run `npm run check` in [crates/tree-sitter-resin/](crates/tree-sitter-resin/README.md) |
| Imports, exports, or entry visibility | [modules.rs](tests/modules.rs), [cli.rs](tests/cli.rs) |
| Typing, conversions, or IR invariants | [nominal_types.rs](tests/nominal_types.rs), [typing-rule tests](crates/resin-types/src/tests.rs), [verifier tests](crates/resin-lir/src/verify/tests.rs) |
| Explicit type holes and return inference | [inference.rs](tests/inference.rs), [inference example](examples/inference.resin) |
| Structs, aliases, unions, and typed errors | [results.rs](tests/results.rs), [errors example](examples/errors.resin) |
| Automatic destruction, scope exits, and copying | [shared.rs](tests/shared.rs), [ownership example](examples/ownership.resin), [C execution tests](tests/c_backend.rs) |
| Host code generation or C interop | [c_backend.rs](tests/c_backend.rs), [foreign.rs](tests/foreign.rs), [printing.rs](tests/printing.rs) |
| Standard-library Results and native failure cleanup | [stdlib.rs](tests/stdlib.rs) (no GPU or windows required) |
| Console input, byte handling, and allocation failures | [console.rs](tests/console.rs), [input example](examples/input.resin) |
| Compilation and artifact reuse | [build_cache.rs](tests/build_cache.rs), [cli.rs](tests/cli.rs) |
| Source formatting, file traversal, or format checks | [formatting.rs](tests/formatting.rs), [format_cli.rs](tests/format_cli.rs), [LSP formatting tests](tests/lsp.rs) |
| Source versions, import caching, editor queries, or recovery | [compiler API tests](crates/resin-compiler/tests/public_api.rs), [loader tests](crates/resin-source/tests/files.rs), [analysis.rs](tests/analysis.rs) |
| LSP protocol, buffer versions, or watched files | [lsp.rs](tests/lsp.rs) |
| Zed syntax features | [zed_queries.rs](tests/zed_queries.rs) |
| Shader generation or execution | [glsl_backend.rs](tests/glsl_backend.rs), [gpu_backend.rs](tests/gpu_backend.rs), [window_backend.rs](tests/window_backend.rs) |

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
cargo test -p resin-compiler -p resin-lsp
cargo test -p resin --test lsp
```

[CI](.github/workflows/build.yml) checks the grammar's JavaScript types, lint,
and formatting, and builds, tests, and lints native Rust packages. Automatic pull
request and push checks run on Linux; manually dispatched checks also run macOS
and Windows. The jobs run a host example and check example formatting with
`--format --check examples`. Window-opening tests are excluded; GPU checks may
skip when facilities are absent. A passing native matrix therefore does not
establish full GPU compatibility.

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
