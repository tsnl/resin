# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Architecture and style

- Make the compiler educational to read. Prefer explicit data and direct control flow;
  a reader should be able to tell what a pass consumes, produces, and computes locally.
- Follow one direction: syntax → AST → HIR → LIR → verified LIR → C/GLSL → native tools.
  Put each phase's public language definitions and operations in `lib.rs`. Keep incoming
  translation in private `lower` modules and textual rendering in private `print` modules.
  A crate's complete public contract should be discoverable from its entry point.
- HIR is a typed, desugared tree with resolved bindings, calls, and operations. Source
  scopes, method namespaces, inference variables, and recovery belong to HIR construction.
  LIR describes storage, cleanup, stack operations, and explicit control-flow blocks.
  Keep LIR verification in private `crates/resin-lir/src/verify/` modules, with the
  verification API beside the language in `lib.rs`; constructing LIR does not verify it.
- Keep resolved types and layout rules in `crates/resin-types`; they must not depend on a
  frontend, backend, or verifier. Keep the public contract in `lib.rs`, with substantial
  representation algorithms in private `types.rs` and concrete checking/conversion rules
  in private `typer.rs`. The compiler driver sequences passes and the toolchain
  owns external processes. Printers consume their own language, without reaching upstream.
- Each compiler phase is an unpublished workspace crate under `crates/`: `resin-cst`,
  `resin-ast`, `resin-hir`, `resin-lir`, and `resin-codegen`,
  with `resin-source` for immutable sources and loading, and `resin-types` for concrete types. Directory names match Cargo package names;
  keep compilation orchestration in `resin-compiler`, external processes and build caches in
  `resin-toolchain`, source identities and import discovery in `resin-source`, and the parser in
  `tree-sitter-resin`.
  Use `publish = false` and local path dependencies. Keep dependencies acyclic and explicit;
  do not work around a boundary with public implementation modules or reverse dev-dependencies.
  Language nodes are public data. Solvers, scopes, builders, and traversal state stay private;
  expose a small set of lowering, printing, and query operations instead.
- The root manifest is both the `resin` package and the workspace. Root `src/` contains
  only the CLI; library crates live under `crates/`. One `resin` executable builds/runs
  programs, formats source, and serves LSP with `--lsp DIR`. `resin-lsp` is a library,
  so editor and compilation services ship in the same binary. Do not re-export compiler
  implementation modules through the CLI crate.
- Keep the complete Tree-sitter package together. Editor integrations live under
  `editors/`; the Zed WASI extension has its own Cargo workspace in `editors/zed`.
  Crates own their isolated tests; root `tests/` exercises the complete executable and
  cross-crate behavior. Shared examples and documentation stay at the root. Resin
  libraries live under `resin/`, with the standard library in `resin/std/`.
  Keep the root package as the default member for root Cargo commands.
- Keep `resin-source` and `resin-types` independent of compiler phases and of each other.
  Define phase-specific errors in their producing crate; source locations and rendered
  source diagnostics are shared transport, not a common compiler-error enum.
  Keep `resin-common` minimal: it owns the shared `define_id!` macro, imported directly
  by consumers instead of re-exported through `resin-types`. Use `tempfile` for temporary
  files. Do not move domain types, source diagnostics, or phase state into `resin-common`
  to break a dependency cycle.
  C and GLSL each have a private target language, lowering, and printing inside `codegen`.
- Use canonical crate and module names; do not rename dependencies or language types
  for brevity. Prefer a qualified name when two phases use the same type name.
  Import needed vocabulary privately with `use resin_source::prelude::*;` and
  `use resin_types::prelude::*;`; do not
  re-export another crate's types merely because they appear in public signatures.
- Design interfaces for information hiding, not just public/private visibility. A caller
  should understand what an operation accepts, returns, and guarantees without learning
  how its module stores data or performs the work. Expose domain concepts and completed
  results; hide caches, builders, solvers, bookkeeping, and incidental dependencies.
- Hide complexity honestly: an abstraction may do substantial work behind a simple
  interface, but its descriptive names, types, and ownership must make that behavior
  predictable. Do not depend on callers reading documentation to discover intended
  usage, hidden side effects, or surprising restrictions. Make mistakes impossible
  by construction where practical, and make correct usage the natural path.
  Judge boundaries by whether they reduce cognitive load and help readers find their
  bearings, not by how many wrappers or private modules they introduce. Documentation
  explains rationale and detail; it must not repair a misleading interface.
- Keep each crate's complete public interface in `lib.rs`. Keep the interface small;
  the file itself may be large. Review the interface in isolation: names should explain responsibilities, and signatures
  should show the direction of data flow. Do not merely re-export implementation modules
  or add forwarding types that expose the same internals under another name.
- Prefer large, cohesive single-file modules when they make the interface and its
  implementation easier to follow. Put private state directly on the public type in
  `lib.rs`; avoid forwarding twins such as `Compilation { data: Data }` or a `Cache`
  wrapper around a compiler's own fields. Crate boundaries enforce information hiding;
  a separate file does not improve an abstraction by itself. Extract a private module
  only for substantial, coherent work, not to shorten a file or scatter one type's methods.
  Delineate major API sections with three-line comment blocks (`//`, a title, `//`).
- Prefer struct-style enum variants with descriptive named fields over tuple payloads,
  including single-field variants such as `Pack { args }`. Names should explain the
  payload at construction and at pattern matching sites.
- Language trees may expose their data directly; service objects and retained results
  should protect their invariants. Require the least information an operation needs.
  Code generation consumes verified LIR; the toolchain consumes an on-disk Ninja project
  and explicit build settings. Neither requires mutable compiler caches or rereads Resin sources.
- Treat `Source` as immutable named text, never implicitly as a file. Clones share a
  source version; a changed version is a new value. Logical identities are independent
  of diagnostic names. Loaders interpret imports and return cached instances when
  unchanged; filesystem paths, standard-library discovery, and editor buffers belong
  to `resin-source` and applications. `resin-compiler` depends on the concrete
  `resin_source::Loader`; do not add loader traits or callback adapters without a
  concrete need. The compiler has no overlays or file notifications.
  Resolve the import graph on every `compile()` before reusing a result, including when
  the entry is unchanged. Preserve old sources and compilations for their consumers.
- Prefer small functions with descriptive names, ideally fewer than ten lines of logic.
  Split by a meaningful operation, not an arbitrary line count. Exhaustive language
  dispatch and simple data definitions may be longer when that keeps the cases together.
- Use concrete representations and a few explicit passes rather than callback frameworks,
  hidden cross-pass state, or speculative abstractions. Explain invariants and non-obvious
  ownership choices close to the code that needs them.
- Use [the Bitwise taste guide](doc/bitwise.md) when designing and refactoring code.
  It explains the motivation through concrete Ion examples: visible data, operations
  that establish complete guarantees, and control flow that teaches the problem.
  Adapt that taste to Resin's behavior, explicit phase boundaries, and idiomatic Rust.

## Cost control

- Run automatic CI for pull requests and pushes on Linux only. Keep Windows and macOS
  support and tests; reserve their hosted runners for manually requested checks.
- Before publishing a release, manually run the `Build` workflow against the release
  candidate ref and require Linux, Windows, and macOS jobs to pass. Use Actions > Build >
  Run workflow, or `gh workflow run build.yml --ref <release-candidate-ref>`.
- GitHub Free includes 2,000 Actions minutes per month for private repositories, shared
  across the repository owner's account. At the September 2026 standard Linux x64 rate
  of $0.006/minute, 2,000 Linux minutes have about $12 of compute value.
- Budget roughly 1x for Linux, 2x for Windows, and 10x for macOS. These are rounded cost
  comparisons: current standard rates are $0.006, $0.010, and $0.062/minute respectively
  (about 1x, 1.67x, and 10.33x), not exact billing multipliers. Recheck
  [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
  when changing CI coverage or budgets.

## Development Practices

- Target 64-bit Linux, macOS, and Windows (MSVC with LLVM Clang for emitted C).
  Keep host builds independent of a Vulkan SDK or GPU. GPU execution still requires the
  runtime's Vulkan features; MoltenVK discovery does not imply full GPU compatibility.
- Keep the native C ABI in `crates/resin-runtime/` and language-facing modules in `resin/std/`.
  Add other Resin libraries as sibling directories under `resin/`.
  Examples import standard-library functionality through `$/std/` paths; imports without
  a leading `$` resolve relative to their importer. Each file has a private scope with
  explicit exports; do not reintroduce textual inclusion.
- Host entries take unit or `(int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)` for argc/argv/envp.
  Startup inputs are deep-copied before Resin entry and borrowed until process exit; treat
  them as read-only. Keep environment lookups on the supplied snapshot, not live OS state.
  `--` separates run arguments from compiler options; execution arguments stay out of build requests.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Keep `src/main.rs` as a wrapper around `resin::cli::main`; argument-to-`Mode` dispatch
  lives in `src/cli/`. Capture process settings through the platform toolchain's
  `Environment`, then choose CLI defaults and the build profile explicitly.
  Validate file selections and output destinations in the CLI. `Compiler::compile`
  consumes immutable sources and a loader to produce a `Compilation`. The separate
  `resin_codegen::generate` operation takes verified LIR and writes C, GLSL, and a Ninja
  build graph to disk in one call. C/GLSL ASTs and individual target emitters stay private.
  `resin-toolchain` stages that directory, configures native tools, and invokes Ninja.
  The graph compiles GLSL to SPIR-V, runs the same Resin binary with `--embed` to write
  aligned byte-array headers, then compiles C. Ninja owns ordering and incremental builds.
  Inspect cached intermediates or immutable `Compilation` results; do not add CLI inspection modes.
  Toolchain APIs consume explicit settings; execution is separate and retains the build-cache lock.
  Capture the Resin executable with `std::env::current_exe()` rather than resolving it on
  PATH. Do not run a blanket native-tool preflight: report failures when a build needs the
  tool. Document Ninja, a C compiler (`CC`/`--cc`), and `glslc` (`GLSLC`/`--glslc`) as installation
  requirements; `NINJA` selects the build runner. Embedding preserves arbitrary bytes and
  their exact logical length, including empty inputs, without appending a NUL.
- Without `-o`, host compilation uses the debug cache and runs the program. With `-o`,
  build and copy the optimized executable without running it.
- Every IR function reserves local zero for its parameter, including unit and tuple parameters
  and foreign declarations. The verifier rejects functions with no locals.
- Functions use `def`, nominal records use `struct`, transparent aliases use `type`, and local value bindings use `var`, including
  uninitialized locals. Record initializers and parameters do not take these keywords. Foreign functions use
  `extern "header.h" def name(...) -> Type;`.
- `impl` adds functions to the defining module's nominal type namespace; aliases retain
  that origin. `value.method(args)` supplies the receiver as the first argument,
  while `(value.field)(args)` calls a field value. Receiver parameter names are ordinary
  identifiers. Desugar method calls into ordinary functions before IR; method namespaces
  and module origins belong to frontend metadata. Compiler-provided methods use the
  same declaration lookup, argument checking, and editor analysis as source methods;
  register their signatures and intrinsic operations in `crates/resin-hir/src/lower/context.rs`.
  HIR construction recognizes `drop` as a hook; direct calls remain ordinary calls.
- Reading existing values performs compiler-defined copying. Function and type
  applications consume their argument results; operators do the same, and aggregate
  constructors consume their field initializers. `impl` defines inherent methods and
  `drop(self: Ptr<T>)` hooks. There is no static move checking or borrow checker.
  Native wrappers must make their copying safe or expose Arc-based ownership;
  `pointer.replace(replacement)` can disarm a native owner during deliberate transfer.
  See `doc/lifetimes.md` for lifecycle rules.
- String literals have primitive type `str`, distinct from `Span<ubyte>` and the owned
  nominal `String`. They expose `data` and `length` over static NUL-terminated bytes;
  length excludes the appended terminator. Literal storage may be shared; treat it as read-only.
  `Span<ubyte>(text)` explicitly borrows literal bytes; never implicitly convert a `str`
  to a span or construct a `str` from arbitrary bytes. `String.from_str(text)` copies a
  `str`, and `String.from_bytes(bytes)` copies a raw byte span. Both append a NUL outside
  their logical length. `fmt(format, arguments)` returns `String`, wrapping
  `Arc<Span<ubyte>>`; formatting and reference counting are host-only.
  `print(text)` and the ordinary `Io.stdout().write(text)` / `Io.stderr().write(text)` methods
  accept `str`, `Span<ubyte>`, and `String` and write bytes verbatim. Use `.data` when
  passing literal storage to C. Ordinary byte arrays contain exactly their declared
  elements, without a sentinel; nested array stride follows the packed shared layout.
  Empty C byte arrays reserve a placeholder byte that is outside the logical array.
  Device-backed byte arrays and spans use 8-bit storage; shader literals need an addressable
  constant-storage implementation and are currently rejected explicitly.
- Numeric suffixes are case insensitive: `b/h/i/l` select signed 8/16/32/64-bit integers,
  `ub/uh/ui/ul` select unsigned widths, and `f/d` select float32/float64. The formatter
  emits lowercase suffixes preceded by an underscore. Suffixes fix literal types and retain
  range checking. Hex accepts integer suffixes; signed `b` requires an underscore so bare
  `b/B` remains a digit, and hex `d/D/f/F` always remain digits. Unsuffixed numeric literals
  infer their type from context, defaulting to `long` for integers and `float64` for floats.
  One-armed `if` is equivalent to an explicit `else {}` and requires a unit-valued body.
- Function result annotations default to unit when omitted, including foreign functions.
  Explicit `_` holes opt into inference in local annotations and function results, including
  nested type positions. Keep parameters, type definitions, and foreign signatures fully explicit.
  Check source expressions into HIR, then lower that tree to LIR in a separate pass.
  Resolve dependency groups and all inference variables before handing the tree to lowering;
  keep inference solvers and deferred emission callbacks out of the lowering pass.
- Shader entries use `@compute_shader`, `@vertex_shader`, or `@fragment_shader` decorators.
  Compute entries take `(ulong, Ptr<T>)` and return unit; their index is the global X invocation
  index. Their signatures are checked at declaration; helpers need no decoration and remain host-callable.
  `function.spirv` requests embedded `Span<ubyte>` bytes from a decorated declaration, never
  from a runtime function alias. Keep shader definitions inline in examples.
- Arrays, `Span<T>`, and `str` provide indexing with `items.at(index)`, returning `Ptr<T>`
  (`Ptr<ubyte>` for `str`);
  its index parameter is `ulong`, with explicit conversions for other integer types.
  Use `items.at(index).*` to read or write. The earlier `items(index)` spelling remains supported.
  Bounds checking is not part of the indexing contract. Host indexing diagnoses invalid indices;
  shader indexing is unchecked, and callers must stay within valid storage.
  `Place<T>` is a compiler expression category, not a source type; pointer-returning user
  functions support the same access rules. Spans have `data` and `length` fields. Pointer arithmetic
  is forbidden; explicit pointer/`ulong` casts permit low-level byte arithmetic. The C ABI retains
  pointer/length pairs; language-facing pipeline creation accepts spans.
- Unions contain value types and use module-wide u32 type IDs, not variant positions.
  Type IDs index one canonical table of nominal, primitive, and structural definitions;
  host and shader emission share it. Union tags are the active payload's table index.
  `None` is a builtin singleton type and value; `T | None` expresses optionality. Postfix `!`
  removes `None` or traps, preserving the other members; it does not unwrap Results.
  `Result<T, E>` is first-class; `ok`/`err` construct it, exhaustive `match` handles it, and postfix
  `?` returns early on error. Error holes collect the least union of propagated errors (`Never`
  if empty). Keep mutable pointers invariant; implicit widening only copies union/Result values.
- Initialized owners are destroyed in reverse scope order on normal exit and `?`;
  preserve returned values before cleanup. Chain expressions with no tail yield unit.
  Standard-library wrappers return Results and keep integer-status C declarations private;
  public operations use static and instance methods on resource types. `RuntimeError` is a union of named status errors.
  Standard-library resource handles now retain shared owners and clean up automatically;
  do not register manual native destruction for them. `commands.submit()` and `commands.cancel()`
  clear the shared native handle; presentation returns `ok(false)` for skipped frames.
  Reference counting and custom destruction are host-only; shader consumption of
  managed values is rejected.
- Always commit and push completed changes to a task branch and open a pull request,
  including in future sessions. Do not push changes directly to `main`.
  Preserve unrelated local changes.
- Use the development environment in `shell.nix` on Linux/macOS for builds, tests, parser generation, and
  examples. Enter it with `nix-shell` from the repository root, or run a command non-interactively
  with `nix-shell --run 'cargo test --workspace --all-features'`. On Windows, use a Visual Studio
  developer PowerShell with Rustup, LLVM Clang, and CMake on PATH, as described in `README.md`.
- The shell supplies Rustup, a C compiler, CMake, Ninja, GLFW's native build dependencies, `glslc`,
  and, on Linux, Vulkan tools and libraries. On macOS, GPU execution uses the Vulkan SDK's loader
  and MoltenVK. Cargo builds and statically links GLFW via `glfw-sys`. Rustup uses
  `rust-toolchain.toml`. Keep the shell's library paths; do not hardcode Nix store paths.
- Use `RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1` when validating the full GPU path so missing
  dependencies do not silently skip tests. A working Vulkan driver is still required.
- Window tests also need a display (desktop or Xvfb) and `RESIN_REQUIRE_WINDOW=1` to prevent
  skips. For Xvfb, set `DISPLAY` and `XDG_SESSION_TYPE=x11`; unsetting `WAYLAND_DISPLAY`
  alone does not prevent GLFW from finding a default Wayland socket. Test actual window
  operations in subprocesses so they run on the process main thread, as GLFW requires.
  Run a shared Xvfb test display with `-noreset` so its last client disconnecting does not
  reset the server while another window test connects.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `crates/tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- The Tree-sitter grammar, generated parser, and Rust bindings live in `crates/tree-sitter-resin/`
  as ordinary files in this repository. Commit grammar changes and regenerated files together.
