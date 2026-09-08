# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Architecture and style

- Make the compiler educational to read. Prefer explicit data and direct control flow;
  a reader should be able to tell what a pass consumes, produces, and computes locally.
- Follow one direction: syntax → AST → HIR → LIR → verified LIR → C/GLSL → native tools.
  Put each language's definitions in `language.rs`, its incoming translation in `lower`,
  and its textual rendering in `print`. Keep module entry points short and navigable.
- HIR is a typed, desugared tree with resolved bindings, calls, and operations. Source
  scopes, method namespaces, inference variables, and recovery belong to HIR construction.
  LIR describes storage, cleanup, stack operations, and explicit control-flow blocks.
  Keep `crates/lir-verifier` independent; LIR definitions do not invoke their verifier.
- Keep common resolved types and layout rules in `crates/common/src/types`; they must not depend on a
  frontend, backend, or verifier. The compiler driver sequences passes and the toolchain
  owns external processes. Printers consume their own language, without reaching upstream.
- Each compiler phase is an unpublished workspace crate under `crates/`: `cst`, `ast`,
  `hir`, `lir`, `lir-verifier`, and `codegen`, with `common` for shared vocabulary.
  Use `publish = false` and local path dependencies. Keep dependencies acyclic and explicit;
  do not work around a boundary with public implementation modules or reverse dev-dependencies.
  Language nodes are public data. Solvers, scopes, builders, and traversal state stay private;
  expose a small set of lowering, printing, and query operations instead.
- Keep `common` narrow: source locations, diagnostics, concrete types, layout, and small
  utilities used by multiple phases. Do not move a phase's state there just to break a cycle.
  C and GLSL each have a target language, lowering, and printing inside `codegen`.
- Prefer small functions with descriptive names, ideally fewer than ten lines of logic.
  Split by a meaningful operation, not an arbitrary line count. Exhaustive language
  dispatch and simple data definitions may be longer when that keeps the cases together.
- Use concrete representations and a few explicit passes rather than callback frameworks,
  hidden cross-pass state, or speculative abstractions. Explain invariants and non-obvious
  ownership choices close to the code that needs them.
- Take inspiration from [Bitwise](https://github.com/pervognsen/bitwise): visible data,
  direct constructors, and code that teaches how the system works. Preserve Resin's
  behavior and idiomatic Rust; the reference is an ethos, not a mandate to copy C idioms.

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
- Keep the native C ABI in `resin-runtime/` and language-facing modules in `stdlib/`.
  Examples import standard-library functionality through `std/` paths. Each file has a private
  scope with explicit exports; do not reintroduce textual inclusion.
- Host entries take unit or `(int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)` for argc/argv/envp.
  Startup inputs are deep-copied before Resin entry and borrowed until process exit; treat
  them as read-only. Keep environment lookups on the supplied snapshot, not live OS state.
  `--` separates run arguments from compiler options; execution arguments stay out of build requests.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Keep `src/bin/resin.rs` as a wrapper around `cli::main`; argument-to-`Mode` dispatch
  lives in `src/cli/`, including environment/default and build-profile resolution.
  Construct validated requests with `compiler::Request::new`; `Session::compile` owns
  analysis and dispatches checked IR to the backend. Every compilation generates GLSL for
  requested shaders, compiles SPIR-V, embeds it in C, then builds an executable. Inspect
  cached intermediates or library snapshots; do not reintroduce artifact targets or CLI inspection modes.
  Toolchain APIs consume explicit settings; execution is separate and retains the build-cache lock.
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
  register their signatures and intrinsic operations in `crates/hir/src/lower/builtin_methods.rs`.
  HIR construction recognizes `drop` as a hook; direct calls remain ordinary calls.
- Reading existing values performs compiler-defined copying. Function and type
  applications consume their argument results; operators do the same, and aggregate
  constructors consume their field initializers. `impl` defines inherent methods and
  `drop(self: Ptr<T>)` hooks. There is no static move checking or borrow checker.
  Native wrappers must make their copying safe or expose Arc-based ownership;
  `pointer.replace(replacement)` can disarm a native owner during deliberate transfer.
  See `doc/lifetimes.md` for lifecycle rules.
- String literals are `Span<ubyte>` over static NUL-terminated bytes, with the terminator
  excluded from length. `fmt(format, arguments)` returns the builtin nominal `String`,
  wrapping `Arc<Span<ubyte>>`; formatting and reference counting are host-only.
  `print(text)` and the ordinary `Io.stdout().write(text)` / `Io.stderr().write(text)` methods
  write strings verbatim. Use `.data` when passing literal storage to C.
  Device-backed byte spans use 8-bit storage; shader literal spans need an addressable
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
  Their signatures are checked at declaration; helpers need no decoration and remain host-callable.
  `function.spirv` requests embedded `Span<ubyte>` bytes from a decorated declaration, never
  from a runtime function alias. Keep shader definitions inline in examples.
- Arrays and `Span<T>` provide indexing with `items.at(index)`, returning `Ptr<T>`;
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
- The shell supplies Rustup, a C compiler, CMake, GLFW's native build dependencies, `glslc`,
  and, on Linux, Vulkan tools and libraries. On macOS, GPU execution uses the Vulkan SDK's loader
  and MoltenVK. Cargo builds and statically links GLFW via `glfw-sys`. Rustup uses
  `rust-toolchain.toml`. Keep the shell's library paths; do not hardcode Nix store paths.
- Use `RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1` when validating the full GPU path so missing
  dependencies do not silently skip tests. A working Vulkan driver is still required.
- Window tests also need a display (desktop or Xvfb) and `RESIN_REQUIRE_WINDOW=1` to prevent
  skips. For Xvfb, set `DISPLAY` and `XDG_SESSION_TYPE=x11`; unsetting `WAYLAND_DISPLAY`
  alone does not prevent GLFW from finding a default Wayland socket. Test actual window
  operations in subprocesses so they run on the process main thread, as GLFW requires.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- The Tree-sitter grammar, generated parser, and Rust bindings live in `tree-sitter-resin/`
  as ordinary files in this repository. Commit grammar changes and regenerated files together.
