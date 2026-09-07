# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Development Practices

- Target 64-bit Linux, macOS, and Windows (MSVC with LLVM Clang for emitted C).
  Keep host builds independent of a Vulkan SDK or GPU. GPU execution still requires the
  runtime's Vulkan features; MoltenVK discovery does not imply full GPU compatibility.
- Keep the native C ABI in `resin-runtime/` and language-facing modules in `stdlib/`.
  Examples import standard-library functionality through `std/` paths. Each file has a private
  scope with explicit exports; do not reintroduce textual inclusion.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Keep `src/bin/resin.rs` as a wrapper around `cli::main`; argument-to-`Mode` dispatch
  lives in `src/cli/`, including environment/default and build-profile resolution.
  Construct validated requests with `compiler::Request::new`; `Session::compile` owns
  analysis and dispatches checked IR to the backend for native/C/GLSL/SPIR-V generation.
  Toolchain APIs consume explicit settings; execution is separate and retains the build-cache lock.
- Without `-o`, host compilation uses the debug cache and runs the program. With `-o`,
  build and copy the optimized executable without running it, even with `--output run`.
- Every IR function reserves local zero for its parameter, including unit and tuple parameters
  and foreign declarations. The verifier rejects functions with no locals.
- Functions use `def`, nominal records use `struct`, transparent aliases use `type`, and local value bindings use `var`, including
  uninitialized locals. Record initializers and parameters do not take these keywords. Foreign functions use
  `extern "header.h" def name(...) -> Type;`.
- Numeric suffixes are case-sensitive: `b/B`, `h/H`, `i/I`, and `l/L` select signed/unsigned
  8/16/32/64-bit integers; `f/d` select float32/float64. Suffixes fix literal types and retain
  range checking. Hex literals only accept suffixes that are not hex digits (`h/H`, `i/I`, `l/L`).
  One-armed `if` is equivalent to an explicit `else {}` and requires a unit-valued body.
- Function result annotations default to unit when omitted, including foreign functions.
  Explicit `_` holes opt into inference in local annotations and function results, including
  nested type positions. Keep parameters, type definitions, and foreign signatures fully explicit.
  Inference resolves dependency groups before IR lowering; never put inference variables in IR.
- Shader entries use `@compute_shader`, `@vertex_shader`, or `@fragment_shader` decorators.
  Their signatures are checked at declaration; helpers need no decoration and remain host-callable.
  `function.spirv` requests embedded `Span<ubyte>` bytes from a decorated declaration, never
  from a runtime function alias. Keep shader definitions inline in examples.
- Arrays and `Span<T>` use call syntax for checked indexing: `items(index)` returns `Ptr<T>`;
  use `items(index).*` to read or write. Spans have `data` and `length` fields. Pointer arithmetic
  is forbidden; explicit pointer/`ulong` casts permit low-level byte arithmetic. The C ABI retains
  pointer/length pairs; language-facing pipeline creation accepts spans.
- Unions contain nominal structs and use module-wide u32 type IDs, not variant positions.
  `Result<T, E>` is first-class; `ok`/`err` construct it, exhaustive `match` handles it, and postfix
  `?` returns early on error. Error holes collect the least union of propagated errors (`Never`
  if empty). Keep mutable pointers invariant; implicit widening only copies union/Result values.
- `defer expression;` is a chain-prefix statement that discards the deferred value. It runs in reverse
  order on normal exit and `?`. Bind names at registration, read values at exit, and preserve
  the returned value before cleanup. Deferred expressions cannot propagate with `?`.
  Chain expressions with no tail yield unit; `defer { ... };` uses ordinary block syntax.
  Standard-library wrappers return Results and keep integer-status C declarations private;
  public operation names omit `resin_`. `RuntimeError` is a union of named status errors.
  Register cleanup after successful acquisition. Submit/cancel take `&commands` and clear the
  consumed handle; presentation returns `ok(false)` for skipped frames. There is no automatic resource ownership.
- Commit and push completed changes directly to `main` by default, including in future
  sessions. Do not open a pull request unless asked. Preserve unrelated local changes.
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
  skips. Test actual window operations in subprocesses so they run
  on the process main thread, as GLFW requires.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- The Tree-sitter grammar, generated parser, and Rust bindings live in `tree-sitter-resin/`
  as ordinary files in this repository. Commit grammar changes and regenerated files together.
