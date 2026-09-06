# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Development Practices

- Target 64-bit Linux with native Vulkan. macOS/MoltenVK support is on hold.
- Keep the native C ABI in `resin-runtime/` and language-facing modules in `stdlib/`.
  Examples import standard-library functionality through `std/` paths. Each file has a private
  scope with explicit exports; do not reintroduce textual inclusion.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Functions use `def`, nominal types use `type`, and local value bindings use `var`, including
  uninitialized locals. Record initializers and parameters do not take these keywords. Foreign functions use
  `extern "header.h" def name(...) -> Type;`.
- Function result annotations default to unit when omitted, including foreign functions.
  Keep parameter types and function types explicit; do not infer return types from bodies.
- Commit and push completed changes directly to `main` by default, including in future
  sessions. Do not open a pull request unless asked. Preserve unrelated local changes.
- Use the development environment in `shell.nix` for builds, tests, parser generation, and
  examples. Enter it with `nix-shell` from the repository root, or run a command non-interactively
  with `nix-shell --run 'cargo test -p resin --test correctness'`.
- The shell supplies Rustup, a C compiler, CMake, GLFW's native build dependencies, `glslc`,
  and Vulkan tools and libraries. Cargo builds and statically links GLFW via `glfw-sys`. Rustup uses
  `rust-toolchain.toml`. Keep the shell's library paths; do not hardcode Nix store paths.
- Use `RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1` when validating the full GPU path so missing
  dependencies do not silently skip tests. A working Vulkan driver is still required.
- Do not run window-spawning tests or windowed examples unless explicitly requested by the user.
  Full-suite runs must skip `windows_present_resize_and_release_resources`,
  `resin_window_example_uses_bundled_glfw`, and `resin_particles_example_computes_and_presents`.
  Headless GPU tests are fine.
- When explicitly requested, window tests need a display and `RESIN_REQUIRE_WINDOW=1` to prevent
  skips. Run actual window operations in subprocesses on the process main thread, as GLFW requires.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- The Tree-sitter grammar, generated parser, and Rust bindings live in `tree-sitter-resin/`
  as ordinary files in this repository. Commit grammar changes and regenerated files together.
