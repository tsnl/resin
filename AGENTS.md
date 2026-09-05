# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Development Practices

- Commit and push completed changes directly to `main` by default, including in future
  sessions. Do not open a pull request unless asked. Preserve unrelated local changes.
- Use the development environment in `shell.nix` for builds, tests, parser generation, and
  examples. Enter it with `nix-shell` from the repository root, or run a command non-interactively
  with `nix-shell --run 'cargo test --workspace --all-features'`.
- The shell supplies Rustup, a C compiler, `glslc`, and Vulkan tools and libraries. Rustup uses
  `rust-toolchain.toml`. Keep the shell's library paths; do not hardcode Nix store paths.
- Use `RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1` when validating the full GPU path so missing
  dependencies do not silently skip tests. A working Vulkan driver is still required.
- Window tests also need a display (desktop or Xvfb) and `RESIN_REQUIRE_WINDOW=1` to prevent
  skips. `shell.nix` supplies GLFW. Test actual window operations in subprocesses so they run
  on the process main thread, as GLFW requires.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- We use `tree-sitter` to generate a parser. The repo `tree-sitter-resin` is checked out as a 
  submodule in the root of the repo. Be careful to update both this and that repo if needed when you
  make changes.
