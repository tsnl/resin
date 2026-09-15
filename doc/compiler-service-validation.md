# Compiler service implementation validation

## Phase 0 — grouped extern preamble

Validated on Linux using `shell.nix`, Rust 1.96.0, 24 CPU threads, and an NVIDIA
GeForce RTX 5090. Xvfb supplied an X11 display with `-noreset`. Cargo build output
and native temporary files used a task-specific memory filesystem because disk
space was limited. Rust debug information and incremental compilation were disabled;
debug assertions and overflow checks remained enabled.

### Acceptance evidence

| Criterion | Evidence |
| --- | --- |
| P0.1 | Tree-sitter corpus and Rust grammar tests cover optional/empty/multiple groups, ordering, trailing commas, malformed input, old-syntax rejection, and opaque foreign types. |
| P0.2 | AST span tests, HIR foreign-signature tests, existing foreign/module/editor tests, and native imported-type/visibility tests pass. |
| P0.3 | Regenerated parser and node types; migrated libraries, examples, benchmark support, fixtures, editor outline queries, help, and guide. Rust, grammar, example, benchmark, and library formatting checks pass. |
| P0.4 | Eight CST preamble tests cover complete/recovering input and misplaced top-level clauses without collecting function-body strings. A review harness exercised 173 malformed inputs through CST → AST → HIR without panics. |
| P0.5 | Full host/GPU/window suite passes. Native regression tests rebuild after transitive header edits/deletions and reject a removed header declared by an empty group. |

### Completed checks

All Cargo commands ran inside `nix-shell` from the repository root.

- `cargo nextest run --workspace --all-features --test-threads 24 --no-fail-fast`:
  **984 passed, 0 skipped**, in 49.171 seconds after compilation. Set
  `RESIN_REQUIRE_SPIRV_TOOLS=1`, `RESIN_REQUIRE_GLSLC=1`, `RESIN_REQUIRE_GPU=1`,
  `RESIN_REQUIRE_WINDOW=1`, `DISPLAY` to the Xvfb display, and
  `XDG_SESSION_TYPE=x11`. The particle test retained its full default workload.
- `cargo test --workspace --all-features --doc`: **20 passed**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `resin --format --check examples benchmarks resin`: passed.
- Tree-sitter generation and corpus checks: **22 corpus cases passed**;
  grammar TypeScript, ESLint, and Prettier checks passed.
- `git diff --check`: passed.

The full suite includes focused parser, CST, AST, HIR, LIR, codegen, language-server,
native-toolchain, and runtime tests. Windows/macOS hosted checks remain manual under
the repository's CI policy; this validation record claims Linux execution only.
