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

## Phase 1 — immutable caches and asynchronous passes

Validated in the same Linux development environment described above, using
`shell.nix`, Rust 1.96.0, 24 logical CPU threads, and the RTX 5090. Cargo artifacts
used `/dev/shm/resin-service-target`; native temporary files used
`/dev/shm/resin-service-tmp`. `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`, and `CARGO_INCREMENTAL=0` reduced storage use without
disabling debug assertions or overflow checks. Measurements below used unoptimized
Rust test builds and are observations from this machine, not throughput guarantees.

### Acceptance evidence

| Criterion | Evidence |
| --- | --- |
| P1.1 | `tests/async_pipeline.rs` calls CST, per-file AST, graph assembly, HIR, LIR, verification, and code generation through their public async APIs. CLI, native-toolchain, and runtime suites exercise local build/run behavior. Applications own pass sequencing. |
| P1.2 | Source and CST tests cover full-text/edit identity convergence and predecessor preservation. `cold_and_incremental_successors_match_without_changing_recovery_inputs` compares syntax, recovery, AST diagnostics, and formatting for cold and incremental successors. |
| P1.3 | Public API tests assert `Send + Sync` for completed source, syntax, AST, HIR, LIR, verified LIR, and generated-project values, and `Send` for pass futures. Concurrent cache, parser, LIR, and code-generation tests retain and inspect earlier results. |
| P1.4 | Source tests force a digest collision and preserve distinct logical modules. Graph/HIR tests cover reconstructed graphs, changed import bindings, and editor facts. LIR key tests cover canonical entries and options. Source, CLI, and LSP tests use independent physical checkouts to verify canonical cache reuse with separate diagnostic origins, including library paths and non-UTF-8 names. Native tests cover header content/search changes and tool identity. |
| P1.5 | `crates/resin-cache/tests/generations.rs` covers deduplicated misses, shared hits, refreshed recency, deterministic eviction despite completion order, zero/exact/overflow capacities, automatic overflow warnings, later shrinking, cheap rebase with non-`Clone` values, and unchanged predecessor maps. |
| P1.6 | Cache/executor tests exercise bounded overlapping builders, nested CPU work, responsive async I/O, queued/running cancellation, callback failures, panics, and dropped futures. Native concurrency tests cover staging-lock cancellation, process descendants, runtime shutdown, and holding execution capacity until owned work stops. |
| P1.7 | Codegen and native concurrency tests retain artifact A while B builds, retain outputs after inputs or staging slots disappear, and verify final-owner cleanup cannot remove another generation. Failed builds preserve published artifacts. |
| P1.8 | The measurements and defaults below record cold/unchanged/edit work, cache reuse, map costs, resident memory, execution overlap, and cooperative cancellation latency. |

### CPU passes and cache measurements

`tests/cache_performance.rs` passed all three measurement tests. Its compiler
fixture contains 17 sources and 1,041 declarations. Each request constructs its
source snapshot and explicitly updates CST, per-file AST, HIR, and verified-LIR
caches. The edit changes one literal in one dependency and reverses source
discovery order. These timings exclude filesystem acquisition, code generation,
and native tools.

| Request | Elapsed | Cumulative miss-builder calls: CST / AST / HIR / verified LIR |
| --- | ---: | ---: |
| Cold | 77.480 ms | 17 / 17 / 1 / 1 |
| Unchanged | 0.387 ms | 17 / 17 / 1 / 1 |
| One-file edit | 92.881 ms | 18 / 18 / 2 / 2 |

The unchanged request reused the same HIR handle. The edit rebuilt one file's
syntax and AST, then the graph's HIR and LIR. Its elapsed time exceeded the cold
request in this run; this measurement demonstrates reuse without claiming every
edit is faster. Process RSS was 21,848 KiB after the cold request and 28,772 KiB
with the cold, unchanged, and edited generations retained.

For a separate cache containing 10,000 entries with 1,024-byte payloads each
(10,240,000 payload bytes), cloning the map averaged **0.683 ms** over 100 clones.
A hit-only update requesting all entries averaged **5.472 ms** over 10 updates.
Process RSS changed from 15,580 KiB before allocation to 26,480 KiB afterward.
These RSS observations include allocator and process overhead; they are not
measurements of exact cache ownership. A weak-reference assertion separately
verifies that an evicted value remains alive through its old snapshot and is
released when that snapshot is dropped.

With an explicit four-job execution limit, four synthetic 40 ms CPU loops completed
in **40.211 ms**, with an observed peak of **four concurrent jobs**. Cancelling a
started loop that continuously checks its token, then awaiting `wait_idle()`, took
**9 µs**. This measures a cooperative checkpoint loop; it does not establish a
latency bound for a complete compiler pass, native process cleanup, or blocked
operating-system calls. Separate executor tests verify async I/O responsiveness.

### Native build measurements

Three separate CLI invocations built and ran a tiny host program in a fresh
temporary directory: first returning `42`, then the unchanged source, then an edit
returning `43`. These elapsed times include the local program run; tool invocation
counts are cumulative across the three requests.

| Request | Elapsed | C compilations | Preprocessor runs |
| --- | ---: | ---: | ---: |
| Cold | 348.704 ms | 1 | 1 |
| Unchanged | 160.166 ms | 1 | 2 |
| Source edit | 360.207 ms | 2 | 3 |

The unchanged request avoided another C compilation. Each request still ran the
preprocessor to capture current native inputs before deciding whether to rebuild.
Native RSS was not captured in this measurement.

### Defaults and retention limits

- `Execution::default()` uses `available_parallelism()`, falling back to one job;
  this machine reported 24. CPU work uses a bounded blocking-worker semaphore.
  Async cache builders have a separate concurrency bound of the same size, so a
  builder can await CPU work without consuming its CPU permit first.
- Each native build reserves one execution slot and invokes Ninja with `-j 1`.
  Independent builds can use different slots.
- The local LSP runtime has two async worker threads. Its retained cache capacities
  are 4,096 CST entries, 4,096 per-file AST entries, and 64 HIR graphs. Compiler
  libraries carry no hidden global cache.
- Capacity counts entries in a cache generation. All distinct requested entries
  survive an overflowing update and the update logs a warning. Old snapshots and
  completed outputs can retain their own dependencies beyond current membership;
  capacity therefore does not bound process RSS. There is no TTL or `prune()`.
- **Phase 2 retention work remains:** the local LSP's long-lived `Loader` still
  retains acquired file/origin records and cached text for previously visited paths.
  Closing an editor buffer clears its supplied text but does not remove that record.
  Cache capacities alone do not bound total LSP memory. Phase 2 must bound source
  acquisition lifetime while preserving open-buffer and symlink identities.

### Final validation status

- `cargo nextest run --workspace --all-features --test-threads 24 --no-fail-fast`:
  **1,064 passed, 0 skipped**, in 53.636 seconds after compilation. This includes
  the regression for upgrading legacy staged native metadata without clearing
  caches. Full GPU/window validation used the same explicit dependency flags and
  X11 settings recorded for Phase 0; particle tests retained their full workload.
- `cargo test --workspace --all-features --doc`: **20 passed**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- All 30 toolchain tests pass, including retained artifacts, process cancellation,
  captured-header races, header search changes, GCC/Clang behavior, and mixed raw
  and preprocessed C projects. Root benchmark and host/GPU/window tests pass.
- `git diff --check`: passed.

Toolchain Rust API cross-checks passed for Windows GNU and macOS ARM64. The macOS
cross-check used `blake3/pure` because the Linux shell's C compiler cannot build
Darwin NEON objects. These are compile checks, not platform execution. Windows and
macOS runtime checks remain manual under the repository's CI policy.
