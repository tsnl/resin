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

## Phase 2 — shared local LSP caches and concurrent requests

Validated on Linux in the same `shell.nix`, Rust 1.96.0, 24-thread, RTX 5090, and
Xvfb environment as Phase 1. Cargo and native temporary files used the same memory
filesystem locations and debug-information settings. Compiler passes and native
builds run locally; separate CLI and LSP processes begin sharing a service in
Phase 3.

### Acceptance evidence

| Criterion | Evidence |
| --- | --- |
| P2.1 | Worker tests show one root reusing per-file results from two earlier callers and equivalent checkout graphs sharing HIR with separate navigation destinations. Build tests retain different contents for the same logical `main.resin` name and verify repeated builds reuse HIR, verified LIR, and generated-project handles. |
| P2.2 | Nine publication tests cover barrier-controlled disjoint/equal misses, a 16-task race across two runtime threads, a hit evicted before retry, current recency, and rejected stale history. Completed builders are not repeated; equal-key competitors return the published handle. The large-map race requires exactly one rebase. Native building occurs after cache selection, outside the retry loop. |
| P2.3 | Source snapshot tests verify release of old/closed text while preserving open physical identities. Worker tests query retained HIR after eviction and with reconstructed source identities. Build tests retain generated files through all-head eviction, then verify final-owner cleanup removes only the corresponding directory; copied executables remain runnable. The sustained-edit test releases every observed HIR after eviction and final-consumer release. |
| P2.4 | Protocol and executable tests cover coalesced rapid edits, close/reopen epochs, invalid code and repair, dependency edits, dirty buffers across saves, parent imports, missing-file creation/deletion, and alias changes. Prepared replies are rechecked against current revisions; reused wire IDs cannot receive cancelled requests' late replies, and diagnostic clearing tracks what the client actually received. |
| P2.5 | Protocol tests exercise admission saturation and cancellation without prematurely freeing queued capacity. Formatting completes while registration is blocked. A real LSP test holds native compilation while editor queries, independent cancellation, and shutdown remain responsive. Worker tests bound root/registration/diagnostic tasks and release accepted text after outstanding snapshots drop. |
| P2.6 | The publication and sustained-edit measurements below record map-copy/publication costs, a controlled CAS retry, active-task peaks, retained HIR handles, and reclamation after close, capacity eviction, and final release. |

### Publication measurements

The integrated `resin-lsp` publication tests used 10,000-entry maps. Ten hit-only
selections, including immutable cache construction, requested-handle selection,
and publication, averaged **9.499 ms** each; no miss builder ran.

A barrier-controlled race started two requests from the same 10,000-entry head.
Each requested the original keys and one different new key. Both completed in
**24.203 ms** combined, with **two original miss-builder calls**, **one rebase**,
and **10,002 retained entries**. The barrier and absence of further writers establish
the single retry. These are unoptimized local test timings, including ordinary-map
copy costs, rather than a bound on all requests or contention patterns.

### Sustained editing and retained results

The scheduler measurement opened four roots sharing one disk dependency, with
64 helper declarations per root. It then completed 100 edits in turn across those
roots, settling analysis after each revision. This measures completed successive
revisions; separate protocol tests exercise coalescing a rapid edit stream.

| Work | Elapsed |
| --- | ---: |
| Initial analysis of four roots | 6.694 ms |
| Unchanged disk refresh | 1.126 ms |
| 100 successive edits | 742.570 ms |

The unchanged refresh reused the original HIR handles. Observed peaks were **four
root tasks**, **one registration task**, and **one diagnostic task**. For this test,
source/CST/AST/HIR capacities were reduced to **16 / 16 / 16 / 8**; those were also
their final membership counts after editing.

Weak references tracked 104 distinct HIR results. Three selected old results were
deliberately retained by consumers throughout the eviction portion:

| Observation point | Tracked HIR values still alive |
| --- | ---: |
| After 100 edits | 11 |
| After closing all documents | 11 |
| After subsequent compilations caused capacity eviction | 3 |
| After releasing the final three consumers | 0 |

Closing documents removes active roots and supplied registrations; it does not
evict warm cache entries by itself. Subsequent cache updates evicted those entries,
and the retained consumers stayed usable until explicitly released. Weak-reference
counts establish ownership and reclamation, not process RSS or exact retained bytes.

### Defaults and current limits

- Application-owned atomic heads retain source, CST, and AST caches of 4,096 entries
  each, HIR and verified-LIR caches of 64 each, and a generated-project cache of 32.
  Publication uses safe `ArcSwap` CAS; published maps and values remain immutable.
- At most four root analyses run concurrently, with one registration task and one
  diagnostic preparation task. Request/reply channels and request admission have
  capacity 64. Accepted editor snapshots and prepared diagnostics use replaceable
  latest-value mailboxes. Execution retains Phase 1's CPU/native-job bounds.
- Each capture receives an independent loader containing current supplied sources
  and explicit bindings. Registration snapshots discard disk-only and closed history
  and retain no ancient cached text. Open epochs preserve physical identities through
  symlink changes; reopening acquires a new registration.
- Identical open aliases share their physical registration until the final alias
  closes. Conflicting text for the same physical source is rejected with both URI
  names. Reopening a retargeted path resolves it again while older aliases retain
  their captured identity.
- An initial in-flight root whose dependencies are not known yet may restart after
  an unrelated edit. Ready roots preserve their results across unrelated buffer
  edits. Disk/save/watch events conservatively invalidate all roots.
- A client that stops reading replies can apply transport backpressure. Responsive
  receive handling does not promise progress against a blocked output transport.
- Capacity counts entries, not bytes. Old consumers, dependency graphs, open buffers,
  and admitted work can retain additional memory. Oversized requested sets still
  survive with a warning. Atomic publication establishes neither a hard RSS bound
  nor a lock-free compiler. There is no periodic eviction task.

### Completed checks

- `cargo nextest run --workspace --all-features --test-threads 24 --no-fail-fast`:
  **1,091 passed, 0 skipped**, in **52.418 seconds** after compilation. GPU and window
  checks used the required-tool flags and X11 settings recorded for Phase 0.
- Final alias-ownership review added three worker regressions and corrected
  closing one of several open aliases. After that correction, the complete source
  and LSP suites passed together: **85 passed, 0 skipped**, in **2.274 seconds**.
  This includes the additional parent-import creation/deletion/alias test.
- Those final 85 tests comprise **41 source tests**, **9 publication tests**,
  **2 real build-handler tests**, **11 worker tests**, **6 protocol-state tests**,
  **1 text-position test**, and **15 executable LSP tests**. The executable tests
  include queries during blocked native work and cancellation that reaps children.
- `cargo test --workspace --all-features --doc`: **20 passed**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo fmt --all -- --check`: passed.
- `git diff --check`: passed.

These execution results are Linux-only. Windows/macOS hosted runtime checks remain
manual under the repository's CI policy.

## Phase 3: HTTP service and merged client

The root executable now delegates to `resin-client`. Its normal dependency graph
contains source/CST/execution/transport libraries only; semantic compiler libraries,
native tools, and shared cache heads belong to `resin-server`. `resin-protocol` is
strict wire data. Build and analyze handlers visibly sequence their compiler passes.

### Acceptance evidence

- **Connection and snapshots:** CLI tests cover build/run/LSP with absent, invalid,
  and unreachable `RESIN_SERVER`; formatting/embedding remain local. Acquisition
  tests cover parent/nested imports, cycles, aliases, missing-file repair, malformed
  bodies, percent-encoded local names, and immutable supplied dependency versions.
  Server tests prove omitted user uploads are never repaired from its filesystem.
- **Cross-client reuse:** the executable LSP regression
  `saved_editor_revision_is_a_cache_hit_for_an_independent_relocated_cli` analyzes
  an unsaved helper, saves those bytes, then runs a separate CLI against a relocated
  checkout. Source/CST/AST/HIR builder counts do not increase. Changing the imported
  file adds exactly one Source, CST, AST, and HIR computation. Native work remains
  independently selected by the explicit build contract.
- **Restart and editor correctness:** full/delta clients recover when a service
  restarts, including unchanged managed snapshots. The no-edit LSP restart test
  changes a managed function from `int` to `bool`; the unchanged user buffer gets
  fresh hover, diagnostics, and a new definition mirror. Older mirror bytes remain
  unchanged. Response validation rejects mismatched query kinds, invalid spans,
  and completion edits targeting a different document.
- **Native headers:** complete binary directory bundles retain nested/non-`.h`
  inputs, ordered include roots, source-scoped duplicate basenames, empty groups,
  and whole-bundle additions/changes/deletions. Runtime includes have explicit
  protected bindings. Contained symlinks flatten; escaping directory symlinks need
  a selected enclosing root. Portable device names and case aliases reject; exclusive
  staging also detects platform-specific filesystem aliases.
- **Preprocessing:** GCC and Clang tests validate actual compiler depfiles before
  compiling captured C. Ambient include-directory/absolute-file escapes reject even
  with spoofed `#line` text, and prior executable generations remain usable. This
  validation occurs after preprocessing; it is not an operating-system sandbox.
- **Ownership and concurrency:** HTTP admission includes response/error bodies.
  Tests cover DELETE before POST admission, malformed/incomplete bodies, shutdown,
  and connection loss. A real compiler PID is reaped after TCP disconnect without
  an explicit cancellation request and produces no artifact. An unread 32 MiB
  executable stream retains its complete length/hash while another build evicts its
  generated-cache entry. Phase 2's protocol freshness, independent request handling,
  cache publication, native cancellation, and source alias cases remain exercised.
- **Local outputs:** downloaded length/hash/revision/target/kind checks precede
  atomic destination replacement; failure preserves existing files. CLI tests cover
  local argv/env/cwd, existing run/build defaults, and service-owned tool overrides.
  Both executables' byte-embedding helpers preserve exact bytes and logical lengths.
- **Deployment:** the supplied Dockerfile built successfully with a rootless
  container engine. Its UID-10001 service compiled a host example without mounting
  a client workspace; the local client printed `fibonacci(10) = 55`. Both systemd
  templates passed `systemd-analyze verify` with their installation-specific
  executable path replaced by the built local binary; no unit was installed/started.
  The CI host smoke script starts an explicit service and runs the same example.

### Final validation

- `cargo nextest run --workspace --all-features --test-threads 24 --no-fail-fast`:
  **1,149 passed, 0 skipped**, in **53.019 seconds** after compilation. GPU/window
  tests used all required-tool flags, X11, and the full default particle workload.
- `cargo test --workspace --all-features --doc`: **20 passed**.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
- Explicit-service CI host smoke and Docker-hosted compilation/execution: passed.
- systemd unit syntax/dependencies: passed with the local executable path substituted.

The broad run includes the final restart, managed navigation, portable-path, native
lifecycle, download-integrity, and header-dependency regressions. Linux execution is
verified here; Windows/macOS hosted execution remains manual under the CI policy.
The Docker/systemd examples are manual deployment pathways, with no installer.
