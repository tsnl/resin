# Compiler service benchmarks

These benchmarks measure the compiler and its clients, including source acquisition,
HTTP requests, editor diagnostics, and native builds. They are separate from the
[CPU/GPU workload benchmarks](README.md), which measure generated programs.

## Run

Build optimized client and server binaries, then run the Python standard-library
harness from the repository root in the development environment:

```sh
nix-shell --run 'cargo build --locked --release -p resin -p resin-server'
nix-shell --run 'python3 benchmarks/service.py --output build/service-benchmark'
```

The harness starts its own loopback service with a fresh temporary native cache for
each trial. It terminates the processes it starts. It never uses an existing
`RESIN_SERVER`. No GPU or display is required. Python 3 is required in addition to
the development toolchain. On Windows, use the development PowerShell described in
the [guide](../doc/guide.md#development).

The default binary directory is `$CARGO_TARGET_DIR/release`, or `target/release`.
Use `--bin-dir PATH` to select another build. `--suite http`, `--suite lsp`, and
`--suite build` run individual groups; the default runs all three. `--label TEXT`
records a label alongside the machine, revision, fixture, and sample metadata.

Results contain every sample, sample counts, median, and nearest-rank p95. A p95
computed from three cold trials is just the largest sample; increase `--trials`
before drawing conclusions about tail latency. The output directory also holds
fixtures, downloaded executables, and diagnostic logs. Keep it outside tracked
source directories or under `build/`.

For a quick correctness check of the harness:

```sh
nix-shell --run 'python3 benchmarks/service.py --trials 1 --samples 2 --edit-samples 2 --build-samples 2 --build-edit-samples 2 --output build/service-benchmark-smoke'
```

There are no timing thresholds and no automatic performance CI jobs. Keep other
builds and CPU workloads quiet while collecting samples.

## What is timed

The synthetic project has 17 source files, 1,041 declarations, and about 52 KiB of
source text. Sixteen helper files are imported by one entry file. Many declarations
are unused by the executable but still require frontend analysis. The build suite
also includes a one-file program. These fixtures establish repeatable workloads;
they do not represent every application or import topology.

| Measurement | Timed work |
| --- | --- |
| Service startup | Process spawn through the first successful capabilities response, including polling delay. |
| Cold HTTP analysis | First source upload and analysis in a fresh service. Excludes service startup and local dependency discovery. |
| Warm HTTP analysis, full | A complete source upload to an already analyzed graph. Includes JSON encoding, loopback transfer, server work, and JSON decoding. |
| Warm HTTP analysis, delta | An unchanged selection referring to a retained input handle. Still includes import metadata and server graph/cache work. |
| Edited HTTP analysis | A new literal in one dependency, submitted as a source replacement. Each edit is distinct and must finish before the next. |
| Concurrent warm analysis | Eight clients analyzing the same warm graph. This is a small hot-graph workload, not a multi-project scalability claim. |
| LSP open | `didOpen` through the matching diagnostic publication. Includes local acquisition and HTTP analysis; excludes LSP initialization. |
| LSP hover | A request through its matching reply after initial analysis completes. |
| LSP edit | A distinct entry-body edit through diagnostic publication for that exact version. Includes local acquisition and server analysis. |
| CLI build | A separate CLI process through publication of its downloaded executable with `-o`. Includes capabilities negotiation, local acquisition, upload, native work, and download. Excludes service startup and running the output. |

Both service binaries and parser benchmarks should use Rust release builds. CLI
`-o` requests optimized native output; this is independent of how the compiler
itself was built. The harness checks that analysis succeeds, hover returns a result,
diagnostics correspond to the requested version, and the final downloaded program
returns the expected value. Validation failures abort the run.

“Cold” means a fresh service and native cache. The OS page cache is not cleared.
HTTP timings use loopback, without TLS, proxying, or remote network latency. RSS,
when available on Linux, is a process observation after the recorded workload; it
includes allocator overhead and retained generations, and is not a cache byte budget.

## Isolate local parsing

```sh
nix-shell --run 'cargo bench --bench service-parse'
```

This uses the same default synthetic source shape, parses each complete source
without an incremental predecessor, and reads each CST preamble. All sources are
available in one bounded parallel batch. Timing includes source copies, worker
scheduling, and CST destruction; it excludes filesystem reads, dependency discovery
waves, header acquisition, HTTP, and server analysis.

Use `--samples COUNT` and `--warmup COUNT` after Cargo's `--` to control sampling.
To parse the exact fixture written by the HTTP harness instead:

```sh
nix-shell --run 'cargo bench --bench service-parse -- --directory build/service-benchmark/fixture --samples 30 --warmup 1'
```

This is an estimate of parsing work, not a measurement of the complete acquisition
step. It also does not establish how much a serialized-CST protocol would save:
that protocol would introduce encoding, transfer, decoding, and validation costs.

## Interpret comparisons

Use identical compiler profiles, fixtures, native toolchains, sample settings, and
hardware for before/after comparisons. Preserve raw samples and repeat runs; small
differences can reflect scheduling, filesystem state, and CPU frequency changes.
Do not compare these optimized HTTP timings directly with the earlier unoptimized
compiler-only measurements in [the implementation validation record](../doc/compiler-service-validation.md).

Current cache reuse is per-file for CST/AST and per complete source graph for HIR.
A source edit therefore still rebuilds whole-graph semantic analysis. Warm builds
still validate native inputs, preprocess C, invoke Ninja, retain files, and hash and
download the artifact. A cache hit does not yet reduce the complete build request
to an artifact lookup. Cache capacities count entries, and retained generations
can keep dependencies alive after eviction from the current cache.

## Recorded Linux run — 2026-09-15

[Raw samples and metadata](results/service-linux-2026-09-15.json) record optimized
compiler binaries from `9a84be2c`, running on an Intel Core Ultra 7 270K Plus with
24 logical CPUs, in `shell.nix`. The run used the default sample counts and a
loopback service. The file also identifies both benchmark sources by SHA-256.

All project rows below use the 17-file fixture. Cold rows contain only three
trials, so the table omits their p95 rather than suggesting a reliable tail estimate.

| Measurement | Samples | Median | p95 |
| --- | ---: | ---: | ---: |
| Full CST parsing and preamble extraction, isolated | 30 | 0.565 ms | 0.753 ms |
| Cold HTTP analysis | 3 | 29.233 ms | — |
| Warm HTTP analysis, full upload | 90 | 2.097 ms | 2.680 ms |
| Warm HTTP analysis, empty source delta | 90 | 2.404 ms | 2.999 ms |
| HTTP analysis after a dependency edit | 60 | 29.273 ms | 31.874 ms |
| LSP initial open to diagnostics | 3 | 30.532 ms | — |
| LSP warm hover | 90 | 2.489 ms | 4.612 ms |
| LSP entry edit to diagnostics | 60 | 45.466 ms | 48.325 ms |
| Cold CLI build and download | 3 | 412.464 ms | — |
| Unchanged CLI build and download | 24 | 223.914 ms | 379.787 ms |
| Edited CLI build and download | 15 | 446.950 ms | 654.490 ms |

The one-file program measured 334 ms cold, 199 ms unchanged, and 364 ms edited
at the median. Both native fixtures downloaded executables of about 6.1 MiB on
this host. Native timing variance was substantial; inspect the samples before
treating small differences as improvements.

After the LSP sequence (initial analysis, one warmup edit, and 20 measured edits),
service RSS was approximately 123 MiB and client RSS was 14–15 MiB. These observations
include retained semantic generations and do not establish a steady-state memory
bound. Eight concurrent clients completed 64 warm analyses at about 3,179 requests
per second in this short, shared-graph workload; this is not a cluster capacity estimate.

The clearest result is reuse of completed semantic analysis. A source edit still
costs roughly as much server analysis as the initial request. Local parallel parsing
is below one millisecond for this fixture, while the complete unchanged build still
costs hundreds of milliseconds. These results support investigating semantic
incrementality and native validation/artifact retention before changing the wire
protocol to carry CSTs.
