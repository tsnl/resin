# Cranelift host backend prototype

The server can emit host object code directly with Cranelift. Select it when
starting the service:

```sh
nix-shell --run 'cargo build --locked --release -p resin -p resin-server'
nix-shell --run 'target/release/resin-server --listen 127.0.0.1:8080 --host-backend cranelift'
```

In another shell, use the ordinary client:

```sh
export RESIN_SERVER=http://127.0.0.1:8080
target/release/resin example.resin -o example
```

`--host-backend c` is the default. Backend selection belongs to the server;
source upload, analysis, target selection, and artifact download use the existing
protocol. Builds still require the configured C compiler driver (`CC`) for linking.
The Cranelift path does not invoke a C frontend or Ninja.

## Supported subset

This is an opt-in experiment for scalar host programs:

- Zero-argument entries returning `int` or unit.
- Scalar integers, booleans, floats, unit, scalar pointers, and function values.
- Local storage, calls, specialized generic scalar functions, conditionals, and loops.
- Numeric operations and conversions with Resin's wrapping and trapping rules.

Aggregates, managed owners, C header/extern dependencies, and GPU operations are
unsupported. Unsupported constructs report a Cranelift error. They do not select
another backend automatically. Runtime traps terminate the generated process;
the prototype does not reproduce the C backend's runtime diagnostic text.

Objects target the server's host platform with baseline CPU features. This is
not a cross-compiler. Validation for this draft is on x86-64 Linux; macOS and
Windows execution still need platform validation.

## Ownership and reuse

`resin-codegen::generate_native` consumes verified LIR and returns immutable object
bytes. Each bounded worker owns its mutable Cranelift builders. The compiler knows
nothing about HTTP or server caches. Linking is a separate
`resin-toolchain::Toolchain::link_object` operation, using the existing asynchronous
process cancellation and temporary-file ownership.

The server caches completed objects using the complete LIR key, selected entry,
optimization level, and target. This permits reuse across identical uploads from
different client directories. Debug and release code generation have distinct
keys while sharing frontend results. The cache's entry capacity uses the server's
generated-output capacity.

Every request still links an independently owned executable, hashes it, and sends
it to the client. There is no executable cache on this path. Cancellation is
checked between functions and native process operations; an individual Cranelift
function compilation is not interruptible. Existing C/SPIR-V compilation remains
available through the default backend.

## Compare build latency

Use the same optimized binaries and fixture for both backends:

```sh
nix-shell --run 'python3 benchmarks/service.py --suite build --host-backend c --output build/compare-c'
nix-shell --run 'python3 benchmarks/service.py --suite build --host-backend cranelift --output build/compare-cranelift'
```

The harness validates downloaded executables and records all samples. See
[compiler service benchmarks](../benchmarks/service.md) for timing boundaries.
The fixtures exercise scalar compilation, not generated-program throughput or
full-language support. The C path links the runtime archive; this headerless
prototype does not, so artifact size and transfer work also differ.

### Recorded comparison — 2026-09-15

Both runs used optimized binaries from `3cfe98f4`, a clean checkout, the same
`shell.nix` toolchain, and an Intel Core Ultra 7 270K Plus on x86-64 Linux. Each
fixture has three cold samples, 24 warm samples, and 15 edited samples. C was
measured first, then Cranelift; these are local observations rather than a general
speedup guarantee. Raw samples: [C](../benchmarks/results/service-cranelift-c-linux-2026-09-15.json),
[Cranelift](../benchmarks/results/service-cranelift-native-linux-2026-09-15.json).

| Fixture / request | C median | Cranelift median |
| --- | ---: | ---: |
| One file, cold | 333.1 ms | 144.2 ms |
| One file, warm | 241.7 ms | 141.4 ms |
| One file, edited | 375.5 ms | 142.2 ms |
| 17 files, cold | 498.1 ms | 173.9 ms |
| 17 files, warm | 311.0 ms | 149.4 ms |
| 17 files, dependency edited | 532.4 ms | 181.5 ms |

For the 17-file fixture, warm p95 was 623.2 ms for C and 155.9 ms for Cranelift;
edited p95 was 740.6 ms and 187.6 ms respectively. Three cold trials are too few
to characterize tail latency. The executable was 6,360,776 bytes with C and
16,416 bytes with Cranelift. Runtime archive omission contributes to this
difference; it is not an executable-size comparison at feature parity.

These measurements establish end-to-end build latency for this subset. They do
not isolate object generation or attribute all saved time to external tools.
Warm Cranelift requests still include client acquisition, HTTP, cache selection,
the external linker, output hashing, and download.

## Validation

- Workspace formatting and Clippy (`--all-targets --all-features`, warnings denied).
- 1,166 unit/integration tests and 21 documentation tests across the workspace.
- Seventeen new tests cover C/Cranelift agreement in both optimization modes,
  unsupported constructs, traps, cancellation, object/executable ownership, and
  service cache reuse across edits, profiles, and client roots.
- Both benchmark runs validated every downloaded executable's result.

The initial concurrent test run and its retry hit `ETXTBSY` ("Text file busy") in
the existing `printing::c_runtime_accepts_empty_buffers_and_pointer_values` test.
All 21 printing tests then passed both serially and under concurrent syscall
tracing. The unchanged file-copy code already documents a related process-fork
descriptor-inheritance race; tracing did not capture a failing instance, so that
cause remains an inference. No copy/process refactor is included here. GPU/window
availability was not required for this prototype's validation.
