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
