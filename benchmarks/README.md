# CPU and GPU benchmarks

This suite measures optimized Resin programs using dedicated workloads. Run CPU
and GPU benchmarks separately from the repository root:

```sh
nix-shell --run 'cargo bench --bench cpu'
nix-shell --run 'cargo bench --bench gpu'
```

The CPU target does not compile GLSL or initialize a GPU. The GPU target requires
`glslc` and a compatible Vulkan device with timestamp support; missing tools or
unsupported devices cause a failure rather than a skipped measurement. No window
or display is needed. On Windows, run the Cargo commands in the development
PowerShell described in the root [README](../README.md).

## Workloads

Each file in [workloads/](workloads/) defines one kernel shared by the host and
device backends. The CPU entry calls that kernel sequentially for every element;
the GPU dispatch assigns one element to each invocation.

| Name | Work | Default elements | Iterations per element |
| --- | --- | ---: | ---: |
| `branch_heavy` | Integer recurrence with a data-dependent branch inside a loop | 262,144 | 128 |
| `arithmetic` | Integer recurrence with a loop and no data-dependent branch | 262,144 | 128 |
| `stream` | Read a `uint`, multiply and add, then write a `uint` | 4,000,000 | 1 |

`branch_heavy` preserves the synthetic shader used to compare structured LIR with
the previous control-flow dispatcher. It stresses control flow and should not be
treated as a prediction of renderer performance. `arithmetic` provides a compute
comparison without the divergent branch, while `stream` performs little arithmetic
per byte accessed. These are compute benchmarks; graphics and presentation are
outside the current suite.

All arithmetic uses wrapping unsigned 32-bit integers. Counts and iteration limits
arrive through runtime parameters, and outputs are consumed by correctness checks.
The harness checks a hash of every CPU output against an independent Rust scalar
reference using an FNV-style hash over `uint` values. GPU outputs are compared element by element with that reference
before and after measurement. A mismatch fails the run.

## Selecting work and recording results

Cargo passes arguments after `--` to the selected benchmark:

```sh
nix-shell --run 'cargo bench --bench cpu -- --list'
nix-shell --run 'cargo bench --bench cpu -- --workload branch_heavy --workload arithmetic --samples 40 --warmup 10 --json cpu.json --label baseline'
nix-shell --run 'cargo bench --bench gpu -- --list-devices'
nix-shell --run 'cargo bench --bench gpu -- --device 0 --workload branch_heavy --samples 40 --warmup 10 --json gpu.json --label baseline'
```

| Option | Behavior |
| --- | --- |
| `--workload NAME` | Select an exact workload name from the table above; repeat to select several. By default, run all three. |
| `--size COUNT` | Override the element count for every selected workload. Accepts 1 through 4,194,240. |
| `--samples COUNT` | Number of measured samples; defaults to 30. |
| `--warmup COUNT` | Number of untimed warmup runs; defaults to 8. |
| `--json PATH` | Write metadata and raw timing samples for later comparison. |
| `--label TEXT` | Attach a descriptive label to the results. |
| `--list` | List available workloads. |
| `--device INDEX` | GPU only: select a device index from `--list-devices`. By default, use the first suitable device. |
| `--list-devices` | GPU only: list Vulkan devices. |

A short correctness and harness check can use a smaller input and fewer samples:

```sh
nix-shell --run 'cargo bench --bench cpu -- --size 65 --samples 2 --warmup 1'
nix-shell --run 'cargo bench --bench gpu -- --size 65 --samples 2 --warmup 1'
```

These short runs do not provide meaningful performance measurements. There are no
performance thresholds: CI correctness checks should not assert timing or imply
that shared runners produce comparable performance results.

## What the timings include

CPU samples time the compiled Resin workload in a single thread, inside its running
process, using a monotonic clock. Process startup, allocation, and checksum
calculation are outside the measured interval.

GPU samples use Vulkan timestamps around one dispatch, including runtime GPU barriers. CPU command submission and
waiting, pipeline creation, and output readback are outside that interval. The
results measure GPU execution rather than end-to-end application latency.

Both targets reuse allocations and perform warmups before collecting samples.
Repeated samples run with warm data and caches; `stream` is not a cold-memory or
host-to-device transfer benchmark. CPU and GPU timings have different boundaries
and should not be used alone to decide whether offloading an application pays off.

For before/after comparisons, use the same hardware, workload sources, input sizes,
warmup and sample settings, build profile, native compiler flags, and shader flags.
Build each revision in its own checkout, record a revision label, and retain the
JSON metadata and raw samples. Repeat runs and consider run-to-run variation before
attributing small differences to compiler changes. Keep other CPU and GPU work
quiet during measurement.

## Adding a workload

1. Add a dedicated Resin module under `workloads/`, exporting `Root`, `kernel`, and
   `cpu`. Preserve the common root layout: `count: uint`, `iterations: uint`,
   `input: Ptr<uint>`, and `output: Ptr<uint>`. The decorated compute `kernel`
   takes `(index: ulong, root: Ptr<Root>)`; `cpu` takes `Ptr<Root>` and invokes the
   same kernel for each element. Check the index against `count` before access.
2. Add the name to the CLI allowlist and `WORKLOADS` in [suite.rs](suite.rs), setting
   its default count and iteration count; implement its independent scalar reference
   in `expected()` there. Keep
   runtime inputs observable and define exact output semantics; do not use an
   example program as the benchmark implementation.
3. Run small CPU and GPU correctness checks, including a count that is not a
   multiple of the 64-invocation workgroup size. Document the workload and timing
   interpretation here, then collect ordinary-sized measurements separately.
