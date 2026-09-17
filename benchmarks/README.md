# Benchmarks

See the [CPU and GPU benchmarks](../doc/benchmarks.md) chapter in the Resin Manual.

Compiler and cache benchmarks retain the large workloads formerly run by the
regression suite: 512 sequential conditional/error-propagation pairs in both C and
SPIR-V, a 17-file/1,041-declaration cold/warm/edit fixture, 10,000-entry cache maps,
and execution overlap/cancellation. Their correctness assertions still run with
each measurement. PR CI keeps 32-branch backend regressions and a small pass-reuse
fixture, alongside the cache and executor crates' ownership/concurrency tests.

```sh
cargo build --workspace --release --locked
RESIN_REQUIRE_SPIRV_TOOLS=1 cargo bench --bench compiler --locked -- --samples 5 --warmup 1 > compiler.json
```

This uses the pinned stable Rust toolchain and requires Ninja, a C compiler,
`spirv-opt`, and `spirv-val`; it does not need a GPU or display. `--branches N`
changes the synthetic workload size. Frontend, code generation, native compilation,
and host execution have separate timings. Each native sample uses fresh temporary
projects. Cache timings preserve their intended cold/warm/edit sequence. RSS is
process-wide and available only on Linux, so it is not a per-workload allocation
measurement.

The **Compiler benchmarks** GitHub Action runs on Linux nightly at **09:17 UTC**
and can be started through **Actions → Compiler benchmarks → Run workflow** (or
`gh workflow run benchmarks.yml --ref <branch>`). Scheduled runs start after the
workflow lands on the default branch. Each run adds median timings to the job
summary and retains JSON samples, commit information, and tool versions as an
artifact for 30 days. The job runs independently of pull-request checks; it has no
timing thresholds because hosted-runner performance varies.
