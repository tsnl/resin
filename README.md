# resin

A small, portable compiler for tensor programs: trace expression graphs in
Rust, differentiate them, and run them on the CPU or on the GPU via
WebGPU/WGSL. See [docs/GOAL.md](docs/GOAL.md) for the long-term vision.

```rust
use resin::{Tree, dsl::Tensor, jit::{DeviceArray, HostArray, CpuJit, Jit}};

#[derive(Tree)]
struct Pair<T> {
    a: T,
    b: T,
}

let add = CpuJit.jit(|p: &Pair<Tensor>| p.a.clone() + p.b.clone());
let out = add
    .call_host(&Pair {
        a: HostArray::from_f32(&[2], &[1.0, 2.0]),
        b: HostArray::from_f32(&[2], &[10.0, 20.0]),
    })?
    .host()?;
assert_eq!(out.data(), &[11.0, 22.0]);
```

## Arrays

| Type | Role |
| --- | --- |
| `HostArray` | Process memory — staging for IO, tests, dataloaders. |
| `Jit::Array` | Device-resident storage for a backend. On CPU this is `HostArray`; on wgpu it is a GPU buffer (`jit::Array`). |
| `Jit::upload` / `upload_tree` | Host → device (no GPU wait on wgpu). |
| `DeviceArray::host` / `scalar` | Device → host (syncs GPU on wgpu). |
| `JittedFn::call` | Runs with device arrays (keep weights resident). |
| `JittedFn::call_host` | Uploads a host tree, then calls (demos / all-host pipelines). |

## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Trace | `resin::dsl` | Build an immutable `Tensor` expression graph; reverse-mode autodiff (`grad_wrt`). |
| Lower | `resin::ir::lower` | Graph → `resin::ir::Program`: a queue of kernel dispatches over flat f32 buffers and strided views. Broadcast/transpose/squeeze are pitch tricks on accessors; strided sinks are densified up-front so optimize can see those copies. |
| Optimize | `resin::ir::optimize` | IR→IR passes (kernel fusion; tiling planned). |
| Run | `resin::jit` | `CpuJit` interprets the IR; `WgpuJit` emits one WGSL compute shader per kernel and dispatches through wgpu. |

Structured inputs/outputs are pytrees: `#[derive(Tree)]` on any struct or enum
with one type parameter, and the JIT maps leaves to buffers in declaration
order. Element types are f32 (default) and u32.

## Examples

```sh
cargo run --example interp_add                       # smallest end-to-end run
cargo run --example demo_front                       # DSL graph pretty-printer
cargo run --example train_mnist -- --quick 2         # MNIST MLP on the CPU
cargo run --example train_mnist -- --backend wgpu 2  # same, through WGSL
cargo bench --bench mnist_train                      # train-step: cpu/wgpu × opt on/off
```

`train_mnist` downloads MNIST to `~/.cache/resin/mnist` on first run
(override the shared root with `RESIN_DATA_DIR`).

The `mnist_train` Criterion bench times one full MLP train step (forward + MSE +
grads + SGD). Matrix: **backend** (`cpu` interpreter / `wgpu`) × **IR opt**
(none / `fuse_elementwise`). wgpu cells are skipped if no adapter is available.
Use `--save-baseline` / `--baseline` to track regressions.

## Layout

- `src/tree.rs`, `crates/resin-macros` — the `Tree` pytree trait and derive.
- `src/ops.rs` — scalar operator vocabulary shared by DSL and IR.
- `src/dsl/` — tensor graph, autodiff, debug printer.
- `src/ir/` — buffers, accessors, kernels, validation, optimization passes.
- `src/jit/` — the `Jit` trait, tracing/caching, DSL→IR lowering, and the
  `cpu` / `wgpu` backends (`wgpu` is a default feature; disable with
  `--no-default-features`).
