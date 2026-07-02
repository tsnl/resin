# `resin`

[![Check](https://github.com/tsnl/resin/actions/workflows/check.yml/badge.svg)](https://github.com/tsnl/resin/actions/workflows/check.yml)

A metaprogramming experiment: a minimal **Rust** DSL for GPGPU (ML, graphics, and more).

Users compose a graph of nodes; symbolic transformations (autodiff, optimization) operate on the graph. Standard PyTorch-like ops cover elementwise, matmul, scatter-gather, reduce, etc.; the WGPU backend lowers graphs to WGSL and runs them on the GPU.

Design ethos: ultimate minimalism. Elegant abstractions.

Project goals and known gaps: [docs/GOAL.md](docs/GOAL.md).

## Crates

| Crate | Role |
| --- | --- |
| `resin-core` | Element types, accessors, `Tree` |
| `resin-dsl` | Identity-keyed `View` / `NodeRef` graph |
| `resin-ir` | In-place `IrProgram` building |
| `resin-grad` | Reverse-mode autodiff (`grad_wrt` / `grad_view`) |
| `resin-nn` | `Linear` / `Mlp`, losses, `sgd_tree` |
| `resin-dataset` | MNIST download and batching |
| `resin-jit-wgpu` | WGSL codegen, lowering, `PipelineFactory` / `Pipeline`, examples & GPU tests |

## Development setup

Prerequisites:

- [Rust toolchain](https://rustup.rs/) (stable)
- A GPU / WGPU-capable adapter (CI uses Mesa Vulkan soft GPU)

```sh
# Run the full check suite (tests, fmt, clippy):
make check

# Format:
make format

# GPU interpreter integration tests only:
cargo test -p resin-jit-wgpu --tests

# Examples:
cargo run -p resin-jit-wgpu --example demo_front
cargo run -p resin-jit-wgpu --example demo_interp
cargo run -p resin-jit-wgpu --example interp_add
cargo run -p resin-jit-wgpu --example train_mnist --release -- 3
```

MNIST data is cached under `~/.cache/resin/mnist` (override with `RESIN_MNIST_DIR`).

## References

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
