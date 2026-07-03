# `resin`

[![Check](https://github.com/tsnl/resin/actions/workflows/check.yml/badge.svg)](https://github.com/tsnl/resin/actions/workflows/check.yml)

A metaprogramming experiment: a minimal **Rust** DSL for GPGPU (ML, graphics, and more).

Users compose a graph of nodes; symbolic transformations (autodiff, optimization) operate on the graph. Standard PyTorch-like ops cover elementwise, matmul, scatter-gather, reduce, etc. A JIT backend (in progress) will lower graphs to WGSL and run them on the GPU.

Design ethos: ultimate minimalism. Elegant abstractions.

Project goals and known gaps: [docs/GOAL.md](docs/GOAL.md).

## Crates

| Crate | Role |
| --- | --- |
| `resin-core` | Element types, accessors, `Tree` (host param trees) |
| `resin-derive` | `#[derive(Tree)]` proc-macro |
| `resin-front` | DSL graph (`dsl`), reverse-mode autodiff (`grad`), NN helpers (`nn`) |
| `resin-util` | Host utilities (`dataset`: MNIST download and batching) |

`resin-jit` (GPU lowering and execution) is not in the workspace yet; the previous `resin-ir` / `resin-jit-wgpu` crates were removed to make room for a fresh JIT design.

## Development setup

Prerequisites:

- [Rust toolchain](https://rustup.rs/) (stable)

```sh
# Run the full check suite (tests, fmt, clippy):
make check

# Format:
make format

# Host-only DSL demo (no GPU):
cargo run -p resin-front --example demo_front

# GPU examples stub until resin-jit lands (panic with TODO message):
cargo run -p resin-front --example demo_interp
cargo run -p resin-front --example interp_add
cargo run -p resin-front --example train_mnist
```

MNIST data is cached under `~/.cache/resin/mnist` (override with `RESIN_MNIST_DIR`) when using `resin_util::dataset`.

## References

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)