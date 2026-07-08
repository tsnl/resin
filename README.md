# `resin`

A metaprogramming experiment: a minimal **Rust** DSL for GPGPU (ML, graphics, and more).

Users compose a graph of tensor ops in plain Rust; symbolic transformations
(autodiff, optimization) operate on the traced graph. Standard PyTorch-like ops
cover elementwise, matmul, reduce, gather/scatter, and views; backends lower
graphs to WGSL (via wgpu) or interpret them on the CPU.

Design ethos: ultimate minimalism. Elegant abstractions. **Composition over
custom kernels**: the op vocabulary stays small and general, and higher-level
capabilities (prefix sums, sorting, rasterization) are built by *composing*
those ops with ordinary Rust functions, not by dropping to hand-written WGSL.

Project goals and known gaps: [docs/GOAL.md](docs/GOAL.md).
Roadmap: [TODO.md](TODO.md).

## Crates

| Crate | Role |
| --- | --- |
| `resin-core` | Element types and operators, `Accessor` (strided views), `Tree` |
| `resin-dsl` | Identity-keyed `Tensor` graph, reverse-mode autodiff (`grad` / `grad_wrt`) |
| `resin-ir` | `IrProgram`: kernel queue over buffers/views, validation, optimization passes |
| `resin-jit` | DSL → IR lowering, `Jit` trait, compile cache; CPU interpreter and WGPU (WGSL) backends |
| `resin-macros` | `#[derive(Tree)]` for user parameter/output structs |
| `resin-dataset` | MNIST download and batching |

The root `resin` crate re-exports the public surface and hosts the examples.

## Development setup

Prerequisites:

- [Rust toolchain](https://rustup.rs/) (stable)
- For the `wgpu` backend: a GPU / WGPU-capable adapter (GPU tests skip
  gracefully when no adapter is present)

```sh
# Tests:
cargo test --workspace

# Lints / formatting:
cargo clippy --workspace --all-targets
cargo fmt --all

# Examples:
cargo run --example demo_front
cargo run --example interp_add
cargo run --example train_mnist --release
```

MNIST data is cached under a local cache directory (see
`examples/train_mnist.rs`).

## References

- [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
