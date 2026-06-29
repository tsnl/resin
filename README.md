# `resin`

[![Check](https://github.com/tsnl/resin/actions/workflows/check.yml/badge.svg)](https://github.com/tsnl/resin/actions/workflows/check.yml)

A metaprogramming experiment: a minimal Python DSL for GPGPU: ML, graphics, and more.

Users compose a graph of nodes; symbolic transformations (autodiff, optimization) operate on the graph. Custom WGSL nodes handle things like sorting (e.g. for 3D Gaussian Splatting), while standard PyTorch-like nodes cover elementwise ops, scatter-gather, reduce, etc.

Design ethos: ultimate minimalism. Elegant abstractions.

## Development setup

Prerequisites:

- Python 3.14
- [uv](https://docs.astral.sh/uv/)
- Rust toolchain (`rustup`)

```sh
# Install Python dependencies:
uv sync --group dev

# `resin-rt-pybind` is a workspace member (see `pyproject.toml`), but `uv sync` only
# installs it as an editable package. The native PyO3 extension still needs to be
# compiled separately:
uv run --directory crates/resin-rt-pybind maturin develop

# Run checks:
uv run python scripts/check.py
```

## References

-   [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
