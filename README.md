# `resin`

[![Check](https://github.com/tsnl/resin/actions/workflows/check.yml/badge.svg)](https://github.com/tsnl/resin/actions/workflows/check.yml)

A metaprogramming experiment: a minimal Python DSL for GPGPU: ML, graphics, and more.

Users compose a graph of nodes; symbolic transformations (autodiff, optimization) operate on the graph. Custom WGSL nodes handle things like sorting (e.g. for 3D Gaussian Splatting), while standard PyTorch-like nodes cover elementwise ops, scatter-gather, reduce, etc.

Design ethos: ultimate minimalism. Elegant abstractions.

## References

-   [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
