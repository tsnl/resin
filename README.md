# `resin`

> **CONCLUSION**
>
> A highly successful experiment, but needs more time in the oven.
>
> See `TODO.md`.

A metaprogramming experiment: a minimal Python DSL for GPGPU: ML, graphics, and more.

Users compose a graph of nodes; symbolic transformations (autodiff, optimization) operate on the graph. Custom WGSL nodes handle things like sorting (e.g. for 3D Gaussian Splatting), while standard PyTorch-like nodes cover elementwise ops, scatter-gather, reduce, etc.

Design ethos: ultimate minimalism. Elegant abstractions first, clean abstractions are straightforward to optimize.

```bash
$ ./setup-repo.py
```

## References

-   [PyTorch Internals](https://blog.ezyang.com/2019/05/pytorch-internals/)
