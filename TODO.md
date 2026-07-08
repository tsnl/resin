# TODO

## Completed foundation

- [x] Static feed-forward tensor graph DSL (`Tensor` / `TensorKind`, identity-keyed).
- [x] Reverse-mode autodiff over the traced graph (`grad` / `grad_wrt`).
- [x] DSL → IR lowering (kernels write buffers; views are accessors), IR validation.
- [x] CPU interpreter backend; naive WGPU backend (IR → WGSL).
- [x] `Jit` trait + compile cache; MNIST training demo (end-to-end).

> [!NOTE]
>
> GPU-side loops add a lot of complexity. Keep the graph feed-forward only:
> host Rust is the control language, the graph is straight-line.

---

## Project: 3DGS

**Goal:** build a 3DGS rendering (and later training) pipeline in Resin,
composed from general tensor ops — the way you'd write it in PyTorch — with
**zero domain-specific kernels**.

### Design stance: composition over custom kernels

The previous plan (see `archive/main-v13`) reached 3DGS through custom WGSL
nodes: `PrefixSumNode`, `SortNode`, `GaussianBlendNode`, plus multi-output
port machinery and per-port DCE to support them. That works, but every custom
kernel is a hole in the abstraction: opaque to autodiff, opaque to the
optimizer, and a maintenance liability.

This plan replaces all of it with three layers:

1. **Primitive breadth (Phase 1).** A slightly wider vocabulary of *general*
   ops — integer dtype, bitwise/compare/select, gather/scatter — each with a
   straightforward lowering and adjoint. These are reusable everywhere, not
   3DGS-specific.
2. **Combinators (Phase 2).** Higher-order Rust functions that *trace* graphs:
   `scan` takes an ordinary Rust closure for its binary operator and unrolls
   log-depth passes of existing elementwise ops; radix `argsort` is built from
   `scan` + `gather`/`scatter`. Because combinators expand into ordinary
   differentiable nodes, autodiff works through them for free, and sink-rooted
   lowering gives dead-code elimination for free (nothing unreachable is ever
   lowered — no per-port DCE machinery needed). Multi-output "ports" are
   unnecessary: a combinator just returns several `Tensor`s.
3. **Domain library (Phase 3).** 3DGS as a library crate of pure graph
   functions over the DSL. The renderer is data flow: project → sort by depth
   → accumulate transmittance (a `scan`!) → blend (a reduction).

Performance follows from *structure*, not hand-tuning: strided views make
slicing/broadcast free, one kernel per materializing node keeps dispatch
counts O(ops), and the fusion pass collapses elementwise chains into single
RPN kernels. Target is "right order of magnitude" (~80% of a hand-fused
pipeline), with headroom recovered later inside the compiler — invisibly to
user code.

### Phase 1: primitive breadth (dtypes, integer ops, gather/scatter)

- [x] **1.1 U32 end-to-end.** `ElementType::U32` in the DSL (IR already has
  `U4`); `constant_u32` / `iota`; typed CPU interpreter (element-kind aware
  RPN evaluation instead of f32-only).
- [x] **1.2 Element ops.** Bitwise (`&`, `|`, `^`, `<<`, `>>`), comparisons
  (`eq/ne/lt/le/gt/ge` → 0/1 mask), `min`/`max`, `floor`/`sqrt`, value cast
  (`f32 ↔ u32`) and `bitcast`, library-level `select(mask, a, b)`. CPU + WGSL
  execution for all; adjoints where meaningful (compare/floor: zero grad).
- [x] **1.3 Slicing as views.** `Tensor::index` (slice/single keys) lowers to
  a pure accessor view (offset arithmetic, no kernel); `scatter_index`
  (pad/embed into a larger zero tensor) lowers to a scatter-view kernel.
- [x] **1.4 Row gather/scatter.** `gather_rows(src, idx)` (`out[i,…] =
  src[idx[i],…]`) and `scatter_rows(src, idx, len, op)` (`out[idx[i],…] ⊕=
  src[i,…]`, `⊕ ∈ {write, add, min, max}`) lowering to `IrRemapKernel`;
  execution on CPU and WGSL (u32 `atomicAdd`, f32 CAS loop for add);
  adjoints: gather ↔ scatter-add.

### Phase 2: combinators, not kernels

- [x] **2.1 `scan`.** Higher-order inclusive/exclusive scan along an axis:
  `scan(x, axis, identity, |a, b| …)` traces the closure per step and unrolls
  Hillis–Steele (`log₂ n` shift + combine passes of existing ops). `cumsum` /
  `cumprod` wrappers. Autodiff through `scan` needs no new rules — verify with
  gradient tests.
- [x] **2.2 `argsort` / `sort_by_key`.** LSD radix sort composed from
  bit-extract + `scan` + `scatter_rows` (stable 1-bit split per pass,
  configurable key width); `float_sort_key` (order-preserving f32 → u32
  bijection via bitcast + sign trick). Sorted *values* stay differentiable
  through `gather_rows`; the permutation itself is integer data.
- [x] **2.3 Elementwise fusion pass.** IR optimization: fuse single-consumer
  elementwise chains into one RPN kernel (the RPN kernel already takes N args
  and an arbitrary expression). Keeps combinator-generated graphs at a sane
  dispatch count without changing their semantics.

### Phase 3: 3DGS renderer (library crate)

- [x] **3.1 Linalg helpers.** Batched quaternion → rotation, `scale_rot_to_cov3d`,
  view/projection application, symmetric 2×2 inverse — all `[N, …]` tensor
  ops built from Phase 1 primitives.
- [x] **3.2 `preprocess_gaussians`.** 3D means/scales/rotations → 2D means,
  depths, conics, radii + validity mask (masking replaces culling: shapes stay
  static, invalid gaussians contribute zero alpha).
- [x] **3.3 Forward render (untiled).** Depth `argsort` → `gather_rows` all
  attributes → per-pixel alpha `[N, H, W]` from conics → transmittance
  `T = exclusive_scan(*, 1-α)` → image = `Σᵢ colorᵢ · αᵢ · Tᵢ` (reduction).
  Golden tests against a scalar CPU reference; PPM demo example.
- [x] **3.4 Backward + training smoke.** No new code paths — `grad_wrt`
  through the whole renderer (scan/gather/scatter adjoints compose). MSE
  reconstruction loss, a few optimization steps on a tiny scene, loss
  decreases.
- [x] **3.5 Tiling.** `render_tiled`: fixed-capacity instance duplication
  (`iota` div/mod + gathers), stable radix sort by tile id over
  `⌈log₂(T+1)⌉` bits (depth order preserved within tiles), per-tile
  segment starts from a prefix sum, capacity-padded `[T·C]` gather lists,
  `[T, C, th, tw]` blend, constant-permutation assembly. One pure traced
  graph — differentiable end-to-end, zero custom kernels. Caps are the
  static-shape trade (D truncates giant spans, C drops the deepest
  overflow); per-frame adaptive caps via host readback + compile-cache
  bucketing are the tracked follow-up. Benchmarked ~5× faster and ~5.5×
  smaller than dense at N=1024, 64² on the CPU backend.

### Phase 4: real checkpoints, performance, training loop

- [x] **4.1 Camera as data.** `render_view` / `preprocess_view` take `[4, 4]`
  view/proj *tensors*; compile once, move the camera every call (basis for
  the viewer and multi-view training). Fixed the inherited projection
  convention (`w = +depth`, screen y down) in both tensor and reference
  paths.
- [x] **4.2 Checkpoints.** Minimal 3DGS PLY loader (INRIA layout, SH DC only)
  with standard activations; `data/hf/dylanebert-3dgs` HuggingFace submodule
  (shallow; `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1`);
  parity tests render the real `luigi` checkpoint.
- [x] **4.3 Chunked renderer.** Host-side fold over depth-sorted chunks
  carrying `(image, transmittance)` — compositing is associative, so this is
  exact. Peak memory `O(K·H·W)` instead of `O(N·H·W)`; renders full
  checkpoints. Forward-only (the fold crosses program boundaries).
- [x] **4.4 Benchmarks.** `bench_3dgs` example: dispatch counts and buffer
  bytes of optimized programs (backend-independent) + first/steady wall
  times for scan, sort, dense render, render+grad, chunked checkpoint;
  `--backend wgpu` on GPU machines.
- [x] **4.5 Training path.** `RawGaussianCloud` (log-scale / logit
  parameterization) + `activate()`; the backward pass is `grad_wrt` on the
  forward — no renderer adjoint code. Multi-view SGD test converges ~100×;
  `train_3dgs` example.
- [x] **4.6 Viewer.** `resin-viewer` (winit + softbuffer): drag-orbit /
  scroll-zoom over the compiled camera-as-parameter renderer; `--offscreen`
  headless mode; `--ply` loads checkpoints.

### Deferred / non-goals for this arc

- Spherical harmonics (flat RGB only until the pipeline is solid).
- GPU-side dynamic compaction (validity masks instead; static shapes).
- Hand-written WGSL escape hatch (`CustomNode`): explicitly *not* part of this
  arc. If a real gap survives Phases 1–3, that's the signal to revisit.
