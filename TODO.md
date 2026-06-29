# TODO

## Completed foundation

- [x] Static feed-forward graph DSL (`Node` + `View` / `Accessor`).
- [x] IR lowering, WGSL codegen, and `WgpuInterp` GPU runtime.
- [x] Autodiff for elementwise, matmul, reduction, and affine scatter.
- [x] MNIST training demo (end-to-end).

> [!NOTE]
>
> GPU-side loops add a lot of complexity. Keep it feed-forward only.

---

## Project: 3DGS

**Goal:** build a 3DGS training and rendering pipeline in Resin.

**Data:** `deps/gaussian_splatting` — [Voxel51/gaussian_splatting](https://huggingface.co/datasets/Voxel51/gaussian_splatting) (git submodule; PLY scenes). *(Add as submodule locally if not present.)*

### Phase 1: basic compiler and language features

- [x] **PR 1.1:** bitwise ops; float ↔ int (`floor` / `ceil` / `bitcast` as needed).
- [x] **PR 1.2:** `IrRemapKernel` / `RemapNode` + `df_do`; GPU tests.

### Phase 2: custom kernels, multi-output ports, DCE, **radix sort subgraph**

- [x] **PR 2.1:** multi-output `Node` / `View.port` / `grad_port` / `CustomNode` protocol.
- [x] **PR 2.2:** `PrefixSumNode` (+ adjoint).
- [x] **PR 2.3:** **`View.sort(max_chunk=...)`** inlines a **radix-sort subgraph**
  (sortable keys → 4× hist / **`PrefixSumNode`** / scatter → **remap gather** for
  values). End users only call `.sort()`; digit helpers live in
  `resin.dsl.radix_subgraph` (stdlib, not app-authored CustomNodes).
  `max_chunk` reserved for future segmented policies.
- [x] **PR 2.4:** port-level DCE from sink reachability.

### Phase 3: 3DGS renderer (`src/resin/lib/gaussians`)

- [x] **PR 3.1–3.4:** linalg, preprocess, untiled blend + grad smoke.
- [x] **PR 3.x (intermezzo):** pygame-ce viewer, pixel-space EWA, dense gnomen, HiDPI.
- [x] **PR 3.5:** tiling — **GPU binning** in `GpuForwardSession`:
  library `TileCount` → builtin **`prefix_sum`** → library `TileFill` → builtin
  **`.sort()` radix subgraph** → **remap gather** → library `TileRanges` →
  `TiledGaussianBlendNode`. Host `build_tiled_layout` kept for CPU parity tests only.
  Prefer builtins; CustomNodes only for irregular expand/range (library-owned).

### Next performance attacks (priority order)

1. **Avoid full-image `read_sink` every frame** (present without host readback).
2. **PLY loader** from `deps/gaussian_splatting` → pressure-test large N.
3. **Parallel hist/scatter** digit passes (workgroup atomics) for huge instance lists.
4. **Shared-memory per-tile blend** (paper-style cooperative load).
5. **Training demo** (colors/opacities SGD); 3D params; tiled grad.

### Training (beyond 3.4 smoke)

- [ ] Demo training step: SGD on **colors / opacities** via untiled `grad`.
- [ ] Backprop into **3D** params (means, scales, quats).
- [ ] Tiled **grad** parity with untiled.
