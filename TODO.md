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

### Phase 1: basic compiler and language features

- [x] **PR 1.1:** bitwise ops; float ↔ int (`floor` / `ceil` / `bitcast` as needed).
- [x] **PR 1.2:** `IrRemapKernel` — accessor + indices keys, gather/scatter direction;
  per-operand etypes (`f4` payload, `u4` indices); accumulate operator for scatter-add;
  migrate `ScatterNode` / `IrScatterKernel`; `RemapNode` + `df_do` adjoint rules; GPU tests.

### Phase 2: custom kernels, multi-output ports, DCE

- [x] **PR 2.1:** multi-output `Node` (dense buffer per port); `View` carries node + output key;
  multi-output IR / `WgpuProgram` schema; `grad_port[(node, port)]`;
  `df_do(node, df_douts: dict[port, View])` composes backward graph nodes;
  `CustomNode` protocol (forward + `df_do` hook).
- [x] **PR 2.2:** `PrefixSumNode` custom kernel (+ `df_do` / grad adjoint).
- [x] **PR 2.3:** `SortNode` custom kernel, two output ports (sorted values, permutation);
  `df_do` adjoint via remap scatter-reduce (not opaque WGSL-only grad).
- [x] **PR 2.4:** codegen DCE per output port — forward reachability from sinks;
  elide unused stores in kernels (e.g. sorted values when only perm is needed).

### Phase 3: 3DGS renderer (`src/resin/lib/gaussians`)

- [x] **PR 3.1:** quaternion and projection linalg helpers.
- [x] **PR 3.2:** `preprocess_gaussians()` — 3D pos/scale/rot → 2D mean, depth, conic,
  opacity, color (no SH).
- [x] **PR 3.3:** `GaussianBlendNode` (untiled, global sort); save forward tape
  (`final_T`, `n_contrib`).
- [x] **PR 3.4:** `GradGaussianBlendNode` via `df_do`; reconstruction loss smoke;
  golden forward/backward GPU tests.
- [ ] **PR 3.5:** tiling — `tile_counts`, prefix-sum offsets, `DuplicateWithKeys` (TBD:
  custom vs DSL), tile+depth sort, `identify_tile_ranges`, tiled blend/grad;
  parity vs untiled (3.3–3.4).