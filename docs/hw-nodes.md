# Hardware nodes: ray tracing, rasterization, and the Vulkan backend

> Status: design + first implementation (draft PR). Stacked on the 3DGS phase 1
> primitives (#60): `U32`, `gather_rows` / `scatter_rows`, `index`-as-view,
> compare/select. Those are exactly the ops that let hardware node *outputs*
> stay minimal while shading stays in the differentiable graph.

## Why fixed-function nodes at all

`docs/GOAL.md` draws the line: prefer general compute over tensors, but "add
hardware raster / ray-tracing nodes when fixed-function is the right tool".
Ray-triangle traversal against an acceleration structure and depth-tested
triangle rasterization are the two places where GPUs offer dedicated hardware
(RT cores, ROPs) that a tensor program cannot approach by composition. The 3DGS
arc deliberately avoided custom kernels because scans and sorts *compose*;
BVH traversal and the depth test do not — they are the signal case the TODO
reserved: "if a real gap survives Phases 1–3, that's the signal to revisit."

The design keeps the composition philosophy intact by making the nodes
**fixed-function and discrete**: they output *visibility* (which primitive, at
what parameter, with what barycentrics), never *appearance*. Shading, material
lookup, interpolation, sampling, and accumulation remain ordinary graph ops —
differentiable where the math is differentiable, exactly like the 3DGS
renderer. This mirrors how modern engines split visibility buffers from
deferred shading, and how differentiable renderers treat visibility as
piecewise-constant (zero gradient) while appearance gradients flow.

## The two nodes

### `trace_rays`

```text
trace_rays(origins   : [N, 3] f32,
           directions: [N, 3] f32,
           t_min     : [N]    f32,
           t_max     : [N]    f32,
           vertices  : [V, 3] f32,
           triangles : [T, 3] u32) -> [N, 5] f32
```

Per ray, the closest-hit record `(t, u, v, prim, hit)`:

- `t`: ray parameter of the closest hit (`0` on miss),
- `u, v`: barycentric coordinates of the hit w.r.t. vertices 1 and 2
  (`w = 1 − u − v` for vertex 0),
- `prim`: triangle index as an exact integer-valued f32 (`0` on miss —
  in-bounds for gathers; mask with `hit`). Exact for `T < 2^24`.
- `hit`: `1.0` if the ray hit, else `0.0`.

Columns are extracted with `index` (a pure view), `prim` becomes gather keys
via `cast(U32)`, and surface attributes come from `gather_rows` on
`triangles` / `vertices` / material tables. The adjoint of the node is zero
(visibility is piecewise constant); gradients w.r.t. materials and shading
flow through the gathers, whose adjoints are scatter-adds (#60).

Both t interval tensors are per-ray so shadow/occlusion rays are expressible
(`t_max` = distance to light).

### `rasterize`

```text
rasterize(clip_positions: [V, 4] f32,
          triangles     : [T, 3] u32,
          height, width : usize) -> [H, W, 4] f32
```

A **visibility buffer**: per pixel `(prim, hit, u, v)` for the depth-nearest
triangle covering the pixel center, with perspective-correct barycentrics.
The "vertex shader" is graph code (positions are *clip-space* tensors — a
matmul away from model space), the "fragment shader" is graph code (gather +
interpolate + shade); only primitive assembly, clipping, coverage, and the
depth test are fixed-function. Depth at a pixel is reconstructable in-graph
from the gathered clip positions and `(u, v)`.

Conventions: pixel `(r, c)` samples NDC at
`x = (c + 0.5)/W · 2 − 1`, `y = 1 − (r + 0.5)/H · 2` (row 0 = top),
right-handed NDC depth `0..1`, depth test `<` (closer wins), no backface
culling. Primitives are clipped against `0 < w` and the NDC box by the
hardware; the CPU reference clips identically.

## IR

Two new kernels alongside `ElementwiseRpn / Matmul / Reduction / Remap`:

```text
IrKernel::TraceRays(IrTraceRaysKernel   { arg_accessors: 6, shape: [N, 5], … })
IrKernel::Rasterize(IrRasterizeKernel   { arg_accessors: 2, shape: [H, W, 4], … })
```

Validation pins the shapes/dtypes listed above. Hardware paths consume raw
device buffers, so lowering **densifies** any argument whose accessor is not
dense C-contiguous (an identity elementwise copy — the same trick as
materializing a view). The IR optimizer ignores non-elementwise kernels, so
fusion (#61) composes safely around these nodes.

## Backends and execution modes

| node        | cpu                    | wgpu                                   | vulkan                                 |
| ----------- | ---------------------- | -------------------------------------- | -------------------------------------- |
| trace_rays  | brute-force reference  | **HW ray query** (experimental wgpu RT) or **compute BVH fallback** | **HW ray query** (`VK_KHR_ray_query`) or the same compute BVH fallback |
| rasterize   | scanline reference     | render pipeline (RGBA32Float + D32)    | render pass (RGBA32Float + D32)        |

- **CPU** implementations are deliberately simple and serve as the golden
  oracle for parity tests.
- The **compute fallback** builds a BVH on the host (reading back geometry if
  it was produced on-device), uploads flat nodes, and traverses in a WGSL
  kernel — this is the "compute shader fallback" that runs on any adapter.
- **Hardware ray query**: wgpu's experimental acceleration-structure API
  (`EXPERIMENTAL_RAY_QUERY` + `EXPERIMENTAL_RAY_TRACING_ACCELERATION_STRUCTURE`)
  and, on the native Vulkan backend, `VK_KHR_acceleration_structure` +
  `VK_KHR_ray_query` driven through `ash`. There is no ray-tracing *pipeline*
  and therefore no shader binding table: with all shading in the graph, a
  ray query from a compute shader is the SBT-free formulation (per-primitive
  "shaders" are gathers over material tables — the same mask/gather style the
  3DGS renderer uses instead of compaction).
- Mode selection is automatic (hardware if the adapter/device offers it,
  else fallback) and overridable for testing.

## The Vulkan backend (`VulkanJit`)

Analogous to `WgpuJit`, one feature flag over: `resin-jit/vulkan` (pulls
`ash` + `naga`). The key economy: **it reuses the WGSL codegen** — the shared
emitter in `backends/wgsl` produces the same kernels the wgpu backend runs,
and `naga` translates WGSL → SPIR-V at lowering time. Ray-query WGSL
translates to `SPV_KHR_ray_query` (naga capability `RAY_QUERY`), so even the
hardware-RT kernel is written once in WGSL and shared by both GPU backends.

Runtime: one compute queue; per-dispatch descriptor sets (binding 0 = output,
1.. = args, matching the WGSL emitter); full memory barriers between
dispatches (correctness-first, like the naive wgpu runtime); host-visible
allocations (upload/readback simplicity; UMA on the lavapipe CI driver).
Rasterize records a render pass; trace_rays in hardware mode builds
BLAS/TLAS with `vkCmdBuildAccelerationStructuresKHR` per invoke (geometry is
graph data — it may change every call).

Everything is testable on CI without a GPU: Mesa lavapipe implements Vulkan
1.4 *including* `VK_KHR_ray_query`, both through wgpu and through ash.

## Domain library: `resin-render`

`crates/resin-render` composes the nodes into renderers, in the style of
`resin-gaussians` (#62):

- **Path tracer** (`pathtrace`): camera ray generation (tensor math), bounce
  loop unrolled in Rust (the host is the control language), `trace_rays` per
  bounce, hit shading = gathers over per-triangle materials (albedo /
  emission), cosine-hemisphere sampling driven by *input* random tensors
  (the graph stays pure), miss = environment term, Monte-Carlo accumulation
  as a mean over samples. Differentiable w.r.t. materials out of the box —
  gradients flow through shading, not visibility.
- **Deferred raster** (`raster`): `rasterize` → gather vertex ids →
  `gather_rows` attributes → barycentric interpolate → shade (e.g. Lambert).
- **References** (`reference`): scalar Rust path tracer and rasterizer with
  deliberately identical semantics for golden tests.

## Testing

- IR validation unit tests for both kernels (shape/dtype rules).
- Node parity: wgpu-fallback vs CPU, wgpu-HW vs CPU, vulkan(-HW/-fallback) vs
  CPU on randomized scenes (tolerances allow watertightness differences at
  triangle edges; raster parity ignores edge-adjacent pixels).
- Vulkan backend parity with CPU across the *existing* kernel suite
  (elementwise / matmul / reduction / gather / scatter).
- naga validation of every emitted WGSL kernel, including the ray-query
  kernel (validated with the `RAY_QUERY` capability) — no GPU required.
- `resin-render`: golden image vs reference on all backends; autodiff
  gradient flow to materials (finite-difference check); one SGD step on
  albedo decreases loss.
- Examples: `demo_pathtrace`, `demo_raster` (`--backend cpu|wgpu|vulkan|auto`),
  writing PPM.

## Known limitations (v1)

- Geometry read-back for the compute-fallback BVH build (and a full
  device-host sync) on every *trace step* — a multi-bounce graph pays it per
  bounce even though the geometry buffers are identical. Hardware AS builds
  stay on-device but are likewise rebuilt per step (no refit/update mode, no
  per-invoke cache yet). Caching by geometry-buffer identity inside one
  invoke is the natural follow-up.
- `prim` ids as integer-valued f32 in packed records caps primitives at 2^24
  (columns of a single output buffer must share a dtype; revisit with a
  multi-output story).
- Naive 1-D dispatches cap a single `trace_rays` at ~4.19M rays and the wgpu
  `rasterize` readback at ~4.19M pixels (2048×2048); both fail with a clear
  error rather than driver validation. Splitting across dispatch y is the
  fix when someone needs it.
- Out-of-range triangle indices clamp on the CPU and compute-fallback paths
  (gather semantics), but the hardware AS build consumes indices verbatim —
  out-of-range indices there are undefined behavior per the Vulkan spec.
  Keep indices in-bounds (the DSL builders can't check data-dependent
  values).
- No ray-tracing pipeline / SBT path: ray queries only. If per-hit *kernels*
  (not gathers) ever become necessary, that is a separate node family.
- Rasterization: no MSAA, no conservative raster, pixel-center sampling
  only; the CPU reference drops triangles touching `w ≤ 0` instead of
  clipping, so keep parity scenes in front of the camera.
- The two nodes are opaque to the IR optimizer (correct, just not fused).
- Mode override for testing (`RESIN_TRACE_FORCE_FALLBACK`) is process-global;
  a per-`Jit` constructor knob is the cleaner future shape.
