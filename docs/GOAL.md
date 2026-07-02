# Resin goals

## Vision

Resin aims to be a modern, portable foundation for **scriptable GPU-accelerated
compute and graphics**—as easy to *compose* as PyTorch, as expressive as writing
shaders by hand, and not locked to CUDA.

### The problem

Libraries like **SDL2** made drawing in 2D delightfully simple: a small set of
ideas, immediate feedback, almost no plumbing. Modern 3D and “GPU content”
pipelines are more diverse than ever—bespoke lighting on meshes, Gaussian
rasterization, ML models for screen-space postprocessing, differentiable and
inverse rendering—yet the tooling split is painful:

- **Bespoke shaders + low-level APIs** (Vulkan, raw WebGPU) deliver full power
  and fixed-function hardware, but orchestration is verbose and brittle.
- **CUDA-centric stacks** (including much of the ML ecosystem) are productive
  but not platform-agnostic.
- **Specialized differentiable frameworks** (e.g. Dr.Jit / Mitsuba-class tools)
  pursue “write math, get gradients and a renderer,” but they are not always a
  general substrate for mixed ML + graphics pipelines, and they typically reach
  **fixed-function** hardware (raster, ROP, ray tracing) only through
  vendor-locked paths—e.g. ray tracing via CUDA/OptiX, which does drive NVIDIA’s
  RT Cores but is not portable—rather than as portable, WebGPU-class nodes.

### The promise

What if GPU work were composed like tensor programs—**Views, linear algebra,
reductions, gathers/scatters, autodiff**—and a compiler lowered that graph to
**WGSL** (portability via WebGPU/wgpu) or, later, **native Vulkan** (utmost
performance)? What if the *primitives* of “drawing” were not only blittable
surfaces, but **tensors and structured buffers**, with orchestration libraries
on top for each application domain?

That is Resin’s bet:

| Property | Intent |
| --- | --- |
| **Easy to compose** | PyTorch-shaped programming model: graphs of ops, modules as `ParamTree`s, reverse-mode autodiff where rules exist. |
| **Powerful** | Same expressiveness trajectory as hand-written WGSL/Vulkan *compute* (and, over time, optional fixed-function nodes). Compiler fusion and scheduling close the performance gap where possible. |
| **Portable** | Primary path: emit WGSL and run on native wgpu today; browser WebGPU is the portability target (packaging still open—see gaps). Later: direct Vulkan (or other) backends for performance-critical native apps. |
| **Versatile** | Tensors and linear algebra as the foundation for scriptable GPU pipelines—not a fixed 2D blit API. |

SDL2 is a useful *analogy for delight and minimal ceremony*, not a product
checklist. Resin does not replace windowing, input, or audio; it replaces the
**“I need a custom GPU pipeline and I refuse to write 2000 lines of glue”**
experience. Presentation (swapchain, windows) can live in a **helper library**—
think GLU relative to GL, or a thin layer over winit + wgpu—without bloating
the core DSL.

### Layering (core vs application)

The multi-stage core (DSL graph → IR → backend artifact → admit/run) is
**intentional**. Delightful orchestration is not the job of the lowest layer.

Application-facing helpers sit **on top** of the core:

- **ML:** today `sgd_tree` (a free function) applies an SGD step over a
  `ParamTree`; the intended object form is `SgdOptimizer`, later
  `AdamOptimizer`, `MuonOptimizer`, and a `Trainer`—all special cases of
  *minimize a scalar loss over a `ParamTree` of parameters*, with data
  iteration and logging as library code.
- **Graphics / inverse rendering (later):** session objects that bind cameras,
  scene tensors, and losses; still “compile a graph, run with bindings,” not a
  different language.

The core stays small: **identity-keyed graphs, ParamTrees, autodiff, lowering,
backend-agnostic `Interp`.** Framework ergonomics grow as libraries, not as a
monolithic runtime.

### Near-term product focus

- **Gaussians first** (structured buffers + compute), not a full mesh engine.
- Mesh lighting, classic vertex attributes, and fixed-function raster/RT are
  **follow-ons**. Prefer **general compute over tensors** for attribute and
  per-primitive work where it fits; add **hardware raster / ray-tracing nodes**
  when fixed-function is the right tool, not before.
- Custom user kernels and richer backends are on the roadmap, not day-one
  requirements for the vision to be coherent.

---

## Known gaps

Honest inventory of what the current stack does *not* yet provide relative to
the vision above. None of these invalidate the architecture; they bound what
demos and APIs should claim.

### Core vs orchestration

| Gap | Notes |
| --- | --- |
| No high-level `Trainer` / optimizer objects | Host loops still wire `admit`, `write_param`, `run`, `read_sink`, and optional `new_model` commits. Intended to be filled by ML helpers atop the DSL, not by bloating IR. |
| Multi-stage mental model | Users who only want “train a net” must learn graph build + compile + admit. Mitigated by domain libraries; the core remains a compiler. |

### Expressiveness (ops and graphs)

| Gap | Notes |
| --- | --- |
| Closed op set | Elementwise, matmul, reduce, remap/scatter-add, views—strong for ML-style compute, not a full shader language. |
| No user-defined WGSL/SPIR-V nodes yet | “As powerful as bespoke shaders” requires an extension path (opaque/custom kernels) or a much larger built-in library. |
| No hardware raster / RT nodes yet | Explicitly deferred; compute-on-tensors first (Gaussians). |
| Autodiff only for implemented reverse rules | New ops need VJPs; discontinuous rendering ops need special treatment later. |
| Scatter-accumulate path is F4-focused | Asserted for atomic u32↔f32 CAS; other dtypes need design work. |
| Indexing | Strided `narrow` / `index` with ranges; no negative slice steps while pitch is unsigned. |

### Types, packaging, and the “compiled function” idea

| Gap | Notes |
| --- | --- |
| `View` is a graph node + accessor, not a resident device tensor | Mental model is still “build a program,” not “array with values on GPU.” |
| Limited dtypes / weak static shapes | F4/F2/U4; shapes are runtime. Typing ladder and shape checks can return later. |
| Packaging / deploy story thin | Native wgpu works; browser WebGPU packaging, versioned artifacts, and language bindings are open. |

**Proposed direction:** the compiler returns a **higher-order callable**—a
bound program object that accepts a **`ParamTree` of GPU buffers** (and
similarly structured inputs) and returns a **`ParamTree` of GPU buffers**
(outputs / updated params).

This is the packaging and typing hinge, with the following properties:

1. **Shape of the tree is fixed at compile time.** The `ParamTree` structure
   (paths, leaf ranks/dtypes) must match what was registered when the graph was
   built—same contract as today’s named params/sinks, but structured.
2. **Leaves are buffer handles, not host `Vec<f32>`.** The callable binds
   already-admitted device storage (or uploads once into owned slots). That
   unifies “module weights,” “batch inputs,” and “outputs” under one tree API
   and removes ad hoc stringly `write_param("model.layers.0.weight")` from
   app code.
3. **It is the natural type of an admitted program.** Today: admit by value,
   then `Interp::param` / `sink` by name. Tomorrow: `let step = compile(graph);
   let outs = step(&inputs_tree);` where `step` owns the backend artifact and
   buffer table layout.
4. **It does not by itself fix static shapes or dtypes**—but it *localizes*
   them: the callable’s type (or a schema object) can carry leaf metadata
   (shape, `ElementType`, read-only vs mutable) for checking at bind time.
5. **In-place vs functional updates** must be chosen: optimizers may want
   mutable parameter leaves (write back into the same buffers) while pure
   inference returns new output leaves. Both fit a ParamTree API if mutability
   is part of the leaf policy.
6. **Host orchestration** (`Trainer`, data loaders, presentation) still sits
   outside: the HOF is the **GPU step**, not the full application.

This gives a compile-time fixed `ParamTree → ParamTree` (or
`&ParamTree → ParamTree` / in-place `&mut ParamTree`) object—a sound target
for packaging and a cleaner type boundary than free-floating buffer ids. It
aligns with existing `ParamTree`, `AdmitProgram`, and `Interp` directions
rather than replacing them.

### Backends and presentation (explicitly non-blocking for now)

| Gap | Notes |
| --- | --- |
| Single production backend (wgpu / WGSL) | Acceptable; Vulkan/native paths can follow once the IR and callable packaging stabilize. |
| No swapchain / window in core | Intended for a **presentation helper** (winit/wgpu glue), not the DSL crate. |
| Custom kernels, hardware raster, hardware RT | Roadmap after Gaussians/compute foundations; extension nodes, not a reason to stall the core. |

### Product narrative

| Gap | Notes |
| --- | --- |
| “PyTorch ease” is not full framework parity | No ecosystem of optimizers, data, distributed, etc.—by design at the core layer; grow via libraries. |
| Pure Rust surface today | Fine for systems work; optional language bindings can wrap the same compiled-function object later. |

---

## Summary

Resin’s goal is a **portable, tensor-native language for GPU programs**—easy to
compose, powerful enough for modern ML and compute-style graphics, with a path
to fixed-function and multi-backend performance. The core stays a minimal
compiler/runtime; **Trainers, optimizers, presentation, and scene APIs** are
libraries. **Gaussians and compute-on-tensors** lead; mesh and hardware
raster/RT follow. The medium-term packaging target is a **compiled higher-order
function over `ParamTree`s of GPU buffers**, which fits the current architecture
and addresses both ergonomics and a clearer type boundary for deployment.
