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
| **Easy to compose** | PyTorch-shaped programming model: graphs of ops, modules as `Tree`s, reverse-mode autodiff where rules exist. |
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

The stack is deliberately staged: **trace a graph, compile once, call many
times.** Orchestration (dataloaders, logging, presentation) is not the job of
the lowest layer.

| Crate | Role |
| --- | --- |
| `resin-core` | Element types, accessors, `Tree` |
| `resin-derive` | `#[derive(Tree)]` |
| `resin-front` | DSL graph (`dsl`), autodiff (`grad`), NN helpers (`nn`) |
| `resin-util` | Host utilities (`dataset`, …) |
| `resin-jit` *(planned)* | Lowering, optimization, WGSL codegen, GPU execution |

Application-facing helpers sit **on top** of `resin-front`:

- **ML:** today `sgd_tree` (a free function) applies an SGD step over a
  `Tree`; later `SgdOptimizer`, `AdamOptimizer`, `MuonOptimizer`, and a
  `Trainer`—all special cases of *minimize a scalar loss over a `Tree` of
  parameters*, with data iteration and logging as library code.
- **Graphics / inverse rendering (later):** scene/camera/loss session objects;
  still “compile a graph, call with `Tree<Buffer>`,” not a different language.

`resin-core` + `resin-front` stay small: **identity-keyed graphs, Trees,
autodiff.** `resin-jit` owns compilation and the callable runtime; framework
ergonomics grow as libraries, not as a monolithic runtime.

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
| No `resin-jit` yet | Legacy `resin-ir` / `resin-jit-wgpu` removed; GPU examples are TODO stubs until the new callable JIT lands. |
| No high-level `Trainer` / optimizer objects | Intended ML helpers atop `resin-front`, not inside the JIT. |
| Multi-stage mental model | Users who only want “train a net” must learn graph build + compile + call. Mitigated by domain libraries; the core remains a compiler. |

### Expressiveness (ops and graphs)

| Gap | Notes |
| --- | --- |
| Closed op set | Elementwise, matmul, reduce, remap/scatter-add, views—strong for ML-style compute, not a full shader language. |
| No user-defined WGSL/SPIR-V nodes yet | “As powerful as bespoke shaders” requires an extension path (opaque/custom kernels) or a much larger built-in library. |
| No hardware raster / RT nodes yet | Explicitly deferred; compute-on-tensors first (Gaussians). |
| Autodiff only for implemented reverse rules | New ops need VJPs; discontinuous rendering ops need special treatment later. |
| Scatter-accumulate path is F4-focused | Asserted for atomic u32↔f32 CAS; other dtypes need design work. |
| Indexing | Strided `narrow` / `index` with ranges; no negative slice steps while pitch is unsigned. |

### Types and packaging

| Gap | Notes |
| --- | --- |
| `View` is a graph node + accessor, not a resident device tensor | At JIT boundaries, leaves become `WgpuBuffer` handles; tracing stays on `View`. |
| Limited dtypes / weak static shapes | F4/F2/U4; shapes are runtime. Typing ladder and shape checks can return later. |
| Packaging / deploy story thin | Native wgpu first; browser WebGPU packaging, versioned artifacts, and language bindings are open. |

---

## JIT callable API (north star)

The DSL/tracing side is already JAX-shaped. The gap is **packaging**: JAX treats
compilation as **function specialization** (`jit(f)(*pytrees)`); resin should
expose the same contract with **`Tree<WgpuBuffer>`** leaves at call time.

### Mental model

```
JAX:     trace → compile → cache → call(pytree) → pytree
Resin:   trace → compile → call(Tree<Buffer>) → Tree<Buffer>
```

No user-managed `Pipeline` instances, `factory.create()`, `param_tree` /
`sink_tree`, `copy_tree`, `flush_binds`, or string-keyed `write_param` in
application code.

### User-facing API

```rust
// Once, at setup (cached by input/output Tree structure + leaf shapes/dtypes)
let train_step = compile(&TrainStepIn::schema(), &TrainStepOut::schema(), &session)?;

// Every step — caller-owned WgpuBuffer leaves, no factory/create/bind/flush
let out: TrainStepOut<WgpuBuffer> = train_step.call(&step_in)?;
```

| JAX | Resin target |
| --- | --- |
| `jit(f)(*pytrees)` | `compiled.call(&In::Map<Buffer>)` |
| Leaves are device arrays | Leaves are `WgpuBuffer` (optional subrange views) |
| Returns new pytree | Returns `Out::Map<WgpuBuffer>` |
| Buffer assignment hidden | `MemoryPlan` + arena layout hidden inside `call` |
| Async; block when needed | `call` submits; `block_until_ready()` / `CallFuture::wait()` when host needs values |

**Properties:**

1. **Tree shape is fixed at compile time.** Input/output `Tree` structure
   (paths, leaf ranks, dtypes) must match the schema registered when the graph
   was built.
2. **Leaves at the boundary are GPU buffers, not host `Vec<f32>`.** Unifies
   weights, batch inputs, and outputs under one tree API.
3. **Compile returns a callable, not an interpreter handle.** One object;
   invoke like a function.
4. **`IoSchema` per leaf** (compile-time metadata): `Param { path, shape,
   etype, mutable }` maps to input-tree leaves; `Sink { path, shape, etype }`
   for true outputs (`loss`, …). If `mutable`, the output aliases the input
   handle (in-place SGD).
5. **Host orchestration stays outside.** Dataloaders, logging, presentation
   are libraries; the callable is the **GPU step**.

### Stateless API, pooled memory (implementation)

**Stateless API** does not mean “allocate from the driver on every tensor.”
Distinguish what callers see from what the runtime caches:

```
┌─────────────────────────────────────────────────────────┐
│  CompiledFn (immutable, shareable, Arc)                 │
│  - WGSL + compute pipelines (created once)              │
│  - Static dispatch queue                                │
│  - MemoryPlan (arena peak size + slot offsets)          │
│  - IoSchema: In/Out Tree paths, shapes, mutability      │
└─────────────────────────────────────────────────────────┘
         │
         │  .call(session, &inputs)   — no user-visible Pipeline
         ▼
┌─────────────────────────────────────────────────────────┐
│  Per-call ephemeral state (stack/local)                   │
│  - arena Buffer (or bump from session pool)             │
│  - bind groups for this call’s buffer handles           │
│  - command encoder + submit → CallFuture                  │
└─────────────────────────────────────────────────────────┘
```

**Per-call buffer policy:**

| Kind | Policy |
| --- | --- |
| **Inputs** (params, minibatch) | **Borrow** caller’s `WgpuBuffer` handles; do not realloc. |
| **Intermediates** (forward/backward temps) | One **arena** per call; slots assigned at compile time via liveness / reuse (XLA-style `MemoryPlan`). WGSL binds use `BufferBinding { buffer: &arena, offset, size }`. |
| **Outputs** | **In-place** for mutable params (SGD writes `model.*` in the trace, not a separate `new_model` sink). **Fresh small buffers** for true sinks (`loss`, …). Returning arena subranges is not JAX-like (invalid after next call). |

**Efficiency notes:**

- One `create_buffer(peak_arena)` per call is acceptable; one buffer **per IR
  node** per call is not — compile-time liveness must collapse intermediates
  into the arena.
- A **session bump pool** (reuse one arena allocation, reset offset each call)
  keeps the API stateless while avoiding driver churn every batch.
- Bind groups may be rebuilt each `call` initially (~O(dispatches)); long-term,
  dynamic buffer offsets can reuse layouts.

**Async:** `call` submits work and returns a `CallFuture` holding output
handles. Host readback (`loss` scalar) is opt-in, once per epoch — not per
batch.

### Implicit buffer motion

Application code reads like a **pure graph language**: build `View` graphs,
`compile`, `call` with tree-shaped buffers. Uploads, binds, arena wiring,
and in-place param updates belong **inside** `call` — not as explicit
`write_buffer`, `copy_tree`, or string-keyed readbacks in training loops.

**Allowed exceptions** where explicit control may surface:

- Pipelining / overlap (`submit` without immediate `wait`)
- External systems (swapchain, filesystem, debug readback in tests)

### MNIST loop at the north star

```rust
let step = compile(&inputs, &outputs, …)?;
let mut model = init_model_buffers(&step, seed)?;

for epoch in 0..epochs {
    for batch in loader.batches(&dataset, epoch) {
        let out = step.call(&TrainStepIn { minibatch: batch, model: model.clone() })?;
        // With in-place SGD in the trace, model handles are unchanged; no new_model copy.
    }
    let loss = out.loss.read_f32().wait()?;  // once per epoch
}
```

### `resin-jit` implementation order

The author will implement `resin-jit` by hand. Suggested sequence (DSL
unchanged):

1. **`MemoryPlan`** in lowering — liveness + arena offsets; stop planning one
   device buffer per graph node at runtime.
2. **`CompiledFn::call`** — `Tree<Buffer>` I/O; replace user-visible pipeline
   instances.
3. **In-place param policy** in traces (`sgd_tree` writes `model.*`, not
   separate sinks).
4. **Session bump pool** (optional) — amortize arena alloc without exposing
   state.
5. **Dynamic bind offsets** — if bind-group rebuild shows up in profiles.

### Backends and presentation (explicitly non-blocking for now)

| Gap | Notes |
| --- | --- |
| Single production backend (wgpu / WGSL) | Acceptable; Vulkan/native paths can follow once `resin-jit` and callable packaging stabilize. |
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
to fixed-function and multi-backend performance. **`resin-core` + `resin-front`**
own tracing and autodiff; **`resin-jit`** (planned) owns lowering and a
**stateless JAX-style callable** — `compile` once, then
`call(Tree<WgpuBuffer>) → Tree<WgpuBuffer>` with compile-time `MemoryPlan`,
per-call arena allocation for intermediates, and in-place param updates where
the trace allows. **Trainers, optimizers, presentation, and scene APIs** are
libraries on top. **Gaussians and compute-on-tensors** lead; mesh and hardware
raster/RT follow.
