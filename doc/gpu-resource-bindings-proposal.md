# Proposal: explicit GPU resource bindings

**Status:** Direction adopted; initial typed-buffer implementation available.

**Date:** 2026-09-17.

**Baseline:** `main` at `ac500e9888b3c6112ee15404c42ddc74510c8b5f`.

This document records the direction selected in the design discussion: make GPU
resources explicit, bind them through typed resource structs, and represent
relationships in shared buffer data with arrays, indices, and offsets rather than
interchangeable CPU/GPU pointers. The motivation is predictable boundary costs,
portability, and simpler resource semantics. Exact syntax and implementation
mechanisms remain open.

The [typed resource bindings guide](resource-bindings.md) documents the implemented
API and its executable CPU/GPU example. It supplies read-only and writable buffer
views, one borrowed resource record, bounds behavior, and recording guarantees.
The initial lowering uses Vulkan device addresses; descriptor-based lowering and
the broader resource kinds described below remain future work. Names and syntax
in the design pseudocode below are illustrative; use the guide for current APIs.

## Decision

Adopt a typed **resource bundle** as the portable shader interface. Its type
declares the resources a shader can access; its host instance supplies the actual
resource views and small value parameters at dispatch or draw. The compiler derives
the binding layout and binding operations from this contract. Start with one bundle
per entry point, separate from invocation built-ins and inter-stage values.
Multiple logical groups can be considered later.

GPU allocations and views should be explicit GPU resources, not ordinary host
`Ptr<T>` values. Buffer elements contain portable value data, including indices and
offsets, but not host pointers or opaque resource handles. Ordinary pointer-free
structs remain shareable. This is not a proposal to remove pointers from CPU code
or all references from shader code.

Preserve sharing of ordinary functions and compatible value types between CPU and
GPU. **Relax the requirement that a GPU shader entry point itself must also be an
ordinary host-callable function.** A CPU adapter and a GPU adapter may share the
actual algorithm without sharing an entry-point ABI.

The language concept is a resource bundle, not a literal Vulkan descriptor set.
Descriptor sets are an initial backend implementation, not a requirement imposed on
every target.

## Where Resin is today

The current model already imposes important restrictions:

- Host `GpuPtr<T>` and `GpuSpan<T>` are owning views over an opaque `GpuView`.
  Shader code instead uses `Ptr<T>` and `Span<T>`.
- Typed pipelines retain the shader root type. Recording projects a matching host
  launch record into a shader root, translates GPU views, preserves subrange
  offsets, and retains referenced allocations.
- Projection handles the launch record, including nested value records. It
  **does not recursively traverse objects reached through buffer contents**.
  GPU buffer elements already reject pointers, spans, managed owners, and drop hooks.
- The current Vulkan profile requires buffer device addresses and 64-bit shader
  integers. Decorated shader entries are currently host-callable.

See [GPU buffers](gpu-buffers.md), [shader requirements](shaders.md#gpu-requirements),
and the [native GPU ABI](../crates/resin-runtime/include/resin_runtime/gpu.h).
The device-address representation should not be conflated with a descriptor array
that emulates every pointer access. In particular, this proposal does not claim
that current Resin performs recursive pointer-graph relocation on every launch.

The ergonomic problem is real nonetheless: host and shader records use different
pointer/view families, while the pointer-shaped interface suggests stronger
interchangeability. Generics can share operations but do not, by themselves,
eliminate the representation boundary.

## Alternative considered: one virtual pointer space

The alternative was to replace host `GpuPtr<T>` values with ordinary `Ptr<T>`
values drawn from canonical host virtual-address ranges:

1. A host-visible allocation supplies a mapped CPU address range.
2. A device-only allocation reserves an inaccessible range of the corresponding
   size, such as a `PROT_NONE` reservation on a suitable host OS.
3. The GPU object maintains an interval map from ranges to allocation/buffer
   metadata. A lookup recovers the allocation and byte offset, including interior
   pointers and subranges.
4. At a CPU/GPU boundary, the runtime translates that identity and offset into
   a device address or a buffer binding plus offset.

This can make explicit launch arguments convenient. The interval map is translation
metadata, however, not shared hardware virtual addressing. It does not make host
pointer bits meaningful when a shader loads them from another buffer.

### Vulkan mapping is not the blocker

Vulkan permits host-visible memory to remain mapped during GPU execution; dispatch
does not require unmapping. Mapping belongs to `VkDeviceMemory`, so suballocated
buffers normally share an allocator-managed mapping. Mapping the same memory object
again while it is already mapped is invalid.

Access still requires synchronization. Non-coherent host writes require flushing;
reading device writes requires completion, visibility, and invalidation operations.
Coherent memory removes explicit cache-maintenance calls, not ordering requirements.
Unmapping does not flush non-coherent writes. It ends access through the host mapping,
not the buffer's device-memory binding. An ordinary remap does not guarantee the same
CPU address. See the [Vulkan memory specification][vulkan-memory].

The proposed pointer scheme therefore needs policies for mapping lifetime, address
stability, allocation reuse, page granularity, and stale pointers. An OS fault on
accidental CPU access is not ownership or synchronization. Even a stable host
mapping does not establish CPU/device address equality.

### The decisive gotcha: pointers embedded inside data

Consider this conceptual structure:

```text
struct Node {
    value: f32,
    next: Ptr<Node>,
}
```

Translating the root locates the first node. It does not translate the `next` value
stored inside that node. GPU traversal requires every embedded pointer it follows
to have a GPU-meaningful representation. Trees, pointer tables, and nested spans
have the same issue; ordinary pointer-free structs do not.

General relocation must discover reachable objects, identify pointer fields,
translate targets, and preserve aliases, cycles, nulls, and interior offsets.
Raw pointers alone do not describe allocation lengths or dynamic object types, so
this needs restrictions or explicit metadata. A visited-object map handles cycles
and sharing; naive recursive traversal is insufficient.

Patching shared bytes in place leaves the CPU with device pointer values. Separate
host/device representations avoid that conflict but require storage, fixup writes,
and a consistency policy. GPU changes to links can require reverse translation
when returning a usable graph to the CPU.

### Why this is a performance concern

For `P` pointer fields and `A` registered allocations, independently resolving each
address through a balanced interval map entails approximately `O(P log A)` lookup
work, in addition to graph discovery, fixups, bookkeeping, and any copies. GPU-side
relocation is possible, but consumes GPU work and adds scheduling dependencies.
These are algorithmic considerations, not measurements of Resin.

Rewalking or repatching a large changing graph at every boundary would make a small
dispatch depend on application-data size and topology. Avoiding that hidden cost in
rendering, simulation, and training loops is a principal reason for this decision.

**Relocation is not necessarily required on every dispatch.** An immutable graph
with stable allocations can be relocated once. Dirty tracking or explicit fixup
lists can restrict subsequent work. Scalar payload changes do not inherently
require relocating unchanged links. A deliberately GPU-native pointer graph can
remain on the GPU. These are valid alternatives, but require additional machinery
or weaken the transparent ordinary-pointer contract.

A software pointer encoded as allocation ID plus offset could instead be decoded
at shader access. That avoids rewriting pointer bits, but retains a resource table,
translation and lifetime rules, and potential per-access indirection. It is not an
ordinary native CPU pointer that happens to work everywhere.

The selected direction avoids requiring either deep relocation or general software
address translation for the portable baseline. It does not claim that descriptors
universally outperform native GPU pointers, or that today's bounded launch-record
projection is intrinsically expensive.

## Proposed programming model

### Resource identity is separate from buffer contents

Names in this table are illustrative, not final API commitments:

| Concept | Host meaning | Shader meaning |
| --- | --- | --- |
| `GpuBuffer<T>` | Owns a typed GPU allocation | Not an ordinary shader value |
| `Buffer<T, Read>` | Retained read-only binding view with a range | Indexed reads through a declared binding |
| `Buffer<T, ReadWrite>` | Retained writable binding view with a range | Indexed reads and writes through a declared binding |
| Resource struct | Typed binding recipe plus value parameters | Compiler-provided resource interface |
| Plain record or array | Value data with a checked transfer layout | Data interpreted under that layout |

A resource struct is **not a byte-compatible struct of CPU handles to upload**.
Resource fields lower to bindings; small value fields use an explicit constants
channel. Host ownership bookkeeping does not execute in shaders. Constructing and
copying bundles is initially a host operation; shaders use declared bindings, not
arbitrary first-class resource objects stored in memory.

Projection still exists, but is bounded, schema-directed binding and parameter
encoding. It must not follow pointers or resource references in buffer contents.

### One declaration and one bundle at the call site

The following is design pseudocode, not accepted Resin syntax or a tested program.
The decorator, resource types, access modes, and allocation/view method names are
provisional. It illustrates the interface, not an implementation commitment.

```text
struct Particle {
    x: f32,
    velocity: f32,
}

// Shared value-level helper: usable by CPU and GPU adapters.
fn advance(p: Particle, dt: f32) -> Particle {
    Particle { x = p.x + p.velocity * dt, velocity = p.velocity }
}

@resources
struct StepBindings {
    particles: Buffer<Particle, ReadWrite>,
    dt: f32,
}

@compute_shader
fn step(index: u64, resources: StepBindings) {
    if (index < resources.particles.length) {
        let p = resources.particles:load(index);
        resources.particles:store(index, advance(p, resources.dt));
    };
}

// Host code. Initialize the storage before dispatch; omitted here.
let particles = gpu:alloc_buffer::<Particle>(count)?;
let arguments = StepBindings {
    particles = particles:read_write_view(),
    dt = f32(0.016),
};
let pipeline = gpu:create_compute_pipeline(step)?;
commands:dispatch(pipeline, arguments, groups, 1, 1)?;
```

The compiler derives a storage binding and a value-parameter layout from
`StepBindings`; the host does not separately author a matching descriptor layout.
The invocation index retains today's `u64` spelling for illustration, not a promise
of native `u64` support on every future backend.

[Blade's `ShaderData` derive][blade-macro] is an applicable precedent: its
[implementation][blade-shader-data] derives a layout from named struct fields and
emits their binding operations. The inspiration is a single typed contract for
resource declaration and binding, not Rust macros or Vulkan descriptor-management
calls in Resin source.

Start with one bundle for compute and one compatible bundle shared by graphics
stages. Invocation built-ins, varyings, and render-target attachments retain their
separate roles. Nested resource groups can be statically flattened; this does not
imply following a runtime object graph.

Resource identity must remain statically resolvable in the initial shader model.
Helpers may be specialized for particular binding paths, or operations may remain
in a thin GPU adapter. Dynamically choosing, returning, or persisting arbitrary
buffer handles is a separate feature, not an implicit consequence of struct syntax.

### Arrays and indices replace cross-boundary links

A linked structure can use indices into an explicitly bound table instead:

```text
struct Node {
    value: f32,
    next_index: u32,
}

@resources
struct ListBindings {
    nodes: Buffer<Node, Read>,
    first_index: u32,
}
```

Define a null sentinel and validate each index before loading. CPU and GPU adapters
interpret the same integer relative to the same logical node table. Moving or
copying the table does not change its indices. Sharing and cycles need no address
fixups, although traversal termination, bounds, and ownership remain obligations.

Similarly, a BVH can store child indices; mesh records can store offset/count pairs
into separately bound vertex and index arrays. Prefer a small number of arenas over
a descriptor per node. Reordering an arena can require updating indices: an index
is not automatically a stable object ID.

A stored range means `(element_offset, element_count)` relative to a known resource.
A launch view additionally carries host resource identity. Specify whether persistent
indices are allocation-relative or view-relative; slicing must not silently change
that interpretation. Selecting unrelated buffers by integer resource ID is not
ordinary array indexing and belongs to a separate resource-table facility.

Subranges must respect binding alignment. A backend can bind an aligned containing
range and pass a base-element offset plus logical length. Shader access then applies
the base and respects the logical range. Do not promise that every element offset
is a legal descriptor offset, or silently copy data to repair alignment. Vulkan
constrains storage-buffer descriptor offsets through
[`minStorageBufferOffsetAlignment`][vulkan-descriptors].

## Portability and lowering

| Backend | Intended mapping | Qualification |
| --- | --- | --- |
| Vulkan | Storage/uniform descriptors and pipeline layouts; initially a descriptor set plus a constants channel | Ordinary buffer access need not require device addresses; binding limits and alignment still apply |
| Metal | Argument buffers or direct resource slots selected by the backend | Residency, lifetime, and synchronization still require correct handling |
| WebGPU | Bind-group layouts, bind groups, and shader resources | Supported layouts, scalar types, access modes, and binding limits constrain the interface |

These are proposed mappings to [Vulkan descriptors][vulkan-descriptors],
[Metal resource bindings][metal], and [WebGPU resource bindings][webgpu], not
implemented Resin backends. One logical bundle need not be one identical physical
object on every target.

WGSL has address-space-qualified pointers, but they are not host-address values
that can be stored in host-shareable buffer records. Resource bindings and
compatible value layouts provide the relevant boundary; saying WebGPU has no
pointers would be incorrect. See [WGSL resource and memory rules][wgsl].

This change alone does not make all current Resin shaders portable. Integer widths,
record padding, storage/uniform layouts, atomics, subgroup operations, textures,
and graphics/ray-tracing capabilities require their own contracts. Diagnose
unsupported layouts and operations rather than silently promising emulation.

Mapping is also backend-specific. Unlike Vulkan's persistent mappings, a mapped
WebGPU buffer cannot be used by the GPU and must be unmapped before work using it
is submitted. Keep upload, readback, and mapping explicit so a backend can use
staging where appropriate. See [WebGPU buffer mapping][webgpu-mapping].

Bindless arrays and physical device pointers may remain explicit,
capability-gated facilities for workloads that justify them. Neither needs to be
the representation of every buffer in the portable interface.

## What remains shared between CPU and GPU

Arithmetic, RNG/hash routines, geometry calculations, and other helpers can remain
shared when their operations and value types are supported on both targets. A CPU
loop over host storage and the GPU entry above can both call `advance`.

Algorithms that manipulate storage may need different adapters or a future generic
view abstraction. Existing generic specialization is relevant, but address spaces,
access modes, and GPU execution semantics still need checking. Do not require
universal pointers, general traits, or a new polymorphism system to implement the
initial resource interface.

Local references and helper calls need not disappear, but must preserve their
storage/address-space rules and cannot leak into serialized buffer data. Workgroup
cooperation and invocation built-ins are additional reasons not to equate GPU
entries with normal host calls.

The revised promise is **one language with reusable CPU/GPU algorithms**, not one
indistinguishable address space or a CPU implementation of every shader entry.

## Cost and correctness contract

For a bundle containing `B` resource views and `C` bytes of value parameters,
recording should depend on the bundle and required resource tracking, not on the
number of objects reachable inside its buffers. This is a design requirement, not
a benchmark result. Explicit payload uploads, readbacks, and format conversion
still have their ordinary costs.

Generate schemas at compilation or pipeline creation. Reuse compatible layouts
and binding objects where beneficial, without mutating descriptors used by
in-flight work. Device-address backends can cache metadata too: compare against
the actual current implementation, not an artificially naive pointer baseline.

Preserve the useful guarantees of [current recording lifetimes](lifetimes.md):

- Validate device identity, buffer usage, element layout, ranges, and declared
  shader access. Read-only bindings must reject stores through that binding;
  aliases with separate write permissions do not disappear.
- Snapshot recorded bindings and parameter values, retaining referenced resources
  and pipelines through completion or cancellation. Later changes to the original
  host bundle must not change recorded work.
- Preserve host/device exclusion, barriers, and completion rules. Declared access
  modes aid tracking but do not prevent races between shader invocations.

Binding limits, descriptor allocation/update overhead, parameter packing, and
specialization are real costs. Large resource sets may need arenas, batching, or
optional bindless support. Indexed traversal can still have poor locality. The
rationale is avoiding an obligatory hidden relocation/translation system and
improving portability, not an unconditional throughput claim.

## Comparison

| Question | Current typed pointer projection | Proposed unified virtual `Ptr` | Selected resource bundles |
| --- | --- | --- | --- |
| Host GPU identity | Owning `GpuPtr` / `GpuSpan` | Canonical range plus allocation map | Explicit buffer and binding view |
| Entry interface | Shader pointer root and projected host record | Ordinary pointers translated at the boundary | Typed resource bundle |
| Embedded links | Pointers rejected in buffer elements | Need rejection, relocation, or software-address interpretation | Values, arrays, indices, and offsets |
| Launch work | Root projection and retention | Bounded for roots; potentially graph-sized for deep relocation | Bundle encoding, binding, and retention |
| Shared code | Helpers and host-callable entries | Aims for pointer interchangeability | Shared helpers; target-specific entry adapters allowed |
| Portability | Current Vulkan device-address profile | Additional address and mapping machinery | Conventional resource interfaces, subject to target capabilities |

## Migration and validation

A subsequent implementation should proceed in reviewable steps:

1. Specify field kinds, access modes, element-layout rules, constants, and the
   logical binding schema. Keep resource structs distinct from storage structs.
2. Add one compute entry using one bundle, typed pipeline/dispatch validation,
   and Vulkan storage-buffer lowering. Reuse allocation ownership and recording
   guarantees rather than rebuilding lifetime management.
3. Port a small example, then particle simulation/rendering. Keep shared value
   helpers and explicit CPU adapters. Extend the contract to graphics without
   conflating resources with varyings. Track ray-tracing resource requirements
   separately before removing device-address support globally.
4. Measure recording/launch CPU time, descriptor churn, GPU kernel time,
   allocations, transferred bytes, and emitted code against the pointer-root path.
   Include unchanged bindings, sliced buffers, many small dispatches, and indexed
   arenas. With a fixed bundle, vary payload size to check that recording does not
   inspect buffer contents.
5. Deprecate the pointer-shaped GPU interface only after equivalent workflows and
   lifetime guarantees are covered. Evaluate Metal/WebGPU lowering before promising
   universal support. Update the manual, examples, and relevant repository guidance
   alongside the implementation, not in this proposal-only change.

Tests should cover mismatched bundle types, read-only stores, device mismatches,
layout/alignment failures, out-of-range views, aliasing, cancellation, and resource
retention. Compare CPU/GPU helper results and verify that indexed data moves between
targets without link fixups. Specify expected diagnostics for attempts to put host
pointers or binding handles inside buffer elements. Do not invent a speedup target
before establishing a representative baseline.

### Open details, not open direction

Type/decorator names, constant packing, bounds behavior, empty views, resource-group
composition, cross-stage compatibility, and the portable scalar/index profile still
need specification. The direction for this draft is explicit: **resources at the
execution boundary, portable values within buffers, and no automatic traversal of
arbitrary pointer graphs.**

## References

Repository references describe the pinned baseline. External references establish
API distinctions, not evidence of a measured Resin speedup.

- [Current GPU ownership and projection](gpu-buffers.md).
- [Current shader interfaces and Vulkan requirements](shaders.md).
- [Ownership and recording lifetime rules](lifetimes.md).
- [Native GPU ABI](../crates/resin-runtime/include/resin_runtime/gpu.h).
- [Blade `ShaderData` declaration][blade-macro] and
  [layout/binding generation][blade-shader-data].
- [Vulkan mapping and cache maintenance][vulkan-memory].
- [Vulkan descriptors][vulkan-descriptors].
- [Apple: mapping resource bindings to Metal][metal].
- [WGSL: host-shareable types, pointers, and resource interfaces][wgsl].
- [WebGPU resource bindings][webgpu] and [buffer mapping][webgpu-mapping].

[blade-macro]: https://github.com/kvark/blade/blob/main/blade-macros/src/lib.rs
[blade-shader-data]: https://github.com/kvark/blade/blob/main/blade-macros/src/shader_data.rs
[vulkan-memory]: https://docs.vulkan.org/spec/latest/chapters/memory.html
[vulkan-descriptors]: https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html
[metal]: https://developer.apple.com/videos/play/wwdc2023/10125/
[wgsl]: https://www.w3.org/TR/WGSL/
[webgpu]: https://www.w3.org/TR/webgpu/
[webgpu-mapping]: https://www.w3.org/TR/webgpu/#buffer-mapping
