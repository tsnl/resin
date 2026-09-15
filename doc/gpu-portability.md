# Vulkan and Metal

The intended GPU targets are Vulkan and native Metal. This change narrows the shader language and
records the implementation sequence for Metal and hardware ray tracing. Vulkan is
still the only implemented backend; its Vulkan 1.3 device requirements are unchanged.

## Language boundary established here

Pipeline creation accepts decorated shader declarations:

```resin
var compute = gpu.create_compute_pipeline(kernel)?;
var graphics = gpu.create_graphics_pipeline(vertex, fragment)?;
```

The compiler requests each shader's reachable dependencies, preserves its stage and
root type, and includes its compiled representation in the executable. Functions no
longer expose `.spirv`. Raw artifact expressions have been removed from HIR and LIR,
including their editor completions. Build-cache artifacts and the codegen crate's
Vulkan artifact accessors remain available for compiler inspection.

Shader pointer casts are also removed: `ulong(pointer)`, `Ptr<T>(address)`, and
reinterpretation between different pointer element types are host-only. The rule
applies after generic specialization, to every reachable shader helper, and to
direct LIR clients through verification. Shader code uses typed dereferences,
field access, and array/span indexing. Numeric GPU addresses are a backend detail.

This is a shader restriction. Host FFI, host pointer casts, ordinary host function
values, records, unions, Results, and generics retain their existing behavior.

| Contract | Current behavior |
| --- | --- |
| Shader references | Direct decorated declarations at pipeline creation; aliases, parameters, and returned shader values still need the work below |
| Shader values | `ubyte`, `int`, `uint`, `long`, `ulong`, `float32`, booleans, records, arrays, unions/Results, and nominal types subject to the existing storage rules |
| Shared buffer elements | Fixed-layout values; no pointers, spans, owners, or drop hooks inside buffer elements |
| Launch arguments | Host GPU views project to typed shader pointers/spans; referenced allocations remain retained |
| Shader pointers | Typed access and direct helper calls; no pointer casts or numeric address construction |
| Local addresses | Existing Vulkan lowering permits immediate access but rejects escape into stored pointers, calls, or returned values |
| Execution | Structured branches/loops and direct, nonrecursive calls; no foreign calls or managed-value consumption |
| Existing omissions | No workgroup storage, barriers, atomics, indirect device calls, or integer division/remainder/shifts |

Keep the existing byte and 64-bit integer types: excluding WebGPU removes the need
to narrow to its numeric baseline. The Metal implementation must validate every
supported operation and shared layout, including mixed byte/64-bit records, before
claiming compatibility. Shader `float64` is already unsupported.

## Implementation sequence

Each step must preserve the Vulkan path and pass its acceptance checks below.
The sections below describe planned work, not APIs introduced by this change.

### 1. First-class shader references and artifact selection

**Owners:** `resin-hir`, `resin-types`, `resin-lir`, `resin-codegen`,
`resin-toolchain`, the runtime C ABI, and `resin/gpu.resin`.

Stage annotations must survive value typing. Represent a shader reference with its
stage and concrete signature; assignment, function parameters/results, and branch
selection must preserve both. A plain host function pointer cannot supply a GPU
implementation. Define the source type spelling and any conversion to an ordinary
host callable explicitly before changing inference. Retain direct host calls to
decorated declarations.

Lower a host shader value to a reference to an immutable compiler-generated shader
descriptor. It names a concrete specialization, stage, root contract, binding
metadata, and available backend artifacts. Reserve its identity before lowering
dependent functions, as for current function instances. Reachability must include
shader values stored or returned by reachable functions, without trying to predict
which runtime branch will execute. Shader code still cannot make indirect calls.

Pipeline creation consumes that descriptor. The runtime selects the artifact for
its device and acquires/caches native library/function/pipeline objects per device
and full pipeline configuration. Cache keys include specialization, stage, artifact
version, and relevant render state. Resources must not cross devices. Missing
artifacts or unsupported requirements produce an explicit Result error.

Replace the current structural byte-span factory bridge in `resin/gpu.resin` and
`crates/resin-codegen/src/c/lower/pipeline.rs` together with the native ABI. Do not
rename `.spirv` to another bytecode property. Backend bytes, source text, and build
paths belong in generated artifacts, not Resin function values.

**Acceptance:** a helper accepts a compute shader value, selects between two
compatible kernels, returns the selection, and creates/dispatches it. Wrong stages
and root types fail type checking. Repeated references share artifacts and native
cache entries; ordinary host calls require no shader tools. Run this on Vulkan
before introducing Metal.

### 2. Explicit backend targets and direct MSL generation

**Owners:** CLI build settings, `resin-codegen`, `resin-toolchain`.

Add an explicit GPU backend to project generation/build settings and cache identity.
Keep host-only compilation independent of GPU SDKs. Retain the current native
process boundary: codegen writes a completed project; the toolchain runs its build
graph. Preserve atomic generation on failure and dependency-based cache rebuilding.

Add a private Metal Shading Language (MSL) target language, lowering, and printer
inside `resin-codegen`. Consume verified LIR directly. The existing SPIR-V emitter's
physical-pointer representation is not an input language for Metal. Keep shared
shader semantics in `resin-types`/LIR and representation constraints in each backend.

The first compiler milestone emits a compute kernel with one scalar buffer, then
expands to all existing shader operations. Map structured LIR control flow directly
and preserve checked numeric conversions, failure exits, union tags, and cleanup
restrictions. Stage wrappers adapt compute indices, vertex output, and fragment
input/output without changing the Resin entry signatures.

For an initial runnable backend, embed generated MSL and let the Metal runtime
compile/cache its library. This allows source generation tests on Linux. Add
offline `.metallib` compilation through Apple's tools as a separate toolchain step
once execution works; include language version, compile options, and SDK/tool
identity in the appropriate caches. The runtime owns device-native object creation
in either case.

**Acceptance:** generate MSL on Linux and compile it with Apple's compiler on a Mac.
Cover numeric boundary cases, structured control flow, arrays, and Results in the
compiler fixtures. Execution follows with the runtime below; generated-text tests
alone cannot establish Metal support.

### 3. Resource bindings and Metal runtime

**Owners:** codegen layout/projection, `crates/resin-runtime/src/gpu/`,
`crates/resin-runtime/src/gpu_view.rs`, native GPU headers.

The main ABI change is the launch root. Current C projection writes Vulkan device
addresses into a root allocation. Introduce backend binding metadata derived from
the completed root projection: scalar fields, resource slots, byte offsets,
lengths, and access requirements. The runtime binds each retained allocation with
its view offset. Metal can encode resources through an argument buffer; it must not
interpret host pointers or Vulkan numeric addresses as Metal pointers.

Lower the shader's typed root accesses using that binding plan. MSL address spaces
(`device`, `thread`, and `constant`) need explicit tracking in private Metal
lowering, including helper signatures and projected fields. Do not assume one MSL
pointer type represents both a local address and device storage. Ordinary buffer
elements keep the shared layout contract: emit matching padding/representation or
diagnose an unsupported layout. Launch resource bindings need not share the current
Vulkan root's byte encoding.

Keep the public runtime handles backend-independent, with private Vulkan/Metal
implementations. Use target-specific `objc2`, `objc2-metal`, and required framework
dependencies on macOS. These bindings provide retained Objective-C objects; buffer
typing, offset bounds, synchronization, and GPU lifetimes still need explicit Rust
invariants. See the [binding safety contract](https://docs.rs/objc2-metal/latest/objc2_metal/).

Implement device/queue creation, buffers and host access, compute pipelines,
command encoding, copies, submission, and cancellation first. Preserve current
synchronous completion, recorded-use exclusions, ownership, and Result behavior.
Add native render pipelines, offscreen textures, and readback next. Presentation
through a Metal layer is a final graphics milestone; Cocoa/window work must remain
on the process main thread. Native Metal must not require MoltenVK.

**Acceptance:** allocation/range/copy/lifetime tests; two buffers and a sliced span
in one launch root; `gradient.resin`; offscreen `triangle.resin`; then resize,
presentation, cancellation, and resource release. Test cross-device rejection and
errors as well as successful rendering. Compare host/Vulkan/Metal results for the
compiler fixtures, including numeric boundary cases and shared layouts.

### 4. Native ray tracing

**Owners:** runtime acceleration structures, shader intrinsics/type checking, both
shader emitters, and a language-facing ray-tracing module.

Ray tracing is an optional device capability. Raster/compute creation must keep
working when it is unavailable. Start with triangle acceleration structures and
instances, closest-hit and occlusion operations inside compute shaders, and
ordinary hit results (distance, primitive/instance identity, barycentrics, and face
orientation). Define ray intervals, masks, transforms, and hit semantics explicitly
and test them across backends. Equal-distance hit ordering must not be relied on.

| Operation | Vulkan | Metal |
| --- | --- | --- |
| Geometry/instance acceleration structures | `VK_KHR_acceleration_structure` builds and instances | Primitive and instance acceleration structures |
| Closest-hit / occlusion | `VK_KHR_ray_query` traversal inside the calling shader | MSL `intersector` with triangle intersection settings |
| Later candidate inspection | Ray-query proceed/confirm/generate operations | MSL `intersection_query` next/commit operations |

Vulkan ray queries and ray-tracing pipelines are separate features. Query support
does not require ray-generation/miss/hit shader stages or shader binding tables.
Enable and test the acceleration-structure and ray-query features and their
dependencies explicitly. See the [Vulkan ray-tracing guide](https://docs.vulkan.org/guide/latest/extensions/ray_tracing.html).

Metal exposes both an intersector and explicit intersection queries. Prefer the
intersector for the initial closest-hit/occlusion API: Apple's M3/A17 Pro guidance
explains that explicit queries can increase traversal state and lose intersection
function reordering, while still using intersection hardware. Apple GPU family 9
introduced dedicated acceleration. Native ray-tracing API availability on earlier
Apple GPUs is separate from dedicated hardware support; do not equate
`supportsRaytracing` with HWRT. See [Apple's architecture guidance](https://developer.apple.com/videos/play/tech-talks/111375/)
and [feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf).

Build/scratch buffers, acceleration structures, referenced geometry, and instance
dependencies must survive their recorded GPU uses. Add build-to-trace barriers and
validate alignment, size, and update rules. Start with immutable builds; add refit,
compaction, custom primitives, or candidate inspection only with separate tests.
An explicit traversal cursor would need function-local lifetime/copy restrictions
in HIR/LIR; it cannot be an ordinary freely copied Resin record.

There is no Resin software BVH builder/traversal fallback in this scope. A device
without native ray-query support returns Unsupported. Native APIs may themselves
implement traversal on shader cores; HWRT performance acceptance requires actual
accelerated hardware. Full ray-tracing pipelines, shader binding tables, recursion,
and WebGPU are outside this sequence.

**Acceptance:** deterministic triangle and instanced-scene hits, misses, occlusion,
masks, transforms, and boundary rays on a Vulkan RT device and an M3-or-newer Mac.
Validate resource lifetimes and unsupported-device behavior. Add a small compute
ray-traced image example and compare results with a CPU reference, allowing documented
floating-point tolerance.

## Validation and remaining portability work

Linux can run source/type/LIR/MSL-generation tests, Vulkan validation, and native
Vulkan execution. It cannot validate Apple's driver, Metal synchronization, or HWRT.
Use a borrowed or rented Apple-silicon Mac for compile/execution checks; an M3-or-newer
GPU is required for the dedicated-HWRT acceptance case. Keep automatic CI Linux-only
and run macOS checks manually, following the repository's runner budget policy.

The first Metal support baseline should be Apple silicon. Choose the minimum macOS
and MSL versions against the actual bindings, resource-binding implementation, and
RT features during the compute milestone; document and check them before release.
Intel Macs, older AMD Metal devices, and Metal 4-only features do not define this
initial baseline.

Vulkan portability needs a separate capability audit: the current runtime requires
graphics and compute queues, Vulkan 1.3, and several features even for simple compute.
Separate compute, graphics/presentation, and RT requirements; derive shader capability
requirements from emitted operations. Evaluate a Vulkan 1.2 plus extensions path
with a matching SPIR-V target and validator, and determine whether maintenance4 is
actually used. Keep buffer device addresses until an alternative binding lowering
exists. Removing `.spirv` and pointer casts does not by itself make older Vulkan
devices compatible.
