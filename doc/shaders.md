# Shaders and graphics

Run either demo like any other Resin program:

```sh
cargo run -- examples/gradient.resin
cargo run -- examples/triangle.resin
```

They write `gradient.png` and `triangle.png` in cwd. Their Resin `main` functions allocate
resources, create pipelines, record dispatch/draw commands, submit, write PNGs, and free resources.
There are no compiler-side graphics/image execution modes.

Shader entry points are ordinary functions with declaration decorators:

```resin
export { main };
import { "$/gpu.resin" };

@compute_shader
fn kernel(index: u64, output: Ptr<u64>) {
	output.* = index;
}
fn main() -> (() | Err<_>) {
	let gpu = gpu_new()?;
	let pipeline = gpu:create_compute_pipeline(kernel)?;
	(())
}
```

`@compute_shader`, `@vertex_shader`, and `@fragment_shader` register shader candidates and
validate their stage signatures. Each function accepts one shader decorator. Helpers require
no decorators, and decorated functions remain ordinary host-callable functions. Decorators
currently describe compiler-defined entry points; user-defined compile-time transformers are
not implemented yet.

Create typed pipelines from shader declarations with
`gpu:create_compute_pipeline(kernel)` and
`gpu:create_graphics_pipeline(vertex, fragment)`. Creation requests their embedded
shader representation automatically and preserves the shader stage and root type.
Creation currently requires direct declarations, including imported declarations; runtime
function aliases are not accepted yet. A reachable pipeline creation site requests its shaders
even if its branch is not executed. Merely declaring a shader or calling it on the host
requires no shader optimizer. There is no `.spirv` property; inspect artifacts in the build
cache instead. The current Vulkan implementation's private C ABI uses pointer/length pairs.

Resin lowers the entry and its reachable named helpers directly to SPIR-V. The toolchain runs
`spirv-opt -O --target-env=vulkan1.3` and embeds the optimized binary in generated C headers.
Shader objects are deduplicated and retained with their generated project;
imported helper changes invalidate them. Copied executables need the Vulkan loader/device,
but neither Resin, source files, nor `spirv-opt` at runtime.

Pass a typed pipeline and its host arguments to
`commands:dispatch(pipeline, arguments, x, y, z)` or
`commands:draw(pipeline, arguments, count)`. The compiler checks the arguments against
the pipeline and projects GPU views internally. Shader entries keep a typed pointer
as their second parameter:

```resin
struct Params { values: Span<f32>, scale: f32 }

@compute_shader
fn kernel(index: u64, root: Ptr<Params>) -> ()  {
    if (index < root.values.length) {
        let mut p: RefMut<f32> = root.values:at_mut(index);
        p = p * root.scale;
        ()
    } else { () }
}
```
The entry interfaces are:

- Compute takes `(u64, Ptr<T>)` and returns `()`. Call `gpu:compute_workgroup_size()`
  to get the `u64` number of invocations per workgroup for that `Gpu`. The runtime
  selects it from the device's reported default subgroup size, bounded by its workgroup
  limits, and specializes each compute pipeline to match. The value stays fixed for
  that GPU's lifetime. With `let mut workgroup_size = gpu:compute_workgroup_size();`, compute
  dispatch groups with `u32((count + workgroup_size - u64(1)) / workgroup_size)`.
  The index is the global X invocation index. Dispatch only in X (`y = z = 1`) and guard
  any excess invocations in the function, as above.
- Vertex takes an `i32` vertex index, optionally paired with `Ptr<T>`, and returns
  `Vertex` with `position: Position` and `color: Color` fields.
  Position has `f32` fields `x, y, z, w`; Color has `r, g, b, a`, in those orders.
- Fragment takes Color, optionally paired with `Ptr<T>`, and returns Color.

Allocate typed GPU storage with `gpu:create(value)?` (an inferred `GpuPtr<T>`) or
`gpu:alloc::<T>(count)?`. A host launch record replaces shader `Ptr<T>`
and `Span<T>` fields with `GpuPtr<T>` and `GpuSpan<T>` values:

```resin
let values = gpu:alloc::<f32>(1024)?;
let mut index: u64 = 0;
while (index < values.length) {
    values:at(index):store(f32(1.0));
    index = index + u64(1);
};
struct HostParams { values: GpuSpan<f32>, scale: f32 }
let pipeline = gpu:create_compute_pipeline(kernel)?;
let commands = gpu:start_command_recording()?;
commands:dispatch(pipeline, HostParams { values = values, scale = f32(2.0) }, 16, 1, 1)?;
commands:submit()?;
```

Projection checks the shader root layout, translates owning views internally, and
retains every referenced allocation. Use `commands:draw(pipeline, None, count)` for graphics
shaders without a root. Successful recording retains arguments and allocations
through synchronous submission or cancellation. They must belong to the recording's
GPU. See [GPU buffers](gpu-buffers.md).

Device pointers support loads, stores, record fields, typed indexing, and passing to
ordinary helpers. Pointer reinterpretation and pointer/integer conversions are host-only. Shared storage supports `u8`, `i32`, `u32`, `i64`, `f32`, `u64`, pointers, nonempty
records, arrays, spans, and nominal wrappers. Scalars align to their size; records align to their largest
member, with member and trailing padding. This matches C and Vulkan's base alignment rules without requiring
scalar-block-layout support. Generated C asserts sizes, alignments, and member offsets.
Spans occupy 16 bytes (address and length) with alignment 8; arrays retain their element alignment.
Storage containing booleans, unit, or other numeric widths is rejected for now.

Host `GpuPtr` and `GpuSpan` operations retain their allocation, including indexing,
and slicing. `load`, `store`, and `replace` access elements on the host.
`:read_only()` and `:write_only()` narrow per-view
access permissions. Host accesses check bounds, alignment, mapping, permissions,
and pending recorded GPU use. `:copy_from(Span<T>)` uploads a bounded host span into the beginning of a writable,
host-visible GPU span; its source length must fit the destination.
`:copy_to(Span<T>)` copies into caller-owned host
memory. GPU views cannot be converted to raw `Ptr` values; the compiler's shader
projection is the host-to-device address conversion boundary. GPU buffer elements
must have a shared layout without pointers, spans, owners, or drop hooks.

Shader bodies support `u8`, 32-bit numbers, `i64`, `u64`, booleans, records, nominal types, local mutation,
branches, loops, and direct calls to named Resin helpers. Integer `/`, `%`, `<<`,
and `>>` work for `u8`, `i32`, `u32`, `i64`, and `u64`. Signed division truncates
toward zero; remainder follows the dividend's sign. `MIN / -1` wraps to `MIN` and
`MIN % -1` yields zero. Right shift is arithmetic for signed integers and logical
for unsigned integers. Left shift discards bits beyond the type's width.

Division or remainder by zero and shifts outside `0..width` fail the invocation
before executing the invalid operation. Failure propagates through helper calls;
earlier stores remain visible and later stores do not run. These cases trap on
the CPU. Neither target unwinds destructors on a trap.

Foreign calls, recursion, and indirect calls are rejected in shaders. Arrays and
spans support unchecked `:at()` and `:at_mut()` indexing. Local `Ref` and `RefMut`
arguments can cross helper calls, including references to fields and indexed
elements. Returning local references or selecting different local referents
across control-flow edges remains unsupported. Local references cannot produce
pointers. `fmt` and `print` are host-only.

Invocations must avoid racing on shared buffers. Workgroup-local storage, shader barriers, and
atomics are not exposed yet. For multi-pass algorithms, record separate dispatches: the runtime
inserts memory barriers before dispatches and rendering, including compute-to-vertex reads.
Submission currently waits for completion, making mapped results readable by the host.

Build a GPU program with `-o` to inspect its unoptimized and optimized SPIR-V without executing GPU work,
for example `cargo run -- examples/gradient.resin -o dist/`. Only the host entry selected by
`FILE:ENTRY` needs to be exported; passing a private shader declaration to pipeline creation
inside its module does not require exporting that shader. `resin-server --spirv-opt PATH` selects the service shader optimizer.


## GPU requirements

The runtime uses conventional Vulkan compute and graphics pipelines, with dynamic rendering
and a dynamic viewport/scissor. Graphics currently target one RGBA8 UNORM color attachment,
triangle lists, one sample, and no blending or depth/stencil testing.

A Vulkan 1.3 device must support graphics and compute, buffer device addresses, 64-bit shader
integers, timeline semaphores, synchronization2, dynamic rendering, and maintenance4.
`VK_KHR_maintenance8` and its `maintenance8` feature are also required and enabled
when creating the device. This makes signed shader remainder well-defined for
negative operands. Devices missing the extension or feature are reported as
unsuitable; GPU creation returns `unsupported` if no suitable device is available.
Shader objects, map_memory2, maintenance5, and maintenance6 are not required. Optional memory-priority and pageable-memory features are enabled when supported.
Shader capabilities are limited to the profile Resin emits; externally supplied SPIR-V must
fit that profile too. The emitted byte profile additionally enables supported `storageBuffer8BitAccess` and `shaderInt8` features.

For background on the graphics API direction, see Sebastian Aaltonen’s
[No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api).

For triangle acceleration structures and ray-generation/miss/closest-hit stages,
see [Ray tracing pipelines](ray-tracing-pipelines.md).
