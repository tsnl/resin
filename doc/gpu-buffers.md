# GPU allocations, pointers, and mapping

GPU allocations have an owning handle and separate borrowed views. `gpu:create(value)?`
returns `GpuPtrMut<T>` for one value; `gpu:alloc::<T>(count)?` returns `GpuSpanMut<T>`
for a sequence. Copies, indexed views, and slices retain the allocation and its GPU.
Their `:read_only()` counterparts are `GpuPtr<T>` and `GpuSpan<T>`.
A `GpuPtr` view denotes one element and cannot be sliced; keep the `GpuSpan`
when further indexing or slicing is needed.

Import `$/gpu.resin` for allocation and mapping, and `$/span.resin` for ordinary
span operations. The GPU wrappers are source structs over the opaque `GpuView` owner.

## Two addresses, ordinary pointer types

| Operation | One value | Sequence |
| --- | --- | --- |
| Writable allocation `:device()` | `PtrMut<T>` | `SpanMut<T>` |
| Read-only allocation `:device()` | `Ptr<T>` | `Span<T>` |
| Writable allocation `:map()?` | `PtrMut<T>` | `SpanMut<T>` |
| Read-only allocation `:map()?` | `Ptr<T>` | `Span<T>` |

`device()` borrows a Vulkan buffer device address for shader use. `map()` borrows
its coherent CPU mapping. The two addresses may differ. Neither operation copies
elements, retains the owner, translates nested pointers, or waits for GPU work.

Raw pointers are one machine word; spans add an element count. They carry no device
or address-space tag. The caller keeps the allocation alive, supplies addresses
valid for the processor and device executing the code, and synchronizes accesses.
This is the same unchecked lifetime contract as other borrowed Resin pointers.

Mappings are persistent for the allocation lifetime and reused by repeated `map()`
calls. There is no per-view unmap operation: allocations may share a mapped backing
slab. Releasing the last allocation owner ends the borrowed pointer's lifetime.
Mapping uses ordinary Vulkan host-visible memory; `VK_EXT_map_memory_placed` and
identical CPU/GPU virtual addresses are not required.

## Allocate, dispatch, map

The [gradient example](../examples/gradient.resin) uses one root type containing
`SpanMut<u32>`. It supplies `pixels:device()` when recording, waits through
`commands:submit()?`, then uses `pixels:map()?` to write a PNG. The owning `pixels`
handle stays alive throughout.

A launch argument with exactly the shader root's type is copied into a retained
root snapshot. Its pointer bits are preserved. **The snapshot does not retain the
allocations referenced by raw pointers.** Keep those owners alive until submission
finishes or the recording is canceled. Graphics and ray tracing use the same rule.

Shader functions remain ordinary host-callable functions. A host call supplies a
host root containing host pointers; a GPU launch supplies device pointers. See
[workgroups](workgroups.md) for an example that shares a cooperative algorithm.

## Nested pointers

GPU elements may contain ordinary pointers and spans. Construct device graphs by
storing addresses obtained through `device()` in their nodes. Vulkan buffer device
addressing lets shaders follow those pointers directly, without descriptor bindings
or recursive relocation.

Mapping a node exposes its bytes unchanged. If its `next` field contains a device
pointer, that field is still a device pointer when read through the CPU mapping.
Obtain the child's host mapping explicitly before accessing the child on the CPU.
Uploading an arbitrary host pointer graph does not make its pointers GPU-accessible.

Supported elements use the shared host/shader layout: `u8`, `i32`, `u32`, `i64`,
`u64`, `f32`, pointers, nonempty arrays, and records of supported fields. Owners,
drop hooks, booleans, and unions are not supported buffer elements. Choose `f32`
explicitly for shared floating-point data; the default `f64` is not supported here.
Shader pointer/integer casts and pointer reinterpretation remain unsupported;
use typed fields and indexing.

`gpu_enumerate_devices(infos)` similarly accepts a `SpanMut<ResinGpuDeviceInfo>`
so device enumeration keeps its output capacity with its address.

## Memory modes

`gpu:alloc_in::<T>(count, memory)` selects:

- `memory_default`: persistently mapped coherent memory, preferring device-local
  storage when available.
- `memory_readback`: persistently mapped coherent memory intended for GPU results.
- `memory_gpu`: device-local storage without a CPU mapping. `map()` returns
  `Err(Unsupported {})`; it does not silently allocate staging memory.

Allocation checks size arithmetic and alignment. Zero-length spans are valid;
they have no accessible elements. Slices preserve offsets, lengths, and permissions
for both address queries. A read-only view cannot grant a writable mapping.

## Checked owner access and retained launch arguments

The owning wrappers also offer immediate checked access: `load`, mutable `store`
and `replace`, bounded `copy_from`, and `copy_to`. These check mapping, range,
alignment, permissions, and outstanding recorded uses known to the runtime.
`write_only()` restricts a writable owner at runtime; reads and address queries
that would grant read access fail. Existing aliases keep their own permissions.
Raw escaped pointers are not tracked by these checks.

Existing retained launch records remain supported: registered `GpuPtr`/`GpuSpan`
fields project to corresponding shader `Ptr`/`Span` fields, with mutable variants
preserving write permission. This convenience path retains those explicit owners
with the recording. It translates only the launch record, never a pointee graph.
An explicit root containing borrowed addresses avoids that translation.

Pipeline tokens preserve root type, stage, and native owner identity. Pipeline
creation requires decorated shader declarations, and dispatch/draw validate the
token before recording. Rootless graphics pipelines use `None`. Recorded pipelines,
images, and root snapshots remain alive until submission or cancellation.
