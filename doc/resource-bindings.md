# Typed resource bindings

Import `$/buffer.resin` to give a shader a single resource record containing
`Buffer<T>` (read-only), `BufferMut<T>` (read/write), and plain value parameters.
The shader borrows that record through `Ref<Resources>`. Pipeline creation and
recording use the same record type; no second host parameter struct is needed.

The existing `gpu:alloc::<T>()` returns an owning GPU allocation view.
`:read_buffer()` and `:write_buffer()` produce retained bindings over that view.
Slice the allocation before constructing a binding to select a subrange. Copies
of host bindings retain the allocation; constructing or copying binding objects
inside shaders is unsupported. Shader helpers borrow them instead.

## Share a storage-using algorithm

This complete program uses the same `advance` helper for host spans and GPU
bindings. The helper reads and writes arrays as well as performing arithmetic.
Generic specialization selects the appropriate `load` and `store` overloads.

```resin
{{#include ../examples/resource_bindings.resin:resource_bindings}}
```

Run `examples/resource_bindings.resin:cpu_main` to test the algorithm on ordinary
host spans, or `examples/resource_bindings.resin` to execute the Vulkan pipeline.
The CPU checkpoint needs no GPU. Both use the normal compiler service described
in [Getting started](getting-started.md).

## Access and bounds

`Buffer<T>` has `load(index)`. `BufferMut<T>` has both `load(index)` and
`store(index, value)`. Both take a read-only reference to the binding object:
writable element access belongs to `BufferMut`, independently of whether the
record containing that binding can be reassigned. Constants in the borrowed
resource record remain read-only. An alias with write permission may still modify
the same allocation; read-only is an access permission, not global immutability.

Binding reads outside `0..length` return the element's zero value; out-of-range
writes do nothing. This applies to CPU binding access and GPU access, including
empty views and very large indices. Bounds handling does not terminate shader
invocations, so it cannot strand peers at a workgroup barrier. In-range accesses
still require correct synchronization between invocations.

The `Span<T>` adapters in this module retain ordinary span semantics: their
indices must be in range. The shared helper above establishes that condition
before either target accesses its arrays.

## Resource and recording guarantees

Buffer elements must have a supported plain shared layout: scalar values, arrays,
and records without pointers, bindings, owners, or destruction hooks. Store links
as indices or offset/count pairs relative to an explicitly identified buffer.
Nested resource records are supported; they describe bindings, not buffer contents.

Recording validates element layouts, ranges, alignment, device identity, and
required access permissions. Constructing a `BufferMut` around a read-only native
view does not grant write permission. Recording snapshots value parameters and
bindings and retains their allocations through completion or cancellation. The
existing [host/device access exclusion](gpu-buffers.md#recording-and-lifetime)
also applies to these bindings.

Host `load` and `store` require host-visible memory and reject access while a
recording retains the allocation for GPU work. Use explicit uploads/readbacks for
device-only allocations. Ordinary host calls remain available when their operations
are supported on the CPU; shader entry signatures need not serve as a general CPU
execution API. Helpers carry the reusable algorithm.

## Current backend and remaining work

The first implementation uses the existing Vulkan device-address backend and its
device requirements. The compiler encodes each buffer binding as an internal
address slot and logical length in the recorded argument storage, retaining source
field offsets. Host ownership bytes are not uploaded as usable shader handles.
Shaders can access a binding only through its checked load/store contract; no
native pointer is exposed by the binding API.

Descriptor-based lowering, texture/sampler bindings, dynamic resource selection,
and additional backends remain follow-ups in the [resource proposal](gpu-resource-bindings-proposal.md).
The existing pointer-root API remains available. This implementation makes no
claim of faster execution or support for devices outside the current Vulkan profile.
