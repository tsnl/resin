# GPU buffers

`GpuBuffer` is a shared handle that owns a GPU allocation and retains its device.
Pass the handle to command recording; shader entry points receive a pointer to the
allocation's first byte, interpreted as their declared root type.

```resin
def dispatch(self: GpuCommands, root: GpuBuffer, x: uint, y: uint, z: uint) -> Result<(), RuntimeError>;
def draw(self: GpuCommands, root: GpuBuffer | None, count: uint) -> Result<(), RuntimeError>;
```

For example, [gradient](../examples/gradient.resin) uses:

```resin
commands.set_pipeline(pipeline)?;
commands.dispatch(root, groups, 1, 1)?;
commands.submit()?;
```

Its compute entry remains `kernel(index: ulong, root: Ptr<Root>)`.
[Particles](../examples/particles.resin) passes its root buffer to both dispatch
and draw. [Triangle](../examples/triangle.resin) uses `commands.draw(None, 3)?`
because its graphics shaders do not dereference a root.

## Command roots and lifetime

A supplied root must belong to the recording's GPU; a different device produces
`InvalidArgument` before recording the command. The library obtains the buffer's
device address internally. `None` supplies zero for a graphics pipeline that does
not access a root. These methods accept allocation-base roots.

After successful recording, the commands retain the root until synchronous submit,
cancellation, or destruction of the unfinished recording. A root created in a
nested scope can therefore outlive that scope through the recording. Recording
failure does not retain it. Submit consumes the recording even on failure and
releases its retained resources. Aliases share the consumed command state.

The handle does not establish a shader root's type, size, alignment, or
initialization. The caller supplies storage matching the shader parameter and
coordinates host writes, GPU execution, and readback.

Root retention does not retain allocations referenced by raw pointers or spans
inside the root. In gradient, `root` and `pixels` are separate allocations; the
caller keeps `pixels` alive through submission. The runtime does not walk the
root's data or infer referenced owners. See [lifetime rules](lifetimes.md).

## Borrowed pointers inside shared data

| Operation | Result | Meaning |
| --- | --- | --- |
| `buffer.host_pointer()` | `Ptr<ubyte>` | Borrowed host mapping; may be null for `Memory.gpu()` |
| `buffer.device_pointer()` | `Ptr<ubyte>` | Borrowed device pointer to the allocation's first byte |
| `gpu.host_to_device_pointer(host)` | `Result<Ptr<ubyte>, RuntimeError>` | Translate a mapped host pointer, preserving its interior byte offset |

Use device pointers when constructing data for shader access. For example:

```resin
data.pixels := Span<uint> {
    data = Ptr<uint>(pixels.device_pointer()),
    length = count,
};
```

The cast selects the pointee type; it preserves the pointer's bits. Host and device
pointers both have source type `Ptr<T>`, but casting never translates between their
address spaces. A device pointer must not be dereferenced on the host. A host
mapping must not be stored as a shader pointer. Pointees must remain inside live,
properly aligned storage with enough bytes for each access.

Buffer pointer queries borrow their receiver. A temporary buffer owner survives
through the containing scope, but returning the pointer beyond that scope carries
no ownership. Host-to-device translation searches mapped memory and is not an
allocation-liveness check; a successful query does not retain an owner.

These source APIs use the existing private native address operations. The
[C ABI](../crates/resin-runtime/include/resin_runtime/gpu.h) continues to expose
`ResinDeviceAddress` for unsafe native callers. Shader pointer representation and
the root push constant layout are unchanged.
