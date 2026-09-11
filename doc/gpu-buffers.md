# GpuBuffer API cleanup proposal

Status: proposal against `e9908832`; this document does not change the API.

Use `GpuBuffer` handles to supply command roots, and keep `Ptr<T>` as the shader
entry parameter. Command recording should obtain the device address internally and
retain the root allocation through completion. A host API built around resource
handles fits the existing shader model.

There are two distinct cleanups: supplying the root buffer to a command, and
constructing device pointers inside the root's data. The first can be implemented
entirely in the Resin library. The second needs an explicit choice about how raw
device pointers should appear in host code.

## What the current values mean

| Value | Current representation | Meaning |
| --- | --- | --- |
| `GpuBuffer` | `Arc<BufferOwner>` | Shared ownership of an allocation and its GPU |
| `buffer.handle` | `Ptr<ResinAllocation>` | Native allocation object, used by the host runtime |
| `buffer.host_pointer()` | `Ptr<ubyte>` | Borrowed host mapping; may be null for GPU-only memory |
| `buffer.device_pointer()` | `ulong` | Device virtual address of the allocation's first byte |
| `dispatch` / `draw` root | `ulong` | Device address sent as one 64-bit push constant |
| Shader root | `Ptr<T>` | Pointer used by the shader to access the root record |

The offending integer is a device address, rather than the allocation handle.
These representations are visible in [the library](../resin/gpu.resin),
[the C ABI](../crates/resin-runtime/include/resin_runtime/gpu.h), and
[the native allocation and command implementation](../crates/resin-runtime/src/gpu/mod.rs).
The C ABI already names its integer address type `ResinDeviceAddress`.

[The shader entry wrapper](../crates/resin-codegen/src/spirv/entry.rs) loads the
64-bit push constant and passes its bits as the source-level pointer argument.
[SPIR-V pointer loads and stores](../crates/resin-codegen/src/spirv/types.rs)
convert the internal integer representation to a physical storage pointer at the
memory access. Keeping those implementation details does not require exposing an
integer root in the host library.

Vulkan itself distinguishes the buffer object from the address returned by
[`vkGetBufferDeviceAddress`](https://docs.vulkan.org/refpages/latest/refpages/source/vkGetBufferDeviceAddress.html).
Resin can preserve this distinction without adopting a pointer-oriented host API
as a requirement of the [No Graphics API](https://www.sebastianaaltonen.com/blog/no-graphics-api)
inspiration discussed in [the design document](design.md).

## First implementation: command roots are handles

Proposed public method signatures:

```resin
def dispatch(self: GpuCommands, root: GpuBuffer, x: uint, y: uint, z: uint) -> Result<(), RuntimeError>;
def draw(self: GpuCommands, root: GpuBuffer | None, count: uint) -> Result<(), RuntimeError>;
```

For example, [gradient](../examples/gradient.resin) would record its dispatch as:

```resin
commands.dispatch(root, groups, 1, 1)?;
```

[Particles](../examples/particles.resin) would pass `root` to both dispatch and
draw, eliminating its cached integer `address`. The rootless
[triangle](../examples/triangle.resin) would use `commands.draw(None, 3)?`.
The compute entry remains ordinary pointer-based shader code:

```resin
@compute_shader
def kernel(index: ulong, root: Ptr<Root>) = {
    if (index < root.pixels.length) {
        root.pixels.at(index).* := pixel(uint(index));
    };
};
```

Implement the library methods in this order:

1. Check that a supplied root belongs to the command recording's GPU. Report
   `InvalidArgument` before recording a command on a different device.
2. Obtain its device address privately and call the existing native operation.
   Translate `None` to zero for rootless graphics.
3. After successful recording, retain the root using the existing
   `RecordedResources` buffer slot. Preserve release on synchronous submit,
   cancellation, and recording destruction, including submission failure.

The method argument owns a reference during recording, so the root remains alive
until the command's retained reference is installed. Keep the existing shared
command state: submit/cancel consume the native recording once across all aliases.

This is a library change with source-call migration. The native C/Rust unsafe APIs,
push constant layout, shader signatures, and compiler passes can stay as they are.
Existing native tests and Rust benchmarks continue to use their raw address API.

### Limits that belong in the contract

Retaining the root does not retain allocations reached through its raw pointers or
spans. For example, gradient's `root` and `pixels` are separate allocations. The
caller must keep `pixels` alive through completion even when the command owns
`root`. The runtime does not walk the root's data or infer the referenced owners.

A handle also does not establish that a buffer has the type, size, alignment, or
initialization required by a particular shader. Pipeline construction currently
accepts SPIR-V bytes without a Resin root type. Preserve the caller's obligations
instead of describing handle-based recording as a typed launch API. `None` is only
appropriate when the graphics stages do not dereference a root.

The first slice supplies allocation-base roots. The old raw address API also
permits interior roots and zero compute roots. Repository Resin examples do not
need these cases; the source-level change intentionally removes them from the
ordinary command methods. If an application needs interior roots, add an owning
buffer view with a checked byte offset. Keep the allocation in that view so command
retention still works; avoid an integer-root overload that loses its owner.

## Second decision: pointers stored inside shared data

The awkward conversion appears separately in gradient and particles:

```resin
data.pixels := Span<uint> {
    data = Ptr<uint>(pixels.device_pointer()),
    length = count,
};
```

Changing dispatch to accept a handle does not remove this expression. The root's
`Span<uint>` still needs device address bits. Storing `GpuBuffer` there would store
a managed host owner, which the shader cannot consume as a buffer pointer.

| Option | Effect | Cost and limitation |
| --- | --- | --- |
| Return `Ptr<ubyte>` from the explicit `device_pointer()` query | Removes the public integer conversion; the caller casts bytes to the required pointee type | Library-only; a device pointer is still invalid to dereference on the host |
| Return a nominal `GpuAddress` with an explicit pointer extraction operation | Distinguishes addresses from ordinary integers and pointers until extraction | Adds a public type and conversion; extraction still has no ownership or address-space guarantee |
| Introduce typed buffer/view operations | Could carry element type, size, and ownership into construction | Requires a separate design for compiler-supported operations or source generics |

Recommend the first option as a small follow-up if removing the integer round trip
is the remaining objective. Keep command roots as handles. An explicit raw pointer
query can coexist with that API without making pointers the resource ownership
model. Its implementation would perform the existing integer-to-pointer conversion
inside `gpu.resin`; the private native declaration would continue to return `ulong`.
The cast to `Ptr<uint>` above would then be a byte-pointer reinterpretation.

Make `gpu.host_to_device_pointer(host)` consistent by returning
`Result<Ptr<ubyte>, RuntimeError>`. Preserve interior byte offsets. This translation
currently searches mapped heap blocks, including space that may no longer belong
to a live allocation; it is not an allocation-liveness check. Its successful result
must not be used to fabricate an owning buffer handle.

Preserve the pointer receiver on borrowed buffer accessors. The existing
[borrowed-receiver test](../tests/stdlib.rs) checks that even a temporary buffer
owner remains alive through the containing scope. A returned pointer still carries
no owner after that scope. Host and device pointer casts preserve bits and never
translate an address; separate address-space checking would be a language change.

For nested buffers whose lexical owners end before submission, a future explicit
`commands.keep_alive(buffer)` operation could retain those owners. It should check
recording state and device identity and release them with the recording. Add it
when supporting that use case; do not imply that root retention solves it already.

## Implementation scope and checks

The first slice touches `resin/gpu.resin`, the Resin dispatch/draw call sites,
wrapper tests, and the API/lifetime documentation. It needs no parser, HIR, LIR,
type-system, native ABI, or SPIR-V change. Keep memory-mode naming, mapping APIs,
allocation alignment rules, asynchronous submission, and typed pipeline design
outside that slice.

For command roots, extend [the mocked library tests](../tests/stdlib.rs) to verify:

- The native command receives the allocation's device address, never its native
  object pointer; dispatch dimensions and draw counts are forwarded unchanged.
- A root created in a nested scope remains alive until submit/cancel/drop, then is
  released exactly once after all owners are gone. Cover fresh temporary roots.
- Recording failure does not install an extra retained root. Submission failure,
  cancellation, and aliases of the consumed recording release retained resources
  according to the existing lifecycle contract.
- Roots from another GPU are rejected before the native command is called.
- Rootless drawing forwards zero and retains no buffer. Source calls passing
  arbitrary integers fail type checking; shader entries still accept `Ptr<Root>`.

If implementing the pointer-query follow-up, additionally cover host/device address
values that differ, preserved interior offsets, query failures, temporary receiver
lifetimes, and editor completion/signatures in [analysis tests](../tests/analysis.rs).
Host mapping can remain null for `Memory.gpu()`; changing the return spelling must
not promise mapping, stronger alignment, or ownership.

Run library, analysis, and shader generation tests in `nix-shell`. Exercise existing
compute, graphics, nested-buffer, and synchronization paths with
`RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1`, plus
`--all-features`, so missing tools or GPU support fail explicitly. Validate gradient
and triangle readback; validate particles with a display and
`RESIN_REQUIRE_WINDOW=1`. Use the repository's Xvfb settings when headless.

Update [the library reference](../resin/README.md),
[lifetime rules](lifetimes.md), and [host/GPU design](design.md) with the implemented
contract. Keep the remaining nested-pointer ownership responsibility explicit.
The two implementation slices can land separately: command handles remove the
root-address ceremony and establish retention; pointer queries remove the raw
integer from shared-data construction without changing shader semantics.
