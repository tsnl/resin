# GPU pointers, spans, and shader arguments

`GpuPtr<T>` owns a view into a GPU allocation. `GpuSpan<T>` adds an element count.
Copies, indexed pointers, and slices retain the allocation and its GPU. Both are
ordinary generic source structs over an opaque `GpuView` primitive. Neither
exposes a raw host pointer or a device-address query. Import `$/gpu.resin` for
these wrappers and device operations. Shader roots using borrowed `Span<T>` also
need an explicit `$/span.resin` import.

```resin
var gpu = Gpu.new()?;
let mut scalar = gpu.create(42)?;                  // GpuPtr<long>, inferred from the value
let mut values = gpu.alloc::<float32>(1024)?;
let mut index = 0_ul;
while (index < values.length) {
    values.at(index).store(1.0_f);
    index = index + 1_ul;
};
let mut first = values.slice(0, 16);            // GpuSpan<float32>, same owner
let mut readable = first.read_only();
```

`gpu.create(initial)` initializes one element and infers its type from the value
or result context; `gpu.create::<T>(initial)` supplies it explicitly.
`gpu.alloc::<T>(count)` allocates uninitialized storage with checked layout and
size arithmetic. Initialize elements before reading or using them in a shader.
Both use default host-visible memory. Pass the exported `int` constants
`memory_default`, `memory_gpu`, or `memory_readback` to `gpu.alloc_in::<T>(count, memory)`
to select a memory mode. Allocation methods are ordinary generic source methods and
report a typed `RuntimeError`.

GPU elements have the same host and shader layout and cannot contain pointers,
spans, managed owners, or custom destruction hooks. Supported scalar storage is
`ubyte`, `int`, `uint`, `long`, `ulong`, and `float32`, with arrays and records of
these types. Unsuffixed floating literals require `float32` context or an `_f`
suffix; the default `float64` has no supported shader storage layout.

## Checked host access

`pointer.load()`, `pointer.store(value)`, and `pointer.replace(value)` perform
checked host access; `replace` returns the previous value. `span.at(index)`
returns an owning pointer, and `.slice(start, length)` returns an owning span.
Indexing and slicing check bounds. To update a field, load its containing record,
edit the local value, then store the record back. Host GPU views do not produce
places or raw field addresses.

Each view carries host read/write permissions. `.read_only()` and `.write_only()`
remove the other permission and cannot restore previously removed permissions.
Their checks apply to loads, stores, replacements, and copies. Existing aliases
keep their own permissions. Access also checks the allocation range, alignment,
host mapping, and whether a recording currently holds the allocation for GPU work.
An invalid host access traps. These are compiler/runtime checks, not OS page
protection or a static borrow checker.

Use `span.copy_to(destination)` to copy into an ordinary `Span<T>` whose memory the
caller owns. This checks GPU read permissions and the destination length; it does
not expose a raw pointer into the GPU allocation. [Gradient](../examples/gradient.resin)
and [triangle](../examples/triangle.resin) copy completed GPU output into host
memory before writing a PNG.

## Typed pipelines and compiler projection

Create pipelines from decorated shader declarations. The compiler preserves their
root type and stage, then checks host arguments when recording a dispatch or draw:

```resin
struct Params { values: Span<float32>; scale: float32; }

@compute_shader
fn kernel(index: ulong, root: Ptr<Params>)  {
    if (index < root.values.length) {
        let mut value: Ref<float32> = root.values:at(index);
        value = value * root.scale;
    };
}

struct HostParams { values: GpuSpan<float32>; scale: float32; }
var pipeline = gpu.create_compute_pipeline(kernel)?;
let mut commands = gpu.start_command_recording()?;
commands.dispatch(pipeline, HostParams { values = values, scale = 2.0_f }, 16, 1, 1)?;
commands.submit()?;
```

The inferred pipeline type is `GpuComputePipeline<Params, GpuPipelineOwner>`.
`GpuGraphicsPipeline<Params, GpuPipelineOwner>` is the corresponding graphics type;
its factory takes a vertex and fragment declaration. Both graphics stages must use
the same root type when both have a root parameter. Rootless graphics pipelines use
`None` as their root type and accept `commands.draw(pipeline, None, count)`.

Pipeline wrappers retain their shared native owner across copies and ordinary
function calls. Explicit parameter types can name `GpuPipelineOwner`, exported
by the GPU module. Each wrapper stores an opaque `GpuPipelineContract` containing
the originating root type, owner type, and shader stage. Dispatch and draw validate
that contract before projecting arguments. Changing a wrapper annotation cannot
authorize a different root or stage.

Dispatch and draw derive the host record from the pipeline's declared root type: a
shader `Ptr<T>` field receives a host `GpuPtr<T>`, and a shader `Span<T>` field
receives a host `GpuSpan<T>`. Scalars and nested records keep their values. The
compiler validates explicit projection declarations on the source wrappers,
creates a separate shader root, translates GPU views internally, and
retains every referenced allocation. An indexed or sliced view preserves its byte
offset. Projection occurs inside recording; the public GPU API exposes no untyped
projected root to construct or reuse.

Pipeline creation requires decorated declarations directly. Runtime function aliases
and arbitrary shader bytes are not accepted. Compiled representations are private to
code generation and the runtime; shader functions have no `.spirv` property. Pointers inside GPU buffer elements are rejected:
projection handles the launch record, not recursively mapped pointer graphs. Raw
host pointers cannot substitute for GPU views. Current shader pointers allow both
reads and writes, so projection requires views with both permissions. Shader pointer
casts are rejected, including pointer/integer conversions and reinterpretation of a
pointer's element type. Use typed indexing for buffer access.

## Recording and lifetime

`commands.dispatch(pipeline, arguments, x, y, z)` and
`commands.draw(pipeline, arguments, count)` bind the supplied pipeline and project
its checked host arguments. Host records can be reused across compatible pipelines,
as in [particles](../examples/particles.resin); each recording creates its own root.

Successful recording retains the root and its referenced allocations through
synchronous submission or cancellation. Those allocations reject CPU access while
recorded work can use them. Submit consumes the recording, and aliases observe the
same consumed state. Dropping an unfinished recording cancels it. Resources from a
different GPU are rejected before use. If submission fails and a fallback device-idle
wait cannot confirm completion, the runtime terminates before releasing resources
that the GPU may still use.
See [lifetime rules](lifetimes.md).

The unsafe [native C ABI](../crates/resin-runtime/include/resin_runtime/gpu.h)
retains its explicit native handles and device addresses. The Resin API obtains
shader addresses through compiler projection.

The GPU module implements pipeline creation and recording through explicitly
decorated compiler bridges. Their native signatures are checked separately from
the typed public calls. Bridge implementations handle native owners and internal
`GpuArguments`, and must preserve resource retention through recording completion.
