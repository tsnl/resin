# GPU pointers, spans, and shader arguments

`GpuPtr<T>` owns a view into a GPU allocation. `GpuSpan<T>` adds an element count.
Copies, indexed pointers, slices, and addresses of fields retain the allocation and
its GPU. The compiler controls their representation and construction; neither type
converts to an ordinary host pointer or exposes a device-address query.

```resin
var gpu = Gpu.new()?;
var scalar = gpu.new(42)?;                  // GpuPtr<long>, inferred from the value
var values = GpuSpan<float32>.allocate(gpu, 1024)?;
var index = 0_ul;
while (index < values.length) {
    values.at(index).* := 1.0_f;
    index := index + 1_ul;
};
var first = values.slice(0, 16);            // GpuSpan<float32>, same owner
var readable = first.read_only();
```

`gpu.new(value)` initializes one element and infers its type, including from result
context. `GpuPtr<T>.new(gpu, value)` provides an explicit element type.
`GpuSpan<T>.allocate(gpu, count)` allocates uninitialized storage with a checked byte
count. These constructors use default host-visible memory. The low-level byte
allocator `gpu.malloc(bytes, alignment, memory)` returns `GpuPtr<ubyte>`.

GPU elements have the same host and shader layout and cannot contain pointers,
spans, managed owners, or custom destruction hooks. Supported scalar storage is
`ubyte`, `int`, `uint`, `long`, `ulong`, and `float32`, with arrays and records of
these types. Unsuffixed floating literals require `float32` context or an `_f`
suffix; the default `float64` has no supported shader storage layout.

## Checked host access

`pointer.*` loads or stores an element. `pointer.at(index)` and `span.at(index)`
return owning pointers; `.slice(start, length)` returns an owning span. Indexing and
slicing check bounds. GPU field addresses preserve their owner: `&pointer.field`
returns another `GpuPtr`, never an ordinary `Ptr`.

Each view carries host read/write permissions. `.read_only()` and `.write_only()`
remove the other permission and cannot restore previously removed permissions.
Their checks apply to dereferences, field accesses, and copies. Existing aliases
keep their own permissions. Access also checks the allocation range, alignment,
host mapping, and whether a recording currently holds the allocation for GPU work.
An invalid host access traps. These are compiler/runtime checks, not OS page
protection or a static borrow checker.

Use `span.copy_to(destination)` to copy into an ordinary `Span<T>` whose memory the
caller owns. This checks GPU read permissions and the destination length; it does
not expose a raw pointer into the GPU allocation. [Gradient](../examples/gradient.resin)
and [triangle](../examples/triangle.resin) copy completed GPU output into host
memory before writing a PNG.

## Compiler projection

Shader entries keep ordinary pointer parameters. A decorated shader declaration
provides `.project(gpu, arguments)`, which produces `Result<GpuArguments, RuntimeError>`
for the standard GPU allocator:

```resin
struct Params { values: Span<float32>, scale: float32 };

@compute_shader
def kernel(index: ulong, root: Ptr<Params>) = {
    if (index < root.values.length) {
        var value = root.values.at(index);
        value.* := value.* * root.scale;
    };
};

// Host launch records contain owning GPU views.
var root = kernel.project(gpu, { values = values, scale = 2.0_f })?;
var commands = gpu.start_command_recording()?;
commands.set_pipeline(pipeline)?;
commands.dispatch(root, 16, 1, 1)?;
commands.submit()?;
```

Projection uses the shader's declared root type to check the host record: a shader
`Ptr<T>` field receives a host `GpuPtr<T>`, and a shader `Span<T>` field receives a
host `GpuSpan<T>`. Scalars and nested records keep their values. It creates a
separate shader root, translates GPU views internally, and retains every referenced
allocation. An indexed or sliced view preserves its byte offset.

Projection requires a decorated declaration directly; runtime function aliases do
not provide `.project`. A shader without a root pointer has nothing to project.
Pointers inside GPU buffer elements are rejected: projection handles the launch
record, not recursively mapped pointer graphs. Raw host pointers cannot substitute
for GPU views. Current shader pointers allow both reads and writes, so projection
requires views with both permissions.

## Recording and lifetime

`commands.dispatch(root, x, y, z)` and `commands.draw(root, count)` accept
`GpuArguments`. Bind the pipeline whose shader root matches those arguments.
`commands.draw(None, count)` is available for shaders without a root. A projected
root may be shared by shaders with the same root layout, as in
[particles](../examples/particles.resin).

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
