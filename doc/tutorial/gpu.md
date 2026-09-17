# 3. Run the solver on the GPU

The small checkpoints have established the algorithm. Now open the
[complete explorer](../../examples/eg011_mandelbrot.resin). It adds a configurable
`Plot` and `Mandelbrot`, interior shortcuts, and a work
partition usable by both CPU and GPU. Its sections put the math first, GPU helpers
second, and the application last.

## Choose a unit of work

One invocation evaluates a horizontal segment of up to 32 pixels:

```resin
{{#include ../../examples/eg011_mandelbrot.resin:evaluate_segment}}
```

A segment owns disjoint output bytes. Its final pixel may reach the right edge
of the image, so the loop checks both the segment length and image width. An
invocation beyond the batch's segment count returns immediately; dispatch rounds up to a
whole workgroup.

Thirty-two pixels is an application choice, not a language requirement or a
claim of optimal GPU scheduling. One invocation per pixel is also possible.
Division and remainder turn the segment index into a row and column, so no origin
buffer is needed. Each dispatch supplies a starting segment and count; the CPU
uses the same indexing. Measure a different partition before treating it as faster.

## Keep the shader entry small

```resin
{{#include ../../examples/eg011_mandelbrot.resin:compute}}
```

`@compute_shader` supplies the global X invocation index and a pointer to the
parameter block. Dereferencing `root` supplies the place borrowed by the
`Ref<Parameters>` helper parameter. There is no reference-to-pointer conversion.

The CPU entry calls exactly that same evaluator:

```resin
{{#include ../../examples/eg011_mandelbrot.resin:dispatch_host}}
```

Both eventually call `evaluate_pixel`, `escape_iterations`, and `palette`.
There is no second `solve_cpu` to keep synchronized. The compiler specializes
ordinary helpers for the selected execution target. Calls remain source-level
calls; the helpers need no shader annotation.

## What references promise

A `Ref<T>` permits read-only access to a `T`; it does **not** promise that a pointer can be
materialized. That contract is the same for a local, a field, and a value reached
through a device pointer. A reference-returning helper does not hand its caller
permission to take an address either.

Both of these complete examples are rejected:

```resin,compile_fail
fn address(value: Ref<i32>) -> Ptr<i32> {
    &value
}
```

```resin,compile_fail
fn needs_pointer(value: Ptr<i32>) {}
fn caller(value: Ref<i32>) {
    needs_pointer(value);
}
```

No call-site convenience, cast, specialization, or LIR conversion silently turns
`Ref<T>` into `Ptr<T>`. The compiler verifies the distinction before code emission.
A helper that needs a pointer must say so in its signature. Reading an *existing*
pointer value through `Ref<Ptr<T>>` is allowed; that reads the stored capability.

`RefMut<T>` permits writes and can weaken to `Ref<T>`. Neither kind grants an
address or exclusive access. For indexing, choose the required access explicitly:

| Receiver | `:at(i)` | `:at_mut(i)` | `:lea(i)` |
| --- | --- | --- | --- |
| Array place | `Ref<T>` | `RefMut<T>` if writable | Unavailable |
| Pointer to an array | `Ref<T>` | `RefMut<T>` | `Ptr<T>` |
| `Span<T>` | `Ref<T>` | `RefMut<T>` | `Ptr<T>` |
| `str` | `Ref<u8>` | Unavailable | `Ptr<u8>` to read-only literal bytes |

A pointer to an array already grants access to addressable storage, so `:lea`
can produce an element pointer. A local array does not grant that capability.
A span carries a pointer to its elements, regardless of where the descriptor lives.
See [references](../references.md) for the complete indexing rules.

## Allocation and target limits

The host's `HostParameters` contains owning `GpuSpan` handles. The shader's
`Parameters` contains borrowed `Span` descriptors. Dispatch validates the pipeline
contract and projects the handles into device-accessible descriptors. This
allocation, upload, and dispatch machinery lives in the GPU helper section.
It is overhead around the algorithm that can be measured and improved separately.

Local shader references may cross helper calls, including references to fields
and dynamically indexed array elements. They need no device-buffer layout:
borrowing a local `bool` works. There are still backend limits: returning local
references from helpers and selecting distinct local referents across branches
are unsupported. Those are [target diagnostics](../diagnostics.md), separate from
the language-wide prohibition on converting references to pointers.

This example uses `f32` on both targets. Deep zooms lose distinguishable
coordinates; a higher iteration budget cannot repair that. CPU/GPU results can
also differ slightly due to floating-point arithmetic. The tests compare their
rendered bytes with a tolerance of one, rather than requiring bit-identical output.

Try the complete application without a window:

```sh
cargo run -- examples/eg011_mandelbrot.resin -- --gpu --width 640 --height 480 --output gpu.png
cargo run -- examples/eg011_mandelbrot.resin -- --cpu --width 640 --height 480 --output cpu.png
```

GPU mode needs the runtime's Vulkan features. CPU headless mode needs no Vulkan
driver. [GPU requirements](../shaders.md#gpu-requirements) lists the current device
requirements; local shader-reference support does not make every host operation
shader-compatible.
