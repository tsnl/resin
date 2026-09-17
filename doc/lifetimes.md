# Ownership, moves, and lifetimes

Resin copies ordinary values by default and checks ownership moves for move-only
values. A struct copies when all its stored fields copy and it has no custom
`drop` hook. The same rule propagates through tuples, arrays, unions, and `Err`
payloads. A noncopyable field makes its enclosing value move-only.

Copying creates a separate value; it does not duplicate pointed-to storage.
Copying an Arc handle retains the same allocation with an additional reference
count. This is broader than Rust's `Copy`, whose copies cannot run retain logic.
There is no user-defined `Copy` hook. Library `clone` functions remain ordinary
visible overloads for explicitly duplicating move-only values or requesting a
library-specific copy.

```resin
struct Item { value: i32 }
fn read(item: Item) -> i32 {
	item.value
}
fn example() -> i32 {
	let item = Item { value = 21 };
	let mut copied = item;
	copied.value = 7;
	read(item) + item.value // 42; passing item did not consume it.
}
```

To opt out of copying without implementing cleanup, include `PhantomBox` from
`$/ownership.resin`. This ordinary library marker owns no allocation; its empty
drop hook makes it noncopyable, and the field rule does the rest:

```resin
import { "$/ownership.resin" };
struct Ticket { serial: u64, ownership: PhantomBox }
fn example() {
	let ticket = Ticket { serial = 1, ownership = PhantomBox {} };
	let transferred = ticket;
	// ticket is moved and cannot be read again.
}
```

A struct with a custom drop hook is already move-only. It needs no marker.
`Ptr<T>` remains a copyable, nonowning pointer; it does not allocate or free `T`.

`let` bindings are immutable unless their identifier has `mut`. Moving an
immutable value is allowed. An uninitialized immutable binding may be initialized
once; assigning again requires `mut`. Assignment returns unit and destroys the
previous initialized destination before installing its replacement. Function
parameters and match-arm binders use the same per-identifier `mut` modifier.

## Where ownership is checked

Ownership analysis completes each function during HIR construction, after type
inference, in `crates/resin-hir/src/lower/elaborate.rs`. It checks unused definitions
as well as entry-point dependencies. The pass follows runtime evaluation order and
tracks initialization and moved field paths for each resolved local binding.

A read, address, or reference argument must name available initialized storage.
Moving a field invalidates that field and overlapping whole-value uses, while
leaving disjoint fields usable. Reinitializing a moved field makes it available
again. Fields cannot be moved out of a struct with a custom drop hook.

At branches, a value must be available on every reachable incoming path. Loop
backedges and `continue` paths must restore values needed on another iteration;
`break` and early returns contribute only to their reachable destinations.
Generic bodies retain operation and type relations until specialization.
Unconstrained type parameters cannot assume implicit copying inside a generic
body: consuming `T` still moves it there. A known wrapper such as `ArcPtr<T>`
can copy for every `T`, because its stored owner handle copies. A `Cell<T>` with
a stored `T` field copies only when that field is known to copy. `Ref<T>` allows
generic code to borrow repeatedly; reading its value is checked for copyability
after specialization.

Completed HIR records moves explicitly. LIR specialization resolves concrete
operations and rejects noncopyable reads through pointers or references. Storage
lowering emits transfers and cleanup, including partial-field cleanup; the LIR
verifier independently checks those storage operations. LIR does not repeat source
initialization analysis.

## Places and references

A place denotes storage: a local, a field, or a dereferenced pointer.
`pointer.*` exposes addressable storage; `&pointer.field` obtains a field pointer.
Locals and `Ref` referents cannot have their addresses taken. A `Ref<T>` parameter
aliases a place without consuming its value. Colon calls insert the receiver as
the first ordinary argument.

```resin
struct Item { value: i32 }
fn read(item: Ref<Item>) -> i32 {
	item.value
}
fn example() -> i32 {
	let item = Item { value = 21 };
	read(item) + item:read()
}
```

References and pointers have unchecked lifetimes and permit aliasing. `Ref<T>`
is read-only; `RefMut<T>` grants writable access without exclusivity. They cannot transfer a noncopyable referent by reading it. Use
`replace(pointer, replacement)` to return the old value and leave a new one in
initialized storage, without destroying the returned value. This is also available
as `pointer:replace(replacement)`.

Ref arguments require initialized places. Bind computed values to locals before
borrowing them; their cleanup follows those locals’ scopes. See
[references](references.md) for argument rules and representation limits.

## Shared ownership

| Access or ownership | One value | Sequence |
| --- | --- | --- |
| Nonowning | `Ptr<T>` | `Span<T>` |
| Shared host ownership | `ArcPtr<T>` | `ArcSpan<T>` |
| Weak host ownership | `WeakPtr<T>` | `WeakSpan<T>` |
| GPU ownership | `GpuPtr<T>` | `GpuSpan<T>` |

Except for `Ptr<T>`, these are ordinary generic library structs. Import
`$/span.resin`, `$/shared.resin`, or `$/gpu.resin` as appropriate. `ArcPtr<Span<T>>`
owns a descriptor; `ArcSpan<T>` owns its elements. There are no unsized payloads.

```resin
import { "$/shared.resin", "$/status.resin" };
fn example() -> i32 | Err<OutOfMemory> {
	let owner = arc_ptr_alloc(i32(42))?;
	let retained = owner; // Retain the same allocation automatically.
	owner:get().* + retained:get().*
}
```

`arc_ptr_alloc(initial)` consumes one value into a shared allocation. If allocation
fails, the consumed initializer is destroyed. `arc_span_alloc(count, initial)`
repeats a copyable initializer, checks allocation arithmetic, and returns only when
all elements are initialized. Move-only initializers cannot be repeated. Empty
allocations are valid. Final release destroys elements in reverse order.

`get` borrows raw access without retaining ownership. An ordinary value copy
retains another strong handle, as does an explicit `clone`; `downgrade` creates a weak handle; `upgrade` returns a retained strong
handle or `None`. The final strong release destroys the payload, while remaining
weak handles retain only bookkeeping. Strong cycles require weak links. Atomic
reference counting does not synchronize payload access or validate raw aliases.

## Destruction

A visible free `fn drop(value: RefMut<Item>) { ... }` is registered as the nominal
type's destruction hook. The hook runs before fields are destroyed in reverse
order. Generic hooks bind their owning type's parameters. Hooks return unit and
handle fallible cleanup locally. Calling `drop` directly is an ordinary call and
does not suppress later automatic destruction; explicit native release must disarm
the stored handle.

Locals are destroyed in reverse scope order. Moves transfer responsibility to the
destination and suppress cleanup of the moved source. Returned values survive
cleanup. Discarded owned results are destroyed, and `?` cleans up acquired owners
and pending expression values on its early-return path. Reference arguments require explicit places; the compiler does not extend
temporary lifetimes. Traps, aborts, and
process termination do not unwind destructors.

Native wrappers can own raw handles safely under these move rules, or use Arc
payloads when callers need shared ownership. Raw pointers, references, and spans
still require their users to keep storage valid, including across reallocation.

## Host and GPU boundary

Reference counts and destructors run on the host. GPU code can address records
containing opaque `StrongOwner`/`WeakOwner` slots, including ordinary neighboring fields. Loading,
storing, copying, upgrading, or otherwise consuming a managed value is rejected
during concrete shader checking with a source-located diagnostic. This includes whole
aggregates containing managed values. Custom destruction is also host-only in this
implementation. `T | None` is available in shader-local values when `T` is compatible.

Managed handles use an opaque 64-bit slot in shared layouts. Their pointee layouts
need not be GPU-compatible. No pointer graph is translated or mirrored across the
host/device boundary. Compiler projection translates owning GPU views in a host launch record into the
shader's raw pointer/span representation, retaining all referenced allocations.

## Standard library

`Gpu`, `GpuPtr<T>`, `GpuSpan<T>`, `GpuImage`, `GpuComputePipeline<Root, Owner>`,
`GpuGraphicsPipeline<Root, Owner>`, `GpuCommands`, `Window`, `InputLine`,
and `ImageData` use shared owners. Copying them retains their shared owners. Explicit `clone` operations remain available. GPU resources
retain their device; a presentation device retains its window. Recorded pipelines,
images, copy buffers, and shader root buffers remain owned until synchronous
submission, cancellation, or destruction of the unfinished recording. Submit/cancel
clear the shared native command handle, so aliases see the same consumed state.
Submission errors return normally when device completion can be confirmed. If both
submission waiting and the fallback device-idle wait fail, the runtime terminates
before releasing resources that may still be in use.

`commands:dispatch(pipeline, arguments, x, y, z)` and
`commands:draw(pipeline, arguments, count)` check host arguments against the typed
pipeline, project them internally, and retain the root and its referenced
allocations after successful recording. `commands:draw(pipeline, None, count)`
supplies no root for a rootless graphics pipeline. Shader entry
parameters remain raw `Ptr<T>` values. Host `GpuPtr` and `GpuSpan` values are owning
GPU views. Indexing, slicing, and explicit cloning retain their owner. Host access
uses checked `load`, `store`, and `replace` methods; it cannot obtain a raw host pointer. Allocation-wide recording state rejects CPU access until work completes or
is canceled; per-view permissions additionally control host reads and writes.
See [GPU buffers](gpu-buffers.md) for projection and layout requirements.

The standard library and its examples no longer require `gpu_destroy`, `gpu_free`,
`gpu_free_pipeline`, `gpu_free_image`, `window_destroy`, `free_input`, or `image_free`.
Those language-facing release functions have been removed. The native C ABI remains
explicit and unsafe. Native-handle fields remain an interoperability escape hatch;
fabricating or mutating them can violate the wrapper's lifecycle invariants.

Use struct destruction hooks for native resource cleanup, as shown in the
[ownership example](../examples/ownership.resin). Do not combine explicit native
destruction with automatic ownership of the same resource.
