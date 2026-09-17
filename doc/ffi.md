# Foreign functions

Standard-library modules keep native declarations private and export Resin wrappers that
check statuses before returning out-parameter values. For example:

```resin
export { Gpu, gpu_new };
extern {
	"resin_runtime.h": {
		fn resin_gpu_create(gpu: Ptr<Ptr<ResinGpu>>) -> int;
		fn resin_gpu_destroy(gpu: Ptr<ResinGpu>);
	},
};
import { "$/status.resin", "$/shared.resin" };

extern type ResinGpu;
struct GpuOwner { handle: Ptr<ResinGpu> }
fn drop(owner: Ref<GpuOwner>) {
	if (ulong(owner.handle) != 0_ul) {
		resin_gpu_destroy(owner.handle);
	};
}
struct Gpu { owner: ArcPtr<GpuOwner> }
fn gpu_new() -> Gpu | Err<RuntimeError> {
	let owner = arc_ptr_alloc(GpuOwner { handle = Ptr<ResinGpu>(0_ul) })?;
	runtime_status_from_code(resin_gpu_create(&owner:get().handle))?;
	Gpu { owner = owner }
}
```
Declare foreign functions in a top-level `extern` block after any `export` clause
and before any `import` clause. Each header names a group of ordinary `fn`
signatures. Groups use comma separators with an optional trailing comma; the block
ends with `};`. The block and individual groups may be empty. Every declared header
remains a native dependency even when none of its functions is called, including
empty groups. The old `extern "header.h" fn ...;` spelling
is no longer accepted.

Header groups do not introduce a scope: their functions have the module's usual
export rules, and signatures can use types from its imports and declarations.
Opaque foreign types remain standalone `extern type Name;` declarations.

Foreign headers resolve through ordered client `-I` / `--include-root` directories,
the declaring file's directory, advertised managed header roots, then target system
headers. Local absolute spellings must resolve on the client. Selected directories are
uploaded as immutable bundles; their transitive includes must remain within those
bundles or configured service toolchain roots. See [C header directories](compiler-service.md#c-header-directories).
Use forward slashes in header paths, including Windows paths such as `C:/SDK/include/api.h`.

The prototype targets 64-bit hosts. Foreign functions accept scalar/pointer parameters and return a scalar, pointer, or unit.
The wrapper forwards each Resin parameter as a separate C argument. Opaque `extern type`
declarations name C structs and may only be used behind pointers; aggregates by value,
variadic calls, and C callbacks are not supported yet.

This is an unchecked C boundary: declarations must match the header's ABI, and callers own
pointer validity, lifetimes, buffer lengths, and synchronization. `&place` takes an address only for storage reached through a pointer;
locals and `Ref` referents cannot expose their addresses.
`pointer.*` dereferences it. On the host, explicit casts allow pointer-to-pointer and
pointer-to-`ulong` roundtrips. There is no borrow checker; borrowed pointers and references must not outlive their storage.
Pointer arithmetic is forbidden. Use array or span indexing, or explicitly convert a pointer
into `ulong`, perform **byte** arithmetic, and convert back when low-level address manipulation
is necessary on the host. Shaders reject pointer casts; use typed pointers and indexing.
Host pointer casts and all raw pointer dereferences remain unchecked.

Arrays, spans, and `str` use `:at(index)` for indexing and return `Ref<T>` (`Ref<ubyte>` for `str`).
The index parameter is `ulong` (unsigned 64-bit); unsuffixed literals infer this type, while
other integer values need an explicit conversion, such as `:at(ulong(i))`:

```resin
let values = arc_ptr_alloc([10_i, 20, 30])?;
values:get():at(1) = 42;
let view = Span<int> { data = values:get():lea(0), length = 3_ul };
let element = view:at(1);
print(fmt("{0}\n", (element,)));
```
Import `$/shared.resin` and `$/span.resin` for this allocated view.
`:lea(index)` returns a pointer for pointers to arrays, spans, and `str`.
Local arrays support `:at(index)` but cannot produce element pointers.

Use `let element: Ref<int> = view:at(1);` to retain an alias instead of copying the
value. Reference parameters and results expose the same place semantics in user
functions; see [references](references.md) for binding, lifetime, and migration rules.

Arrays retain the original `values(index)` spelling; source spans use `:at(index)`. `Span<T>` has `data: Ptr<T>`
and `length: ulong` fields. Host indexing checks
the array or span length and terminates with a diagnostic for negative or out-of-range indices,
before forming an element address. This failure does not unwind automatic cleanup.
Shader array and span indexing is unchecked: callers must keep indices within valid storage;
out-of-range access has undefined behavior. Constructing a span does not validate its pointer,
allocation size, or lifetime.

Initialize output slots before passing their addresses: Resin does not infer initialization
effects from foreign calls. String literals are NUL-terminated; pass their storage with a
span data field, such as `path.data` for `let mut path = "triangle.png";`.
