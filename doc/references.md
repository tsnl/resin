# References

`Ref<T>` is a fixed, nonowning alias to initialized storage containing a `T`.
It exposes a place: assignment writes the referent, but `&` is forbidden.
Reading a copyable referent copies its value. Moving a noncopyable value through
an alias is rejected. Use `replace(pointer, replacement)` when an operation needs
to transfer a value from explicitly addressable storage.

```resin
import { "$/span.resin", "$/shared.resin" };

fn first<T>(items: Ref<Span<T>>) -> Ref<T> {
	items:at(u64(0))
}
fn increment(value: Ref<i32>) {
	value = value + 1;
}

fn example() -> i32 | Err<_> {
	let items = arc_ptr_alloc([i32(10), i32(20)])?;
	let view = Span<i32> { data = items:get():lea(u64(0)), length = u64(2) };
	let reference: Ref<i32> = first(view);
	increment(reference);
	first(view) = 42;
	reference
}
```

## Bindings and arguments

`let value = expression;` binds an immutable value. `let mut value = expression;`
permits direct reassignment. An explicit `Ref<T>` annotation retains an alias;
unannotated locals and plain `_` holes infer value types.

```resin
let mut value: i32 = 1;
let reference: Ref<_> = value;
let copied = reference; // int copies; copied remains 1.
reference = 2;          // Writes value through the alias.
```

A reference binding cannot be rebound. Assignment writes its referent even when
the binding itself is immutable. References provide unchecked mutable access;
there is no shared/exclusive borrowing distinction.

Passing a place to a `Ref<T>` parameter aliases that place without moving it.
Passing a reference onward aliases the same storage. Referent types must match
exactly; they cannot widen unions or error payloads. `value:operation(args)` is
exactly `operation(value, args)`, including its argument rules. Receiver names
have no special meaning and structs contain only fields.

Reference arguments require initialized places: named locals, fields of places,
pointer dereferences, or expressions that already return `Ref<T>`. A temporary
value cannot bind to a reference parameter. Give it a local first:

```resin
struct Item { value: i32 }
fn read(item: Ref<Item>) -> i32 {
    item.value
}
fn example() -> i32 {
    let item = Item { value = 42 };
    item:read()
}
```

The same rule applies to receivers, overloaded operators, generic calls,
reference initializers, and reference-returning function bodies. For example,
`print(fmt(...))` needs a named formatted string before `print` can borrow it.
The compiler does not create hidden storage to extend an argument's lifetime.
Branches can select places:

```resin
fn choose<T>(left: Ref<T>, right: Ref<T>, flag: bool) -> Ref<T> {
	if (flag) {
		left
	} else {
		right
	}
}
```

A `-> _` annotation infers a value result. Use `-> Ref<_>` to infer the referent
while preserving the reference result.

A destruction hook takes `Ref<T>` as well: `fn drop(value: Ref<Item>)`.
It may mutate the dying value but cannot turn a local value into a pointer.

## Ownership and unchecked lifetimes

References neither retain shared owners nor destroy their referents. They do not
track lifetimes or prevent aliasing. Returning a reference to a local, storing a
reference to a local that has left scope, or using a reference after its owner
is moved or destroyed can leave a dangling alias. Callers must keep storage alive,
initialized, and writable. String literal storage is read-only.

Use `ArcPtr<T>` or `ArcSpan<T>` when a value must retain shared ownership. Explicit
`owner:clone()` creates another owning handle. A raw reference or pointer obtained
from an Arc still does not retain it; keep an owning handle alive during access.
Atomic reference counts do not synchronize writes to the payload.

`Ptr<T>` is an ordinary copyable pointer. `pointer.*` accesses its pointee and
`&pointer.field` obtains a pointer to a field of that pointee. Local values, their
inline fields, and `Ref` bindings cannot have their addresses taken. A `Ref`
parameter or result keeps this restriction even when its referent came from a
pointer. Functions that expose an address must accept or return `Ptr<T>` explicitly.
Reading a pointer through a reference and dereferencing it still accesses
addressable storage; `Ref<Ptr<T>>` does not remove the stored pointer’s capability.
`Ref<Ptr<T>>` aliases a pointer slot: assignment changes the stored pointer.

## Indexing and representation

Arrays, `Span<T>`, and `str` support `items:at(index)` returning `Ref<T>` (or
`Ref<u8>` for `str`). The index is `u64`; receivers and indices evaluate once.
Use `items:at(i) = value` to write. `:lea(i)` returns an element pointer for
pointers to arrays, `Span<T>`, and `str`; it is unavailable on local array values
and references to arrays. Thus an allocated array can produce pointers without
allowing an inline local array to expose its address. Both methods check host
indices before producing a result.
Host bounds diagnostics and unchecked shader indexing retain their existing rules.
The low-level `pointer_index` intrinsic returns a pointer. `ArcPtr<T>:get()` also
returns a pointer. Host `GpuSpan<T>:at()` returns an owning `GpuPtr<T>` supporting
`load`, `store`, and `replace`.

HIR, specialization, and LIR retain distinct reference and pointer types.
`LocalRef` borrows a local; `Borrow` turns access through an existing pointer into
a reference. There is no reverse operation. Loads, stores, and projections preserve
these capabilities, and verification rejects references passed or returned as
pointers, pointer casts of references, and pointer-only indexing on references.
C emission uses pointer representations for references without granting Resin
pointer operations. In SPIR-V, device references use
physical pointers. Helpers receiving local references are specialized for their
root object and field/index path; dynamic indices are passed separately. This
preserves aliases without copying a field in and out. Shader-local references
can cross helper calls and bind local aliases, but returning them from a helper
or selecting distinct local referents across branches remains unsupported.
Local reference referents require shader value support, independently of device
buffer layout. Helpers can borrow local `bool` values, boolean arrays, and records
containing booleans. Explicit device pointers still require a shared storage layout. Local projection specialization has a 16,384-variant
budget and a 32-call nesting guard.

References are limited to bindings and function signatures. Reference fields,
array elements, union members, error payloads, nested references, `Ptr<Ref<T>>`, and
reference-valued generic arguments are rejected, including through aliases.
Function values with reference parameters/results may be stored in aggregates.
Foreign declarations use pointer ABI types. There is no `Ref<T>(value)` constructor.
