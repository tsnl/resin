# References

`Ref<T>` is a fixed, nonowning alias to initialized storage containing a `T`.
It exposes a place: assignment writes the referent, but `&` is forbidden.
Reading a copyable referent copies its value. Moving a noncopyable value through
an alias is rejected. Use `replace(pointer, replacement)` when an operation needs
to transfer a value from explicitly addressable storage.

```resin
import { "$/span.resin", "$/shared.resin" };

fn first<T>(items: Ref<Span<T>>) -> Ref<T> {
	items:at(0_ul)
}
fn increment(value: Ref<int>) {
	value = value + 1;
}

fn example() -> int | Err<_> {
	let items = arc_ptr_alloc([10_i, 20_i])?;
	let view = Span<int> { data = items:get():lea(0_ul), length = 2_ul };
	let reference: Ref<int> = first(view);
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
let mut value = 1_i;
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

A function argument may also be a temporary value. The compiler materializes its
storage and destroys it at the end of the full expression: a binding initializer,
expression statement, block tail, or loop condition. Nested calls share that
expression's temporary lifetime, and early exits clean up acquired temporaries.

```resin
struct Item { value: int }
fn read(item: Ref<Item>) -> int {
	item.value
}
fn example() -> int {
	Item { value = 42 }:read()
}
```

This argument rule does not extend local reference initializers. An explicit
`let reference: Ref<T> = expression;` still requires an initialized place, as does
a reference-returning function body. Branches can select places:

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
reference returned from a borrowed temporary, or using a reference after its owner
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
`Ref<ubyte>` for `str`). The index is `ulong`; receivers and indices evaluate once.
Use `items:at(i) = value` to write. `:lea(i)` returns an element pointer for
pointers to arrays, `Span<T>`, and `str`; it is unavailable on local array values
and references to arrays. Thus an allocated array can produce pointers without
allowing an inline local array to expose its address. Both methods check host
indices before producing a result.
Host bounds diagnostics and unchecked shader indexing retain their existing rules.
The low-level `pointer_index` intrinsic returns a pointer. `ArcPtr<T>:get()` also
returns a pointer. Host `GpuSpan<T>:at()` returns an owning `GpuPtr<T>` supporting
`load`, `store`, and `replace`.

HIR retains reference types and explicit use relations. LIR specialization chooses
address and read operations after resolving dependent signatures. References use the existing pointer ABI in C. In SPIR-V, device references use
physical pointers. Helpers receiving local references are specialized for their
root object and field/index path; dynamic indices are passed separately. This
preserves aliases without copying a field in and out. Shader-local references
can cross helper calls and bind local aliases, but returning them from a helper
or selecting distinct local referents across branches remains unsupported.
The current LIR pointer representation also requires reference referents to have
a shared host/device layout; shader-local types such as `bool` cannot yet be
borrowed across calls. Local projection specialization has a 16,384-variant
budget and a 128-call nesting guard.

References are limited to bindings and function signatures. Reference fields,
array elements, union members, error payloads, nested references, `Ptr<Ref<T>>`, and
reference-valued generic arguments are rejected, including through aliases.
Function values with reference parameters/results may be stored in aggregates.
Foreign declarations use pointer ABI types. There is no `Ref<T>(value)` constructor.
