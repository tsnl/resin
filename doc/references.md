# References

`Ref<T>` is a fixed, nonowning **read-only** alias to initialized storage.
`RefMut<T>` is its **writable** counterpart: assignment writes the referent.
Neither permits taking the referent's address with `&`.

| Contract | Read | Write the referent | Take its address |
| --- | --- | --- | --- |
| `Ref<T>` | Yes | No | No |
| `RefMut<T>` | Yes | Yes | No |
| `Ptr<T>` | Yes | No | Already a pointer |
| `PtrMut<T>` | Yes | Yes | Already a pointer |

A `RefMut<T>` may weaken to `Ref<T>` at a call, binding, or return. A `Ref<T>`
never strengthens to `RefMut<T>`. Both reference kinds are aliasable; writable
does not mean exclusive. Read-only access does not promise that other aliases
cannot change the value.
Reading a copyable referent copies its value. Moving a noncopyable value through
an alias is rejected. Use `replace(pointer, replacement)` when an operation needs
to transfer a value from explicitly addressable storage.

```resin
import { "$/span.resin", "$/shared.resin" };

fn first_mut<T>(items: Ref<SpanMut<T>>) -> RefMut<T> {
	items:at_mut(u64(0))
}
fn increment(value: RefMut<i32>) {
	value = value + 1;
}

fn example() -> i32 | Err<_> {
	let items = arc_ptr_alloc([i32(10), i32(20)])?;
	let view = SpanMut<i32> { data = items:get():lea(u64(0)), length = u64(2) };
	let reference: RefMut<i32> = first_mut(view);
	increment(reference);
	first_mut(view) = 42;
	reference
}
```

## Bindings and arguments

`let value = expression;` binds an immutable value. `let mut value = expression;`
permits reassignment and writable borrowing. An explicit `Ref<T>` or `RefMut<T>` annotation retains an alias;
unannotated locals and plain `_` holes infer value types.

```resin
let mut value: i32 = 1;
let reference: RefMut<_> = value;
let copied = reference; // i32 copies; copied remains 1.
reference = 2;          // Writes value through the alias.
```

A reference binding cannot be rebound. Assigning through `RefMut` writes its
referent even when the reference binding itself lacks `mut`. Adding `mut` to a
`Ref` binding does not grant permission to mutate its referent.

Borrowing a local or inline field as `RefMut` requires a `let mut` binding (or a
`mut` parameter). Borrowing through an existing `RefMut` or writable pointer uses
that access permission. `Ref` can borrow an immutable local.

Passing a place to a `Ref<T>` parameter aliases that place without moving it.
Passing a reference onward aliases the same storage. Referent types must match
exactly; they cannot widen unions or error payloads. `value:operation(args)` is
exactly `operation(value, args)`, including its argument rules. Receiver names
have no special meaning and structs contain only fields.

Reference arguments require initialized places: named locals, fields of places,
pointer dereferences, or expressions that already return either reference kind. A temporary
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
while preserving the reference result; `-> RefMut<_>` preserves writable access.

A destruction hook takes `RefMut<T>`: `fn drop(value: RefMut<Item>)`.
It may mutate the dying value but cannot turn a local value into a pointer.

## Ownership and unchecked lifetimes

References neither retain shared owners nor destroy their referents. They do not
track lifetimes or prevent aliasing. Returning a reference to a local, storing a
reference to a local that has left scope, or using a reference after its owner
is moved or destroyed can leave a dangling alias. Callers must keep storage alive,
and initialized; writes also require writable backing storage. String literal storage is read-only.

Use `ArcPtr<T>` or `ArcSpan<T>` when a value must retain shared ownership. Copying an owner creates another retaining handle;
`owner:clone()` also remains available. A raw reference or pointer obtained
from an Arc still does not retain it; keep an owning handle alive during access.
Atomic reference counts do not synchronize writes to the payload.

`Ptr<T>` is an ordinary copyable read-only pointer; `PtrMut<T>` grants write
permission. `pointer.*` accesses its pointee, and `&pointer.field` preserves the
pointer's permission when addressing a field. A `PtrMut<T>` can weaken to `Ptr<T>`
at a call, binding, or return. Pointee types remain invariant: this does not convert
`PtrMut<PtrMut<T>>` into `Ptr<Ptr<T>>`.

Local values, inline local fields, and either reference kind cannot have their
addresses taken. A reference parameter or result keeps this restriction even when
its referent came from a pointer. No implicit reference-to-pointer conversion exists.

Permission belongs to each access, separately from binding mutability.
`let p: PtrMut<T>` can write its pointee; `let mut p: Ptr<T>` can replace the pointer
binding but cannot write its pointee. `Ref<PtrMut<T>>` can read the stored writable
pointer and write through it, but cannot replace the pointer slot. Similarly,
`Ptr<PtrMut<T>>` cannot replace its first pointee, although the inner pointer can
write its own pointee. This is aliasable access, not a borrow checker.

| View | Read-only | Writable |
| --- | --- | --- |
| Borrowed pointer | `Ptr<T>` | `PtrMut<T>` |
| Borrowed sequence | `Span<T>` | `SpanMut<T>` |
| Retained GPU pointer | `GpuPtr<T>` | `GpuPtrMut<T>` |
| Retained GPU sequence | `GpuSpan<T>` | `GpuSpanMut<T>` |

The span and GPU families are ordinary source structs. Their `:read_only()` method
explicitly weakens a writable view; there is no implicit conversion between nominal
view structs. Slicing and `:lea()` preserve access. Host shared owners' `:get()`
returns `PtrMut` or `SpanMut`; copying these borrowed views does not retain the owner.
Raw pointer/integer casts and foreign declarations remain low-level unchecked
boundaries, not a memory-safety guarantee. A direct cast cannot strengthen a `Ptr`
into `PtrMut`.

## References do not grant pointers

Both of these complete programs are rejected:

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

## Indexing and representation

Use `Ptr<T>` for one value and `Span<T>` for a sequence; use their `Mut` variants
when writes are needed. A span keeps the element count with the address, including
for paths and text slices. Preserve that length across application and library calls.
NUL-terminated pointers and pointer/count pairs belong at native or compiler boundaries.

Direct pointer arithmetic (`p + n`, `p - n`, or `p - q`) is rejected. Obtain an
element address through `view:lea(index)` and a subview through `view:slice(start, length)`.
Explicit pointer/integer casts remain available for low-level host interop; ordinary
indexing should not discard bounds by converting addresses to integers.

Arrays, both span kinds, and `str` support `items:at(index)` returning `Ref<T>` (or
`Ref<u8>` for `str`). The index is `u64`; receivers and indices evaluate once.
`:at_mut(index)` returns `RefMut<T>` for writable array access and `SpanMut<T>`.
An inline array needs a writable place; a read-only array reference cannot call
`at_mut`. String literals have no `at_mut`. Use `items:at_mut(i) = value` to write. `:lea(i)` returns an element pointer for
pointers to arrays, both span kinds, and `str`; it is unavailable on local array values
and references to arrays. Thus an allocated array can produce pointers without
allowing an inline local array to expose its address. These methods check host
indices before producing a result.
Host bounds diagnostics and unchecked shader indexing retain their existing rules.
The low-level `pointer_index` intrinsic returns a pointer. `ArcPtr<T>:get()` also
returns a pointer. Host `GpuSpan<T>:at()` returns `GpuPtr<T>`; `GpuSpanMut<T>:at()` returns
`GpuPtrMut<T>`. Both can load; only the mutable pointer can store or replace.
String literal bytes and process argument/environment snapshots expose read-only pointers.

HIR, specialization, and LIR retain reference permissions separately from pointer
capabilities. `LocalRef` borrows local storage as `RefMut`; HIR checks whether a
source binding permits that access. `Borrow` preserves a pointer’s permission in the resulting reference.
`ReadOnly` weakens a writable reference or pointer. There are no reverse operations. Loads, stores, and projections preserve
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
