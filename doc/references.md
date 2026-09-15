# References

`Ref<T>` is a fixed, nonowning reference to initialized storage containing a `T`.
A function returning `Ref<T>` exposes that storage to its caller. Reading it copies
`T`; assignment writes the referent; `&` obtains its address.

```resin
import { "$/span.resin" };

def first<T>(items: Span<T>) -> Ref<T> = { items.at(0_ul) };
def increment(value: Ref<int>) = { value := value + 1; };

def example() -> int = {
    var items = [10_i, 20_i];
    var view = Span<int> { data = &items.at(0_ul), length = 2_ul };
    var reference: Ref<int> = first(view);
    increment(reference);
    first(view) := 42;
    reference
};
```

The source spelling is **Ref**. A **place** is the compiler's category for an
expression with an address, such as a local, a field, or a dereferenced pointer.

## Bindings and calls

Initialized locals accept an optional annotation: `var name: Type = expression;`.
Use `Ref<T>` explicitly to retain a reference. Unannotated locals and plain `_`
holes infer value types, including when their initializer returns a reference.

```resin
var value: int = 1;
var reference: Ref<_> = value; // Alias value's storage; infer int as the referent.
var copy = reference;         // Copy int into independent storage.
reference := 2;               // Update value. copy remains 1.
```

A reference binding cannot be rebound. Passing a place to a `Ref<T>` parameter
binds to that place without copying its contents. Passing a reference onward binds
to the same place. A `T` parameter instead receives an ordinary value copy.
Reference referent types must match exactly; they cannot widen unions or Results.

A reference needs an initializer, and known local storage must be initialized
before it can be bound. Literal values, arithmetic results, and calls returning
ordinary values cannot initialize references. Functions returning references check
their body in a reference context, including branch and match results:

```resin
def choose<T>(left: Ref<T>, right: Ref<T>, flag: bool) -> Ref<T> = {
    if (flag) { left } else { right }
};
```

Generic functions, function values, source methods, and dependent method calls
support reference parameters and results. A method can borrow its receiver with
`self: Ref<Cell<T>>`. This receiver requires a place, just like a reference argument.
A result annotation of `-> _` infers a value result; use `-> Ref<_>` to request a
reference with an inferred referent type.

## Pointers and lifetime

`Ptr<T>` is an ordinary pointer value. Use `pointer.*` to access its pointee and
`&reference` to get a pointer to a reference's referent. Neither conversion is
implicit. `Ref<Ptr<T>>` aliases a pointer slot: assignment changes the stored
pointer, and reading the reference yields that pointer.

```resin
def refer<T>(pointer: Ptr<T>) -> Ref<T> = { pointer.* };
```

References do not retain shared owners or destroy referents when their bindings
leave scope. Reading a managed referent performs its normal compiler-defined copy;
replacing it performs normal assignment and destruction. Assignment's result and
the stored replacement are separate copies, as for other assignment expressions.

There is no borrow checker or lifetime extension. The caller must keep referenced
storage alive and must only write writable storage. Returning a reference to a
local, or retaining a reference after its owner is destroyed, can leave a dangling
reference. Pointer-derived storage has the same initialization and validity
obligations as raw pointer access. String literal storage is read-only.

## Indexing migration

Arrays, `Span<T>`, and `str` return `Ref<T>` from `.at(index)` (`Ref<ubyte>` for
`str`). Arrays also retain `items(index)`. Receivers and indices are evaluated once;
indexing an array place preserves its storage. `.at()` accepts `ulong`. Host bounds
diagnostics and unchecked shader indexing retain their existing behavior.

| Previous source | Current source |
| --- | --- |
| `var value = items.at(i).*;` | `var value = items.at(i);` |
| `items.at(i).* := value;` | `items.at(i) := value;` |
| `native_call(items.at(i));` with a `Ptr<T>` parameter | `native_call(&items.at(i));` |
| `var p = items.at(i); p.* := value;` | `var r: Ref<T> = items.at(i); r := value;` |

For pointer-valued elements, `items.at(i)` now reads the stored pointer, while
`&items.at(i)` addresses its slot. An additional `.*` dereferences that stored
pointer. Migrate according to the intended level of indirection.

The low-level `pointer_index` intrinsic still returns `Ptr<T>`; `Span<T>.at()`
returns its dereferenced place. Owning accessors such as `ArcPtr<T>.get()` retain
their pointer API. Host `GpuSpan<T>.at()` still returns a `GpuPtr<T>` for
`load`, `store`, and `replace`.

## Representation and current limits

HIR retains reference types and explicit reference-use relations. A dependent
result retains its value-type projection until specialization knows whether the
selected function returns a reference. LIR specialization translates bindings and
reads into address and dereference operations. References then use the existing
pointer storage and calling convention in both C and SPIR-V.

Shader helpers can pass and return references to device storage. Shader-local
addresses retain the backend's existing restrictions: they cannot escape through
calls, stored bindings, or return values. This feature does not add another shader
calling convention or relax shader pointer-cast restrictions.

References are currently limited to bindings and function signatures. Direct
reference fields, array elements, union variants, Result payloads, nested
references, `Ptr<Ref<T>>`, and reference-valued generic arguments are rejected.
Function values with reference parameters or results can still be stored in
aggregates. Transparent aliases preserve these restrictions. Foreign declarations
use the existing pointer ABI types. There is no `Ref<T>(value)` constructor.
