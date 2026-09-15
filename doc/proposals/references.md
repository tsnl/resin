# Proposal: source references and reference-returning indexing

Status: proposed language change. The syntax and APIs below are not implemented.

Expose `Ref<T>` for a nonowning alias to initialized storage containing a `T`.
A function returning `Ref<T>` exposes that storage directly to its caller, so
indexing and user-defined accessors support reads, assignment, and address
formation without a trailing `.*`.

Use **Ref** in source and **place** for the compiler's expression category.
`Place<T>` and `CppRef<T>` would not be additional source spellings. The feature
adopts fixed reference bindings and access to the referent through ordinary
expressions. It does not adopt C++'s complete reference type system.

## Source behavior

```resin
def first<T>(items: Span<T>) -> Ref<T> = {
    items.at(0)
};

def increment(value: Ref<int>) = {
    value := value + 1;
};

// Inside a function with an initialized Span<int> named items:
var copied = first(items);              // Copy the element into an int local.
first(items) := 42;                     // Replace the element.
var alias: Ref<int> = first(items);      // Bind to the element's location.
increment(alias);                      // Pass that location to increment.
var pointer = &first(items);            // Obtain Ptr<int>.
```

The consumer determines how a place is used:

| Consumer | Behavior |
| --- | --- |
| A value of type `T` | Read the referent using Resin's ordinary copying rules. |
| A reference of type `Ref<T>` | Bind to the same location without copying `T`. |
| An assignment destination | Replace the referent using ordinary assignment rules. |
| Unary `&` | Produce a `Ptr<T>` to the referent. |
| A field or method receiver | Look up the referent's members and preserve place access where required. |

Variables already follow this distinction between location and value. A call
returning `Ref<T>` would have the same behavior. A call returning `T` continues
to produce a value; a reference context cannot turn a temporary value into a
reference to newly materialized storage.

### Binding syntax and inference

Add initialized local annotations:

```resin
var value: int = 10;
var reference: Ref<int> = value;
var another: Ref<int> = reference;
var copied = reference;
reference := 20;
```

The last assignment changes `value`. `reference` and `another` remain aliases
of the same location. `copied` is an independent `int` with the value 10.

The current grammar has `var name = expression;` and `var name: Type;`, but no
`var name: Type = expression;`. The initialized annotation is an explicit
syntax addition in this proposal, including for ordinary value types. Its
annotation supplies the initializer's expected type and records the binding's
declared type in the AST.

Unannotated value bindings read their initializer as a value. A reference
annotation is required to retain an alias in a local binding. `_` continues
to infer a value type; `Ref<_>` explicitly requests a reference with an inferred
referent type. A function's result annotation must likewise explicitly request
`Ref<T>`; a plain result hole must not infer reference semantics from a tail
expression.

Reference locals require an initializer. `var reference: Ref<int>;` is rejected;
later assignment always writes the referent and cannot establish or change a
reference binding. No new reference-construction expression is required by this
proposal.

### Parameters, results, and control flow

A parameter of type `Ref<T>` binds to the argument's place. Passing an ordinary
local, a field, a pointer dereference, or another reference-returning call is
allowed. The referent type must match exactly; reference binding cannot widen
unions or perform numeric conversions.

A function with result `Ref<T>` checks its body in a reference context:

```resin
def choose<T>(left: Ref<T>, right: Ref<T>, take_left: bool) -> Ref<T> = {
    if (take_left) { left } else { right }
};
```

Each arm supplies a location. The call returns the selected location without
copying its contents. Blocks and exhaustive matches propagate a reference
context to their result expressions in the same way. Ordinary value contexts
continue to read their selected result as a value.

Reference parameters borrow their referents. Leaving a function does not destroy
those referents. Calls retain callee-first, left-to-right evaluation, and an
accessor used as an assignment destination is evaluated once. Multiple reference
parameters may alias; writes through one must be visible through another.

### Ownership, initialization, and pointers

Binding requires initialized storage. This keeps references from introducing a
second mechanism for output parameters or definite initialization through aliases.
HIR construction checks source-known locals, including in unused definitions.
For places obtained from raw pointers, initialization remains a caller contract;
the compiler does not gain heap-state or alias analysis from a reference type.

A reference neither retains an owning handle nor extends the referent's lifetime.
Reading a managed `T` through a reference still performs the ordinary retain or
recursive copy. Replacing it still destroys the previous value. Binding or
passing the reference performs neither operation on `T`.

There is no borrow checker or new lifetime guarantee. Callers keep the backing
storage alive and stable; returning a reference does not preserve a local or an
allocation whose last owner is destroyed. As with `Ptr<T>`, writes require
writable storage. References into string literal storage must be treated as
read-only; this proposal adds no const-qualified type.

`Ptr<T>` remains an ordinary pointer value. `pointer.*` still produces its
pointee's place, and `&place` still produces a pointer. For example:

```resin
def refer<T>(pointer: Ptr<T>) -> Ref<T> = {
    pointer.*
};
```

There is no implicit conversion from `Ptr<T>` to `Ref<T>`, or from `Ref<T>`
to `Ptr<T>`. Use `pointer.*` and `&reference` respectively. A reference to a
pointer, `Ref<Ptr<T>>`, aliases the pointer slot; reading it yields `Ptr<T>`.
Raw pointer dereferencing retains its existing validity requirements. Merely
giving the result a reference type does not validate an arbitrary address.

## Indexing and API migration

Change array `.at(index)`, the retained array-call indexing syntax, and
`Span<T>.at(index)` to return `Ref<T>`. Change `str.at(index)` to return
`Ref<ubyte>`. Indices and bounds behavior remain as specified today: `.at()`
takes `ulong`, host indexing checks bounds, and shader indexing requires valid
indices supplied by the caller.

The source `Span` method can wrap the existing pointer intrinsic:

```resin
def at(self: Span<T>, index: ulong) -> Ref<T> = {
    pointer_index(self.data, self.length, index).*
};
```

The primitive `pointer_index` contract can continue to return `Ptr<T>`. Pointer
arithmetic, byte views, and native pointer/length ABIs retain their current
representations.

| Current use | Proposed use |
| --- | --- |
| `var value = items.at(i).*;` | `var value = items.at(i);` |
| `items.at(i).* := value;` | `items.at(i) := value;` |
| `var pointer = items.at(i);` | `var pointer = &items.at(i);` |
| `native_call(items.at(i));` for a `Ptr<T>` parameter | `native_call(&items.at(i));` |
| `var pointer = items.at(i); pointer.* := value;` | `var reference: Ref<T> = items.at(i); reference := value;` |

This is a source-breaking API change. Migration must inspect the consumer:
deleting `.*` alone can leave a pointer binding as an unintended value copy.
Indexing pointer-valued elements needs particular care because `items.at(i).*`
would now dereference the element's pointer value.

Update the borrowed indexing callers in `resin/`, examples, benchmarks, tests,
and documentation together. Owning host accessors such as `ArcPtr<T>.get()`
can remain pointer-returning in this change. Host `GpuSpan<T>.at()` continues
to return an owning GPU view with `load`, `store`, and `replace`; it must not
expose a raw host reference to device storage.

## Compiler boundaries

The implementation should preserve the source distinction until its behavior
has been made explicit:

1. Syntax and AST retain initialized binding annotations. The formatter and
   editor grammar recognize `Ref<T>` and the new binding form.
2. HIR construction resolves reference types, checks valid binding locations
   and initialization, and distinguishes binding a reference from using its
   referent as a place. Completed HIR carries these decisions without relying
   on an expected-type inference pass in LIR construction.
3. LIR specialization preserves reference distinctions in source instance
   identities while translating reference transport into pointer storage and
   explicit address/dereference operations. Storage lowering consumes that
   completed concrete tree and retains its existing copying and cleanup rules.
4. Verification and code generation consume the resulting explicit operations.
   Host emission can use its existing pointer representation and C ABI machinery
   for ordinary Resin calls.

Reference-result handling must cover direct calls, function values, source
methods, and dependent method selection. Generic inference should deduce `T`
from a referent for a `Ref<T>` parameter, while a value parameter deduces and
receives the value type. Reference context must not introduce new inference
of dependent receivers or method arguments during specialization.

The current shader backend rejects local addresses stored as values or passed
through calls. This change retains those restrictions and source-located
diagnostics. Device-storage reference access may use the existing device pointer
path, subject to the same shader type and operation checks. A reference does not
permit shaders to read or copy managed host values.

Shader helper variants by storage class, transport of containing locals and
indices, and general local-reference results belong in a separate change.

## Initial scope and validation

Support reference locals, parameters, and function results. A referent may be an
ordinary scalar, pointer, aggregate, or nominal value type. Function value types
may include reference parameters and results.

Defer references stored directly as aggregate fields, array elements, union or
Result payloads, `Ptr<Ref<T>>`, nested `Ref<Ref<T>>`, and supplying a reference
as a generic value-type argument. Reject these forms explicitly, including when introduced
through a transparent alias. Read-only references, temporary lifetime extension,
new foreign-reference signatures, and changes to destruction-hook signatures are
also outside this initial feature.

Implementation validation should cover:

- Parser, formatter, and editor behavior for initialized annotations and
  reference signatures.
- Reads versus alias bindings; mutation and address formation through a returned
  reference; exact referent typing and generic inference.
- Calls through function values and methods, including dependent methods whose
  selected signatures contain reference parameters or results.
- Conditional, block, and match results; single evaluation of effectful accessors;
  argument evaluation order and aliasing between parameters.
- Managed referents: binding causes no retain/drop, value reads copy normally,
  and replacement destroys the previous value exactly once.
- Diagnostics for uninitialized referents, missing reference initializers,
  temporary binding, and unsupported reference-containing types.
- Migrated indexing reads, writes, raw-address consumers, pointer-valued elements,
  and existing host bounds failures.
- Existing device-storage shader access and continued rejection of unsupported
  local-address transport and managed-value reads.

Run the appropriate crate tests and complete executable tests in `shell.nix`,
then the workspace checks required by the implementation. The proposal itself
changes documentation only and does not claim compiler support for these examples.
