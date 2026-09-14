# Remaining template and standard-library design

Status: draft for the work after the implementation through
[#180](https://github.com/tsnl/resin/pull/180). Named function templates, generic
aliases, and the polymorphic HIR/LIR foundation are on `main`. Source generic
structs and method parameters are still rejected during HIR construction, even
though their syntax and the underlying nominal representation exist.

This proposal finishes those source features, then uses them to replace special
compiler-provided library types with ordinary generic definitions. The
[implementation plan](template-implementation.md) describes the remaining
translations, ownership requirements, and PR sequence. The
[architecture guide](architecture.md) describes the compiler as it exists today.

## What has landed

| Area | Implemented foundation |
| --- | --- |
| Nanopass preparation | [#162](https://github.com/tsnl/resin/pull/162), [#165](https://github.com/tsnl/resin/pull/165), and [#166](https://github.com/tsnl/resin/pull/166): per-function LIR state and source origins, declaration identities, and direct completion into HIR. |
| Declaration syntax | [#168](https://github.com/tsnl/resin/pull/168) moves methods inside structs; [#169](https://github.com/tsnl/resin/pull/169) represents named binders and explicit applications, including method arguments. |
| Polymorphic representation | [#171](https://github.com/tsnl/resin/pull/171) gives HIR its own type language; [#172](https://github.com/tsnl/resin/pull/172) adds bounded LIR specialization. |
| Translation guarantees | [#174](https://github.com/tsnl/resin/pull/174) establishes initialization in HIR construction; [#175](https://github.com/tsnl/resin/pull/175) introduces explicit compilation roots; [#176](https://github.com/tsnl/resin/pull/176) selects shader operations in LIR. |
| Symbolic operations and inference | [#177](https://github.com/tsnl/resin/pull/177) retains dependent operations; [#178](https://github.com/tsnl/resin/pull/178) enables function templates and contextual deduction. |
| Type applications | [#179](https://github.com/tsnl/resin/pull/179) enables generic aliases; [#180](https://github.com/tsnl/resin/pull/180) normalizes nominal applications before materializing layouts and hooks. |

The old [Nanopass scope](https://github.com/tsnl/resin/pull/160) is superseded by
these changes. It is no longer a prerequisite PR. HIR retains one polymorphic
body per definition; LIR already owns concrete instances, types, and target
selection. The remaining work extends these languages and translations.

The following contracts are already established and must survive that extension:

- Named `<T>` binders are rigid in definitions. `_` is a weak monomorphic variable,
  which may unify with a bound parameter; it never adds a template parameter.
  A caller cannot finish a definition's unresolved result variable.
- Each function application deduces fresh arguments from operands and expected
  results, or uses explicit `identity::<int>` arguments. Local storage remains
  monomorphic. Unsuffixed numeric literals participate in the same constraints;
  unconstrained integers default to `long`, floats to `float64`, during HIR
  construction. A suffix fixes the literal's type and retains range checking.
- Transparent aliases substitute their resolved bodies in their defining scope.
  They preserve nominal origins, add no method namespace, and resolve in
  declaration order. Recursive expansion is an error, including beneath `Ptr`.
  Alias definitions and nominal fields currently require explicit types.
- HIR can retain operations whose concrete support depends on type arguments.
  LIR resolves determining types and selects supported operations without
  inferring missing arguments or choosing numeric defaults. An unsupported
  monomorph fails when requested; templates need not work for every type/platform.
- `CompilerConfig.max_monomorphs_per_function` defaults to 16,384. Definition,
  normalized arguments, and Host/Shader profile determine reuse. Owner and method
  arguments must participate in those same keys and the same allowance.
  Independent depth and size limits bound type expansion. Memoization does not
  prove that every source program has finitely many possible instances.

## Generic source structs

Enable named parameters on nominal declarations using the syntax already retained
by the AST:

```resin
struct Pair<T> { left: T, right: T };
type PairPtr<T> = Ptr<Pair<T>>;

def swap<T>(pair: Pair<T>) -> Pair<T> = {
    Pair<T> { left = pair.right, right = pair.left }
};
```

This example is proposed source functionality. `Pair<int>` and `Pair<float32>`
are distinct nominal types. Repeated uses of one application share its identity;
equal layouts or identical spelling in another module do not make types equal.
Deduction through `Pair<T>` matches the declaration origin and arguments.
Aliases retain that identity after expansion.

Start with explicit type arguments on record constructors. Constructor argument
deduction, default arguments, and partial argument lists are separate features.
Parameters may stand for complete array types; array lengths remain a separate
compile-time value mechanism, with no general value parameters or arithmetic
constraint solver added here.

Resolve a family's fields against its own binders and defining module. A recursive
field such as `Ptr<Node<T>>` refers to that same family; it must not cause eager,
unbounded expansion merely to identify `Node<T>`. LIR already has lazy nominal
materialization and specialized hooks. Connect source declarations to that
representation while retaining inline-layout and expansion diagnostics.

## Owner and method parameters

Methods remain inside their defining struct, with explicit ordinary receiver
parameters. They inherit their owner's type parameters and may introduce their
own:

```resin
struct Cell<T> {
    value: T,

    def read(self: Cell<T>) -> T = { self.value };
    def replace_with<U>(self: Cell<T>, value: U) -> Cell<U> = {
        Cell<U> { value = value }
    };
};

def example() -> float32 = {
    var cell = Cell<int> { value = 1 };
    var next = cell.replace_with::<float32>(2);
    next.read()
};
```

`T` belongs to `Cell`; `U` belongs to `replace_with`. Resolve their lexical
identities separately. Explicit method arguments supply only the method's own
parameters. Owner arguments come from the receiver or the explicitly applied
owner in an associated call. Alias lookup preserves the original namespace.
There are no blanket implementations, overload sets, or per-instantiation method
registrations.

A method on a nongeneric owner can also bind parameters:

```resin
struct Factory {
    def pair<T>(left: T, right: T) -> Pair<T> = {
        Pair<T> { left = left, right = right }
    };
};

def example() -> Pair<int> = {
    Factory.pair(1, 2)
};
```

These examples use proposed semantic support with existing syntax. Use the
established turbofish spelling for explicit expression arguments:
`Factory.pair::<int>(1, 2)` or `cell.replace_with::<float32>(2)`.
Type applications retain `Cell<int>` syntax. Method calls use the same deduction,
argument-order independence, contextual literals, and fixed-type conflicts as
free functions. Missing type information requires an annotation or explicit
argument during HIR construction.

A receiver whose nominal origin is known can select its method even when owner
arguments are symbolic. Deliver these calls first. Fully dependent lookup, such
as a method on a wholly unknown `T`, remains a separate layer of this plan: HIR
needs explicit determining receiver/signature relations and a dependent call
form. Its existing field-type expression alone does not provide method lookup.
LIR may resolve completed relations but cannot search for type arguments to make
a call succeed. Cases that would require new inference need explicit arguments
or annotations; ordinary methods must not require concrete owner arguments.

Drop hooks inherit owner arguments and keep the existing `Ptr<Owner>` contract;
they cannot require additional arguments that automatic destruction has no way
to supply. Local structs remain field-only. Foreign and decorated shader entry
signatures keep their fixed ABI requirements; generic helpers can serve both
host and shader applications when their concrete operations permit it.

## Ordinary library wrappers over compiler primitives

The agreed boundary is **library wrappers, compiler primitives**. Move `Span<T>`,
`Arc<T>`, `Weak<T>`, `GpuPtr<T>`, and `GpuSpan<T>` into ordinary generic library
definitions. Apply the same review to the typed compute/graphics pipeline
wrappers and every module under [resin/](../resin/). Migrate their users in
examples, benchmarks, documentation, and tests along with each implementation.

Keep primitive pointers, arrays, and Result representations, together with the
existing scalar/aggregate language and the low-level operations needed for
allocation, ownership, access, and dispatch. Intrinsics may remain polymorphic
operations; their existence must not require a separate nominal type family
for each public library wrapper.

`Span<T>` owns its ordinary pointer/length representation and source methods.
The compiler still provides primitive pointer/array access and layout operations.
Preserve host/shader ABI behavior explicitly when replacing the old span variant,
including the literal-byte conversion and packed byte storage rules.

Ownership needs a complete representation decision before replacing `Arc<T>` or
`Weak<T>`. An ordinary struct containing a raw pointer and a destructor is
insufficient: implicit copies must retain shared storage, weak upgrades must
respect the strong count, and final destruction must run the right specialized
hook exactly once. Primitive managed ownership can provide these guarantees
under an ordinary typed wrapper. The exact primitive representation and private
runtime ABI remain implementation decisions; this draft does not introduce a
user-defined copy hook or a new ownership system by implication.

GPU wrappers need the same discipline. Indexing, slicing, and field addresses
must retain their owner and offset. Access must check mapping, permissions, range,
alignment, and recording state, preserving the current
[GPU access contract](gpu-buffers.md). A public unchecked raw pointer would let
reads and writes bypass those guarantees. Choose a narrow primitive access
boundary that ordinary library methods can use; adding more fields alone does
not enforce accesses through an escaped pointer. OS page protection is not a
prerequisite and is not the mechanism promised by this migration.

Ordinary methods should own allocation conveniences, slicing, copying, and error
translation. Preserve the established access spellings where they require
compiler place operations, without adding general operator overloading.
Library import/export policy and any temporary compatibility names belong in the
migration PR that introduces the declarations. Their names must be ordinary
resolved library bindings, rather than reserved compiler type constructors.

## GPU allocation API

Put typed allocation on the GPU value, replacing the awkward
`GpuSpan<T>.allocate(gpu, count)` interface. The intended result shapes, including
normal allocation errors, are:

```text
gpu.create::<T>()       -> Result<GpuPtr<T>, RuntimeError>
gpu.alloc::<T>(count)   -> Result<GpuSpan<T>, RuntimeError>
```

An example inside a function returning a compatible Result is:

```resin
def allocate(gpu: Gpu, count: ulong) -> Result<(), RuntimeError> = {
    var item = gpu.create::<int>()?;
    var items = gpu.alloc::<float32>(count)?;
    var inferred_item: GpuPtr<int>;
    inferred_item := gpu.create()?;
    var inferred_items: GpuSpan<float32>;
    inferred_items := gpu.alloc(count)?;
    ok(())
};
```

These are proposed APIs. `count` determines the allocation length, not the element
type. Context may determine `T` through the ordinary Result/`?` relationship;
without such context, `gpu.create()` needs an explicit argument or annotation.
Neither method receives a default element type.

Specify initialization before implementing the final `create` signature. Current
`gpu.new(value)` initializes an element, whereas `GpuSpan<T>.allocate` reserves
uninitialized storage. A no-argument `create<T>()` cannot silently acquire a
promise of zero initialization or a generic default constructor. The allocation
PR must state its storage/initialization contract and how initialized creation
is expressed. Preserve checked byte-count arithmetic and existing memory-policy
semantics. Low-level byte allocation can remain an implementation facility.

## Dispatch is the projection boundary

The only host-to-device pointer conversions are compiler-builtin projection
operations used for pipeline dispatch and draw. The shader entry point continues
to take `Ptr<Root>`; a public host view or integer handle is not a shader address.

Preserve the existing typed relation from shader `Ptr<T>`/span fields to host
`GpuPtr<T>`/`GpuSpan<T>` arguments, including nested launch records and view
offsets. Ordinary pipeline wrappers still carry the shader root relationship,
stage, and retained owner. Shader declaration identity and construction provenance
must survive the removal of special pipeline type variants.

Projection creates the launch root and retains its allocations for the recording.
It must preserve cross-GPU rejection, access permissions, busy-allocation checks,
and release on submission or cancellation. Do not add a reusable public
`kernel.project` result, a general device-address query, or a raw-pointer cast as
a substitute. Native C internals may continue to carry handles and addresses;
the language-facing library composes the compiler's dispatch operations.

## Completion criteria

The remaining implementation is complete when generic struct and method source
programs use the existing scheme/application pipeline, and the standard library
uses those features for the wrapper families above. Each migration must preserve
its ownership, access, layout, and dispatch guarantees before deleting its old
compiler representation.

The [implementation plan](template-implementation.md) breaks that work into
reviewable PRs. Acceptance includes explicit and inferred method arguments,
recursive generic layouts and drop hooks, imports and aliases, immutable editor
queries and recovery, instance limits, and supported host/shader uses. Re-run the
existing lifecycle and GPU tests for each relevant migration, and add focused
coverage for the new guarantees. Update the language guide as features land;
proposed examples in this draft do not advertise present compiler support.

General value parameters, automatic let-generalization, traits/concepts,
overloading, template specialization, and arbitrary compile-time execution remain
outside this work.
