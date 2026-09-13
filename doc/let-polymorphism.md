# Proposal: let-polymorphism over types

Status: draft design; the examples below describe proposed behavior. The current
compiler still requires concrete parameter types and has no `let` declaration.

Allow a function or eligible immutable binding to be used at independently inferred
types. Keep the compiler's public HIR, LIR, layout rules, and native interfaces
concrete by specializing checked functions during HIR construction. Type inference
does not quantify runtime values; array lengths remain separately determined
compile-time values.

This is a substantial frontend feature. Existing unification and dependency-group
solving provide the foundation, but schemes, binding rules, and specialization are
new machinery. The delivery stages below separate those responsibilities.

## Proposed source behavior

Permit explicit `_` holes in module-level ordinary function parameters, including
nested positions. Each hole starts independently; checking the body establishes
which holes denote the same type. Omitted result annotations continue to mean unit.
Source `impl` methods and `drop` hooks retain their concrete signatures in the
first version; existing compiler-provided method inference continues to work.

```resin
def identity(value: _) -> _ = { value };

def apply(function: (_) -> _, value: _) -> _ = { function(value) };

def main() = {
    let same = identity;
    var number = same(42_i);
    var flag = same(1 == 1);
    var again = apply(identity, number);
};
```

`identity` has the scheme `forall T. T -> T`. Reading `same` instantiates that
scheme afresh. `apply` has `forall A, B. ((A -> B), A) -> B`; each invocation
passes an ordinary function with one concrete signature. Its parameter cannot
itself be used at several unrelated types inside one invocation.

Keep the function-shaped annotation on `apply` explicit in the first version.
The legacy `items(index)` spelling also performs indexing, so an unconstrained
`function: _` followed by `function(value)` is ambiguous. Inferring a function
shape from that spelling requires a separate language decision.

In a mutable binding, `var same = identity`, the initializer instantiates once.
Every later use and assignment must agree on that single type. Later uses can
still determine it, as they determine local holes today. Taking its address uses
the existing rules for concrete mutable function-pointer storage.

Use `let name = initializer;` for initialized immutable locals. There is no
uninitialized `let`. An immutable local and its inline fields cannot be assigned
or exposed through a mutable address. A stored `Ptr` or shared owner still permits
the operations of its pointee type; `let` does not introduce deep immutability.

Initially, only a statically resolved function declaration or a chain of eligible
`let` aliases may produce a generalized local binding. Such an alias has no
runtime storage and cannot be addressed. Other `let` initializers are evaluated
once, remain monomorphic, and follow existing copy and destruction rules.
Aliasing a parameter or `var` never freshens its surrounding type variables.

This is a deliberately conservative first set of eligible bindings. Generalizing
aggregate values or results of arbitrary expressions needs an additional
representation and effect analysis. An immutable name may still refer to shared
mutable storage, so immutability alone cannot justify generalization. This is the
problem addressed by the [value restriction in ML languages](https://ocaml.org/manual/5.4/polymorphism.html).

## Type checking and generalization

The current [scope lookup](../crates/resin-hir/src/lower/scope.rs) returns one
inference type per binding. The [checker](../crates/resin-hir/src/lower/check/mod.rs)
builds bodies against those types, solves dependency groups, and requires every
type to become concrete. Adding a final generalization step alone would leave
callers sharing the callee's inference variables.

Replace that contract with monomorphic bindings and generalized schemes. A scheme
contains a type expression with explicitly quantified type parameters. Those
parameters are distinct from mutable solver variables. Generalization quantifies
free type variables of a completed definition's signature that are not free in
its surrounding environment. Instantiation replaces only those quantified
parameters with fresh solver variables, preserving repeated occurrences.

Discover resolved declaration dependencies before checking dependent bodies.
Check each strongly connected group using monomorphic placeholders for its own
members and fresh instances of previously completed schemes. Solve the group,
then generalize its definitions for external callers. Recursive calls within the
group must agree on one type per function. Polymorphic recursion is excluded.
Dependencies must use declaration identities so local shadowing does not create
false edges.

Check generic bodies once, retaining their checked operations and quantified
types privately. An unused definition must still report invalid operations.
An unresolved variable appearing only inside a body is an ambiguity error, not
an extra hidden specialization parameter. Truly unconstrained signature variables
may generalize: a recursively diverging function with an inferred result can have
`forall T. () -> T`, unlike today's blanket rejection of unresolved results.
Infinite types still fail the occurs check.

Resolve concrete operation constraints before publishing a scheme. Pointer
dereference with an explicit `Ptr<_>` shape and calls with a function shape can
relate generic types. An unknown receiver's method, unknown record fields, or
unresolved arithmetic must require an annotation. Do not retry source overload
selection for each caller and silently turn a checked function into a template.
Type-dependent layout and copy/drop operations that have a defined meaning for
every admissible type remain symbolic until specialization.

Preserve numeric defaults: unconstrained integer and floating literal variables
become `long` and `float64` at their existing definition boundary. Quantify only
unconstrained ordinary type variables; numeric classes do not become implicit
operator constraints. Thus `def one() -> _ = { 1 };` remains `() -> long`.

Error-set inference needs a distinct boundary. Close a concrete propagated error
accumulator to its least union, including `Never` when empty. Do not close an
unknown error input to `Never` merely to finish generalization. In the first
version, the error component of a parameter such as `Result<_, _>` must resolve
to a concrete type from its definition's constraints or an annotation. Merely
carrying a `Result` still imposes the error-type restriction; it does not create
an unconstrained error parameter. Quantified error sets and residual error-inclusion
constraints are deferred. A generic `identity` still accepts an entire `Result`
as its unconstrained `T`.

## Specialization and compiler boundaries

Keep schemes, generic checked bodies, and their substitution operations private
to `resin-hir` construction. [Public HIR](../crates/resin-hir/src/lib.rs) continues
to contain resolved bindings, concrete `Ty` values, and concrete function IDs.
`resin-types` remains independent of inference variables and source declarations.

After checking, instantiate the required functions through an explicit worklist.
Key each instance by the source declaration's stable identity and its canonical
concrete type arguments. Reserve its function ID before visiting its callees so
ordinary recursion reuses that instance. Deduplicate repeated requests, preserve
source locations for diagnostics, and report an instantiation chain if a compiler
limit is reached. Do not use source names as instance identity.

Substitute into the checked body and elaborate concrete HIR. Resolve layout,
intrinsic signatures, and lifecycle-sensitive operations using the resulting
types. A polymorphic alias selects a concrete function reference at each use;
it never becomes a runtime universally quantified function pointer. Passing such
a reference through a monomorphic host function or field uses the existing ABI.

Specialization must not duplicate evaluation of runtime initializers or invent
extra owners. Monomorphic `let` values retain one initialization and one lexical
cleanup obligation; compiler-defined reads still perform the normal copies.
Specializing an ordinary function may select different copy/drop behavior for
`int` and `Arc<int>` while preserving source evaluation order.

Imports retain schemes and access to checked generic bodies, including private
helpers, in the owning compilation. Uses in another module instantiate the same
declaration identity. Editor queries expose scheme labels without publishing
mutable solver state; source diagnostics must identify the declaration and the
failing use. Definition navigation should reach the generic source declaration.

Separate source exports from executable entries: importing an exported generic
function is valid, but selecting it directly as `FILE:ENTRY` requires a concrete
wrapper. Keep foreign declarations and decorated shader entries fully concrete.
The current exported-name to single-`FunctionId` mapping must account for this
distinction without assigning a generic export an arbitrary concrete ABI.

Generic helpers called directly by shaders specialize before device lowering.
Each concrete shader instance still obeys device type, ownership, address-escape,
and recursion restrictions. Managed host values remain invalid on the device.
Arbitrary runtime selection between higher-order GPU callbacks is separate work:
the current backend expects statically resolved function identities.

## Array lengths and adjacent features

The [inference representation](../crates/resin-hir/src/lower/infer/mod.rs) stores
array length in `Head::Array(usize)`, separately from its element type. Preserve
that distinction. In conceptual `Array<T, N>` notation, `T` may be quantified while
`N` remains a concrete compile-time length; this proposal does not add that source
syntax or infer arithmetic relations between lengths.

`identity` can accept arrays of different lengths at separate uses by instantiating
`T` with each complete array type. Expressing a shared unknown length across
parameters is value parameterization and needs its own rules.

User-defined parameterized structs and aliases are adjacent work. Today, type
application recognizes a fixed set of builtin constructors. Generalizing that
lookup also requires parameterized declarations, canonical nominal instances,
and matching method/drop-hook specialization. It is not supplied automatically by
let-generalization. Explicit named type binders can be designed alongside it.
Generic source methods also need scheme-aware method registration and lookup,
even when their receiver is an already concrete nominal type; that extension is
deferred with the other method-specialization work.

Lambdas, nested functions, captured environments, higher-rank arguments, traits,
row polymorphism, and general compile-time evaluation are outside this proposal.

## Delivery and acceptance

Each implementation PR should leave the compiler usable and add tests for the
behavior it introduces. Keep parser changes and generated files together.

1. **Schemes and function instances.** Add ordinary parameter holes, dependency
   checking against schemes, and concrete function specialization together.
   Demonstrate `identity` at two types, repeated-instance deduplication, ordinary
   recursion, body errors in unused functions, infinite-type rejection, and
   concrete HIR on both the C and direct shader-helper paths.
2. **Imports and editor analysis.** Preserve generic exports and private helper
   bodies across modules; separate generic exports from executable entries.
   Cover two importing modules, transitive helpers, shadowing, source navigation,
   scheme hovers, and diagnostics after edits. Until this is complete, reject
   cross-module generic use explicitly.
3. **Immutable bindings.** Add and format `let`, enforce assignment and address
   restrictions, and generalize eligible function aliases. Test independent uses
   of an alias, monomorphic `var` and parameter aliases, nested shadowing, and
   single evaluation and cleanup of effectful monomorphic initializers.
4. **Integration and documentation.** Cover higher-order host calls, inferred
   numeric defaults, concrete Result errors, ownership of `Arc` and custom-drop
   values, and rejection of invalid GPU instances. Update user documentation and
   language instructions to distinguish inferred monotypes from schemes.

Completion requires positive and negative cases, not just two successful calls
to `identity`. In particular, independent uses must not constrain each other,
aliases must not create writable polymorphic storage, and specialization must
preserve effects and cleanup. Existing monomorphic programs retain their numeric,
error, ABI, and ownership behavior; newly valid inferred generic declarations
are an intentional extension.
