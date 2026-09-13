# Proposal: polymorphic definitions and type templates

Status: draft design. The syntax below is illustrative and is not implemented.

Functions and types can declare named type parameters. Each use supplies or
deduces type arguments. `_` introduces a weak, monomorphic inference variable,
never another template parameter. HIR retains schemes,
type constraints, and one body per definition. HIR-to-LIR lowering enumerates
concrete applications and checks whether their operations are supported for each
requested platform. HIR does not constrain the space of all supported monomorphs.
LIR remains monomorphic, and code generation consumes verified LIR.

## Named parameters and weak type variables

Named parameters are explicitly bound by declarations. Each `_` introduces a
fresh weak, monomorphic variable into ordinary inference. It may unify with a
concrete type, an existing bound parameter, or a determining type expression.
It is never generalized, and separate occurrences start independently.

```resin
def do_something<T>(x: T) -> Result<_, String> = { ok(x) };
```

Unification equates `_` with `T`, completing this signature as
`forall T. T -> Result<T, String>`. There is exactly one template parameter.
Remove special hole-solving machinery; the source spelling uses the same weak
variables and constraints as ordinary inference.

Local `var` storage, runtime parameters, and object fields retain one type per
enclosing application. `var value: _;` joins constraints from its uses instead of
making every read independently polymorphic. Only named parameters, in written
order, participate in explicit template argument lists. Foreign declarations and
decorated shader entries retain fixed ABI signatures. There is no new `let`
keyword or automatic generalization of bindings.

The model draws on C++ [call argument deduction](https://eel.is/c++draft/temp.deduct.call)
and [template instantiation](https://eel.is/c++draft/temp.inst). Resin also uses a
call's expected result type to deduce still-undetermined parameters through the
completed result shape, so contextual numeric literals remain convenient. This is
an extension to ordinary C++ call deduction; it does not adopt C++'s complete
overload, conversion, or specialization system.

The [implementation architecture](template-implementation.md) defines structural
HIR checking, weak-variable inference, dependent type constraints, and concrete checking
during LIR lowering. Concrete ordinary functions use schemes with no parameters.

## Function templates

Proposed spelling:

```resin
def identity<T>(value: T) -> T = { value };
def add<T>(left: T, right: T) -> T = { left + right };

def main() = {
    var number = identity(42);       // identity<long>
    var flag = identity(1 == 1);     // identity<bool>
    var sum = add(1, 2);             // add<long>
    var fraction = add(1.0, 2.0);    // add<float64>
    var explicit = identity<int>(7);
    var function = identity<int>;
    var again = function(8);
};
```

`T` is bound by the declaration's type-parameter list. Each reference to `T` in
that declaration denotes the same parameter. It is not a runtime value or a
fresh anonymous variable. An undeclared type name remains an error.

Builtin `+` has the structural scheme `forall T. (T, T) -> T`. HIR can check
`add<T>` without finding which `T`s have addition implementations. An `add<bool>`
request fails during LIR construction if that operation is unsupported; the error
names the operation and requesting use. Implementations may differ between host
and shader platforms. This introduces no user-defined operator overloading or
propagated `SupportsAdd` constraint. A call inside another template can record
an enclosing parameter, such as `add<U>`, until specialization makes it concrete.

Deduction matches annotated parameter shapes against call argument types, retaining
unresolved literal constraints. Repeated occurrences of one template parameter
must agree, including inside `Ptr<T>`, `Span<T>`, records, function types, and
instantiated nominal types.
Preserve nominal identity and mutable-pointer invariance; do not widen conflicting
deductions to manufacture a common `T`. Once type arguments are fixed, check the
ordinary call using Resin's existing conversion rules.

Initially, a call either provides a full list of type expressions or omits the
list and deduces its arguments using the rules below. `_` in a provided type
expression introduces the same ordinary variable, to be determined during HIR
checking. A remaining undetermined argument needs an annotation or explicit type.
Partial argument lists and default template arguments are deferred. Existing
compiler-provided contextual inference joins this same checking machinery.

A template family is not a runtime function value. Initially, storing or passing
a function template requires explicit arguments, as in `identity<int>` above.
The resulting value has one function-pointer signature per enclosing application
and obeys normal assignment and ownership rules. Inside a template, explicit
arguments may themselves be bound parameters, such as `identity<T>`.
`var function = identity` is insufficient, and
later uses do not turn the variable into a family of functions. Deduction from an
expected function-pointer type can be added separately.

Omitted function results continue to mean unit. `-> _` introduces a weak variable
related to the body's type. It may become `T`, a concrete type, or a dependent
expression such as the type of a member of `T`. An undetermined weak variable
requires an annotation rather than becoming a template parameter. Every requested
application must have concrete types before LIR emission.

## Unsuffixed integers and contextual deduction

Ordinary Resin calls and assignments already allow unsuffixed literals to acquire
their type from context. The [existing inference tests](../tests/inference.rs)
cover this behavior for every numeric width, including constraints from later
uses of a local. Templates must preserve it: `42` is initially a numeric literal
constraint, not an already committed `long` argument. Suffixes select a fixed
type when the programmer wants one; they are optional in ordinary examples.

```resin
def total() -> int = { add(1, 2) };          // result context selects int
def increment(value: int) -> int = { add(value, 1) };
def reversed(value: int) -> int = { add(1, value) };
def narrow() -> ubyte = { identity(255) };   // fits the selected type
```

Collect constraints before fixing a call's type arguments:

1. Apply explicit type arguments, or deduce from already determined argument
   types. Keep unsuffixed numeric arguments flexible, including those nested in
   aggregates and those linked through still-unresolved local bindings.
2. Fill remaining parameters from an available expected result type where it
   matches the template's completed result shape unambiguously. Context may come
   from assignment, a function's annotated return, a typed field, or an enclosing
   call parameter. Propagate it through nested calls before defaulting literals.
3. Solve structural constraints using completed callee schemes. Default eligible
   unconstrained numeric components to `long` or `float64` in HIR, after known
   dependencies settle. Keep a literal tied to a dependent expected type as an
   expression over that type producer, without defaulting it prematurely.
4. Retain dependent type relations in HIR without generalizing weak variables.
   LIR substitutes concrete arguments, resolves determining
   expressions, and checks literal ranges and operation support. It cannot infer
   a missing argument, choose another numeric default, or retry a wider instance.

This gives `add(1, 2)` type `long` without context, `add(existing_int, 2)` type
`int`, and `add<int>(1, 2)` type `int`. Swapping argument order cannot change the
result. A typed parameter of an enclosing call can also select `int` for
`identity(42)`; nested template calls with completed result patterns participate
in the same constraint solving without constructing monomorphic bodies.

Result context does not overwrite explicit arguments or types already determined
by concrete values. Once a parameter is fixed, apply the normal result conversion
rules. For example, an `identity<int>(1)` result can widen into `int | None`;
the expected union does not replace `T = int`. The same rule preserves ordinary
Result error widening. Do not search union members or alternative instances to
make a call compile. Existing literal-to-union inference may still use its single
unambiguous numeric candidate where the ordinary checker already permits it.

An expected `Ptr<int>` may determine `T` for a completed result `Ptr<T>`, even
with no value arguments. A result inferred once as `T` is equally useful:

```resin
def inferred_identity<T>(value: T) -> _ = { value };
def contextual() -> int = { inferred_identity(42) };
```

The completed scheme is `forall T. T -> T`; context selects `T = int` without
changing the definition's inference. Meanwhile,
`def plain() -> _ = { 1 };` completes to `() -> long` and stays fixed.
Expected-result matching does not invert arbitrary conversions or dependent
computations such as the type of `T.member`. The compiler never tries different
bodies or type arguments to find an instance that succeeds.

Recursive reuse alone cannot resolve an unknown type:

```resin
def looped() -> _ = { looped() }; // weak R = R; needs a result annotation
```

This does not declare an anonymous result parameter. If a requested application's
type remains unknown after substitution and resolution, report an annotation
error before LIR emission. A derived result such as `MemberType(T, "member")`
remains tied to its determining operation; it is not a free result either.

Concrete numeric values retain their types. `add(1_i, 2_l)` remains a deduction
conflict; suffix-free code does not imply implicit conversion between stored
integer widths. `identity(256)` in `ubyte` context and a negative literal in
`ulong` context are range errors. An unconstrained integer outside the default
`long` range still needs suitable type context. This changes neither the default
width nor the existing floating-point conversion rules, and introduces no runtime
arbitrary-precision integer type.

## Type templates and methods

Apply the same named-parameter model to nominal structs and transparent aliases:

```resin
struct Pair<T> { left: T, right: T };
type PairPtr<T> = Ptr<Pair<T>>;

def swap<T>(pair: Pair<T>) -> Pair<T> = {
    Pair<T> { left = pair.right, right = pair.left }
};
```

`Pair<int>` and `Pair<float32>` have distinct nominal identities. Repeated uses
of `Pair<int>` refer to the same canonical instance. A transparent alias keeps
the identity of its substituted target. Deduction through `Pair<T>` must match
the originating template and its arguments, not merely equal record layouts.
Start with explicit type arguments in record constructors; constructor argument
deduction can be designed separately.

`_` does not make an alias or struct into an anonymous type family. Use named
parameters for generic fields, recursive relationships, and shared element types.
Any weak variables still need a determining type before concrete use.

The current [type decoder](../crates/resin-hir/src/lower/eval.rs) recognizes a fixed
set of builtin constructors. Extend source type lookup to parameterized
declarations while preserving builtin representation rules. Canonical instance
identity, recursive record layout, and alias-cycle checking must be established
before an instance enters the shared concrete type table.

Methods belonging to a generic nominal type use its owner type arguments.
Method lookup continues to use the type's defining module, including through
aliases. Each instantiated drop hook has a concrete `Ptr<Owner>` receiver and
uses the existing lifecycle rules. Method templates with additional type
parameters require their own declaration and call syntax; deliver them after
function and type templates, without implying that existing method registration
already supports templates.

## Checking and instantiation

Resolve names and check definitions into polymorphic HIR. Its immutable scopes
retain declarations and their completed schemes; bodies reference bindings and
definitions directly. There is one body per definition and no retained list of
monomorphs. Diagnose syntax errors, duplicate parameters, unbound non-dependent
names, and inconsistent structural types even in unused templates. Concrete
operation support is checked for requested monomorphs during LIR construction.

Applying a scheme introduces fresh deduction variables for its quantified
parameters. They may resolve to concrete types or expressions over enclosing
parameters. HIR checks structural relationships and retains determining type
expressions for dependent operations. It does not validate a concrete catalog
of supported operator implementations or retain an instantiated function list.

Complete signatures by definition dependency groups before exposing them to
unrelated callers. Recursive groups may solve structural
equations together; calls instantiate the resulting schemes. A receiver's method
namespace remains its defining module's namespace. Caller-local declarations
cannot change the meaning of names in the template body.

This deliberately changes the phase guarantee: structurally valid HIR can contain
an application that is unsupported on a requested platform. LIR construction
reports that error. Unused families need not support all possible substitutions.
Editor analysis distinguishes dependent types from concrete errors, shows named
parameters and determined weak-variable types in hovers, and navigates to source
definitions. Instantiating a scheme freshens only its named parameters; unresolved
weak variables must not become independently selectable types at each use.

HIR-to-LIR lowering owns the instance worklist, keyed by definition, canonical
concrete substitution, and semantic target profile. Reserve an identity before
lowering recursive references; repeated requests reuse it. Function values,
shader artifacts, pipeline bridges, and implicit drop hooks also request work.
LIR construction owns concrete type interning and memoization, resolves field
types and other determining expressions, and checks the resulting ground type
relations and operation implementations. It does not infer new type arguments.

For `value.take(1)`, a concrete method parameter can determine the literal's type
without a new deduction session. If the selected method has its own type parameters,
HIR must already have recorded their substitution. While lookup remains dependent,
use explicit arguments such as `value.take<int>(1)`, or annotate the receiver so
HIR can deduce them. LIR does not match operands to a newly discovered generic
signature to infer those arguments. Every type must be concrete before storage
or instructions depending on it enter LIR. Unknown types are errors, not a prompt
to try another instance.

Validate every requested platform even when an identical concrete body can be
shared. Host success does not imply shader validity. Checks requiring the complete
call graph, such as shader recursion, run after discovery reaches a fixed point.

Importing a template retains its HIR definition and references to private
helpers in the owning compilation. Instances requested in different
modules use the same declaration identity and are deduplicated. This requires
retaining declarations across module checking instead of immediately replacing
every source function with one concrete function ID.

Source exports may expose templates. Selecting an executable `FILE:ENTRY` still
requires a concrete function; use a wrapper to call a template instance. Foreign
declarations and decorated shader entries remain concrete initially. The export
and entry APIs must represent that distinction without assigning a template an
arbitrary ABI.

Change [public HIR](../crates/resin-hir/src/lib.rs) to express schemes, symbolic
types, type constraints, and determining operations. Private solver state stays
in HIR construction. LIR owns concrete `Ty`, type tables, and function identities;
verification, C, and SPIR-V consume concrete programs. Keep template parameters
and deduction state out of `resin-types`, published LIR, and the backends.

## Monomorph allowance

Set `CompilerConfig.max_monomorphs_per_function` to 16,384 by default. The
configurable allowance counts distinct requests per original function definition
over the whole compilation, including owner/enclosing arguments and target
profiles. Repeated requests cost nothing. Check the limit before reserving a new
request; the default rejects request 16,385 and reports its arguments, platform,
and requesting chain. Keep configuration fixed for each compiler's caches.

This permits general specialization with bounded function-request growth; it
does not prove that every program has a finite set of monomorphs. A function
calling itself at `Ptr<T>` repeatedly produces new keys despite memoization.
Independent type-normalization and expansion safeguards are also required. The
[implementation architecture](template-implementation.md) specifies accounting,
configuration ownership, diagnostics, and the limit's practical scope.

## Ownership, errors, and GPU behavior

Each call evaluates its runtime arguments once in source order. Compile-time
instantiation does not run an initializer or allocate runtime storage. Concrete
instances select the existing copy, drop, layout, and ABI rules: `identity<int>`
and `identity<Arc<int>>` may require different cleanup operations.

Error types use the same weak variables and constraints as other types. An input
error variable does not become `Never` merely because it occupies a Result slot.
An error type determined by `?` propagation is tied to the least union of its
contributors; LIR normalizes that union and validates the concrete error
types. A closed empty union is `Never`. Pending contributors cannot be treated
as empty, or their determining relation discarded to generalize another parameter.

Direct shader helper calls can request concrete template instances. Device
lowering still rejects managed host values, illegal escaping addresses, and
recursive shader call graphs. Template inference introduces no host-to-device
pointer conversion; that remains compiler projection during pipeline recording.
Arbitrary runtime selection between GPU callbacks remains separate work because
the SPIR-V backend requires statically resolved function identities.

## Value parameters and scope

Array length remains a separately determined compile-time value. The existing
[inference representation](../crates/resin-hir/src/lower/infer/mod.rs) stores it
in `Head::Array(usize)`, separately from the element type. In conceptual
`Array<T, N>` notation, `T` is a type argument and `N` is a value argument. This
proposal does not add that source syntax, generic value parameters, or a solver
for arithmetic relations between lengths.

`identity` can accept arrays of different lengths at separate uses because its
`T` can denote each complete array type. Expressing a shared unknown length
across several parameters needs the separate value-parameter mechanism.

The first implementation has one template declaration per name. User-written
partial or explicit specializations, overload selection between templates,
SFINAE, concepts/traits, variadic parameters, and arbitrary compile-time execution
are outside this proposal. These restrictions keep the initial instantiation
rules understandable without preventing dependent body checking.

## Delivery and acceptance

Follow the migration in the [implementation architecture](template-implementation.md):
move ordinary programs onto HIR scopes and schemes, move concrete instance
enumeration and checking into LIR lowering, then enable named polymorphic function
and type definitions with ordinary weak-variable inference.
Contextual deduction, imports, and editor analysis are part of the function-template
implementation from the outset.

Language acceptance includes named scheme parameters, weak `_` variables, explicit and
deduced applications, monomorphic local storage, concrete function values,
dependent arithmetic and field access, suffix-free nested calls and later local
uses, argument-order independence, range errors, and fixed-type conflicts.
Preserve union/Result widening, allow completed inferred result patterns to guide
deduction, and reject inversion of dependent results. Verify nominal distinctions,
aliases, methods, and drop hooks across instances and imports. Exercise C and
direct shader-helper paths, including
ownership-sensitive values and rejected managed GPU types.

Every migration step leaves ordinary programs usable. The new architecture must
also pass its boundary tests for resolved identities, scheme isolation, one HIR
body per definition, LIR instance reuse, numeric scheduling, failure recovery,
and immutable editor facts. Test configurable request limits, target-specific
validation, and nonconcrete types producing annotation errors during LIR
construction. Update the language guide and instructions with implemented syntax
and limits.
