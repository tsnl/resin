# Proposal: templates with inferred type arguments

Status: draft design. The syntax below is illustrative and is not implemented.

Allow functions and types to declare named type parameters. Each use supplies or
deduces type arguments. HIR retains polymorphic definitions and their schemes;
checking instantiates schemes as needed, while HIR-to-LIR lowering enumerates
concrete instances. Dependent operations such as arithmetic, field access, and
method calls carry requirements that are checked when their types are known.
This follows the useful core of C++ templates while retaining Resin's language
rules and a monomorphic LIR.

## Templates, inference holes, and let-polymorphism

These are distinct mechanisms:

| Mechanism | Meaning |
| --- | --- |
| Inference hole `_` | Find one type from the constraints of this expression or declaration. |
| Let-polymorphism | Generalize a binding's inferred type, then instantiate its quantified variables at each use. |
| Type template | Declare named type parameters and apply the declaration with chosen or deduced types. |

This proposal adds type templates and argument deduction. Existing `_` holes keep
their inference meaning. There is no new `let` keyword, generalized mutable
storage, or change to the meaning of an ordinary `var` binding. Parameters continue
to have explicit annotations, which may refer to declared template parameters.

The model draws on C++ [call argument deduction](https://eel.is/c++draft/temp.deduct.call)
and [template instantiation](https://eel.is/c++draft/temp.inst). Resin also uses a
call's expected result type to deduce still-undetermined parameters through the
completed result shape, so contextual numeric literals remain convenient. This is
an extension to ordinary C++ call deduction; it does not adopt C++'s complete
overload, conversion, or specialization system.

The [implementation architecture](template-implementation.md) defines polymorphic
HIR, scheme checking, dependent requirements, and specialization during LIR
lowering. Ordinary functions use the same machinery with no type parameters.

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
that declaration denotes the same parameter. It is neither an inference hole nor
a runtime value. An undeclared type name remains an error.

`add` is valid for an instance whose operands support Resin's existing `+`
operation. An `add<bool>` use reports an error at the operation and the use that
requested the instance. The definition retains this dependent requirement;
the compiler does not have to prove `+` exists for every possible `T`. This does
not introduce user-defined operator overloading. A call inside another template
can use its enclosing parameters, such as `add<U>`, until specialization makes
them concrete.

Deduction matches annotated parameter shapes against call argument types, retaining
unresolved literal constraints. Repeated occurrences of one template parameter
must agree, including inside `Ptr<T>`, `Span<T>`, records, function types, and
instantiated nominal types.
Preserve nominal identity and mutable-pointer invariance; do not widen conflicting
deductions to manufacture a common `T`. Once type arguments are fixed, check the
ordinary call using Resin's existing conversion rules.

Initially, a call either provides all type arguments or deduces them from its
arguments and expected result, using the rules below. A parameter that cannot be
deduced requires an explicit type argument. Partial explicit argument lists,
default template arguments, and `_` inside explicit argument lists are deferred.
Existing compiler-provided contextual inference retains its current rules.

A template family is not a runtime function value. Initially, storing or passing
a function template requires explicit arguments, as in `identity<int>` above.
The resulting value has one function-pointer signature per enclosing application
and obeys normal assignment and ownership rules. Inside a template, explicit
arguments may themselves be bound parameters, such as `identity<T>`.
`var function = identity` is insufficient, and
later uses do not turn the variable into a family of functions. Deduction from an
expected function-pointer type can be added separately.

Omitted function results continue to mean unit. Explicit `-> _` infers a result
while checking the polymorphic definition. That result may be `T` or a dependent
type such as the type of a member of `T`; it does not introduce another template
parameter. An unresolved hole without a determining operation remains an error.

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
3. Solve pending constraints and normalize dependent types whose inputs are known.
   Use completed callee schemes without rechecking their bodies. When deduction
   needs a numeric fallback, default unconstrained numeric components to `long`
   or `float64` after their productive dependencies settle. Do not default a
   literal connected to an enclosing bound parameter or a pending external
   result producer. The implementation architecture specifies this scheduling rule.
   An unresolved nonnumeric parameter still requires an explicit type argument.
4. Check literal ranges and other requirements as their types become known.
   Retain requirements that depend on enclosing parameters in HIR. Range failure
   reports an error; it does not retry a wider instance.

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
names, and invalid non-dependent operations even in unused templates.

Applying a scheme introduces fresh deduction variables for its named parameters.
They may resolve to concrete types or expressions over enclosing parameters.
Check operations whose types are known, and retain explicit dependent operations
and requirements for the rest. An inferred member result remains tied to its
receiver and member, never an unconstrained result a caller can choose. Concrete
uses check those requirements ad hoc using HIR's semantic operations.

Ordinary functions have schemes with no quantified parameters. Complete inferred
signatures by definition dependency groups before exposing them to unrelated
callers. Recursive groups may solve their result equations together; callers
cannot invent a result for an ambiguous cycle. A receiver's method namespace
remains its defining module's namespace. Caller-local declarations cannot change
the meaning of names in the template body.

This deliberately changes when some errors can be diagnosed: a bad dependent
operation in an unused template can remain undiagnosed until an instance needs
it. Editor analysis should distinguish unresolved dependent operations from
errors, show declared parameters in hovers, and navigate to the source template.
Failures in instances should show the concrete type arguments and the chain of
uses that requested them.

HIR-to-LIR lowering owns the instance worklist, keyed by definition and canonical
closed type arguments. Reserve a LIR function identity before lowering recursive
references to the same instance; repeated references reuse it. Function values,
shader artifacts, pipeline bridges, and implicit drop hooks also request work.
Use HIR's shared substitution and requirement checking to specialize one body,
lower it, and discard the temporary specialization. Diagnose unbounded expansion
with a bounded work limit and a useful application chain.

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
types, and typed dependent operations. Private inference variables still cannot
escape HIR construction. LIR owns final concrete `Ty` and function identities;
verification, C, and SPIR-V continue to consume concrete programs. Keep template
parameters and deduction state out of `resin-types`, LIR, and the backends.

## Ownership, errors, and GPU behavior

Each call evaluates its runtime arguments once in source order. Compile-time
instantiation does not run an initializer or allocate runtime storage. Concrete
instances select the existing copy, drop, layout, and ABI rules: `identity<int>`
and `identity<Arc<int>>` may require different cleanup operations.

Named error parameters such as `E` in `Result<T, E>` carry the existing error-type
requirements. Inferred error holes collect a symbolic least union of propagated
errors, normalized after substitution and equal to `Never` when empty. Recursive
groups finish their inclusion equations together. Pending contributors cannot be
treated as empty, and an accumulator does not become an extra template parameter.

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
enumeration into LIR lowering, then enable function and type templates.
Contextual deduction, imports, and editor analysis are part of the function-template
implementation from the outset.

Language acceptance includes explicit and deduced instances, concrete function
values, dependent arithmetic and field access, suffix-free nested calls and later
local uses, argument-order independence, range errors, and fixed-type conflicts.
Preserve union/Result widening, allow completed inferred result patterns to guide
deduction, and reject inversion of dependent results. Verify nominal distinctions,
aliases, methods, and drop hooks across instances and imports. Exercise C and
direct shader-helper paths, including
ownership-sensitive values and rejected managed GPU types.

Every migration step leaves ordinary programs usable. The new architecture must
also pass its boundary tests for resolved identities, scheme isolation, one HIR
body per definition, LIR instance reuse, numeric scheduling, failure recovery,
and immutable editor facts. Update the language guide and instructions with
implemented syntax and limits.
