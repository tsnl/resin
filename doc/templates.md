# Proposal: templates with inferred type arguments

Status: draft design. The syntax below is illustrative and is not implemented.

Allow functions and types to declare named type parameters. Each use supplies or
deduces concrete type arguments, and the compiler checks the resulting instance.
Dependent operations such as arithmetic, field access, and method calls are
resolved using those concrete types. This follows the useful core of C++ templates
while retaining Resin's language rules and concrete public HIR.

## Templates, inference holes, and let-polymorphism

These are distinct mechanisms:

| Mechanism | Meaning |
| --- | --- |
| Inference hole `_` | Find one type from the constraints of this expression or declaration. |
| Let-polymorphism | Generalize a binding's inferred type, then instantiate its quantified variables at each use. |
| Type template | Declare named type parameters, choose concrete arguments for a use, and instantiate the declaration. |

This proposal adds type templates and argument deduction. Existing `_` holes keep
their inference meaning. There is no new `let` keyword, generalized mutable
storage, or change to the meaning of an ordinary `var` binding. Parameters continue
to have explicit annotations, which may refer to declared template parameters.

The model draws on C++ [call argument deduction](https://eel.is/c++draft/temp.deduct.call)
and [template instantiation](https://eel.is/c++draft/temp.inst). Resin also uses a
call's expected result type to deduce still-undetermined parameters through the
declared result shape, so contextual numeric literals remain convenient. This is
an extension to ordinary C++ call deduction; it does not adopt C++'s complete
overload, conversion, or specialization system.

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
requested the instance. Checking this dependent operation waits for instantiation;
the compiler does not have to prove `+` exists for every possible `T`. This does
not introduce user-defined operator overloading.

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
The resulting value has one concrete function-pointer signature and obeys normal
assignment and ownership rules. `var function = identity` is insufficient, and
later uses do not turn the variable into a family of functions. Deduction from an
expected function-pointer type can be added separately.

Omitted function results continue to mean unit. Existing `-> _` result inference
may run separately for each concrete template instance; it does not introduce an
additional template parameter. Unresolved local or result holes remain errors.

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

Collect constraints before committing an instance:

1. Apply explicit type arguments, or deduce from already concrete argument
   types. Keep unsuffixed numeric arguments flexible, including those nested in
   aggregates and those linked through still-unresolved local bindings.
2. Fill remaining parameters from an available expected result type where it
   matches the template's declared result shape unambiguously. Context may come
   from assignment, a function's annotated return, a typed field, or an enclosing
   call parameter. Propagate it through nested calls before defaulting literals.
3. Solve the pending constraints in the caller's dependency group to a fixed
   point. Instances whose own type arguments are already fixed may be checked
   now; their completed inferred results can constrain enclosing calls. When no
   further progress is possible, default remaining numeric variables to `long`
   or `float64`. An unresolved nonnumeric parameter still requires an explicit
   type argument.
4. Check literal ranges as their selected types become known, and request the
   remaining concrete instances. Range failure reports an error; it does not
   retry a wider instance.

This gives `add(1, 2)` type `long` without context, `add(existing_int, 2)` type
`int`, and `add<int>(1, 2)` type `int`. Swapping argument order cannot change the
result. A typed parameter of an enclosing call can also select `int` for
`identity(42)`; nested template calls with declared result patterns participate
in the same constraint solving before any instance body is needed.

Result context does not overwrite explicit arguments or types already determined
by concrete values. Once a parameter is fixed, apply the normal result conversion
rules. For example, an `identity<int>(1)` result can widen into `int | None`;
the expected union does not replace `T = int`. The same rule preserves ordinary
Result error widening. Do not search union members or alternative instances to
make a call compile. Existing literal-to-union inference may still use its single
unambiguous numeric candidate where the ordinary checker already permits it.

An expected `Ptr<int>` may determine `T` for a declared result `Ptr<T>`, even
with no value arguments. A `-> _` result provides no such declared pattern:
the compiler must not inspect or repeatedly try the template body to discover
missing type arguments. Infer that result only after the instance is selected,
preserving the existing rule that callers cannot determine a body's inferred
return type. Expected-result matching also does not invert arbitrary conversions
or dependent type computations.

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

Methods belonging to a generic nominal type use its concrete type arguments.
Method lookup continues to use the type's defining module, including through
aliases. Each instantiated drop hook has a concrete `Ptr<Owner>` receiver and
uses the existing lifecycle rules. Method templates with additional type
parameters require their own declaration and call syntax; deliver them after
function and type templates, without implying that existing method registration
already supports templates.

## Checking and instantiation

Keep template declarations and their type parameters private to HIR construction.
Parse all bodies and establish lexical bindings in the defining module. Diagnose
syntax errors, duplicate parameters, unbound non-dependent names, and invalid
non-dependent operations even in unused templates. Retain the source body and
its definition environment for checks that depend on template arguments.

On instantiation, bind the declared parameters to concrete types and check the
body using the ordinary checker. Resolve dependent operators, fields, methods,
layout queries, and intrinsic calls at this point. A concrete receiver's method
namespace remains its defining module's namespace. Caller-local declarations
cannot change the meaning of names in the template body.

This deliberately changes when some errors can be diagnosed: a bad dependent
operation in an unused template can remain undiagnosed until an instance needs
it. Editor analysis should distinguish unresolved dependent operations from
errors, show declared parameters in hovers, and navigate to the source template.
Failures in instances should show the concrete type arguments and the chain of
uses that requested them.

Use an explicit instance worklist keyed by declaration identity and canonical
concrete type arguments. Reserve a function ID before checking recursive calls
to the same instance; repeated calls reuse it. Calls requesting different type
arguments create different instances. Diagnose unbounded instantiation expansion
with a bounded work limit and a useful chain, rather than overflowing the host
stack. Apply the existing dependency-group result inference to concrete recursive
instances; callers must not invent a missing result for an ambiguous cycle.

Importing a template retains its declaration, definition environment, and access
to private helpers in the owning compilation. Instances requested in different
modules use the same declaration identity and are deduplicated. This requires
retaining declarations across module checking instead of immediately replacing
every source function with one concrete function ID.

Source exports may expose templates. Selecting an executable `FILE:ENTRY` still
requires a concrete function; use a wrapper to call a template instance. Foreign
declarations and decorated shader entries remain concrete initially. The export
and entry APIs must represent that distinction without assigning a template an
arbitrary ABI.

Elaborate checked instances into [public HIR](../crates/resin-hir/src/lib.rs) with
concrete `Ty` values and function IDs. HIR-to-LIR lowering, verification, C, and
SPIR-V continue to consume concrete programs. Keep source template parameters and
deduction state out of `resin-types`, LIR, and the backends.

## Ownership, errors, and GPU behavior

Each call evaluates its runtime arguments once in source order. Compile-time
instantiation does not run an initializer or allocate runtime storage. Concrete
instances select the existing copy, drop, layout, and ABI rules: `identity<int>`
and `identity<Arc<int>>` may require different cleanup operations.

Named error parameters such as `E` in `Result<T, E>` are fixed before body checking.
Validate their concrete error types using the existing Result rules. Inferred
error holes inside an instance then collect its least propagated error union,
including `Never` when empty. There is no need to generalize an error accumulator
into an error-set scheme.

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

1. **Explicit function instances.** Add named type parameters and explicit
   function type application. Keep parser changes and generated files together.
   Cover instances at two types, reuse of one instance, concrete function values,
   recursive instance reuse and expansion errors, and diagnostics for a failing
   dependent operation. Reject unsupported generic exports until import support
   is available.
2. **Call deduction, imports, and editor analysis.** Deduce arguments through
   nested parameter and declared result shapes, reject conflicts and missing
   arguments, and preserve delayed numeric defaults. Test suffix-free calls,
   typed arguments in both orders, assignment/return/enclosing-call context,
   nested calls, later local uses, range boundaries, fixed-type conflicts, union
   and Result widening, and rejection of inference through a `-> _` body.
   Retain imported templates and private helpers. Test repeated imports, lexical
   lookup, declared-parameter hovers, and instance diagnostics.
3. **Type templates.** Add parameterized structs and aliases, canonical nominal
   instances, layout/cycle checks, and methods/drop hooks belonging to those
   instances. Then add methods with their own type parameters. Test nested types,
   nominal distinctions, transparent aliases, and deduced `Pair<T>` arguments.
4. **Host/GPU integration and documentation.** Exercise generic arithmetic and
   field access, Result error parameters, `Arc` and custom-drop values, and direct
   shader helpers. Verify argument evaluation and cleanup, concrete HIR, and
   rejection of invalid GPU instances. Update the language guide and instructions
   with the implemented syntax and deduction limits.

Each stage should leave ordinary programs working and add positive and negative
tests for the behavior it introduces. The essential guarantees are independent
instances, stable declaration lookup, concrete public IR, and unchanged runtime
evaluation and ownership semantics.
