# Type inference

Inference fills in omitted types using constraints from a definition's expressions.
It does not introduce dynamic typing or make a local binding generic: a local has
one type for the whole function body.

## Local values and contextual literals

```resin
fn twice(value: int) -> int { value + value }
fn example() -> int {
    let initial = 21;
    let answer = twice(initial);
    answer
}
```

The call requires an `int`, so `initial`'s unsuffixed literal is typed as `int`.
Constraints can come from later assignments or uses, not just an initializer.
Write `let initial: int = 21;` or `let initial = 21_i;` to state that choice directly.

Unconstrained integer literals default to `long`, and floating-point literals to
`float64`, after contextual constraints are considered. A suffix fixes the type;
it cannot be silently changed to make a call fit. This is contextual literal typing,
not an implicit conversion of an existing numeric value.

## Generic arguments

```resin
fn identity<T>(value: T) -> T { value }
fn example() -> int {
    let value: int = identity(42);
    identity::<int>(value)
}
```

The first call determines `T = int` from the operand and expected result. The
second supplies that type explicitly. Generic arguments can be determined from the
whole signature, including the result. A definition's named `T` remains one rigid
parameter while checking that definition; inference does not independently choose
another type at each use of `T`.

## Results belong to their definitions

A declared result type constrains the body. If the annotation is omitted, the
result is unit, `()`; it does not request inference. To infer a result from the body,
write `-> _`. Callers cannot decide a separately defined function's unresolved
result for it. Recursive groups must contain enough information to determine their
types without an unconstrained cycle.

## Explicit inference holes: `_`

Write `_` to request a concrete type inferred from the surrounding code:

```resin
export { main };
import { "$/string.resin" };

fn next(n: int) -> _ {
	n + 1
}

fn main() {
	let mut value: _;
	value = next(41);
	let reference: Ref<_> = value;
	print(fmt("value = {0}\n", (reference,)));
}
```
Holes can nest inside local annotations, local type ascriptions, function results,
and explicit type applications: `Ptr<Ptr<_>>`, `Span<_>`, `(_, Ptr<_>)`,
`(int) -> _`, and `identity::<_>(42_i)`. Each `_` is an independent, weak inference
variable. It may resolve to an enclosing named type parameter, but never creates
a new generic parameter.

Parameters, aliases, struct fields, and foreign signatures remain fully explicit.
Unresolved or infinitely recursive inferred types are errors; `_` is not a
wildcard, unit, or a dynamic type. Inferred types are resolved before code
generation, so host code and shaders use the same inference rules. Run
`cargo run -- examples/inference.resin` for a complete example.
