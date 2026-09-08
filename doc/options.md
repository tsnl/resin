# Optional values

`Option<T>` is either `some(value)` or `none()`. An empty option needs enough type
context to determine `T`; options may contain structs, tuples, functions, or other options.

```resin
def choose(value: Option<int>) -> int = {
    match (value) { some(number) => { number }, none(unused) => { 0 } }
};
```

Matching requires both variants exactly once. Postfix `!` unwraps an option:

```resin
var value = some(42);
var number = value!;
```

Unwrapping evaluates the operand once. `some` yields its payload; `none` traps with
`cannot unwrap none` on the host. A shader stops the invocation and propagates failure
through callers, preserving prior writes, as for bounds and numeric-conversion traps.
Traps do not unwind cleanup. Prefer `match` when absence is expected.

Postfix operations compose from left to right: `nested!!` unwraps two nested options,
and `optional_record!.field` unwraps before selecting a field. Prefix `!` still negates
a Boolean. Options use a tag and inline payload, with tag zero for `none` and one for `some`.
