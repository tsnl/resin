# Unions and errors

Unions are structural sets of value types: `A | B`, `B | A`, and `A | B | A`
are the same type. Aliases preserve the identities of their targets. A variant's u32
tag identifies its member type throughout a compiled program; it is not its position in a
particular union. Tags are not persistent IDs across separate builds.
`Never` is the empty union. Structs and aliases are declared in source order;
a struct can refer to itself through a pointer, but aliases cannot introduce cycles.

```resin
struct DivideByZero {}
struct NegativeInput { value: int }
type CalculationError = DivideByZero | NegativeInput;

fn divide(n: int, d: int) -> (int | Err<DivideByZero>) {
	if (d == 0) {
		Err(DivideByZero {})
	} else {
		(n / d)
	}
}

fn calculate(n: int) -> (int | Err<_>) {
	if (n < 0) {
		Err(NegativeInput { value = n })
	} else {
		(divide(n, 2)?)
	}
}
```

`T | Err<E>` is an ordinary union. Return a plain `T` for success and `Err(error)`
for failure. Error payloads may be any value type, including `str`, owned `String`,
numbers, and user-defined structs. `Err<E>` is itself a value type; `Err(value)`
infers its payload type from the value and context. `Err<Err<int>>` nests wrappers,
whereas nested unions flatten and duplicate members collapse.

An inferred payload `Err<_>` collects the least union of errors that can escape,
including through recursive calls. With no errors it becomes `Err<Never>`.
Success holes remain monomorphic and need a determining value or annotation.

Postfix `?` evaluates its operand once. If the active member is an `Err`, it
returns that wrapper immediately; otherwise it yields the remaining value.
It preserves all non-error members, so `(int | str | Err<E>)?` yields `int | str`.
The enclosing result must include every propagated error. Union and error values
may widen while preserving their ownership transfer; mutable pointers remain invariant.
Handle failures with exhaustive, duplicate-free matches:

```resin
fn describe(result: (int | Err<CalculationError>))  {
    match (result) {
        int(value) => { print(fmt("value = {0}\n", (value,))) },
        Err(error) => {
            match (error) {
                DivideByZero(zero) => { print("division by zero\n") },
                NegativeInput(negative) => { print(fmt("negative: {0}\n", (negative.value,))) },
            }
        },
    }
}
```

Host entry points can return `(() | Err<E>)` or `(int | Err<E>)`; an unhandled
error prints its payload value and exits with status 1. Run `cargo run -- examples/errors.resin`
or `cargo run -- examples/errors.resin:failure` to try both paths.
Helpers using `Err` and `match` also compile to SPIR-V. C uses a tag and a union of
payloads; shader values use a tag and separate payload fields.
Shared host/device buffer layouts for tagged values are not yet supported.

The standard library's `runtime_status_from_code(code)` converts native status integers to
`(() | Err<RuntimeError>)`. `RuntimeError` is a union of named errors such as
`InvalidArgument`, `OutOfMemory`, and `IoError`; `UnknownRuntimeError { code }`
preserves unrecognized codes. `runtime_status_code(error)` and `runtime_status_message(error)`
recover the native code and C diagnostic string. Standard-library operations already
return error unions, so callers normally use `gpu_new()?` rather than converting statuses.
Standard-library resources release themselves on scope exit, including early returns
through `?`. Explicit clones retain shared ownership.

See [Ownership and cleanup](lifetimes.md) for copying, moves, and destruction.
