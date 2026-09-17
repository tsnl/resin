# Unions, optionals, and errors

Unions are structural sets of value types: `A | B`, `B | A`, and `A | B | A`
are the same type. Aliases preserve the identities of their targets. A variant's u32
tag identifies its member type throughout a compiled program; it is not its position in a
particular union. Tags are not persistent IDs across separate builds.
`Never` is the empty union. Structs and aliases are declared in source order;
a struct can refer to itself through a pointer, but aliases cannot introduce cycles.

## Optional values

`None` is a builtin singleton: it names both a type and its only value. An optional
integer is the ordinary structural union `int | None`. Any integer widens into that
union directly; `None` represents absence.

```resin
type OptionalInt = int | None;

fn choose(value: OptionalInt) -> int  {
    match (value) { int(number) => { number }, None => { 0 } }
}
```

`match` covers every member exactly once. A type pattern binds the value of that
type; the singleton case can be written `None => { ... }` without a binding.
There is no user-defined singleton declaration syntax yet.

Union members may be primitive, nominal, or structural types. Aliases are
transparent, nesting flattens, and repeated members collapse: `OptionalInt | None`
is the same type as `int | None`. A union does not distinguish two occurrences of
`None`; use a wrapper struct if those cases need different meanings.

Postfix `!` removes `None` from the operand's possible types:

```resin
fn require_number(value: int | None) -> int  { value! }
fn require_choice(value: int | bool | None) -> int | bool  { value! }
```

The operand is evaluated once. If it is `None`, the host traps with
`cannot unwrap None`. Otherwise the value keeps its member type; the result can
then widen into a larger union at its consumer. Mutable pointer and Span element
types remain invariant: `Ptr<int> | None` does not become `Ptr<int | None>`.

In a shader, failure stops the invocation and propagates through shader callers,
preserving prior writes, as for numeric-conversion traps. This does not
report a panic to the host or roll back a dispatch. Traps do not unwind cleanup.
Prefer `match` when absence is expected. Shader-local unions support only payloads
supported by the shader backend; union buffer layouts are not part of this change.

Postfix operations compose from left to right: `optional_record!.field` excludes
`None` before selecting a field. Prefix `!` still negates a Boolean. A second
postfix `!` is invalid once no `None` remains.

`T | Err<E>` distinguishes success from failure even when `T` and `E` overlap:
`int` and `Err<int>` are different types. In `int | Err<E> | None`, postfix `!`
removes only `None`; postfix `?` propagates only `Err` and preserves `None`.
Use `match` to handle either explicitly. Both use the same ordinary union tags.

Every nominal, primitive, and structural type is interned in one module-wide
vector of type definitions. A type's ID is its index in that vector; an ordinary
union's u32 tag is exactly its active payload's type ID. The host and GPU use the
same table. Aliases share their underlying type's entry, while separate nominal
structs retain distinct entries even when their fields are identical.

For example, `int` has the same tag in
`int | None` and `int | bool | None`, on both the host and the GPU. Widening
preserves that tag. A union type has a table entry, but is never itself a payload:
`(int | None) | bool` is flattened to `int | None | bool` before lowering.
Numeric tag values are local to the compiled program; they are not a stable
serialization format or ABI between separate builds. `Err<E>` has its own type ID, like every other union member.
## Errors and propagation

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

In a `match`, `Variant(_)` ignores the payload. A final `_ => { ... }` arm covers
all remaining variants. Duplicate, unreachable, and non-final wildcard arms are
rejected. Assertions and early returns are covered under [control flow](syntax.md).
