# Optional values

`None` is a builtin singleton: it names both a type and its only value. An optional
integer is the ordinary structural union `int | None`. Any integer widens into that
union directly; `None` represents absence.

```resin
type OptionalInt = int | None;

def choose(value: OptionalInt) -> int = {
    match (value) { int(number) => { number }, None => { 0 } }
};
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
def require_number(value: int | None) -> int = { value! };
def require_choice(value: int | bool | None) -> int | bool = { value! };
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

`Result<T, E>` retains distinct `ok` and `err` cases even when `T` and `E` overlap.
It can be a member of a union, such as `Result<int, Error> | None`; `!` on that
union removes only `None`, leaving the Result intact. Use `?` or `match` to handle
the Result. Postfix `!` on a bare Result is not supported in this change.

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
serialization format or ABI between separate builds. Results have a separate
tag domain for success and failure.
