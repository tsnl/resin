# Errors, assertions, and representations

## Early returns

`return value;` leaves the current function. `return;` returns unit. The value is
preserved before locals and unfinished operands are destroyed in reverse order.
Only paths that continue participate in definite-initialization checks.

## Assertions

`assert(condition);` evaluates a `bool` once and returns unit. False traps with
`assertion failed` on the host, or stops the current shader invocation through
the shader failure path. Assertions remain enabled in optimized builds.

In a `match`, `Variant(_)` ignores the payload. A final `_ => { ... }` arm
covers all remaining variants. Duplicate, unreachable, and non-final wildcard
arms are rejected.

Error payloads can be any value type, including `str`, numbers, tuples, and owned
`String` values. Inferred error sets collect their union; mutable pointers remain
invariant and errors are still owned and destroyed normally.

## Value representations

Import `repr` from `$/string.resin` to obtain an owned `String` describing any
host value. Records show their names and fields, arrays and tuples show their
elements, and unions show their active payload. Strings are quoted and escaped;
pointers show addresses and opaque handles show their type. `fmt` accepts these
values too, while direct string arguments retain their verbatim text behavior.
Unhandled entry-point errors include this representation before cleanup.

A struct may provide `repr_bytes(self: Ref<Self>)` returning the primitive
`(Ptr<ubyte>, ulong)` byte view. Use the actual struct name in
place of `Self`. The view must remain readable while its receiver is alive;
the hook borrows its receiver and must not invalidate it. `String` uses this
hook so formatting and nested representations show its text. Representation
is a host operation and limits nested output to 128 levels.

`Err<E>` is a builtin wrapper for an error payload of type `E`. Construct it with
`Err(value)` or an explicit payload type such as `Err<int>(7)`. Wrappers copy and
destroy their payload normally, have distinct type identities in unions, and
may be nested. `repr(Err("message"))` produces `Err("message")`.

Fallible functions return ordinary unions such as `int | Err<str>`. Return a
plain `int` on success and `Err("message")` on failure. Postfix `?` returns any
`Err` member immediately, after cleaning up the exited scopes; its value type
is the union of all remaining members. The enclosing function must admit every
propagated error. An `Err<_>` result hole collects the least union of error
payloads, using `Never` if none occur. Mutable pointers remain invariant.

Handle failures with `match (operation()) { int(value) => { ... }, Err(error) =>
{ ... } }`. `Err(error)` covers all error wrapper members and binds the union
of their payloads; `Err(_)` discards those payloads. A final `_ => { ... }` can
handle any remaining members. Explicit `Err<E>(value)` type patterns bind the
wrapper itself, like other explicit type patterns.
