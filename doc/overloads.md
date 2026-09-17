# Function overloads and SFINAE

An **overload set** contains the visible functions with a particular name. A call
selects a single applicable signature. **SFINAE** means “substitution failure is
not an error”: if substituting the call's types makes a candidate signature
inapplicable, that candidate is removed from consideration.

For example, a generic `choose<T>(left: T, right: T)` cannot accept an `i32` and a
`bool` together: they cannot both determine the same `T`. A separate
`choose(left: i32, right: bool)` can remain applicable:

```resin
fn choose<T>(left: T, right: T) -> T { left }
fn choose(left: i32, right: bool) -> i32 { if (right) { left } else { 0 } }
fn example() -> i32 { choose(i32(42), true) }
```

If no candidate remains, the call fails; if several remain, it is ambiguous. There
is no “best overload” ranking.

This rule concerns the signature. Once a candidate is selected, an error in its
body is an error in the program. The compiler does not try another overload to
make that body work. Caller imports also cannot add candidates to a generic body
that was defined in a different lexical scope.

## Overload resolution

A visible function name can have several declarations. Calls consider every
argument and the expected result, substituting generic signature parameters to
find applicable candidates. Exactly one candidate must remain. Declaration order
does not break ties, and a concrete signature does not outrank a matching generic
one. Failed signature substitution removes a candidate; errors in a selected
function's body remain errors and do not trigger fallback to another overload.

```resin
fn combine(left: i32, right: i32) -> i32 {
	left + right
}
fn combine(left: i32, right: bool) -> i32 {
	if (right) {
		left
	} else {
		0
	}
}
fn example() -> i32 {
	i32(20):combine(i32(22)) + combine(i32(0), false)
}
```

Imports combine explicitly exported overloads. Exporting a struct does not export
its operations. Local bindings can shadow an operation name. Aliases preserve type
identity but do not introduce namespaces. These lexical rules determine candidates
before specialization; instantiating a generic function cannot add operations from
its caller's scope.

Generic operations declare all their own parameters:

```resin
struct Cell<T> { value: T }
fn take<T>(cell: Cell<T>) -> T {
	cell.value
}
fn replace_with<T, U>(cell: Cell<T>, value: U) -> Cell<U> {
	Cell<U> { value = value }
}
fn example() -> i32 {
	let cell = Cell<u64> { value = 7 };
	cell:replace_with::<u64, i32>(42):take()
}
```

Type arguments can be inferred from arguments and the result, or supplied with
`::<...>`. They are the function's complete parameter list, also in a colon call.
When operand types depend on generic parameters, HIR retains the visible candidate
identities and signature relations. Specialization substitutes concrete types and
selects the operation before storage lowering. It does not search source scopes or
use a candidate's body to establish applicability. Unresolved or ambiguous
applications require an annotation or an unambiguous set of signatures.

## Operator overloading

Operators use visible functions with Python-style names. Both operands participate
in resolution, so relations need not belong to either operand's type.

```resin
struct Vec2 { x: i32, y: i32 }
fn __add__(left: Ref<Vec2>, right: Ref<Vec2>) -> Vec2 {
	Vec2 { x = left.x + right.x, y = left.y + right.y }
}
fn __mul__(scale: i32, value: Ref<Vec2>) -> Vec2 {
	Vec2 { x = scale * value.x, y = scale * value.y }
}
```

Here `left + right` borrows both vectors, and `2 * vector` borrows the vector.
An overload with value parameters instead consumes its noncopyable operands.
Operator functions can be called directly, such as `__add__(left, right)` or
`left:__add__(right)`. Export the functions explicitly to make them available in
another module.

| Expression | Function |
| --- | --- |
| `+value` | `__pos__` |
| `-value` | `__neg__` |
| `~value` | `__invert__` |
| `!value` | `__not__` |
| `left + right` | `__add__` |
| `left - right` | `__sub__` |
| `left * right` | `__mul__` |
| `left / right` | `__truediv__` |
| `left % right` | `__mod__` |
| `left << right`, `left >> right` | `__lshift__`, `__rshift__` |
| `left & right`, `left \| right`, `left ^ right` | `__and__`, `__or__`, `__xor__` |
| `left == right`, `left != right` | `__eq__`, `__ne__` |
| `left < right`, `left <= right` | `__lt__`, `__le__` |
| `left > right`, `left >= right` | `__gt__`, `__ge__` |

Operands are evaluated once, from left to right. Primitive operations and source
overloads share the operation-selection path. Generic functions can retain these
relations until their operand types are concrete; see
[the vector example](../examples/operators.resin). Numeric literals receive
signature context before ordinary numeric defaults apply.

`__not__` is Resin's name for logical negation. Conditions require `bool`.
Each comparison has its own function; `__eq__` does not supply `__ne__`.
Reflected names such as `__radd__` and in-place names such as `__iadd__` have
no special meaning. Assignment, address-of, dereference, indexing, postfix `!`
and `?`, and short-circuit `&&` / `||` keep their built-in behavior. Constant
expressions do not execute source operator functions.


Continue with [type inference](inference.md) for omitted annotations and type arguments.
