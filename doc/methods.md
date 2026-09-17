# Free operations and colon calls

Structs contain fields only. Operations are ordinary functions, declared and
exported independently of the types they use. There are no method namespaces,
inheritance, user-defined traits, or implicit `self` parameters.

```resin
struct Counter { value: int }
fn read(counter: Ref<Counter>) -> int {
	counter.value
}
fn increment(counter: Ref<Counter>) {
	counter.value = counter.value + 1;
}

fn example() -> int {
	let counter = Counter { value = 41 };
	counter:increment();
	counter:read()
}
```

`counter:read()` calls `read(counter)`. The receiver is the first ordinary
argument, and its parameter can have any name. A `Ref<T>` parameter borrows
storage; a value parameter moves a noncopyable argument. References permit
unchecked mutation even through an immutable binding; `mut` controls direct
assignment to the binding, not access through an alias. See [references](references.md).

A dot selects a field: `(value.callback)(argument)` calls a function stored in a
field. A colon selects a visible function: `value:callback(argument)` passes the
value as its first argument. Colon calls do not search a type-owned namespace.
A temporary can bind to a reference parameter and stays alive through the full
expression, so `values:at(index):store(value)` is supported.

## Overload resolution

A visible function name can have several declarations. Calls consider every
argument and the expected result, substituting generic signature parameters to
find applicable candidates. Exactly one candidate must remain. Declaration order
does not break ties, and a concrete signature does not outrank a matching generic
one. Failed signature substitution removes a candidate; errors in a selected
function's body remain errors and do not trigger fallback to another overload.

```resin
fn combine(left: int, right: int) -> int {
	left + right
}
fn combine(left: int, right: bool) -> int {
	if (right) {
		left
	} else {
		0
	}
}
fn example() -> int {
	20_i:combine(22_i) + combine(0_i, false)
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
fn example() -> int {
	let cell = Cell<ulong> { value = 7 };
	cell:replace_with::<ulong, int>(42):take()
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
struct Vec2 { x: int, y: int }
fn __add__(left: Ref<Vec2>, right: Ref<Vec2>) -> Vec2 {
	Vec2 { x = left.x + right.x, y = left.y + right.y }
}
fn __mul__(scale: int, value: Ref<Vec2>) -> Vec2 {
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

## Indexing and destruction

Arrays and `str` have primitive `at` operations; `Span<T>` supplies an ordinary
free overload from `$/span.resin`. `items:at(index)` takes a `ulong` and returns
`Ref<T>` (`Ref<ubyte>` for `str`). Write an element with `items:at(index) = value`
and obtain a pointer with `items:lea(index)` on a pointer to an array, a span,
or `str`. Local arrays support `at` only. Arrays also retain `items(index)`.
Host indexing checks the declared length; shader indexing is unchecked.

A free `fn drop(value: Ref<Item>) { ... }` declared with its nominal type supplies
its destruction hook. A generic hook binds the owner's parameters. Direct calls
remain ordinary calls; see [ownership and cleanup](lifetimes.md).

Free functions may be shader entries or helpers. Shader-local addresses still
cannot escape into a callee, and reference counting and custom destruction remain
host-only. Free operations do not relax these backend restrictions.
