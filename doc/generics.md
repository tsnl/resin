# Types and generics

Resin is statically typed: every expression has a type determined before execution,
and the compiler checks that operations receive the kinds of values they require.
A type describes both a set of possible values and the operations available on them.
A value is a particular inhabitant, such as `42_i` of type `int`.

## Primitive and compound types

Primitive types include fixed-width numbers (`int`, `ulong`, `float32`), `bool`,
and the string-literal type `str`. The unit type `()` has one value, also written
`()`. Its empty result is useful for operations such as printing. Numeric widths
are fixed across supported targets; conversions between already typed numeric
values are explicit, for example `float32(count)`.

Combine types into tuples, arrays, function types, or named structs. A tuple such
as `(int, bool)` is identified by its structure. A named struct introduces a new
identity: two structs with the same fields are still different types. An alias
introduces another spelling for an existing type, without a new identity.

```resin
struct Point { x: int, y: int }
struct Size { x: int, y: int }
type Position = Point;

fn origin() -> Position {
    Point { x = 0, y = 0 }
}
fn coordinates(point: Ref<Point>) -> (int, int) {
    (point.x, point.y)
}
```

`Position` and `Point` are interchangeable; `Size` is a distinct type. Tuple members
are selected with `.0`, `.1`, and so on. An array literal such as `[1_i, 2_i, 3_i]`
has a fixed length and one element type. Function types describe their parameter
sequence and result: `(int, int) -> int` takes two integers and returns an integer.

`Ptr<T>` describes an existing pointer to a `T`; `Ref<T>` grants access to a place
containing a `T` without promising its address. Standard-library constructors such
as `Span<T>` and `ArcPtr<T>` are ordinary generic structs. [References](references.md)
and [ownership](lifetimes.md) explain the different access and lifetime contracts.

## Generic definitions and applications

A generic definition describes a family of concrete definitions. In `Pair<T>`,
`T` stands for a type chosen when `Pair` is used; it is not a runtime value or a
dynamic type tag. Substitution replaces that parameter consistently throughout
the definition. `Pair<int>` and `Pair<bool>` are different concrete types.

Named type parameters bind one type throughout a definition. Each application
substitutes concrete types for those parameters; explicit function arguments use
`::<T>`:

```resin
struct Pair<T> { left: T, right: T }
type View<T> = Ptr<Pair<T>>;

fn first<T>(pair: Pair<T>) -> T {
	pair.left
}
fn identity<T>(value: T) -> T {
	value
}

fn example() -> int {
	let pair = Pair<int> { left = 40, right = 2 };
	first(pair) + identity::<int>(2)
}
```
Struct constructors take explicit type arguments. Different applications retain
distinct nominal types, even if their layouts agree; aliases keep their target's
identity. Pointer fields may recurse through the same generic declaration. Local
structs remain field-only and can use enclosing function type parameters.

Free operations declare their own complete type parameter list. Colon calls
pass the receiver first and can infer generic arguments from the whole signature:

```resin
struct Cell<T> { value: T }
fn replace_with<T, U>(cell: Cell<T>, value: U) -> Cell<U> {
	Cell<U> { value = value }
}
fn take<T>(cell: Cell<T>) -> T {
	cell.value
}
fn example() -> int {
	let original = Cell<ulong> { value = 7 };
	let changed = original:replace_with::<ulong, int>(42);
	changed:take()
}
```

Generic bodies retain their visible overload candidates. Once operand types are
concrete, signature substitution selects one applicable operation. Ambiguity is
an error; a failing body is never used to discard a candidate. Caller imports do
not change the definition's candidate set. See [function overloads](overloads.md).

Next, [function overloads and SFINAE](overloads.md) explain how substitution chooses
an operation. [Inference](inference.md) then explains which annotations and type
arguments can be omitted.
