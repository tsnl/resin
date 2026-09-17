# Types and generics

Resin is statically typed: every expression has a type determined before execution,
and the compiler checks that operations receive the kinds of values they require.
A type describes both a set of possible values and the operations available on them.
A value is a particular inhabitant, such as `i32(42)` of type `i32`.

## Primitive and compound types

Primitive types include fixed-width numbers (`i32`, `u64`, `f32`), `bool`,
and the string-literal type `str`. The unit type `()` has one value, also written
`()`. Its empty result is useful for operations such as printing. Numeric widths
are fixed across supported targets; conversions between already typed numeric
values are explicit, for example `f32(count)`.

Combine types into tuples, arrays, function types, or named structs. A tuple such
as `(i32, bool)` is identified by its structure. A named struct introduces a new
identity: two structs with the same fields are still different types. An alias
introduces another spelling for an existing type, without a new identity.

```resin
struct Point { x: i32, y: i32 }
struct Size { x: i32, y: i32 }
type Position = Point;

fn origin() -> Position {
    Point { x = 0, y = 0 }
}
fn coordinates(point: Ref<Point>) -> (i32, i32) {
    (point.x, point.y)
}
```

`Position` and `Point` are interchangeable; `Size` is a distinct type. Tuple members
are selected with `.0`, `.1`, and so on. An array literal such as `[i32(1), i32(2), i32(3)]`
has a fixed length and one element type. Function types describe their parameter
sequence and result: `(i32, i32) -> i32` takes two integers and returns an integer.

`Ptr<T>` describes an existing pointer to a `T`; `Ref<T>` grants access to a place
containing a `T` without promising its address. Standard-library constructors such
as `Span<T>` and `ArcPtr<T>` are ordinary generic structs. [References](references.md)
and [ownership](lifetimes.md) explain the different access and lifetime contracts.

## Generic definitions and applications

A generic definition describes a family of concrete definitions. In `Pair<T>`,
`T` stands for a type chosen when `Pair` is used; it is not a runtime value or a
dynamic type tag. Substitution replaces that parameter consistently throughout
the definition. `Pair<i32>` and `Pair<bool>` are different concrete types.

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

fn example() -> i32 {
	let pair = Pair<i32> { left = 40, right = 2 };
	first(pair) + identity::<i32>(2)
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
fn example() -> i32 {
	let original = Cell<u64> { value = 7 };
	let changed = original:replace_with::<u64, i32>(42);
	changed:take()
}
```

Generic bodies retain their visible overload candidates. Once operand types are
concrete, signature substitution selects one applicable operation. Ambiguity is
an error; a failing body is never used to discard a candidate. Caller imports do
not change the definition's candidate set. See [function overloads](overloads.md).

## Copying generic values

Value parameters follow the concrete type's ownership rules. A single use can
transfer a move-only value, as in `identity` above. Repeated uses infer a copy
requirement:

```resin
fn twice<T>(value: T) -> (T, T) {
	(value, value)
}
fn example() -> i32 {
	let pair = twice(i32(21));
	pair.0 + pair.1
}
```

`twice` accepts numbers, structs whose fields copy, and shared handles such as
`ArcPtr<T>`. A move-only argument produces an error at the use that needs copying.
No trait annotation is needed. The compiler also considers branches, loops, and
moves of individual fields; assigning a replacement restores availability.

Requirements apply to value uses. A generic callback that specializes to a
`Ref<T>` or `RefMut<T>` parameter borrows its argument and may be called repeatedly
with a move-only local. A failed copy requirement is a body error, so it does not
trigger overload fallback. See [ownership](lifetimes.md) for the complete rules.

Next, [function overloads and SFINAE](overloads.md) explain how substitution chooses
an operation. [Inference](inference.md) then explains which annotations and type
arguments can be omitted.
