# Types and inference

## Type inference

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
Holes can nest inside local annotations, local type ascriptions, and function
return annotations: `Ptr<Ptr<_>>`, `Span<_>`, `(_, Ptr<_>)`, and `(int) -> _`
all use the same inference mechanism. Each `_` is independent. Local constraints
can come from later assignments or uses; numeric literals default to `long` or
`float64` only after those constraints have been considered.

Every function uses the same checker, including functions with no explicit holes.
Later uses can constrain unsuffixed local literals; use an annotation or suffix to fix
a local’s type independently of those uses.

Function results are inferred from bodies in dependency order, checking mutually
recursive groups together. Callers outside a group cannot determine its return
types. A recursive group without enough information is an error, not a generic
function. Parameters, aliases, struct fields, and
foreign signatures remain fully explicit. Unresolved or infinitely recursive
inferred types are errors; `_` is not a wildcard, unit, or a dynamic type.

Omitting a function result annotation still means unit; inference is opt-in.
Types are fully resolved before IR generation, so C and SPIR-V share the same
inference behavior. Run `cargo run -- examples/inference.resin` for an example.


## Generic functions and structs

Named type parameters bind one type throughout a definition. Calls infer their
arguments from values and expected results; explicit function arguments use `::<T>`:

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
not change the definition's candidate set. See [free operations](methods.md).

`_` is a weak inference variable in local annotations, function results, and
explicit applications. It may resolve to a named parameter but never creates
another generic parameter. Unsuffixed literals follow expected types before
falling back to the ordinary integer/float defaults. Template bodies retain their
type relationships; unsupported concrete operations and layouts fail when an
application is required. Specialization substitutes these relations and selects concrete operations; it does
not reopen source inference or use function bodies to deduce signature parameters.
