# Functions, values, and control flow

## Functions and values

```resin
export { main };
import { "$/string.resin" };

fn fibonacci(n: int) -> int {
	if (n <= 1) {
		n
	} else {
		fibonacci(n - 1) + fibonacci(n - 2)
	}
}

fn main() {
	print(fmt("fibonacci(10) = {0}\n", (fibonacci(10),)));
}
```
Functions use `fn` and are top-level, immutable definitions. Parameter types are explicit;
omitting the result annotation means `()`. Non-unit results require `-> T` or explicit inference
with `-> _`; a non-unit tail expression without an annotation is a type error.
Foreign functions can also omit `-> ()` for C `void` results. Function types still spell out
the result, such as `() -> ()`. All function names are in scope before bodies are checked,
so mutual recursion needs no forward declarations. There are no
lambdas, nested function definitions, or captured environments. Ordinary function values can
be stored, passed, and returned on the host.

Functions take a sequence of arguments enclosed in parentheses. `add(1, 2)` passes two
arguments; `add(pair)` passes one value and requires a function with one parameter.
Function types list their parameters before the arrow: `(int, int) -> int` takes two integers,
while `((int, int)) -> int` takes one tuple. `f()` has no arguments; `f(())` passes one unit
value. A trailing comma, as in `f(value,)`, does not create a tuple. Use `f((value,))` to pass
a singleton tuple. Access tuple members by index: `pair.0`, `pair.1`. Calls evaluate the
callee first, followed by arguments from left to right.

Files contain only function, foreign, type, and constant declarations, after their export/import clauses.
There are no global variables or executable top-level statements. Values and mutable state
belong inside functions and are passed explicitly to helpers, by value or pointer.
Local bindings use `let name = value;`, or `let name: Type;` to reserve uninitialized storage.
Use `let mut name = value;` for direct reassignment. Assignment uses `name = value` and returns unit. `struct Point { x: int, y: int }` creates a nominal type;
`type Position = Point;` is a transparent alias for that same type. Only `struct` creates
a new nominal identity. Construct values with `Point { x = 1, y = 2 }`.
Record initializers keep bare `name = value`
fields, and parameters and struct fields keep bare `name: Type` declarations.
Type formers use angle brackets: `Ptr<int>`, `Span<float32>`, and `Ptr<Ptr<int>>`.
Calls and conversions require parentheses: `fibonacci(n)`, `int(n)`, and `process([1, 2])`.
`process [1, 2]` and `process { value }` are not calls. Nominal record construction retains
its dedicated `Name { field = value }` syntax. Anonymous record types and values
are not supported; use a named `struct` or a tuple.
See `examples/` for functions, recursion, records, pointers, and linked lists.

### Constants and `iota`

`const` declares a compile-time numeric, `bool`, or `str` value at module or local scope.
Constants have no mutable storage: assignment, address-taking, and reference binding are errors.
Module constants can be exported and may refer to later constants; cycles are errors.
Local constants become visible after their specification, and can shadow outer names.

```resin
const answer: int = 40 + 2;
const (
	read: uint = 1 << iota; // 1
	write: uint = 1 << iota; // 2
	_ = iota; // Skip index 2.
	execute: uint = 1 << iota; // 8
);
const reset = iota; // 0, with type long.
```

`iota` starts at zero in each `const` declaration and increments once per specification,
including discarded `_` specifications. Every specification requires an explicit
`= expression`, including later rows in a group and discarded names. Each row supplies
its own expression list and optional type annotation. A specification can bind multiple
names, such as `a, b: uint = iota, iota + 10;`; their counts must match. Resin requires
semicolons and uses `: Type` annotations and lowercase value names.

Constants use Resin's fixed-width types. An annotation or numeric suffix selects a type;
otherwise numeric inference finishes at the declaration, defaulting to `long` or `float64`.
Later uses do not change that type. Arithmetic, comparisons, logical and bitwise operators,
numeric conversions, constant references, and type layout queries are allowed. Function calls
and aggregate initializers are not constant expressions. Integer overflow, zero divisors,
non-finite floating-point results, and shifts outside the operand width are compile errors,
including in unused declarations. Floating-point operations round at their declared precision.

### Type sizes

`sizeof(Type)` returns the native value size in bytes as `ulong`, including padding:

```resin
struct Pair<T> { first: T, second: T }
const pair_bytes = sizeof(Pair<int>); // 8
const real_bytes = sizeof(float64); // 8
fn size<T>() -> ulong {
	sizeof(T)
}
```
Only a type operand is accepted; `sizeof(value)` is an error. Generic queries resolve when
the function is specialized. A `const` initializer must determine its size at declaration.
Sizes follow Resin's 64-bit native representations, including the one-byte placeholder for
unit/empty records, an element placeholder for empty arrays, and union tags. Opaque foreign
types and `Ref<T>` bindings have no queryable value layout.
The existing `size_of` and `align_of` builtins retain their narrower shared CPU/GPU storage
layout contract. `sizeof` does not imply that a type supports GPU storage.

Numeric suffixes are case insensitive and fix the literal's primitive type. Prefix an
integer width with `u` for unsigned values. The formatter writes lowercase suffixes with
an underscore separator; the separator is optional in source except for hexadecimal `b`.

| Suffix | Type | Example |
| --- | --- | --- |
| `b` / `ub` | `sbyte` / `ubyte` (8 bits) | `-128_b`, `255_ub` |
| `h` / `uh` | `short` / `ushort` (16 bits) | `-32768_h`, `65535_uh` |
| `i` / `ui` | `int` / `uint` (32 bits) | `42_i`, `42_ui` |
| `l` / `ul` | `long` / `ulong` (64 bits) | `42_l`, `42_ul` |
| `f` / `d` | `float32` / `float64` | `1.5_f`, `1e3_d` |

These widths are the same on every target. For example, `42UI`, `42uI`, and `42_ui`
all mean the same thing, and format as `42_ui`. Uppercase `42L` now means signed `long`;
use `42_ul` for unsigned `ulong`.

Unsuffixed literals take their type from context, including later assignments and uses.
Integer notation can infer any numeric type; decimal-point and exponent notation infer
floating-point types. Unconstrained integers default to `long`, and floats to `float64`.
A suffixed literal cannot be retyped by an annotation or ascription. Out-of-range literals
are errors, including negative unsigned literals and floating-point overflow. Integer
suffixes require integer notation. Hexadecimal literals accept integer suffixes, such as
`0xffff_ffff_ui` and `0xff_ub`. A signed byte suffix needs its separator (`0x7f_b`);
otherwise `b/B` remains a hex digit. Hex `d/D/f/F` always remain digits.
Use an explicit supported width for otherwise unconstrained shader literals, such as
`let mut step = 0_i;`: the shader profile supports `ubyte`, `int`, `uint`, `ulong`, and `float32`.

An `if` without `else` has an implicit unit branch, so its body must also yield unit:

```resin
if (count > 0_ul) {
    count = count - 1_ul;
};
```

It behaves like `if (...) { ... } else {}`. Bindings inside its body remain local, and
assignments made only inside that body do not establish definite initialization afterward.


## Loops

`while` works on both the host and GPU:

```resin
export { main };
import { "$/string.resin" };

fn main() -> () {
	let mut n = 1;
	let mut sum = 0;
	while (n <= 10) {
		sum = sum + n;
		n = n + 1;
	};
	print(fmt("sum = {0}\n", (sum,)));
}
```

`true` and `false` are literals of type `bool`.

The condition must be boolean and is evaluated before every iteration. The body has its own
scope; its result is discarded, and the loop returns `()`. As with other expression statements,
the trailing semicolon is required unless the loop is the enclosing block's final expression.
The body may run zero times, so initializing a variable only in the body does not make it
definitely initialized afterward. `break;` exits the nearest loop; `continue;` starts its next iteration. Both
are valid inside a loop body, including nested branches, and destroy exited
scope owners before transferring control. Loop conditions cannot contain these exits.
