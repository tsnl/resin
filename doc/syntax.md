# Functions, values, and control flow

## Functions and values

```resin
export { main };
import { "$/string.resin", "$/stdio.resin" };

fn fibonacci(n: i32) -> i32 {
	if (n <= 1) {
		n
	} else {
		fibonacci(n - 1) + fibonacci(n - 2)
	}
}

fn main() {
	let text = fmt("fibonacci(10) = {0}\n", (fibonacci(10),));
	print(text);
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
Function types list their parameters before the arrow: `(i32, i32) -> i32` takes two integers,
while `((i32, i32)) -> i32` takes one tuple. `f()` has no arguments; `f(())` passes one unit
value. A trailing comma, as in `f(value,)`, does not create a tuple. Use `f((value,))` to pass
a singleton tuple. Access tuple members by index: `pair.0`, `pair.1`. Calls evaluate the
callee first, followed by arguments from left to right.

Files contain only function, foreign, type, and constant declarations, after their export/import clauses.
There are no global variables or executable top-level statements. Values and mutable state
belong inside functions and are passed explicitly to helpers by value, reference, or pointer.
Local bindings use `let name = value;`, or `let name: Type;` to reserve uninitialized storage.
Use `let mut name = value;` for direct reassignment. Assignment uses `name = value` and returns unit. `struct Point { x: i32, y: i32 }` creates a nominal type;
`type Position = Point;` is a transparent alias for that same type. Only `struct` creates
a new nominal identity. Construct values with `Point { x = 1, y = 2 }`.
Record initializers keep bare `name = value`
fields, and parameters and struct fields keep bare `name: Type` declarations.
Type formers use angle brackets: `Ptr<i32>`, `Span<f32>`, and `Ptr<Ptr<i32>>`.
Calls and conversions require parentheses: `fibonacci(n)`, `i32(n)`, and `process([1, 2])`.
`process [1, 2]` and `process { value }` are not calls. Nominal record construction retains
its dedicated `Name { field = value }` syntax. Anonymous record types and values
are not supported; use a named `struct` or a tuple.
See `examples/` for functions, recursion, records, pointers, and linked lists.

## Uniform function call syntax (UFCS)

Structs contain fields only. Operations are ordinary functions, declared and
exported independently of the types they use. There are no method namespaces,
inheritance, user-defined traits, or implicit `self` parameters.

```resin
struct Counter { value: i32 }
fn read(counter: Ref<Counter>) -> i32 {
	counter.value
}
fn increment(counter: RefMut<Counter>) {
	counter.value = counter.value + 1;
}

fn example() -> i32 {
	let mut counter = Counter { value = 41 };
	counter:increment();
	counter:read()
}
```

`counter:read()` calls `read(counter)`. The receiver is the first ordinary
argument, and its parameter can have any name. A `Ref<T>` parameter borrows
read-only storage; a value parameter moves a noncopyable argument. `RefMut<T>`
permits mutation and requires a mutable place. Writable references remain
aliasable and can weaken to `Ref<T>`; neither reference kind grants an address. See [references](references.md).

A dot selects a field: `(value.callback)(argument)` calls a function stored in a
field. A colon selects a visible function: `value:callback(argument)` passes the
value as its first argument. Colon calls do not search a type-owned namespace.
Reference arguments require initialized places. Give a computed value a named local
before borrowing it; the compiler does not extend temporary lifetimes for calls.

## Constants and `iota`

`const` declares a compile-time numeric, `bool`, or `str` value at module or local scope.
Constants have no mutable storage: assignment, address-taking, and reference binding are errors.
Module constants can be exported and may refer to later constants; cycles are errors.
Local constants become visible after their specification, and can shadow outer names.

```resin
const answer: i32 = 40 + 2;
const (
	read: u32 = 1 << iota; // 1
	write: u32 = 1 << iota; // 2
	_ = iota; // Skip index 2.
	execute: u32 = 1 << iota; // 8
);
const reset = iota; // 0, with type i64.
```

`iota` starts at zero in each `const` declaration and increments once per specification,
including discarded `_` specifications. Every specification requires an explicit
`= expression`, including later rows in a group and discarded names. Each row supplies
its own expression list and optional type annotation. A specification can bind multiple
names, such as `a, b: u32 = iota, iota + 10;`; their counts must match. Resin requires
semicolons and uses `: Type` annotations and lowercase value names.

Constants use Resin's fixed-width types. An annotation or explicit type application selects a type;
otherwise numeric inference finishes at the declaration, defaulting to `i64` or `f64`.
Later uses do not change that type. Arithmetic, comparisons, logical and bitwise operators,
numeric conversions, constant references, and type layout queries are allowed. Function calls
and aggregate initializers are not constant expressions. Integer overflow, zero divisors,
non-finite floating-point results, and shifts outside the operand width are compile errors,
including in unused declarations. Floating-point operations round at their declared precision.

## Type sizes

`sizeof(Type)` returns the native value size in bytes as `u64`, including padding:

```resin
struct Pair<T> { first: T, second: T }
const pair_bytes = sizeof(Pair<i32>); // 8
const real_bytes = sizeof(f64); // 8
fn size<T>() -> u64 {
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

## Numeric literals

Numeric literals have no suffixes. They get their types from context, including
later assignments and uses. An annotation or explicit type application selects a
width when context is insufficient:

```resin
fn example() {
    let count: u32 = 42;
    let step: f32 = 0.5;
    let precise = f64(0.5);
    assert(count == 42 && step == 0.5 && precise == 0.5);
}
```

| Primitive types | Meaning |
| --- | --- |
| `i8`, `i16`, `i32`, `i64` | Signed integers with the named bit width |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers with the named bit width |
| `f32`, `f64` | 32-bit and 64-bit floating-point values |

Widths are the same on every target. Integer notation can infer any numeric type;
decimal-point and exponent notation infer floating-point types. Unconstrained
integers default to `i64`, and floats to `f64`. Out-of-range literals are errors,
including negative unsigned values and floating-point overflow.

Underscores group digits: `1_000` and `0xffff_ffff`. In hexadecimal notation,
`a` through `f` are always digits. Use a supported type for otherwise unconstrained
shader literals: the shader profile supports `u8`, `i32`, `u32`, `u64`, and `f32`.

## Conditionals

`if (condition) { ... } else { ... }` is an expression. Its condition must be a
`bool`, and both branches must produce compatible result types. Only the selected
branch is evaluated.

An `if` without `else` has an implicit unit branch, so its body must also yield unit:

```resin
if (count > u64(0)) {
    count = count - u64(1);
};
```

It behaves like `if (...) { ... } else {}`. Bindings inside its body remain local, and
assignments made only inside that body do not establish definite initialization afterward.


## Loops

`while` works on both the host and GPU:

```resin
export { main };
import { "$/string.resin", "$/stdio.resin" };

fn main() -> () {
	let mut n = 1;
	let mut sum = 0;
	while (n <= 10) {
		sum = sum + n;
		n = n + 1;
	};
	let text = fmt("sum = {0}\n", (sum,));
	print(text);
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


## Early returns

`return value;` leaves the current function. `return;` returns unit. The value is
preserved before locals and unfinished operands are destroyed in reverse order.
Only paths that continue participate in definite-initialization checks.

## Assertions

`assert(condition);` evaluates a `bool` once and returns unit. False traps with
`assertion failed` on the host, or stops the current shader invocation through
the shader failure path. Assertions remain enabled in optimized builds.

## Parallel blocks

`parallel_map` and `parallel_reduce` introduce nonescaping blocks with explicit
parameters and read-only captures. See [Parallel blocks](parallel.md) for their
syntax, ownership rules, and host and cooperative compute schedules.
