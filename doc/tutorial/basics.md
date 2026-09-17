# Your first Resin program

Start with a small counter before adding complex numbers or GPU resources. Each
checkpoint is a complete program. Run the commands from the repository root after
[setting up the compiler service](../getting-started.md).

## 1. An entry point and an import

```resin
{{#include ../../examples/tutorial/basics/hello.resin:hello}}
```

A file is a module. `export { main };` makes its entry visible, and `import` brings
in the exported functions of another module. `$/` selects the standard-library
root; here `$/string.resin` supplies `print`.

`fn` defines a function. Parentheses contain its parameters and braces contain its
body. `main` takes no arguments and has no result annotation, which means it
returns the unit value `()`. The semicolon ends the call used as a statement.
A string literal has type `str`; `\n` is a newline.

```sh
cargo run -- examples/tutorial/basics/hello.resin
```

Expected output: `Hello, Resin!`.

## 2. Values, functions, and mutation

```resin
{{#include ../../examples/tutorial/basics/functions.resin:functions}}
```

Parameters state their types with `name: Type`, and `-> int` states the result.
The last expression, without a trailing semicolon, supplies the function's result.
`add(initial, 22)` passes two arguments, evaluated from left to right.

`let` binds a value. `initial: int` gives an explicit type; `answer` gets its type
from `add`'s result. `let mut` allows direct reassignment. The two bindings hold
separate integer values, so updating `doubled` does not change `answer`.

`assert` checks a boolean condition. `fmt` formats the tuple `(answer, doubled)`
into an owned string, and `print` writes that string. The format arguments are one
tuple value, so the call still has just two arguments: a format and a tuple.

```sh
cargo run -- examples/tutorial/basics/functions.resin
```

Expected output: `answer = 42, doubled = 84`.

## 3. A struct and ordinary operations

```resin
{{#include ../../examples/tutorial/basics/counter.resin:counter}}
```

`struct Counter` creates a named type with one field. Struct bodies contain fields;
the functions operating on them live beside the type. `counter.value` selects a field.

`Ref<Counter>` borrows access to the caller's existing counter. `add` writes that
counter and `read` reads it. Passing by value would consume a named struct; a
reference lets us keep using the same counter. References do not retain ownership
or check lifetimes, and cannot be turned into pointers. The [reference chapter](../references.md)
gives the complete rules.

## 4. Uniform function call syntax

**UFCS**, or uniform function call syntax, lets a free function use its first
argument as a receiver: `counter:add(1)` means `add(counter, 1)`. The function is
still ordinary and visible in the current scope. A colon calls an operation;
a dot selects a field. There is no special `self` parameter or hidden method table.

```resin
{{#include ../../examples/tutorial/basics/counter.resin:counter_main}}
```

The constructor `Counter { value = 40 }` names each field with `=`. Both calls to
`add` mutate that one counter. The binding does not need `mut` because the calls
write through a reference; `mut` would permit direct assignment of the binding.

The `if` expression chooses a value. Both branches supply a `str`, which becomes
`description`. Comparisons produce `bool`; boolean literals are `true` and `false`.

## 5. Loops and early returns

```resin
{{#include ../../examples/tutorial/basics/counter.resin:control}}
```

`return 0;` handles the nonpositive case immediately. Otherwise the `while` loop
checks its condition before each iteration and adds the next integer. `_i` fixes
the initial numeric literal's type to `int`. The loop statement ends with `;`;
the final `counter:read()` is the result expression.

Run the third checkpoint:

```sh
cargo run -- examples/tutorial/basics/counter.resin
```

Expected output: `counter = 42, sum = 55: ready`. Change the limit or increment to
see which assertions depend on the result. The reference chapter on
[functions and control flow](../syntax.md) also covers `break`, `continue`, function
values, and overloads.

Continue with the [Mandelbrot tutorial](index.md), where these same building blocks
become a numeric solver and then an interactive application.
