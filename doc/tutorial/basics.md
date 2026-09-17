# Your first Resin program

This is one complete source file, [`examples/tutorial/basics.resin`](../../examples/tutorial/basics.resin).
Read its comments from top to bottom for functions, values, structs, control flow,
and uniform function call syntax (UFCS).

```resin
{{#include ../../examples/tutorial/basics.resin}}
```

After [starting the compiler service](../getting-started.md), run it from the repository root:

```sh
cargo run -- examples/tutorial/basics.resin
```

Expected output: `counter = 42, sum = 55: ready`.

`$/` selects the standard-library root: `$/stdio.resin` supplies input and output,
and `$/string.resin` supplies owned strings and formatting. Each file is a module;
imports make its exported names available.

The `counter` binding needs no `mut` because `increment` writes through a reference.
References neither retain ownership nor check lifetimes; keep the borrowed value
alive. The [reference chapter](../references.md) explains that contract.

Try changing the limit or increment and predicting which assertions will fail.
Continue with [Mandelbrot theory](theory.md), then turn the recurrence into an image.
