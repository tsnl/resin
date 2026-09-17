# 1. Follow one orbit

The [theory chapter](theory.md) established the recurrence and strict escape test.
Now implement that test: return the first escape iteration, or zero when no escape
was found within the chosen budget.

```resin
{{#include ../../examples/tutorial/fractal.resin:orbit}}
```

Zero is a convention for this renderer, **not a membership proof**. A point that
survives 256 iterations can escape at iteration 257. More iterations reveal more
structure near the boundary and cost more work.

## Read the Resin

`Complex<float32>` comes from [`$/math.resin`](../../resin/math.resin). `let mut`
permits reassignment; `let` alone binds an immutable value. The suffix `_f` chooses
`float32`, and `_i` chooses `int`. Without the suffixes, unconstrained floats and
integers default to `float64` and `long` respectively.

`z:squared():add(c)` uses free functions through method notation:
`add(squared(z), c)`. The math helpers take `Ref<Complex<T>>` parameters, so they
can read local complex values without moving them. The intermediate result is a
temporary that lasts through the expression. These calls do not create a
user-visible pointer to `z`. See [references](../references.md) for the contract.

A function's final expression supplies its result without `return`; `return n`
exits early. `assert(condition)` is a language construct. Booleans are `true` and
`false`.

## Check the boundary before drawing it

For `c = 1`, the nonzero iterates are `1, 2, 5`: escape occurs at iteration three.
For `c = 2`, they begin `2, 6`: escape occurs at two. Check these along with fixed
points and cycles:

```resin
{{#include ../../examples/tutorial/01_orbit.resin:assertions}}
```

Run the complete [checkpoint](../../examples/tutorial/01_orbit.resin):

```sh
cargo run -- examples/tutorial/01_orbit.resin
```

Success produces no output. Try `>=` in the escape test and watch the `-2` and
one-iteration assertions catch the mistake.

The [complete explorer](../../examples/eg011_mandelbrot.resin) additionally skips
iteration for the known interior of the main cardioid and period-two bulb. That
optimization changes how much work it does, not the escape-count interface used
by coloring.
