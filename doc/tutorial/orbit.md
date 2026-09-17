# Follow one orbit

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

`Complex<f32>` comes from [`$/math.resin`](../../resin/math.resin). `let mut`
permits reassignment; `let` alone binds an immutable value. Numeric literals get
their types from context; an unconstrained integer defaults to `i64`, and a float
to `f64`. The `Complex<f32>` annotation chooses the recurrence's precision.

`z:squared():add(c)` uses free functions through UFCS: it means
`add(squared(z), c)`. The math helpers take ordinary values and return new values.
`Complex<f32>` copies because both fields copy, so these calls leave their inputs
unchanged and compose without intermediate locals. Assignment installs the new
value in `z`. See [ownership](../lifetimes.md) for copying and move-only types.

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
