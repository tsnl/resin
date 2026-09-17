# 1. Follow one orbit

For each complex coordinate \\(c\\), start a fresh sequence:

\\[
z_0 = 0, \qquad z_{n+1} = z_n^2 + c.
\\]

The Mandelbrot set contains the coordinates whose sequences stay bounded. The
constant added at every step is **the original coordinate `c`**. It does not change
with `z`. Different pixels usually have unrelated sequences; there is no useful
general-purpose dynamic-programming table shared between their iterations.

## A finite test

An orbit that reaches \\(|z_n| > 2\\) will diverge. Compare the squared magnitude
against four to avoid a square root. Use strict `>`, not `>=`: for `c = -2`, the
orbit is `0, -2, 2, 2, ...` and never escapes.

Why does the radius work? If \\(|c| \leq 2\\), the reverse triangle inequality gives
\\(|z^2+c| \geq |z|^2-2\\). Above two, this is greater than \\(|z|\\); the excess
above two grows, so the orbit cannot settle at a finite limit. If \\(|c|>2\\), the
first iterate already has magnitude \\(|c|\\), and
\\(|z^2+c| \geq |z|^2-|c| \geq |z|(|z|-1)\\) once \\(|z|\geq|c|\\), giving growth
by a factor greater than one. These statements concern exact arithmetic; our
floating-point computation is an approximation.

Choose an iteration budget. Return the first escape iteration, or zero when no
escape was found within that budget:

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
