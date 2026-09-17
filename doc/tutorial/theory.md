# Mandelbrot theory

Each point in the picture is a separate experiment with a complex number. We draw
how quickly that experiment escapes a small circle; the boundary reveals the
fractal. No recursive geometry or subdivision is needed.

## Iterate a coordinate

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
escape was found within that budget. A finite budget can miss eventual escapes;
black pixels are not a general proof of membership.

## What we visualize

Map a sample inside each pixel to a complex coordinate `c`. Color points that
escape according to their first escape iteration; color the others black. The
circle of radius two is a bound on the **iterates `z`**, not a circular viewport or
a boundary crossed by a moving pixel. The coordinate `c` stays fixed throughout
its experiment.

The escape count is useful data independent of the palette. Recoloring can reuse
counts; panning, zooming, or adding subpixel samples generally asks about new
coordinates. Memoizing arbitrary intermediate `z` values is usually unhelpful:
different `c` values define different recurrences, and exact shared floating-point
states are rare. Our explorer simply evaluates each sample independently.

## Known cases

Use these cases before rendering an image. The table counts updates beginning
with `z₁ = c`; it uses a budget of 256 and a strict squared-magnitude test `> 4`.
The zero result means “no escape observed,” not an iteration numbered zero.

| Coordinate `c` | Orbit from `z₀ = 0` | First escape iteration | Solver result |
| --- | --- | --- | --- |
| `0` | `0 → 0 → 0 → …` | Never | `0` |
| `-1` | `0 → -1 → 0 → -1 → …` | Never | `0` |
| `-2` | `0 → -2 → 2 → 2 → …` | Never | `0` |
| `i` | `0 → i → -1+i → -i → -1+i → …` | Never | `0` |
| `1` | `0 → 1 → 2 → 5 → …` | `3` | `3` |
| `2` | `0 → 2 → 6 → …` | `2` | `2` |
| `3` | `0 → 3 → 12 → …` | `1` | `1` |

The `-2` and `2` cases catch an accidental `>=` test. With a budget of only one,
`c = 2` returns zero: its escape has not happened yet. Conjugate coordinates have
conjugate orbits, so their exact escape counts agree; this is another useful check.

We will turn these cases into assertions in [Follow one orbit](orbit.md), then
separate solving, coloring, and blending before running the evaluator on the GPU.
