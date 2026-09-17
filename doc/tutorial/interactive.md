# 5. Explore interactively

Run the complete explorer on a machine with a compatible Vulkan device and display:

```sh
cargo run -- examples/eg011_mandelbrot.resin -- --gpu --samples 4
```

The default view is 960 × 720, with 256 maximum iterations and four samples per
pixel at rest. Use `--cpu` to run the same evaluator on the CPU. Interactive CPU
mode still uses the GPU to present the image.

| Input | Action |
| --- | --- |
| Arrows or WASD | Pan |
| Scroll, `+`, or `-` | Zoom |
| `[` or `]` | Halve or double the iteration limit, within 32–4096 |
| R or Home | Reset the view and iteration limit |
| P | Save a screenshot |
| Escape | Close |

## Keep one presentation path

`Renderer` owns the GPU, pipelines, uploaded samples and segment origins, pixel
buffer, and presentation image. It groups GPU resource management; the recurrence
and color functions do not depend on it.

The CPU path copies its output into the same pixel buffer the compute shader
writes. Both modes then draw that buffer into the same image:

```resin
{{#include ../../examples/eg011_mandelbrot.resin:render}}
```

The shared path makes switching execution modes small and visible. It does add a
host-to-device copy to CPU mode. Include that cost when measuring interactive
latency; measure the evaluator separately when comparing algorithm throughput.
The host-only headless path bypasses this machinery entirely.

## Track what changed

The event loop keeps two booleans:

- `dirty`: the view, dimensions, or iteration limit changed.
- `refine`: the last render used one sample and needs a render at the selected quality.

Wait for events, inspect input, render if either flag is set, then present. The
first changed frame uses one sample for responsiveness. The next quiet frame uses
the requested number. This is a second complete render, not an accumulating or
memoized image. An unchanged view is presented without rerunning the solver.

When the framebuffer changes size, recreate the pixel buffer, segment list, and
image, update the plot dimensions, and mark the view dirty. A minimized window may
have zero width or height: skip rendering until dimensions are nonzero. Held keys
are paced by the event wait, while bracket/reset/screenshot actions use key-press
transitions.

Read [`interactive`](../../examples/eg011_mandelbrot.resin) alongside the
[window library reference](../windowing.md). Keep math changes in `Plot` and
`Mandelbrot`, and keep window and GPU ownership in application helpers. This makes
the rendering policy changeable without forking the solver.

## Keep zoom expectations honest

The plot uses single precision on both targets. The application clamps the
vertical span to at least `height × 1e-6`, and at most eight, to avoid the most
obvious loss of subpixel differences in its usual view. This is an application
limit, not a universal precision guarantee for every possible center coordinate.

A fuzzy edge may need more samples; a region incorrectly painted black may need
more iterations. If nearby coordinates round to the same floating-point number,
neither adjustment helps. A deeper-zoom implementation needs a different numeric
strategy, with corresponding CPU and GPU support.
