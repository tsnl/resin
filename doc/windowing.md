# Windows and presentation

## Windowing

```sh
cargo run -- examples/window.resin
cargo run -- examples/particles.resin
```
The demo renders a triangle until Escape or the close button is pressed. It uses a `while`
event loop and defines its decorated shader functions inline, as does the headless triangle demo.
Resizing scales the fixed-size offscreen image. Building this demo requires `spirv-opt`; running the
resulting executable does not. The PNG demos remain headless.

`particles.resin` seeds **1,000,000 particles** with pseudorandom 3D positions and velocities,
then advects them through a Lorenz attractor field on the GPU. A slowly orbiting camera shows
the two swirling lobes. Small sphere billboards use a blue → cyan → yellow → orange → red speed heatmap,
with smooth per-pixel normals, an upper-left light, and specular highlights. Perspective size
and distance fog distinguish near and far particles. Each sphere uses an eight-triangle octagon
(24 million vertices per frame); the silhouette is approximate, and overlaps still follow draw
order because the renderer has no depth buffer. Compute and graphics share a 24 MB particle
buffer, with constant-time vertex indexing and compute dispatch sized for the selected GPU. Pipelines
and allocations are reused.

The heatmap anchors are **20, 45, 65, 115, and 220 units per simulation second**. These
approximate the settled cloud's 5th, 25th, 50th, 75th, and 95th speed percentiles, sampled
with 50,000 particles across three seeds at frames 400, 1,000, and 2,000. Colors interpolate
between squared-speed anchors and clamp outside the range, keeping the palette useful
without letting rare fast particles stretch it. The scale stays fixed across frames and
reseeds; initial random velocities fall in the blue end. Lighting and fog modify brightness.

**Left-drag** to orbit horizontally and vertically, and **scroll** to zoom (0.35×–3×).
Either control stops automatic rotation; **Home** resets the view and resumes it. The camera
remains interactive while paused. Press **Space** to pause/resume, **R** to reseed the cloud,
and **Escape** to quit. The initial seed is reproducible; each reseed starts a different cloud.
The 1280×800 offscreen image scales with the window. Each frame advances two fixed 0.005-second simulation steps, so playback
speed depends on rendering throughput. Its decorated shader functions, ordinary helpers, and
shared data definitions live alongside the host code in the same file. Initialization accepts
a host `Span<Particle>`; the shaders use the same allocation's device address.

Input is available through `$/window.resin`. Exported `int` constants name GLFW key codes
(`key_w`, `key_space`, `key_left_shift`, `key_escape`) and the eight mouse buttons
(`mouse_button_left`, `mouse_button_right`, `mouse_button_middle`, `mouse_button_4` through `mouse_button_8`).
After polling, `window:key_state(key_w)` and
`window:mouse_button_state(mouse_button_left)` return `ButtonState` records
with `down`, `pressed`, and `released` booleans. A quick tap can set both edge flags;
key repeat does not create another press. `window:key_pressed(key)` queries `down`.

Poll each window once per frame before reading its input. Polling pumps GLFW events for
all windows, then commits that window's snapshot; other windows retain pending input
until their own poll. Edges and scroll reset on the next poll of that window, and repeated
queries read the same snapshot. `window:scroll_delta()` returns accumulated horizontal/vertical
scroll offsets (positive vertical scroll is up). `window:cursor_position()` returns coordinates
in window content units, with a top-left origin and positive y downward, independently of
framebuffer scaling. Both return `((float64, float64) | Err<RuntimeError>)`.

`window:focused()` reports keyboard focus. `window:capture_cursor(capture)` hides and
captures the cursor for camera controls when `capture` is true, with unbounded virtual
coordinates; set it false to restore normal behavior.
GLFW synthesizes button releases on focus loss. These APIs report physical controls;
they do not decode typed text or implement text composition.

Windowing is an ordinary runtime API, exposed by `resin_runtime/window.h` and
`$/window.resin`:

- `window_new(width, height, title: String)` returns a shared window owner; use `string_from_str("Resin")` for a literal title. `window:poll_events()` processes
  GLFW events. Close state, framebuffer size, resizing, and GLFW key codes
  are available through the corresponding window methods. Predicates return `bool`;
  fallible operations return error unions, including framebuffer size as `(width, height)`.
- `gpu_new_for_window(window)` selects a graphics/compute/present-capable GPU for a window.
  The existing GPU constructors stay headless. There is one GPU per window; multiple
  windows can each have their own GPU.
- `gpu:present(image)` blits an already-submitted `GpuImage` to the window, scaling to
  its framebuffer with FIFO presentation. Swapchains are recreated after resize.
  Its `(bool | Err<RuntimeError>)` is `(true)` when presented and `(false)` when
  skipped (minimized, timed out, or out of date): poll events and retry. Other failures propagate.

Create, use, and release windows and their GPUs on the process main thread. Shared
image/pipeline owners retain their GPU, which retains the window while using its surface.
The final owners release native resources in dependency order. Do not mix the runtime
with independently managed GLFW initialization.

GLFW is statically linked into the runtime and generated executables, and initialized only
when creating a window. Headless programs do not need a display. Initialization and window
creation failures print the GLFW error to stderr and return `RESIN_STATUS_WINDOW_UNAVAILABLE`.
Windowed executables need a display, its system libraries, and a suitable Vulkan driver at runtime,
but no GLFW shared library.
Presentation additionally requires `VK_EXT_swapchain_maintenance1` and its instance
dependencies, so presentation fences can safely govern resource reuse and teardown.
This initial path is deliberately synchronous and presents offscreen images; rendering
directly into swapchain images and multiple frames in flight are not implemented.

### Mandelbrot explorer

Run `resin examples/eg011_mandelbrot.resin` with the compiler service running.
The GPU `compute` entry is a one-line call to `evaluate_segment`; `--cpu` calls
that same ordinary function in a loop. Each invocation handles up to 32 adjacent
pixels in one row, clipping the final segment at the right edge. Segment origins
are prepared on the host; large dispatches use batches of descriptors within
Vulkan's workgroup limit. `pixels_per_segment` controls the chunk size. Pixels are
independent, so this grouping is a scheduling choice, not an algorithm requirement.
Orbit iteration, palette evaluation, and color blending are separate functions,
using `Complex<float32>` from `$/math.resin`.

Both interactive paths produce the same RGBA8 GPU buffer: CPU mode uploads its
completed bytes once with `:copy_from`, and GPU mode writes directly. A fullscreen
triangle presents that buffer through the same pipeline. The example groups pipeline
setup, uploads, dispatch, presentation, and readback in its GPU helpers section.
Headless CPU mode writes its host bytes directly, without GPU setup.
Pass example options after Resin's `--` separator:

```sh
resin examples/eg011_mandelbrot.resin -- --help
resin examples/eg011_mandelbrot.resin -- --cpu
resin examples/eg011_mandelbrot.resin -- --output mandelbrot.png --width 1920 --height 1080
resin examples/eg011_mandelbrot.resin -- --cpu --output detail.png --real -0.7435 --imag 0.1314 --span 0.005 --iterations 1024
```

`--output PATH` writes a PNG once and exits without creating a window. GPU output needs
a Vulkan device; CPU output needs neither a display nor a Vulkan device. Interactive
mode uses Vulkan for presentation with either solver. `--screenshot PATH` sets the
interactive screenshot filename, defaulting to `mandelbrot.png`; saving replaces that file.
`--width`/`--height` accept 1–8192, `--iterations` accepts 32–4096, and `--samples`
accepts 1–16. `--real`, `--imag`, and `--span` set the initial view.

The image matches the window's framebuffer resolution and preserves the complex plane's
aspect ratio when resized. Moving uses one sample per pixel; when input stops, a second
pass blends the selected number of subpixel colors (default four; `--samples 1`
disables refinement). Samples use prefixes of a fixed 16-point Halton lookup table,
with radical inverses in bases 2 and 3.
Headless output and screenshots use the selected sample count immediately.
The default budget is 256 iterations
per sample, with early escape and shortcuts for the main cardioid and period-two bulb.
The image is recomputed only after a change or for that refinement pass.
Zoom stops at a vertical span of `framebuffer_height * 1e-6` to leave room for subpixel
offsets with float32 coordinates near the Mandelbrot set.

- **Arrows / WASD:** pan.
- **Scroll / + / −:** zoom around the center.
- **[ / ]:** halve or double the iteration budget (32–4096).
- **R / Home:** reset the view and restore 256 iterations.
- **P:** save the current view as a PNG screenshot at framebuffer resolution.
- **Escape:** close.

Black pixels have not escaped within the chosen budget; this does not prove membership.
Run the exported `test` entry for known orbits, the strict escape-radius boundary,
pixel coordinates, and zoom limits: `resin examples/eg011_mandelbrot.resin:test`.

`window:wait_events(seconds)?` waits for input or a timeout and commits the same input
snapshot as `poll_events`. Use either operation once per frame; the timeout must be finite
and positive. The explorer uses it to avoid spinning while idle or minimized.
