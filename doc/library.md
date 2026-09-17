# Library reference

Import modules with `import { "$/name.resin" };`. `$` names the standard-library
root. Definitions are ordinary source declarations with explicit exports. The API pages below are generated from exported declarations and documentation
comments at this manual's revision; each page also links to its source.

| Module | Main operations | Lifetime and target contract |
| --- | --- | --- |
| [`$/math.resin`](../resin/math.resin) | `Complex<T>`, `complex`, `add`, `mul`, `squared`, `magnitude_squared`, `sqrt`, `sin`, `cos` | Arithmetic helpers borrow inputs. Shader transcendental operations currently use `f32`. |
| [`$/span.resin`](../resin/span.resin) | `Span<T>`, `at`, `lea`, `slice`, `bytes`, `as_bytes` | Borrowed pointer/length view; keep the backing storage alive. CPU and shader indexing have different bounds-check behavior. |
| [`$/shared.resin`](../resin/shared.resin) | `arc_ptr_alloc`, `arc_span_alloc`, `get`, `clone`, `downgrade`, `upgrade` | Host allocation and ownership; borrowed views do not retain an owner. |
| [`$/string.resin`](../resin/string.resin) | `String`, `string_from_str`, `string_from_bytes`, `fmt`, `repr` | Host allocation and formatting; `str` is a distinct primitive literal view. |
| [`$/stdio.resin`](../resin/stdio.resin) | `print`, `io_stdout`, `io_stderr`, `write`, `console_read_byte`, `console_read_line`, `console_print` | Host input and output; stream operations expose error unions and input distinguishes EOF. |
| [`$/process.resin`](../resin/process.resin) | Argument and environment snapshot access | Borrowed immutable startup data, valid until exit. |
| [`$/argparse.resin`](../resin/argparse.resin) | `argparse`, `next`, `named`, `argument_integer`, `argument_number` | Host parser over borrowed arguments; `--size=` describes an option with a value. |
| [`$/image.resin`](../resin/image.resin) | `image_data_read_png`, `image_data_write_pixels`, `write_png` | Host PNG I/O; raw pixel writes take bounded spans. |
| [`$/gpu.resin`](../resin/gpu.resin) | Device creation, allocation, pipelines, commands, dispatch | Host-managed ownership; completion synchronizes device access. |
| [`$/graphics.resin`](../resin/graphics.resin) | `Color`, `Vertex`, image and presentation support | Explicit shader interfaces and rendering resources. |
| [`$/window.resin`](../resin/window.resin) | Window creation, input, resize, polling | Host window operations; the interactive examples run them on the main thread. |
| [`$/status.resin`](../resin/status.resin) | Runtime error types | Named members of ordinary error unions. |

The API index excludes `$/internal/` modules. Read the [library guides](library-guides.md)
for complete usage examples. `$/` resolves from the compiler service's configured
library root (`RESIN_LIBRARY_ROOT` or `--library-root`); ordinary imports resolve
relative to the importing file.
