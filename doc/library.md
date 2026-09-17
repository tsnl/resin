# Library reference

Import modules with `import { "$/name.resin" };`. `$` names the standard-library
root. Definitions are ordinary source declarations with explicit exports. Follow
the source links below for the complete signatures at this manual's revision.

| Module | Main operations | Lifetime and target contract |
| --- | --- | --- |
| [math](../resin/math.resin) | `Complex<T>`, `complex`, `add`, `mul`, `squared`, `magnitude_squared`, `sqrt`, `sin`, `cos` | Arithmetic helpers borrow inputs. Shader transcendental operations currently use `float32`. |
| [span](../resin/span.resin) | `Span<T>`, `at`, `lea`, `slice`, `bytes`, `as_bytes` | Borrowed pointer/length view; keep the backing storage alive. CPU and shader indexing have different bounds-check behavior. |
| [shared](../resin/shared.resin) | `arc_ptr_alloc`, `arc_span_alloc`, `get`, `clone`, `downgrade`, `upgrade` | Host allocation and ownership; borrowed views do not retain an owner. |
| [string](../resin/string.resin) | `String`, `string_from_str`, `string_from_bytes`, `fmt`, `print` | Host allocation and formatting; `str` is a distinct primitive literal view. |
| [io](../resin/io.resin) | `io_stdin`, `io_stdout`, `io_stderr`, reads and writes | Host I/O with ordinary error unions. |
| [process](../resin/process.resin) | Argument and environment snapshot access | Borrowed immutable startup data, valid until exit. |
| [argparse](../resin/argparse.resin) | `argparse`, `next`, `named`, `argument_integer`, `argument_number` | Host parser over borrowed arguments; `--size=` describes an option with a value. |
| [image](../resin/image.resin) | `image_data_read_png`, `image_data_write_pixels`, `write_png` | Host PNG I/O; raw pixel writes take bounded spans. |
| [gpu](../resin/gpu.resin) | Device creation, allocation, pipelines, commands, dispatch | Host-managed ownership; completion synchronizes device access. |
| [graphics](../resin/graphics.resin) | `Color`, `Vertex`, image and presentation support | Explicit shader interfaces and rendering resources. |
| [window](../resin/window.resin) | Window creation, input, resize, polling | Host window operations; the interactive examples run them on the main thread. |
| [status](../resin/status.resin) | Runtime error types | Named members of ordinary error unions. |

Read [strings and I/O](strings.md), [GPU buffers](gpu-buffers.md), and
[windows](windowing.md) for worked usage. The source export lists are authoritative
for available overloads; this index deliberately links to them instead of copying
an API signature catalog that could drift.
