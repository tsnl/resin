# Resin libraries

Libraries written in Resin live under this directory. The current modules provide
platform functionality through C and the native `resin-runtime` ABI, plus shared
graphics types. Add modules or subdirectories here as libraries grow.

Imports beginning with `$/` resolve from this directory: `$/gpu.resin` loads
`gpu.resin`, and a future `$/math/matrix.resin` would load `math/matrix.resin`.
`RESIN_LIBRARY_ROOT` overrides the whole library root. Other imports resolve relative
to the importing file. No manifests or special entry files are required; each file
explicitly imports the functionality it uses and exports its public declarations.

```resin
import { "$/gpu.resin", "$/status.resin" };
```

- `gpu.resin`: devices, allocations, pipelines, GPU images, commands, and window presentation.
- `span.resin`: borrowed spans and literal byte views.
- `shared.resin`: shared pointer and sequence allocation, with corresponding weak owners.
- `string.resin`: owned strings, formatting, and printing.
- `window.resin`: windows, named controls, button snapshots, cursor capture, and scrolling.
- `image.resin`: PNG I/O.
- `status.resin`: native status conversion and named errors.
- `graphics.resin`: shared shader input/output types.
- `io.resin`: stdout and stderr byte output accepting `str`, `Span<ubyte>`, and `String`.
- `console.resin`: byte and line input, and shared input-line ownership and printing.
- `process.resin`: checked argument views and lookups in the frozen startup environment.

Public operations use static and instance methods on the corresponding types; scalar
options and control codes are exported constants. Fallible runtime operations return
`Result<T, RuntimeError>`; console operations use their own error sets. Infallible queries
return values. Resource owners release their native handles automatically.
Native declarations stay private. The C ABI remains unchecked: callers
must uphold pointer validity, lifetimes, and buffer sizes. Importing a module does not
re-export its dependencies. Import `$/status.resin` to name or match errors; inferred
`Result<(), _>` callers do not need that import.
Modules under `internal/` support these wrappers and are not part of the public library API.
`ok` and `err` are compiler builtins; `fmt` and `print` are ordinary exports of
`string.resin`. Shader entries use
`@compute_shader`, `@vertex_shader`, or `@fragment_shader`. Pass these declarations
directly to GPU pipeline creation; compiled representations are handled internally.

Constructors return the new handle, not an integer and an out-parameter:

```resin
export { main };
import { "$/gpu.resin" };

def main() -> Result<(), _> = {
    var gpu = Gpu.new()?;
    var data = gpu.alloc::<uint>(64)?;
    var commands = gpu.start_command_recording()?;
    commands.submit()?;
    ok(())
};
```

Resources use shared owners. Copying a handle retains its allocation, and initialized
locals release ownership in reverse scope order, including through `?`. GPU views
retain their allocation and device through indexing and slicing. Use `load`,
`store`, and `replace` for checked host access.
`gpu.create(value)?` infers `GpuPtr<T>`; `gpu.alloc::<T>(count)?` allocates
uninitialized elements. `.read_only()` and `.write_only()` narrow host access.

Pipeline creation accepts decorated shader declarations and preserves their root
type: `gpu.create_compute_pipeline(kernel)` and
`gpu.create_graphics_pipeline(vertex, fragment)`. Record work with
`commands.dispatch(pipeline, arguments, x, y, z)` or
`commands.draw(pipeline, arguments, count)`; rootless graphics use
`commands.draw(pipeline, None, count)`. The compiler checks host arguments and
projects GPU views to the shader's raw pointers and spans inside recording.
Recorded allocations reject CPU access until submission or cancellation. Use
`.copy_to(Span<T>)` for host readback without escaping a raw pointer into GPU memory.
See [GPU buffers](../doc/gpu-buffers.md).

Submission consumes a recording even on failure. `commands.submit()` and
`commands.cancel()` clear the shared native handle before entering C.
Aliases observe that cleared state. Dropping an unfinished recording cancels it;
dropping one already consumed does not cancel it again. Submission waits for GPU work.

| Type | Constructors and operations |
| --- | --- |
| `ArcPtr<T>` / `ArcSpan<T>` | `.alloc(initial)` / `.alloc(count, initial)`, `.get()`, `.downgrade()` |
| `Span<T>` | `.at(index)`, `.slice(start, length)`, numeric `.as_bytes()` |
| `WeakPtr<T>` / `WeakSpan<T>` | `.empty()`, `.upgrade()` |
| `Gpu` | `Gpu.new()`, `Gpu.new_at(index)`, `Gpu.new_for_window(window)`, `gpu.create_compute_pipeline(kernel)`, `gpu.create_image(...)` |
| `GpuPtr<T>` | `gpu.create(value)`, `.load()`, `.store(value)`, `.replace(value)`, `.slice(start, length)`, `.read_only()`, `.write_only()` |
| `GpuSpan<T>` | `gpu.alloc::<T>(count)`, `gpu.alloc_in::<T>(count, memory)`, `.at(index)`, `.slice(start, length)`, `.copy_to(destination)`, `.read_only()`, `.write_only()` |
| `GpuComputePipeline<Root, Owner>` / `GpuGraphicsPipeline<Root, Owner>` | `gpu.create_compute_pipeline(kernel)`, `gpu.create_graphics_pipeline(vertex, fragment)` |
| `GpuCommands` | `gpu.start_command_recording()`, `commands.dispatch(pipeline, arguments, x, y, z)`, `commands.draw(pipeline, arguments, count)`, `commands.submit()`, `commands.cancel()` |
| `Window` | `Window.new(width, height, String.from_str("Resin"))`, `window.poll_events()`, `window.framebuffer_size()`, input and cursor methods |
| `ImageData` | `ImageData.read_png(path, channels)`, `image.write_png(path)` |
| `Console` / `InputLine` | `Console.read_byte()`, `Console.read_line()`, `Console.print(line)` |
| `RuntimeStatus` | `RuntimeStatus.from_code(code)`, `RuntimeStatus.code(error)`, `RuntimeStatus.message(error)` |

Import `$/gpu.resin` for the `int` constants `memory_default`, `memory_gpu`, and
`memory_readback`, used as `gpu.alloc_in::<uint>(count, memory_readback)`.
Import `$/window.resin` for `key_escape`, `key_w`, `key_space`, and the other GLFW
key codes, plus `mouse_button_left`, `mouse_button_right`, `mouse_button_middle`,
and `mouse_button_4` through `mouse_button_8`. Pass these constants directly to
`window.key_state(key_w)` and `window.mouse_button_state(mouse_button_left)`.
These replace the former `Memory` methods, `Window.key_escape()`, `Window.keys()`,
and `Window.mouse_buttons()`; the `KeyCodes` and `MouseButtons` records are removed.

`ImageData.write_pixels(path, width, height, channels, pixels, stride)` accepts a
borrowed `Span<ubyte>`. It checks dimensions, channel count, row stride, and the
span's capacity before calling the native image writer. A zero stride means packed
rows; a nonzero stride separates row starts. Storage may end at the last pixel,
without padding after the final row. Copy GPU output to owned host
storage with `GpuSpan<T>.copy_to` first, and keep the owner alive through the write.
The instance method `image.write_png(path)` uses the loaded image's dimensions and pixels.

Import `$/shared.resin` for
`ArcSpan<T>.alloc(count, initial) -> Result<ArcSpan<T>, OutOfMemory>`. Each element is
initialized with an ordinary copy of `initial`; the type determines its size.
Allocation size overflow and allocation failure return `OutOfMemory`. Empty
sequences are valid. Copies of the returned handle retain its allocation, and the
last owner destroys the elements in reverse order and frees their storage.
`owner.get()` borrows a `Span<T>` without retaining the allocation.

```resin
var host = ArcSpan<uint>.alloc(pixels.length, 0_ui)?;
pixels.copy_to(host.get());
ImageData.write_pixels(path.data, width, height, 4, host.get().as_bytes(), 0)?;
```

For numeric elements, `Span<T>.as_bytes()` exposes their in-memory bytes explicitly.
`ArcPtr<T>` owns one value, while `ArcSpan<T>` owns a sequence; `WeakPtr<T>` and
`WeakSpan<T>` provide the corresponding weak references. A plain `Span<T>` is still
a borrowed descriptor. `ArcPtr<Span<T>>` shares that descriptor, not its elements.

Some operations return additional information:

- `Gpu.device_count()` returns the count.
- Pipeline factories return typed shared owners. Dispatch and draw project arguments
  internally, preserving interior byte offsets in GPU views.
- `ImageData.read_png(path, channels)` returns a shared `ImageData` owner exposing
  `width()`, `height()`, `channels()`, and `pixels()` methods. The final owner releases the pixels;
  `channels = 0` requests the file's channel count.
- `window.framebuffer_size()` returns `(width, height)`.
  `window.should_close()` and `window.key_pressed(key)` return booleans, not status codes.
- `gpu.present(image)` returns a boolean: presented or skipped. Native `INCOMPLETE`
  becomes a successful skipped frame; poll events and retry. Other failures remain errors.
- `Gpu.enumerate_devices(infos, count)` still writes into caller-owned storage and returns
  `Incomplete` if truncated. Unlike skipped presentation, incomplete enumeration is an error;
  already-written entries remain available.

`RuntimeError` is the union of `InvalidArgument`, `VulkanUnavailable`, `Unsupported`,
`OutOfMemory`, `VulkanError`, `IoError`, `Incomplete`, `WindowUnavailable`, and
`UnknownRuntimeError { code: int }`. `RuntimeStatus.from_code(code)` converts a native status;
`RuntimeStatus.code(error)` recovers its number and `RuntimeStatus.message(error)` returns
a borrowed, NUL-terminated native message. Unknown codes are preserved, not treated as success.
Unhandled entry-point errors print their variant name and exit with status 1 after scope cleanup.
Fallible resource operations return errors for callers to handle or propagate.
Invalid checked pointer access and the infallible formatting/printing operations can trap.

## Console input

`Console.read_line() -> Result<InputLine, InputError>` reads one line from stdin through C's `getchar()`.
The reading loop, buffer growth, newline removal, and ownership handling are written in Resin.
Small native helpers expose standard-stream operations and integer-width conversions.

```resin
export { main };
import { "$/console.resin", "$/string.resin" };

def main() -> Result<(), _> = {
    print("Name: ");
    var name = Console.read_line()?;
    print("Hello, ");
    Console.print(name)?;
    print("!\n");
    ok(())
};
```

The prompt is a separate `print` call, which flushes before input blocks. Unlike Python's
optional prompt argument, `Console.read_line()` takes no arguments. LF and CRLF endings are removed;
other bytes, including whitespace, embedded NULs, and a lone CR, are preserved as delivered by
the C stream. UTF-8 is preserved without decoding or validation. Windows standard streams use
the CRT's default text mode, including its newline and EOF translations.

`InputLine` is a shared owner whose `get()` method borrows a `Span<ubyte>`. Its
dynamically grown allocation has an extra trailing NUL outside `length`. Empty lines
succeed with length zero. EOF before any bytes returns `EndOfInput`; a final nonempty line without a newline succeeds. The other `InputError`
variants are `InputReadError` and `InputOutOfMemory`. Failure frees any partial buffer; consumed
stdin bytes are not restored. Errors remain subject to C's stream error state.

Copying a line retains shared ownership; the final owner frees its allocation. Raw pointers
into that allocation do not retain it. `Console.print(line)` writes all `length` bytes, adds no
newline, flushes stdout, and returns `Result<(), InputWriteError>`. The library function `print` does
not accept `InputLine` values.

`Console.read_byte() -> Result<ubyte, EndOfInput | InputReadError>` reads a single byte, including
NUL and 255, and distinguishes EOF from stream failure. The raw `getchar` binding is private;
callers never need to interpret its negative sentinel. These console APIs are for host execution.


`Io.stdout().write(text)` and `Io.stderr().write(text)` accept `str | Span<ubyte> | String`, write
bytes verbatim, flush, and return `Result<(), WriteError>`. Import `$/io.resin` to use them.
Use `fmt("n = {0}", (n,))` to construct an owned String before writing or storing it.
Literals have type `str` over static bytes; formatting results own an `ArcSpan<ubyte>` allocation. Use
`bytes(literal)` from `$/span.resin` when a raw byte view is needed. An InputLine
can be passed as `line.get()` while
its owner remains live.

`String.from_str(text)` copies a `str` without formatting. `String.from_bytes(span)` copies
arbitrary raw bytes, including non-UTF-8 data and embedded NULs. Both append a NUL outside the
logical length. `Window.new(width, height, title)` takes an owned String from either constructor
or from `fmt`.
