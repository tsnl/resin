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
- `window.resin`: windows, named controls, button snapshots, cursor capture, and scrolling.
- `image.resin`: PNG I/O.
- `status.resin`: native status conversion and named errors.
- `graphics.resin`: shared shader input/output types.
- `io.resin`: stdout and stderr streams with string-only `write` methods.
- `console.resin`: byte and line input, and shared input-line ownership and printing.
- `process.resin`: checked argument views and lookups in the frozen startup environment.

Public operations use static and instance methods on the corresponding types. Fallible runtime operations return
`Result<T, RuntimeError>`; console operations use their own error sets. Infallible queries
return values. Resource owners release their native handles automatically.
Native declarations stay private. The C ABI remains unchecked: callers
must uphold pointer validity, lifetimes, and buffer sizes. Importing a module does not
re-export its dependencies. Import `$/status.resin` to name or match errors; inferred
`Result<(), _>` callers do not need that import.
`fmt`, `print`, `ok`, and `err` are unshadowable compiler builtins. Shader candidates use
`@compute_shader`, `@vertex_shader`, or `@fragment_shader`; `function.spirv` produces a
`Span<ubyte>` accepted directly by the compute/graphics pipeline wrappers.

Constructors return the new handle, not an integer and an out-parameter:

```resin
export { main };
import { "$/gpu.resin" };

def main() -> Result<(), _> = {
    var gpu = Gpu.new()?;
    var data = GpuSpan<uint>.allocate(gpu, 64)?;
    var commands = gpu.start_command_recording()?;
    commands.submit()?;
    ok(())
};
```

Resources use shared owners. Copying a handle retains its allocation, and initialized
locals release ownership in reverse scope order, including through `?`. GPU views
retain their allocation and device through indexing, slicing, and field addresses.
`gpu.new(value)?` infers `GpuPtr<T>`; `GpuSpan<T>.allocate(gpu, count)?` allocates
uninitialized elements. `.read_only()` and `.write_only()` narrow host access.

Compiler projection `kernel.project(gpu, arguments)?` converts host launch-record
GPU views to the shader's raw pointers and spans. The resulting `GpuArguments`
retains all referenced allocations. Commands accept those roots with
`commands.dispatch(root, x, y, z)` or `commands.draw(root, count)`; rootless graphics
use `commands.draw(None, count)`. Recorded allocations reject CPU access until
submission or cancellation. Use `.copy_to(Span<T>)` for host readback without
escaping a raw pointer into GPU memory. See [GPU buffers](../doc/gpu-buffers.md).

Submission consumes a recording even on failure. `commands.submit()` and
`commands.cancel()` clear the shared native handle before entering C.
Aliases observe that cleared state. Dropping an unfinished recording cancels it;
dropping one already consumed does not cancel it again. Submission waits for GPU work.

| Type | Constructors and operations |
| --- | --- |
| `Gpu` | `Gpu.new()`, `Gpu.new_at(index)`, `Gpu.new_for_window(window)`, `gpu.malloc(...)`, `gpu.create_compute_pipeline(code)`, `gpu.create_image(...)` |
| `GpuPtr<T>` / `GpuSpan<T>` | `gpu.new(value)`, `GpuSpan<T>.allocate(gpu, count)`, `.at(index)`, `.slice(start, length)`, `.read_only()`, `.write_only()` |
| `GpuArguments` | `kernel.project(gpu, arguments)` |
| `GpuCommands` | `gpu.start_command_recording()`, `commands.set_pipeline(...)`, `commands.dispatch(root, x, y, z)`, `commands.draw(root, count)`, `commands.submit()`, `commands.cancel()` |
| `Window` | `Window.new(width, height, String.from_str("Resin"))`, `window.poll_events()`, `window.framebuffer_size()`, input and cursor methods |
| `ImageData` | `ImageData.read_png(path, channels)`, `image.write_png(path)` |
| `Console` / `InputLine` | `Console.read_byte()`, `Console.read_line()`, `Console.print(line)` |
| `Memory` | `Memory.default()`, `Memory.gpu()`, `Memory.readback()` |
| `RuntimeStatus` | `RuntimeStatus.from_code(code)`, `RuntimeStatus.code(error)`, `RuntimeStatus.message(error)` |

`ImageData.write_pixels(path, width, height, channels, pixels, stride)` writes from
borrowed host memory; copy GPU output there with `GpuSpan<T>.copy_to` first. The
caller keeps that memory valid.
The instance method `image.write_png(path)` uses the loaded image's dimensions and pixels.

Some operations return additional information:

- `Gpu.device_count()` returns the count. `kernel.project(gpu, arguments)` returns
  an owning shader root whose projected views preserve their interior byte offsets.
- `ImageData.read_png(path, channels)` returns a shared `ImageData` owner exposing
  `width`, `height`, `channels`, and `pixels`. The final owner releases the pixels;
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
The exit-on-failure `check` helper is removed; wrappers never terminate the process.

## Console input

`Console.read_line() -> Result<InputLine, InputError>` reads one line from stdin through C's `getchar()`.
The reading loop, buffer growth, newline removal, and ownership handling are written in Resin.
Small native helpers expose standard-stream operations and integer-width conversions.

```resin
export { main };
import { "$/console.resin" };

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

`InputLine` is a shared owner exposing `data: Ptr<ubyte>` and `length: ulong`. Its
dynamically grown allocation has an extra trailing NUL outside `length`. Empty lines
succeed with length zero. EOF before any bytes returns `EndOfInput`; a final nonempty line without a newline succeeds. The other `InputError`
variants are `InputReadError` and `InputOutOfMemory`. Failure frees any partial buffer; consumed
stdin bytes are not restored. Errors remain subject to C's stream error state.

Copying a line retains shared ownership; the final owner frees its allocation. Raw pointers
into that allocation do not retain it. `Console.print(line)` writes all `length` bytes, adds no
newline, flushes stdout, and returns `Result<(), InputWriteError>`. The builtin `print` does
not accept `InputLine` values.

`Console.read_byte() -> Result<ubyte, EndOfInput | InputReadError>` reads a single byte, including
NUL and 255, and distinguishes EOF from stream failure. The raw `getchar` binding is private;
callers never need to interpret its negative sentinel. These console APIs are for host execution.


`Io.stdout().write(text)` and `Io.stderr().write(text)` accept `str | Span<ubyte> | String`, write
bytes verbatim, flush, and return `Result<(), WriteError>`. Import `$/io.resin` to use them.
Use `fmt("n = {0}", (n,))` to construct an owned String before writing or storing it.
Literals have type `str` over static bytes; formatting results own an Arc allocation. Use
`Span<ubyte>(literal)` when a raw byte view is needed. InputLine
can be passed as an explicit `Span<ubyte> { data = line.data, length = line.length }` while
its owner remains live.

`String.from_str(text)` copies a `str` without formatting. `String.from_bytes(span)` copies
arbitrary raw bytes, including non-UTF-8 data and embedded NULs. Both append a NUL outside the
logical length. `Window.new(width, height, title)` takes an owned String from either constructor
or from `fmt`.
