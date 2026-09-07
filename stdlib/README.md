# Standard library

Resin modules built on C and the native `resin-runtime` ABI. Import only the functionality a file
uses; the `std/` prefix resolves here. `RESIN_STDLIB` overrides this directory for installations.

```resin
import { "std/gpu.resin", "std/status.resin" };
```

- `gpu.resin`: devices, allocations, pipelines, GPU images, and commands.
- `window.resin`: windows and presentation.
- `image.resin`: PNG I/O.
- `status.resin`: native status conversion and named errors.
- `graphics.resin`: shared shader input/output types.
- `console.resin`: byte and line input, and owned input-line printing and release.

Public operation names omit the native `resin_` prefix. Fallible runtime operations return
`Result<T, RuntimeError>`; console operations use their own error sets. Infallible queries
return values and releases return unit.
Native declarations stay private. The C ABI is unchanged and remains unchecked: callers
must uphold pointer validity, lifetimes, and buffer sizes. Importing a module does not
re-export its dependencies. Import `std/status.resin` to name or match errors; inferred
`Result<(), _>` callers do not need that import.
`print`, `ok`, and `err` are unshadowable compiler builtins. Shader candidates use
`@compute_shader`, `@vertex_shader`, or `@fragment_shader`; `function.spirv` produces a
`Span<ubyte>` accepted directly by the compute/graphics pipeline wrappers.

Constructors return the new handle, not an integer and an out-parameter:

```resin
export { main };
import { "std/gpu.resin" };

def main() -> Result<(), _> = {
    var gpu = gpu_create()?;
    defer gpu_destroy(gpu);
    var data = gpu_malloc(gpu, 256, 16, memory_default())?;
    defer gpu_free(gpu, data);
    var commands = gpu_start_command_recording(gpu)?;
    defer gpu_cancel_command_buffer(gpu, &commands);
    gpu_submit(gpu, &commands)?;
    ok(())
};
```

Register each release after successful acquisition, before another fallible operation.
Cleanup runs in reverse order on scope exit, including through `?`.
Deferred expressions discard their values and must handle errors locally; they cannot
propagate with `?` themselves. Blocks can group multiple cleanup actions.
Keep GPU resources live until pending work completes, and avoid releasing handles whose
ownership was transferred. There is no automatic ownership checking.

Submission consumes a recording even on failure. `gpu_submit(gpu, &commands)` and
`gpu_cancel_command_buffer(gpu, &commands)` clear the caller's handle before entering C.
Deferred cancellation therefore releases unfinished recordings and is harmless after
submission. Do not retain or use copies of consumed handles. Submission waits for GPU work.

Some operations return additional information:

- `gpu_device_count()` returns the count; `gpu_host_to_device_pointer(gpu, host)` returns the address.
- `image_read_png(path, channels)` returns `ImageData { width, height, channels, pixels }`.
  Release `pixels` with `image_free`; `channels = 0` requests the file's channel count.
- `window_framebuffer_size(window)` returns `(width, height)`.
  `window_should_close` and `window_key_pressed` return booleans, not status codes.
- `gpu_present(gpu, image)` returns a boolean: presented or skipped. Native `INCOMPLETE`
  becomes a successful skipped frame; poll events and retry. Other failures remain errors.
- `gpu_enumerate_devices(infos, count)` still writes into caller-owned storage and returns
  `Incomplete` if truncated. Unlike skipped presentation, incomplete enumeration is an error;
  already-written entries remain available.

`RuntimeError` is the union of `InvalidArgument`, `VulkanUnavailable`, `Unsupported`,
`OutOfMemory`, `VulkanError`, `IoError`, `Incomplete`, `WindowUnavailable`, and
`UnknownRuntimeError { code: int }`. `status(code)` converts a native status;
`runtime_error_code(error)` recovers its number and `runtime_error_message(error)` returns
a borrowed, NUL-terminated native message. Unknown codes are preserved, not treated as success.
Unhandled entry-point errors print their variant name and exit with status 1 after defers run.
The exit-on-failure `check` helper is removed; wrappers never terminate the process.

## Console input

`input() -> Result<InputLine, InputError>` reads one line from stdin through C's `getchar()`.
The reading loop, buffer growth, newline removal, and ownership handling are written in Resin.
Small native helpers expose standard-stream operations and integer-width conversions.

```resin
export { main };
import { "std/console.resin" };

def main() -> Result<(), _> = {
    print("Name: ", ());
    var name = input()?;
    defer free_input(name);
    print("Hello, ", ());
    print_input(name)?;
    print("!\n", ());
    ok(())
};
```

The prompt is a separate `print` call, which flushes before input blocks. Unlike Python's
optional prompt argument, Resin's `input` takes no arguments. LF and CRLF endings are removed;
other bytes, including whitespace, embedded NULs, and a lone CR, are preserved as delivered by
the C stream. UTF-8 is preserved without decoding or validation. Windows standard streams use
the CRT's default text mode, including its newline and EOF translations.

`InputLine { data: Ptr<ubyte>, length: ulong }` owns a dynamically grown allocation with an
extra trailing NUL outside `length`. Empty lines succeed with length zero. EOF before any bytes
returns `EndOfInput`; a final nonempty line without a newline succeeds. The other `InputError`
variants are `InputReadError` and `InputOutOfMemory`. Failure frees any partial buffer; consumed
stdin bytes are not restored. Errors remain subject to C's stream error state.

Release a successful line exactly once with `free_input`, normally registered immediately with
`defer`. Copying a line copies its pointer, not its allocation; copies and pointers become invalid
after release. `print_input(line)` writes all `length` bytes, adds no newline, flushes stdout, and
returns `Result<(), InputWriteError>`. The builtin `print` does not accept `InputLine` records.

`read_byte() -> Result<ubyte, EndOfInput | InputReadError>` reads a single byte, including
NUL and 255, and distinguishes EOF from stream failure. The raw `getchar` binding is private;
callers never need to interpret its negative sentinel. These console APIs are for host execution.
