# Standard library

Resin modules built on C and the native `resin-runtime` ABI. Import only the functionality a file
uses; the `std/` prefix resolves here. `RESIN_STDLIB` overrides this directory for installations.

```resin
import { "std/gpu.resin", "std/status.resin" };
```

- `gpu.resin`: devices, allocations, pipelines, GPU images, and commands.
- `window.resin`: windows and presentation.
- `image.resin`: PNG I/O.
- `status.resin`: `status(code)` returns `Result<(), RuntimeError>`; `check(code)` still exits on failure.
- `graphics.resin`: shared shader input/output types.
- `console.resin`: `getchar`, dynamically sized line input, and owned input-line printing and release.

Foreign bindings can be exported directly when no wrapper is useful. Native declarations
remain unchecked: callers must uphold pointer validity, lifetimes, and buffer sizes. Importing
a module does not re-export its dependencies. `print`, `shader`, `ok`, and `err` are unshadowable compiler builtins.

Use `status(native_call(...))?` in a Result-returning function. After each successful
resource acquisition, register its release with `defer release(resource);` before another fallible
operation. Cleanup runs in reverse order on scope exit, including through `?`.
Deferred expressions discard their values and must handle errors locally; they cannot
propagate with `?` themselves. Blocks can group multiple cleanup actions.
Keep GPU resources live until pending work completes, and avoid releasing handles whose
ownership was transferred. The existing `check` helper terminates the process on failure,
bypassing defers; existing graphics examples still use it with explicit cleanup.

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

The exported `getchar() -> int` is also available directly: it returns an unsigned byte as an
integer or a negative value for EOF/error. These console APIs are for host execution.
