# Strings and console I/O

## Strings, formatting, and output

String literals have primitive type `str`, distinct from raw `Span<ubyte>` views and owned
`String` values. They expose `data: Ptr<ubyte>` and `length: ulong` over static UTF-8 bytes,
so copying or returning one does not allocate or introduce an owner. The length excludes a
trailing NUL; embedded and explicitly trailing `\0` bytes count toward its length. The escapes are `\n`, `\r`, `\t`,
`\0`, `\"`, and `\\`. Pass `text.data` to C functions that take a NUL-terminated string.
Literal storage may be shared; treat it as read-only.

Import `$/span.resin` and use `bytes(text)` to explicitly borrow the bytes of a `str`; this preserves its pointer
and length without copying. There is no implicit conversion, and arbitrary byte spans cannot
be converted to `str`. `text:at(index)` returns a byte reference (`Ref<ubyte>`) and uses a `ulong` index.

Import `$/string.resin` for `String`, `fmt`, and `print`.
`fmt(format, arguments)` is an ordinary generic host function returning `String`,
a source wrapper with a `storage: ArcSpan<ubyte>` field. Its allocation owns the bytes and an additional
trailing NUL outside their logical length. Cloning a String retains the allocation; the final owner releases it.
Extracting a raw span or pointer does not retain that owner.
`string_from_str(text)` copies a `str` verbatim into an owned String. For raw bytes, use
`string_from_bytes(span)`; the span need not be UTF-8 or have a NUL terminator. Both constructors
preserve embedded NULs and treat braces as ordinary bytes.

```resin
export { main };
import { "$/io.resin", "$/string.resin" };

fn main() -> (() | Err<_>) {
	let n = 42;
	let message = fmt("n = {0}\n", (n,));
	io_stdout():write(message)?;
	io_stderr():write(fmt("diagnostic: {0}", (message:bytes(),)))?;
	print("done\n");
	(())
}
```

The format accepts `str`, `Span<ubyte>`, or `String`. In the argument tuple,
use `view:bytes()` or `message:bytes()` for span and String values. These methods
provide an explicit structural byte view to the formatting primitive. Literal
`str` arguments work directly. Other supported arguments are numbers, booleans,
unit, and pointer addresses. The argument tuple is explicit, including the
trailing comma for a single argument. `{0}`, `{1}`, etc. are zero-based and may repeat;
`{{` and `}}` escape braces. Arguments evaluate once in source order, including unused arguments.
Malformed formats and invalid indices terminate with a diagnostic before any formatted output
is written. Formatting itself performs no output.

`io_stdout()` and `io_stderr()` return ordinary library `Output` values. Their `write` method
has overloads for `str`, `Ref<Span<ubyte>>`, and `Ref<String>`, writes bytes verbatim, flushes, adds no newline, and returns
`(() | Err<WriteError>)`. The library function `print(text)` is a stdout shorthand returning unit;
it terminates on an output error. Neither writer interprets braces. Both `fmt` and `print`
are ordinary source functions and host-only.

Byte arrays and device-backed `Span<ubyte>` values support shader reads and writes using
8-bit storage and arithmetic extensions. The runtime enables the corresponding Vulkan features when available.
Shader `str` literals remain unsupported: the backend does not yet provide addressable
constant storage for the device addresses used by Resin spans. Pass a span of uploaded bytes in the shader root instead.


## Console input

Import `$/console.resin` for `console_read_line()`, a line reader implemented in Resin on top of C's
`getchar()`. It grows its buffer as needed and strips LF or CRLF. Write a prompt with `print`
before reading:

```resin
export { main };
import { "$/string.resin", "$/console.resin" };

fn main() -> (() | Err<_>) {
	print("Name: ");
	let name = console_read_line()?;
	print("Hello, ");
	console_print(name)?;
	print("!\n");
	(())
}
```

The result is a shared `InputLine` owner; `line:get()` borrows its `Span<ubyte>` view.
Explicit clones retain its allocation; the final owner frees it. `console_print(line)` prints the bytes without adding a newline. Empty lines succeed, EOF before any
bytes returns `EndOfInput`, and a final line without a newline succeeds. Read and allocation
failures are also explicit errors. See [the console API](../resin/README.md#console-input) for
ownership and byte semantics, or run `cargo run -- examples/input.resin`.
