# Strings and console I/O

## Strings, formatting, and output

String literals have primitive type `str`, distinct from raw `Span<u8>` views and owned
`String` values. They expose `data: Ptr<u8>` and `length: u64` over static UTF-8 bytes,
so copying or returning one does not allocate or introduce an owner. The length excludes a
trailing NUL; embedded and explicitly trailing `\0` bytes count toward its length. The escapes are `\n`, `\r`, `\t`,
`\0`, `\"`, and `\\`. Pass `text.data` to C functions that take a NUL-terminated string.
Literal storage may be shared; treat it as read-only.

Import `$/span.resin` and use `bytes(text)` to explicitly borrow the bytes of a `str`; this preserves its pointer
and length without copying. There is no implicit conversion, and arbitrary byte spans cannot
be converted to `str`. `text:at(index)` returns a byte reference (`Ref<u8>`) and uses a `u64` index.

Import `$/string.resin` for `String`, `fmt`, and `repr`, and `$/stdio.resin` for `print` and stream I/O.
`fmt(format, arguments)` is an ordinary generic host function returning `String`,
a source wrapper with a `storage: ArcSpan<u8>` field. Its allocation owns the bytes and an additional
trailing NUL outside their logical length. Cloning a String retains the allocation; the final owner releases it.
Extracting a raw span or pointer does not retain that owner.
`string_from_str(text)` copies a `str` verbatim into an owned String. For raw bytes, use
`string_from_bytes(span)`; the span need not be UTF-8 or have a NUL terminator. Both constructors
preserve embedded NULs and treat braces as ordinary bytes.

```resin
export { main };
import { "$/stdio.resin", "$/string.resin" };

fn main() -> (() | Err<_>) {
	let n = 42;
	let message = fmt("n = {0}\n", (n,));
	let stdout = io_stdout();
	stdout:write(message)?;
	let diagnostic = fmt("diagnostic: {0}", (message:bytes(),));
	let stderr = io_stderr();
	stderr:write(diagnostic)?;
	print("done\n");
	(())
}
```

The format accepts `str`, `Span<u8>`, or `String`. In the argument tuple,
use `view:bytes()` or `message:bytes()` for span and String values. These methods
provide an explicit structural byte view to the formatting primitive. Literal
`str` arguments work directly. Other host values use their representation, described below. The argument tuple is explicit, including the
trailing comma for a single argument. `{0}`, `{1}`, etc. are zero-based and may repeat;
`{{` and `}}` escape braces. Arguments evaluate once in source order, including unused arguments.
Malformed formats and invalid indices terminate with a diagnostic before any formatted output
is written. Formatting itself performs no output.

`io_stdout()` and `io_stderr()` return ordinary library `Output` values. Their `write` method
has overloads for `str`, `Ref<Span<u8>>`, and `Ref<String>`, writes bytes verbatim, flushes, adds no newline, and returns
`(() | Err<WriteError>)`. The library function `print(text)` is a stdout shorthand returning unit;
it terminates on an output error. Neither writer interprets braces. Both `fmt` and `print`
are ordinary source functions and host-only.

Byte arrays and device-backed `SpanMut<u8>` values support shader reads and writes using
8-bit storage and arithmetic extensions. The runtime enables the corresponding Vulkan features when available.
Shader `str` literals remain unsupported: the backend does not yet provide addressable
constant storage for the device addresses used by Resin spans. Pass a span of uploaded bytes in the shader root instead.


## Console input

Import `$/stdio.resin` for `console_read_line()`, a line reader implemented in Resin on top of C's
`getchar()`. It grows its buffer as needed and strips LF or CRLF. Write a prompt with `print`
before reading:

```resin
export { main };
import { "$/string.resin", "$/stdio.resin" };

fn main() -> (() | Err<_>) {
	print("Name: ");
	let name = console_read_line()?;
	print("Hello, ");
	console_print(name)?;
	print("!\n");
	(())
}
```

The result is a shared `InputLine` owner; `line:get()` borrows its `Span<u8>`
view. Value copies and explicit clones retain the allocation; the final owner frees it. Raw pointers
and spans do not keep it alive. The allocation has a trailing NUL outside `length`.
Empty lines succeed, EOF before any bytes returns `EndOfInput`, and a final nonempty
line without a newline succeeds. Read and allocation failures return `InputReadError`
and `InputOutOfMemory`; they free any partial buffer without restoring consumed input.

LF and CRLF endings are removed. Other bytes, including whitespace, embedded NULs,
and a lone CR, are preserved as delivered by C. UTF-8 is not decoded or validated.
Windows streams use the CRT's default text mode, including newline and EOF translation.

`console_print(line)` writes all bytes, adds no newline, flushes stdout, and returns
`(() | Err<InputWriteError>)`. `print` does not directly accept an `InputLine`; pass
its borrowed byte span or use `console_print`. A prompt written with `print` is
flushed before input blocks.

`console_read_byte()` returns `u8 | Err<EndOfInput | InputReadError>` and preserves
all byte values, including NUL and 255, while distinguishing EOF from stream failure.
These functions run on the host. Try `cargo run -- examples/input.resin`.

## Value representations

Import `repr` from `$/string.resin` to obtain an owned `String` describing any
host value. Records show their names and fields, arrays and tuples show their
elements, and unions show their active payload. Strings are quoted and escaped;
pointers show addresses and opaque handles show their type. `fmt` accepts these
values too, while direct string arguments retain their verbatim text behavior.
Unhandled entry-point errors include this representation before cleanup.

A struct may provide `repr_bytes(self: Ref<Self>)` returning the primitive
`(Ptr<u8>, u64)` byte view. Use the actual struct name in
place of `Self`. The view must remain readable while its receiver is alive;
the hook borrows its receiver and must not invalidate it. `String` uses this
hook so formatting and nested representations show its text. Representation
is a host operation and limits nested output to 128 levels.
