# PNG images

Import `$/image.resin` for host PNG input and output. The
[Mandelbrot explorer](tutorial/index.md) writes an owned pixel buffer to disk
without GPU execution when run with `--cpu --output mandelbrot.png`.

## Read an image

`image_data_read_png(path, channels)` returns `ImageData | Err<RuntimeError>`.
Paths are bounded `Span<u8>` values, such as `bytes("image.png")` from
`$/span.resin`. A path may be a slice without a trailing NUL: the library copies
exactly its bytes and adds termination at the C boundary. Embedded NULs return
`InvalidArgument` instead of silently truncating the path.
A zero channel count preserves the file's channel count; one through four requests
that number of output channels.

`image:width()`, `image:height()`, and `image:channels()` return dimensions.
`image:pixels()` borrows a packed `SpanMut<u8>`. Keep the image owner alive while
using that view. `image:clone()` retains shared ownership; the final owner releases
the pixels. `image:write_png(path)` writes its dimensions and pixels to a PNG.

## Write a pixel buffer

`image_data_write_pixels(path, width, height, channels, pixels, stride)` accepts
borrowed `Span<u8>` values for both the path and pixels. It validates dimensions,
channel count, row stride, and capacity before entering the native writer. Channels must be between one and four.
A zero stride requests packed rows; a nonzero stride is the byte distance between
row starts and must accommodate one whole row.

Storage may end at the last pixel: padding after the final row is not required.
Numeric spans expose their bytes through `:as_bytes()`. For a GPU image buffer,
first use `:copy_to(host:get())` to read back into owned host storage, then keep
that owner alive through the write. See [GPU buffers](gpu-buffers.md).

Image I/O runs on the host. PNG read and write operations return ordinary runtime
error unions; use `?` to propagate them or `match` to handle failures.
