# PNG images

Import `$/image.resin` for host PNG input and output. The
[Mandelbrot image checkpoint](tutorial/image.md) is a complete example that writes
an owned pixel buffer to disk without GPU execution.

## Read an image

`image_data_read_png(path, channels)` returns `ImageData | Err<RuntimeError>`.
The path is a pointer to a NUL-terminated string, such as `"image.png".data`.
A zero channel count preserves the file's channel count; one through four requests
that number of output channels.

`image:width()`, `image:height()`, and `image:channels()` return dimensions.
`image:pixels()` borrows a packed `Span<ubyte>`. Keep the image owner alive while
using that view. `image:clone()` retains shared ownership; the final owner releases
the pixels. `image:write_png(path)` writes its dimensions and pixels to a PNG.

## Write a pixel buffer

`image_data_write_pixels(path, width, height, channels, pixels, stride)` accepts
a borrowed `Span<ubyte>`. It validates dimensions, channel count, row stride, and
capacity before entering the native writer. Channels must be between one and four.
A zero stride requests packed rows; a nonzero stride is the byte distance between
row starts and must accommodate one whole row.

Storage may end at the last pixel: padding after the final row is not required.
Numeric spans expose their bytes through `:as_bytes()`. For a GPU image buffer,
first use `:copy_to(host:get())` to read back into owned host storage, then keep
that owner alive through the write. See [GPU buffers](gpu-buffers.md).

Image I/O runs on the host. PNG read and write operations return ordinary runtime
error unions; use `?` to propagate them or `match` to handle failures.
