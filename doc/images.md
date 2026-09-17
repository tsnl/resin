# PNG and HDR images

Import `$/image.resin` for host PNG, OpenEXR, and Radiance HDR input and output. The
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

## Linear HDR images

`image_data_read_exr(path)` reads the first RGB(A) OpenEXR layer.
`image_data_read_hdr(path)` reads a Radiance RGBE image. Both return
`FloatImageData | Err<RuntimeError>` with packed, top-left-origin RGBA `f32`
samples. RGB remains linear and unclamped, including values above one and
negative EXR values. Missing alpha becomes one; alpha is not premultiplied.

The same `:width()`, `:height()`, `:channels()`, `:pixels()`, and `:clone()`
operations apply; `:pixels()` borrows a `Span<f32>`. Keep the owner alive through
that borrow. `image:write_exr(path)` preserves the floating-point samples.
`image_data_write_exr_pixels(path, width, height, samples)` requires exactly
`width × height × 4` samples and rejects invalid dimensions/counts before opening
the output file. These operations do not tone map or change color spaces.

See the [Arris tutorial](arris.md) for EXR/HDR environment lighting and saving
linear renderer output alongside a tone-mapped PNG.
