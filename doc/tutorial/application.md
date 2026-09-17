# 6. Finish the application

A useful explorer should also produce an image without opening a window and make
its rendering parameters reproducible. The [options module](../../examples/mandelbrot/options.resin)
separates argument parsing and validation from execution.

## Parse arguments once

The exported entry receives `argc`, `argv`, and `envp`. The process library's
`arguments(argc, argv)` borrows the full argument vector, including the program
name at index zero. The option parser begins at index one. Startup input is
an owned snapshot kept alive for the duration of the program; its borrowed bytes
must be treated as read-only.

[`$/argparse.resin`](../../resin/argparse.resin) takes a small string description:

```text
--help|-h --cpu --gpu --width= --height= --iterations= --samples= --real= --imag= --span= --output= --screenshot=
```

`|` separates aliases; a trailing `=` means the option takes a value. The parser
recognizes options and values. `parse_options` converts numbers, rejects invalid
ranges and non-finite coordinates, and builds an `Options` value. This keeps
validated settings out of the rendering loop's concerns. Run `--help` for the
complete current spelling and bounds.

The entry then selects headless or interactive execution:

```resin
{{#include ../../examples/eg011_mandelbrot.resin:main}}
```

`match` narrows the union in each arm. `Options(options)` unwraps the successful
value; `Err(message)` handles a parse failure and returns exit status two. `None`
means no output path was supplied. Runtime failures use `?` and remain visible
through the entry's `int | Err<_>` result.

## Render reproducibly

First build a release executable to avoid including compilation or debug-C costs
in timing:

```sh
cargo run -- examples/eg011_mandelbrot.resin -o /tmp/mandelbrot
/tmp/mandelbrot --cpu --width 1280 --height 960 --iterations 512 --samples 16 --real -0.5 --imag 0 --span 3 --output overview.png
```

`--output` selects headless mode. With `--cpu`, that mode allocates only host
buffers and writes a PNG: a machine without Vulkan or a desktop can run it.
`--gpu --output overview.png` uses compute and readback without opening a window.
When using `cargo run`, place program flags after Resin's extra `--` separator.

The same dimensions, center, span, iteration limit, sample count, and execution
mode describe a render. CPU/GPU floating-point differences mean this is not a
cross-device promise of byte-identical files.

## Capture the current view

Pass `--screenshot capture.png`, then press P in interactive mode. The application
renders at the selected sample quality when a capture coincides with a view change.
It reads the rendered image in one transfer and writes a PNG:

```resin
{{#include ../../examples/eg011_mandelbrot.resin:screenshot}}
```

Screenshot code belongs next to GPU execution helpers. Its staging allocation and
copy are explicit; the solver and palette never need to know that a file is being
written. The default path is `mandelbrot.png`; writing the same path again replaces
the previous capture.

## What this program teaches about Resin

- A small algorithm can stay in ordinary functions and run under both execution targets.
- `Ref<T>` describes access without granting an address. Explicit pointers keep that
  stronger capability visible at interfaces, including indexing with `:lea`.
- A borrowed descriptor and an owner are different values. Keep owners alive for
  every use of their spans; Resin does not check these lifetimes for you.
- Error unions, `?`, and `match` cover both library failure and application policy.
- GPU support is a subset with target-specific representation limits. Those limits
  should produce source diagnostics, not force a second copy of the algorithm.

The tests check known orbits, segment write boundaries, all Halton prefixes on CPU
and GPU, headless PNGs, and interactive resize and screenshots. The tutorial
checkpoints and the two rejected reference examples are tested too. See
[maintaining the manual](../manual-development.md) for commands.

Possible next experiments: a smooth escape-time palette, a different work
partition, or progressive sampling that retains the previous sum. Change one
layer at a time and compare CPU and GPU outputs before changing the next.
