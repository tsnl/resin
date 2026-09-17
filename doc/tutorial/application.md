# Finish the application

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
--help|-h --cpu --gpu --width= --height= --iterations= --real= --imag= --span= --output=
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
through the entry's `i32 | Err<_>` result.

## Render reproducibly

First build a release executable to avoid including compilation or debug-C costs
in timing:

```sh
cargo run -- examples/eg011_mandelbrot.resin -o /tmp/mandelbrot
/tmp/mandelbrot --cpu --width 1280 --height 960 --iterations 512 --real -0.5 --imag 0 --span 3 --output overview.png
```

`--output` selects headless mode. With `--cpu`, that mode allocates only host
buffers and writes a PNG: a machine without Vulkan or a desktop can run it.
`--gpu --output overview.png` uses compute and readback without opening a window.
When using `cargo run`, place program flags after Resin's extra `--` separator.

The same dimensions, center, span, iteration limit, and execution
mode describe a render. CPU/GPU floating-point differences mean this is not a
cross-device promise of byte-identical files.

## What this program teaches about Resin

- A small algorithm can stay in ordinary functions and run under both execution targets.
- `Ref<T>` describes access without granting an address. Explicit pointers keep that
  stronger capability visible at interfaces, including indexing with `:lea`.
- A borrowed descriptor and an owner are different values. Keep owners alive for
  every use of their spans; Resin does not check these lifetimes for you.
- Error unions, `?`, and `match` cover both library failure and application policy.
- GPU support is a subset with target-specific representation limits. Those limits
  should produce source diagnostics, not force a second copy of the algorithm.

The tests check known orbits, segment write boundaries, CPU/GPU agreement,
headless PNGs, and interactive resize. The tutorial checkpoints and rejected
reference examples are tested too. See [maintaining the manual](../manual-development.md).

## Next steps

Try supersampling: evaluate several positions inside each pixel and average their
colors. A fixed Halton sequence in a lookup table gives repeatable sample positions.
Other extensions include a smooth escape-time palette or a different work partition.
Change one layer at a time and compare CPU and GPU outputs.
