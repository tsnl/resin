# Maintaining the manual

From the repository root:

```sh
nix-shell --run 'mdbook build'
python3 scripts/check-manual.py
nix-shell --run 'mdbook serve --open'
```

The book lives in `doc/`; `doc/SUMMARY.md` owns navigation. Generated HTML belongs
in `target/manual/`. Markdown remains readable in the repository. The build
preprocessor rewrites links to repository files outside `doc/` to the current Git
revision, so source links also work in the generated site.

## Executable examples

Tutorial excerpts use mdBook `include` anchors in complete `.resin` files.
Edit those files, format them, then rebuild the book. Avoid copying source into
Markdown. Run:

```sh
nix-shell --run 'cargo test --test tutorial'
nix-shell --run 'cargo test --all-features --test mandelbrot --test spirv_execution'
```

The first suite compiles and executes the CPU checkpoints, checks their PNG output,
and exercises documented rejection cases. The existing Mandelbrot suite checks
CPU/GPU sample agreement, headless output, resize, and screenshots. GPU execution
requires a working driver; window checks also require a display. Set the required
feature environment variables described in [Development](development.md) to make
missing GPU/window facilities fail rather than skip.

`mdbook test` runs Rust snippets; it does not test Resin. The ordinary Cargo tests
own Resin compilation and execution. Mark explanatory fragments as `text` or
include a tested complete program. Each language rule should have one reference
page; tutorial explanations link to it. Keep implementation restrictions and
language guarantees distinct, and update both tests and documentation in the PR
that changes behavior.

The documentation workflow builds HTML and checks local links, anchors, and assets
on Linux. It uses mdBook 0.5.4; `scripts/manual.py` requires Python 3.9 or later. Its artifact can be reviewed
before deciding where to host the manual.
