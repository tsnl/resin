# Maintaining the manual

Install mdBook 0.5.4 and Python 3.12 or later alongside the [development toolchain](development.md).
mdBook is a Rust program; Python runs this repository’s build and link-checking scripts.
Nix users can enter `nix-shell` for these tools. From the repository root:

```sh
mdbook build
python3 scripts/check-manual.py
mdbook serve --open
```

The result in `target/manual/` is a portable static site: copy that directory to
any static HTTP server or download the `resin-manual` artifact from the Manual CI
workflow. No Resin service or application backend is needed to serve it. MathJax
for mathematical notation is loaded from a CDN. The Manual workflow can also publish
the checked site to GitHub Pages on request.

The book lives in `doc/`; `doc/SUMMARY.md` owns navigation. Generated HTML belongs
in `target/manual/`. Markdown remains readable in the repository; the library index links to source there
and to generated API chapters in HTML. The build
preprocessor rewrites links to repository files outside `doc/` to the current Git
revision, so source links also work in the generated site.

## Generated illustration

Each build compiles the complete Mandelbrot explorer and runs its headless CPU
mode at 640 × 480 with 256 iterations and one sample per pixel. The build script
starts an isolated loopback compiler service on a free port, then stops it and
removes its temporary files. This needs the normal native build tools, including
SPIR-V Tools because the application also contains shaders; it needs no GPU or display.
The PNG is embedded at the top of the Mandelbrot tutorial, keeping the site portable.
A failed render fails the book build.

The fisheye tutorial uses the checked-in `doc/assets/ray-tracing.png`, rendered by
`examples/eg013_ray_tracing.resin`; book builds need no ray tracing hardware.
The first-program illustration shows the console output checked by its test.

## Generated library reference

The preprocessor builds the local CLI and runs `resin --doc` for every public
`resin/*.resin` module. It adds one chapter named by its import path, such as
`$/math.resin`, with exported signatures and attached Markdown. Edit `///` and `//!`
comments in Resin source; do not hand-edit generated API pages. These pages need
neither a compiler service nor GPU/native runtime tools.

`CARGO_TARGET_DIR` selects the Cargo artifact directory. For an already built,
matching CLI, `RESIN_DOC_TOOL=/absolute/path/to/resin mdbook build` skips building the CLI; the illustration still builds the service and runtime.
A source change requires rebuilding that CLI if its documentation behavior changed.
`mdbook serve` watches the book source; restart or rebuild after library-source
changes to refresh generated API text.

## Executable examples

Each tutorial is one page: a title, an example image, and one mdBook `include`
of the entire `.resin` file. Put run commands and explanations in source comments,
beside the code they explain. Keep examples readable as standalone programs;
keep detailed API contracts in the reference chapters. Edit the source, format
it, then rebuild the book. Avoid copying code into Markdown. Run:

```sh
cargo test --test tutorial
cargo test --all-features --test mandelbrot --test spirv_execution
cargo test --features gpu --test ray_tracing
```

The first suite compiles and executes the CPU checkpoints, checks their PNG output,
and exercises documented rejection cases. The existing Mandelbrot suite checks
CPU/GPU pixel agreement, headless output, resize, and screenshots. GPU execution
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
on Linux. It uses mdBook 0.5.4; `scripts/manual.py` requires Python 3.12 or later. Its artifact can be reviewed
as a portable static site. Pushes and pull requests only build and check the book.

## Publishing to GitHub Pages

After the workflow is merged, a repository administrator must select **Settings →
Pages → Build and deployment → Source → GitHub Actions** once. In **Settings →
Environments → github-pages**, allow deployments from the default branch (normally
`main`). No personal access token or repository secret is needed.

Open **Actions → Manual → Run workflow**, select the default branch, check
**Publish the latest default-branch manual to this repository's GitHub Pages**,
and run it. Equivalently, for this repository:

```sh
gh workflow run manual.yml --repo tsnl/resin --ref main -f deploy=true
```

The workflow checks out the latest default branch, builds the book and its generated
reference and illustration, checks links and assets, then deploys the Pages artifact.
Deployment requests from other branches are skipped. Leave the checkbox unchecked
to build a downloadable preview without publishing. Deployments are serialized and
are not cancelled by ordinary book builds.

The deployment job reports the published URL; for `tsnl/resin` without a custom
domain it is <https://tsnl.github.io/resin/>. The workflow reads the current
repository's Pages URL, so project paths, forks with Pages enabled, and custom
domains use their own site configuration. Only the deployment job receives Pages
write and OIDC permissions.
