# Resin for Zed

Language support for `.resin` files: highlighting, comments, brackets,
indentation, outlines, and function/comment text objects. The native `resin --lsp` mode
adds diagnostics, hover, go-to-definition, basic completion, and formatting.

Both this extension and the grammar are ordinary folders in the
[Resin repository](https://github.com/tsnl/resin). No separate repository or
submodule checkout is required.

## Install for development

The tracked `.zed/settings.json` contains shared editor preferences. Keep
host-specific executable paths and launch commands in Zed's **user settings**;
no generated or gitignored project settings file is needed. By default, the
extension launches an installed `resin` from PATH without building it on startup.

From the Resin repository root, use the build environment appropriate to your host.
With Nix on Linux or macOS (NixOS is not required):

```sh
nix-shell --run 'cargo install --path . --locked'
nix-shell --run 'rustup target add wasm32-wasip2'
```

Without Nix, install the platform's [native build dependencies](../../README.md#development),
then run these commands in that development environment (a Visual Studio developer
PowerShell on Windows):

```sh
cargo install --path . --locked
rustup target add wasm32-wasip2
```

Make sure Cargo's installation directory is on the PATH visible to Zed, or configure
the executable's absolute path in user settings as shown below.

The repository's `rust-toolchain.toml` declares the `wasm32-wasip2` target,
so Rustup installs it when activating the project toolchain. If Zed reports
`can't find crate for core`, run the target-install command above from this
repository root, then retry installing the dev extension. Rust targets are
installed per toolchain; adding the target outside the repository can select
a different Rust version.

Launch Zed from your development environment so it inherits the build tools and native
library paths, then run **zed: install dev extension** in the command palette and
select `editors/zed/`. With Nix, use `nix-shell --run 'zed .'` (or
`nix-shell --run 'zeditor .'` on systems whose executable is named `zeditor`).
Zed compiles the extension and downloads the WASI SDK to build the grammar.
Rebuild it from Zed's Extensions view after changing the adapter or queries.
Restarting `resin --lsp` alone does not reload highlighting queries or the pinned
Tree-sitter grammar. If `struct`, `match`, or `impl` still look like ordinary
identifiers, rebuild/reinstall the dev extension from this checkout's `editors/zed/`.
See [Zed's extension development guide](https://zed.dev/docs/extensions/developing-extensions).

The adapter uses a configured binary or finds `resin` on the worktree's PATH,
then passes `--lsp` and the worktree root. A configured `binary.arguments` replaces
that argument list; include `--lsp` and the project directory when overriding it.
For a local debug build, run `cargo build -p resin` in your development environment
(`nix-shell --run 'cargo build -p resin'` with Nix). Open Zed's user settings with
**zed: open settings file** and set an absolute path:

```json
{
  "lsp": {
    "resin-lsp": {
      "binary": {
        "path": "/absolute/path/to/resin/target/debug/resin"
      },
      "initialization_options": {
        "libraryRoot": "/absolute/path/to/checkout/resin"
      }
    }
  }
}
```

On Windows, use the path to `target/debug/resin.exe`. When replacing an earlier
`cargo run` override, also remove its `binary.arguments` so the adapter supplies
`--lsp` and the current worktree root.

The library-root override is useful when switching worktrees or using an
installed server built in another checkout. Alternatively, set `RESIN_LIBRARY_ROOT` in
the environment. Configured `binary.env` entries override the inherited worktree
environment. Restart the language server after changing its configuration or
rebuilding the native executable.

For syntax support alone, Zed's language setting
`"languages": { "Resin": { "enable_language_server": false } }` disables the LSP.

## Formatting

After rebuilding/reinstalling `resin` and restarting the language server, use
**Format Document**. To use the language server for formatting and enable it on
save, add these [language settings](https://zed.dev/docs/configuring-languages):

```json
{
  "languages": {
    "Resin": {
      "formatter": "language_server",
      "format_on_save": "on",
      "hard_tabs": true
    }
  }
}
```

The formatter always indents with hard tabs; `hard_tabs` also makes ordinary editor
indentation use tabs. `tab_size` controls their display width. A trailing comma
forces a list onto multiple lines and is preserved, except for singleton tuples such as
`(x,)` and `(int,)`. Comments and nested multiline content can still force breaks. Comments stay intact, and
multiple blank lines collapse to one. See the [formatting rules](../../crates/resin-lsp/README.md#formatting)
for details. Incomplete syntax is left unchanged until repaired.

## Check the workflow

Open `examples/eg009_imports.resin`. Hover `next`, navigate to its definition in
`examples/lib/counter.resin`, and navigate `RuntimeStatus.from_code` into `resin/status.resin`.
Rename a use without saving to see a diagnostic, then repair it to clear the
diagnostic. Type a prefix inside a function to see visible-name completions.
Edits to an open imported buffer apply to its consumers before saving.

Use **dev: open language server logs** for protocol logs and **zed: open log**
for extension errors. A missing executable produces an installation message.
If `$/` imports fail, check the library root. If the executable cannot
load a native library, restart Zed from your development environment.

`Failed to find wayland-scanner` is a native build failure in GLFW's Linux Wayland
support. It occurs when a `cargo run` launch override builds Resin without the
required tools. Build or install Resin in the development environment above, then
use the installed executable or an absolute binary path. If you keep an automatic
build command in user settings, it must enter that environment too: `shell.nix`
already supplies `wayland-scanner`; without Nix, install it with the other Linux
build dependencies. Run Cargo from the Resin checkout so Rustup selects its pinned
toolchain, and pass the opened project directory to `--lsp`. Restart the language
server after updating its launch settings.

## Build and maintain

```sh
nix-shell --run 'cargo build --manifest-path editors/zed/Cargo.toml --release --target wasm32-wasip2'
nix-shell --run 'cargo test -p resin --test zed_queries'
```

The extension is an independent Cargo workspace depending only on
`zed_extension_api` 0.7. Its grammar pin points to a Resin commit with
`path = "crates/tree-sitter-resin"`. After a grammar change, commit the regenerated
parser in Resin and update that pin. Query tests compile every query and check
captures against representative syntax, examples, and standard-library files.

The extension is not published in Zed's registry yet. A future registry entry
can point to this repository with `path = "editors/zed"`; see the
[publishing guide](https://zed.dev/docs/extensions/publishing/publishing-guide).
Current limitations and compiler architecture are documented in
[crates/resin-lsp/README.md](../../crates/resin-lsp/README.md).

Validated with Zed 1.17.2 on Linux under Xvfb: Zed's own builder compiled and
loaded the extension and pinned grammar; highlighting, imported/standard-library
navigation, hover, and unsaved completion/diagnostic recovery worked in the editor.
The protocol suite additionally covers imported-buffer updates and Unicode edits.

After rebasing the PR stack onto `79e4a26`, Zed rebuilt the declarations-only
grammar and loaded `examples/eg009_imports.resin`. Hover displayed
`next(counter: Ptr<Counter>) -> int`, go-to-definition opened `lib/counter.resin`,
and the outline included both `Counter` and `next`.

The earlier grammar pin at `70a05ac` added `def`/`var`/`type` declarations and optional unit
result annotations. Compiler/LSP regressions and query captures cover this syntax,
including hover/completion for omitted unit results. This revision was validated
through automated tests and WASI builds; the editor smoke test above used `79e4a26`.

The grammar also supports explicit `_` type-inference holes, including nested
local and return annotations. Rebuild `resin` and reinstall the dev extension
after updating. Parser, query, and stdio regressions cover the new syntax and
inferred hover types without launching an editor.

The grammar revision at `847bc7a` added nominal `struct` declarations, transparent `type`
aliases, `A | B` unions, `Result<T, E>`, postfix `?`, and exhaustive `match` arms.
Queries highlight the new syntax and include structs in the outline; semantic tests
cover inferred error sets and match-payload navigation/completion. Rebuild the language
server and reinstall the dev extension together. No new editor smoke test was run.

The ownership grammar adds inherent `impl` methods, `Arc<T>`, `Weak<T>`, and
`Option<T>`, and removes the legacy cleanup statement. Parser, compiler, query,
and recovery tests cover the updated syntax. Reinstall the dev extension after
updating so the grammar and queries stay in sync.

Current queries highlight all declaration and control keywords, label top-level
structs and aliases with `struct`/`type` in the outline, and expose struct bodies
through Zed's class text objects. Query tests cover these captures, including
`Result`, inference holes, `match`, and `impl`.
