# Resin for Zed

Language support for `.resin` files: highlighting, comments, brackets,
indentation, outlines, and function/comment text objects. The native `resin --lsp` mode
adds diagnostics, hover, go-to-definition, basic completion, and formatting.

Both this extension and the grammar are ordinary folders in the
[Resin repository](https://github.com/tsnl/resin). No separate repository or
submodule checkout is required.

## Install for development

From the Resin repository root:

```sh
cargo install --path . --locked
rustup target add wasm32-wasip2
```
The repository's `rust-toolchain.toml` declares the `wasm32-wasip2` target,
so Rustup installs it when activating the project toolchain. If Zed reports
`can't find crate for core`, run the target-install command above from this
repository root, then retry installing the dev extension. Rust targets are
installed per toolchain; adding the target outside the repository can select
a different Rust version.

Launch Zed from your configured development environment so it inherits any native
library paths (Nix users can use `nix-shell`), then run
**zed: install dev extension** in the command palette and select `editors/zed/`.
On systems whose executable is named `zeditor`, use `zeditor .`.
Zed compiles the extension and downloads the WASI SDK to build the grammar.
Rebuild it from Zed's Extensions view after changing the adapter or queries.
Restarting `resin --lsp` alone does not reload highlighting queries or the pinned
Tree-sitter grammar. If `struct`, `match`, or `fn` still look like ordinary
identifiers, rebuild/reinstall the dev extension from this checkout's `editors/zed/`.
An `Error loading highlights query` with `Invalid node type "::"` means the
queries require a newer parser than the extension's grammar pin. Update the
checkout to include the corrected pin, then rebuild/reinstall the dev extension.
See [Zed's extension development guide](https://zed.dev/docs/extensions/developing-extensions).

The adapter uses a configured binary or finds `resin` on the worktree's PATH,
then passes `--lsp` and the worktree root. A configured `binary.arguments` replaces
that argument list; include `--lsp` and the project directory when overriding it.
For a local debug build, run `cargo build -p resin` and set
an absolute path in Zed settings:

```json
{
  "lsp": {
    "resin-lsp": {
      "binary": {
        "path": "/absolute/path/to/resin/target/debug/resin",
        "env": { "RESIN_SERVER": "http://127.0.0.1:7412" }
      },
      "initialization_options": {
        "includeRoots": ["include"]
      }
    }
  }
}
```

Start a [compiler service](compiler-service.md) at the configured URL.
`RESIN_SERVER` is mandatory; LSP initialization negotiates capabilities before
reporting success. Configured `binary.env` entries override the inherited worktree
environment. Standard-library selection belongs to the service's `--library-root`
/ `RESIN_LIBRARY_ROOT`; editor `libraryRoot` is no longer accepted. Optional
`includeRoots` selects client header directories relative to the editor project.
Restart the language server after changing configuration or rebuilding the client.
See [Zed language-server configuration](https://zed.dev/docs/configuring-languages#language-servers).

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
multiple blank lines collapse to one. See the [formatting rules](editor-client.md#formatting)
for details. Incomplete syntax is left unchanged until repaired.

## Check the workflow

Open `examples/eg009_imports.resin`. Hover `next`, navigate to its definition in
`examples/lib/counter.resin`, and navigate `runtime_status_from_code` into `resin/status.resin`.
Rename a use without saving to see a diagnostic, then repair it to clear the
diagnostic. Type a prefix inside a function to see visible-name completions.
Edits to an open imported buffer apply to its consumers before saving.

Use **dev: open language server logs** for protocol logs and **zed: open log**
for extension errors. A missing executable produces an installation message.
If `$/` imports fail, check the library root. If the executable cannot
load a native library, check the inherited development environment and restart Zed.

## Build and maintain

```sh
cargo build --manifest-path editors/zed/Cargo.toml --release --target wasm32-wasip2
cargo test -p resin --test editor_queries
```

The extension is an independent Cargo workspace depending only on
`zed_extension_api` 0.7. Its grammar pin points to a Resin commit with
`path = "crates/tree-sitter-resin"`. After a grammar change, commit the regenerated
parser in Resin and update that pin. Query tests compile every query and check
captures against representative syntax, examples, and standard-library files.
These tests use the local parser, so also verify that the pinned commit contains
the parser changes required by the queries. The grammar supports free functions and UFCS colon calls, field-only structs,
generic type parameters, explicit `::<T>` applications, checked `intrinsic`
declarations, `Err` and `None` unions, and documentation comments.

The extension is not published in Zed's registry yet. A future registry entry
can point to this repository with `path = "editors/zed"`; see the
[publishing guide](https://zed.dev/docs/extensions/publishing/publishing-guide).
Current limitations and compiler architecture are documented in
[the editor-client chapter](editor-client.md).

Parser and query tests cover declaration and control keywords, struct and alias
outlines, documentation comments, and function/class/comment text objects. The
stdio LSP suite checks imported and unsaved-buffer hover, navigation, completion,
formatting, and diagnostics without launching the editor. After changing the
adapter or grammar pin, also rebuild the dev extension in Zed and check the
workflow above; protocol tests alone do not test Zed's loading and rendering.
