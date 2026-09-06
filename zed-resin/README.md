# Resin for Zed

Language support for `.resin` files: highlighting, comments, brackets,
indentation, outlines, and function/comment text objects. The native `resin-lsp`
adds diagnostics, hover, go-to-definition, and basic completion.

Both this extension and the grammar are ordinary folders in the
[Resin repository](https://github.com/tsnl/resin). No separate repository or
submodule checkout is required.

## Install for development

From the Resin repository root:

```sh
nix-shell --run 'cargo install --path resin-lsp --locked'
nix-shell --run 'rustup target add wasm32-wasip2'
```

The repository's `rust-toolchain.toml` declares the `wasm32-wasip2` target,
so Rustup installs it when activating the project toolchain. If Zed reports
`can't find crate for core`, run the target-install command above from this
repository root, then retry installing the dev extension. Rust targets are
installed per toolchain; adding the target outside the repository can select
a different Rust version.

Launch Zed from `nix-shell` so it inherits the native library paths, then run
**zed: install dev extension** in the command palette and select `zed-resin/`.
On systems whose executable is named `zeditor`, use `nix-shell --run 'zeditor .'`.
Zed compiles the extension and downloads the WASI SDK to build the grammar.
Rebuild it from Zed's Extensions view after changing the adapter or queries.
See [Zed's extension development guide](https://zed.dev/docs/extensions/developing-extensions).

The adapter uses a configured binary or finds `resin-lsp` on the worktree's PATH.
For a local debug build, run `nix-shell --run 'cargo build -p resin-lsp'` and set
an absolute path in Zed settings:

```json
{
  "lsp": {
    "resin-lsp": {
      "binary": {
        "path": "/absolute/path/to/resin/target/debug/resin-lsp",
        "arguments": ["--stdio"]
      },
      "initialization_options": {
        "stdlibPath": "/absolute/path/to/resin/stdlib"
      }
    }
  }
}
```

The standard-library override is useful when switching worktrees or using an
installed server built in another checkout. Alternatively, set `RESIN_STDLIB` in
the environment. Configured `binary.env` entries override the inherited worktree
environment. Restart the language server after changing its configuration or
rebuilding the native executable.

For syntax support alone, Zed's language setting
`"languages": { "Resin": { "enable_language_server": false } }` disables the LSP.

## Check the workflow

Open `examples/eg009_imports.resin`. Hover `next`, navigate to its definition in
`examples/lib/counter.resin`, and navigate `check` into `stdlib/status.resin`.
Rename a use without saving to see a diagnostic, then repair it to clear the
diagnostic. Type a prefix inside a function to see visible-name completions.
Edits to an open imported buffer apply to its consumers before saving.

Use **dev: open language server logs** for protocol logs and **zed: open log**
for extension errors. A missing executable produces an installation message.
If `std/` imports fail, check the standard-library path. If the executable cannot
load a native library, restart Zed from the repository's `nix-shell`.

## Build and maintain

```sh
nix-shell --run 'cargo build --manifest-path zed-resin/Cargo.toml --release --target wasm32-wasip2'
nix-shell --run 'cargo test -p resin --test zed_queries'
```

The extension is an independent Cargo workspace depending only on
`zed_extension_api` 0.7. Its grammar pin points to a Resin commit with
`path = "tree-sitter-resin"`. After a grammar change, commit the regenerated
parser in Resin and update that pin. Query tests compile every query and check
captures against representative syntax, examples, and standard-library files.

The extension is not published in Zed's registry yet. A future registry entry
can point to this repository with `path = "zed-resin"`; see the
[publishing guide](https://zed.dev/docs/extensions/publishing/publishing-guide).
Current limitations and compiler architecture are documented in
[resin-lsp/README.md](../resin-lsp/README.md).

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
local and return annotations. Rebuild `resin-lsp` and reinstall the dev extension
after updating. Parser, query, and stdio regressions cover the new syntax and
inferred hover types without launching an editor.

The grammar revision at `847bc7a` added nominal `struct` declarations, transparent `type`
aliases, `A | B` unions, `Result<T, E>`, postfix `?`, and exhaustive `match` arms.
Queries highlight the new syntax and include structs in the outline; semantic tests
cover inferred error sets and match-payload navigation/completion. Rebuild the language
server and reinstall the dev extension together. No new editor smoke test was run.

The grammar revision at `bba45b9` added scoped `defer { ... };` statements. Parser and query tests
cover the keyword and block syntax; compiler analysis tests cover deferred bindings,
inferred local types, and navigation. No window or editor was opened for this change.

`defer expression;` now accepts any expression and discards its value, while remaining
statement-only. Chain blocks may omit their unit tail. Parser, compiler, query, and
recovery tests cover both forms without launching an editor.
