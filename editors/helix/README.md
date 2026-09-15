# Resin for Helix

Language support for `.resin` files: highlighting, comments, indentation, and
function/class/comment text objects. The existing `resin --lsp` server provides
diagnostics, hover, go-to-definition, completion, and formatting.

## Install

From the Resin repository root on Linux or macOS:

```sh
nix-shell --run 'cargo install --path . --locked'
```

On Windows, build/install from the developer PowerShell described in the
[guide](../../doc/guide.md). Ensure `resin` is on the PATH used to launch Helix.

Merge [languages.toml](languages.toml) into your Helix configuration's
`languages.toml`. Use `:config-open` in Helix to locate the configuration directory:
normally `~/.config/helix` on Linux/macOS or `%AppData%\helix` on Windows.
Keep any existing language settings. A project can instead put these entries in
`.helix/languages.toml`.

The server configuration launches `resin --lsp .`. Helix starts the server in
the detected project directory; `.git` is the root marker. For a debug build,
set `command` to the absolute path of `target/debug/resin`.

LSP features work with just this configuration and the executable. To also
install syntax support:

1. Copy the three files in [runtime/queries/resin](runtime/queries/resin) into
   `<helix-config>/runtime/queries/resin/`.
2. Run `hx --grammar fetch`, then `hx --grammar build`, with a C compiler on PATH.
   On Linux/macOS, run these commands inside the repository's `nix-shell`.
   The grammar comes from a pinned commit in this repository.
3. Restart Helix and run `hx --health resin` to check the server, grammar,
   highlighting, text objects, and indentation.

Helix normally fetches/builds all configured grammars. To limit a build to Resin,
temporarily add `use-grammars = { only = ["resin"] }` at the top of
`languages.toml`, before any table headers. Remove it afterward to restore your
usual grammar selection. If you set `HELIX_RUNTIME`, ensure it points to a
writable runtime directory when fetching/building grammars.

See Helix's [language configuration](https://docs.helix-editor.com/languages.html)
and [adding languages](https://docs.helix-editor.com/guides/adding_languages.html)
guides for configuration and runtime details.

## Library imports and formatting

Launch Helix from the development shell on Linux/macOS so the server inherits
the native library paths:

```sh
nix-shell --run 'hx examples/eg009_imports.resin'
```

If the server was built in another checkout, add this table to `languages.toml`:

```toml
[language-server.resin-lsp.config]
libraryRoot = "/absolute/path/to/checkout/resin"
```

This selects the checkout's language library directory for `$/` imports.
Alternatively, set `RESIN_LIBRARY_ROOT` before launching Helix. Restart Helix
after changing its configuration or environment. Use `:lsp-restart` after
rebuilding the server.

Use `:format` for LSP formatting. To format on save, add `auto-format = true`
to the `[[language]]` entry for Resin. The formatter uses hard tabs; the supplied
indentation settings match it. See the [formatting rules](../../crates/resin-lsp/README.md#formatting).

## Check the workflow

Open `examples/eg009_imports.resin`. Use Space-k on `next` for hover and `gd` to open
its definition in `examples/lib/counter.resin`. Navigation on `print` should open
`resin/string.resin`. Change a use to an unknown name without saving and check
that a diagnostic appears; repair it to clear the diagnostic.
In insert mode, use Ctrl-x for completion. Use `:format`
to check formatting, and `maf` to select a function text object.

Use `:log-open` for server or query errors and `hx --health resin` for missing
components. Restart the editor after changing queries or rebuilding the grammar.

## Maintain

```sh
nix-shell --run 'cargo test -p resin --test editor_queries --test lsp'
```

The query suite checks both editors against representative syntax, incomplete
code, examples, and standard-library sources.

The language server and parser are shared with Zed. The Helix queries use its
own highlight names (`variable.other.member`, `constant.numeric`) and indentation
captures (`indent`, `outdent`). When the grammar changes, update both editors'
queries and grammar pins together, then fetch/build the grammar and recopy the
queries. To test an uncommitted parser, replace the grammar's `source` with
`{ path = "/absolute/path/to/checkout/crates/tree-sitter-resin" }` locally.

The integration is installed through user configuration; it is not bundled with
Helix. See the [language server documentation](../../crates/resin-lsp/README.md)
for shared features and limitations.

Validated with Helix 25.07.1 on Linux using an isolated configuration: fetched and
built the pinned grammar, checked all query files with `hx --health resin`, and
verified hover, local/standard-library navigation, completion, unsaved diagnostic
recovery, formatting, function text objects, and tab indentation in the editor.
