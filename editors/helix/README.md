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
usual grammar selection. Compiled grammars are written to
`<helix-config>/runtime/grammars/`.

See Helix's [language configuration](https://docs.helix-editor.com/languages.html)
and [adding languages](https://docs.helix-editor.com/guides/adding_languages.html)
guides for configuration and runtime details.

## Develop from a checkout

Helix supports a [local grammar source](https://docs.helix-editor.com/guides/adding_languages.html#grammar-configuration).
Use it while developing Resin so uncommitted parser changes are available to the
editor. The checked-in `rev` is an installation snapshot of the parser; it does
not pin the language server and does not need updating for compiler/LSP changes.

In your Helix `languages.toml`, keep Resin's `[[language]]` entry and replace the
existing `resin-lsp` server and `resin` grammar entries with these local settings.
Replace `/absolute/path/to/checkout` with the checkout you are developing:

```toml
[language-server.resin-lsp]
command = "/absolute/path/to/checkout/target/debug/resin"
args = ["--lsp", "."]
config = { libraryRoot = "/absolute/path/to/checkout/resin" }

[[grammar]]
name = "resin"
source = { path = "/absolute/path/to/checkout/crates/tree-sitter-resin" }
```

Replace the entire grammar `source` value, including its `git`, `rev`, and
`subpath` fields. These settings belong in your local configuration; keep the
installable snapshot in this repository's `languages.toml`. If you use
`CARGO_TARGET_DIR`, point `command` at that directory's debug executable.
On Windows, use `resin.exe` and forward slashes in the TOML paths.

For query edits, replace your installed `<helix-config>/runtime/queries/resin`
copy with a symlink to `<checkout>/editors/helix/runtime/queries/resin` (or a
directory junction on Windows). Preserve any customized queries before replacing
the directory. Helix will read query changes directly from the checkout when it
starts. Setting `HELIX_RUNTIME` alone does not override queries already present
in the user configuration's runtime directory.

Create `<helix-config>/runtime/grammars/` if it does not exist; Helix writes the
compiled parser there. After changing the grammar source path, close Helix and
remove that directory's `resin.so` (`resin.dll` on Windows) before rebuilding.
Helix checks source timestamps, so an older checkout can otherwise reuse the
previous checkout's compiled parser.

From the checkout root on Linux/macOS:

```sh
nix-shell
cargo build -p resin
hx --grammar build
hx --health resin
hx examples/eg009_imports.resin
```

The grammar build reads the local generated parser; no grammar fetch, commit, or
pin update is needed. The `use-grammars` filter described above can restrict the
build to Resin. On Windows, run the Cargo and Helix commands in the developer
PowerShell described in the [guide](../../doc/guide.md).

| Changed | Development loop |
| --- | --- |
| Compiler or LSP Rust code | `cargo build -p resin`, then `:lsp-restart` in Helix. |
| `grammar.js` | [Regenerate the parser](../../crates/tree-sitter-resin/README.md), run `cargo build -p resin` and `hx --grammar build`, then restart Helix. |
| Query `.scm` files | Restart Helix to reload the linked queries. |

When switching worktrees, update the three configured paths and the query link,
remove the compiled parser as above, then build the server and grammar and
restart Helix. This keeps the server, standard library, parser, and queries on
the same checkout. A second
`hx --grammar build` without parser changes reuses the compiled grammar.

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
captures (`indent`, `outdent`). When publishing updated syntax support, point both
editors' grammar pins at a commit containing the parser required by their queries.
For installations using the pinned parser, fetch/build the grammar and recopy the
queries together. Compiler/LSP-only changes do not require a grammar pin update.
During development, use the
[local checkout workflow](#develop-from-a-checkout) above.

The integration is installed through user configuration; it is not bundled with
Helix. See the [language server documentation](../../crates/resin-lsp/README.md)
for shared features and limitations.

Validated with Helix 25.07.1 on Linux using an isolated configuration: fetched and
built the pinned grammar, checked all query files with `hx --health resin`, and
verified hover, local/standard-library navigation, completion, unsaved diagnostic
recovery, formatting, function text objects, and tab indentation in the editor.
The local development configuration also builds without fetching a grammar and
loads queries through a symlink to the checkout.
