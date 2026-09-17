# Resin for Helix

Language support for `.resin` files: highlighting, comments, indentation, and
function/class/comment text objects. The existing `resin --lsp` server provides
diagnostics, hover, go-to-definition, completion, and formatting.

## Install

Install Helix (`hx`) and the [development tools](getting-started.md#setup),
including a C compiler for grammar builds. Nix users can enter `nix-shell`, which
also provides Helix. From the Resin repository root:

```sh
cargo install --path . --locked
```

On Windows, build/install from the developer PowerShell described in the
[guide](getting-started.md). Ensure `resin` is on the PATH used to launch Helix.

Merge [languages.toml](../editors/helix/languages.toml) into your Helix configuration's
`languages.toml`. Use `:config-open` in Helix to locate the configuration directory:
normally `~/.config/helix` on Linux/macOS or `%AppData%\helix` on Windows.
Keep any existing language settings. A project can instead put these entries in
`.helix/languages.toml`.

The server configuration launches `resin --lsp .`. Helix starts the server in
the detected project directory; `.git` is the root marker. For a debug build,
set `command` to the absolute path of `target/debug/resin`.

Start a [compiler service](compiler-service.md) and export
`RESIN_SERVER=http://127.0.0.1:7412` before launching Helix. LSP initialization
requires successful capability negotiation. To also install syntax support:

1. Copy the three files in [runtime/queries/resin](../editors/helix/runtime/queries/resin) into
   `<helix-config>/runtime/queries/resin/`.
2. Run `hx --grammar fetch`, then `hx --grammar build`, with a C compiler on PATH.
   Use the environment configured during setup.
   The grammar comes from this repository's `main` branch.
3. Restart Helix and run `hx --health resin` to check the server, grammar,
   highlighting, text objects, and indentation.

Helix normally fetches/builds all configured grammars. To limit a build to Resin,
temporarily add `use-grammars = { only = ["resin"] }` at the top of
`languages.toml`, before any table headers. Remove it afterward to restore your
usual grammar selection. Compiled grammars are written to
`<helix-config>/runtime/grammars/`.

Helix requires `rev` for a Git grammar source; `rev = "main"` selects the branch
without pinning a commit. Run `hx --grammar fetch` and `hx --grammar build` to
update the installed parser, then restart Helix. Starting the editor does not
fetch updates. Update the copied queries from `main` at the same time.

Helix 25.07.1 can fetch a new `main` commit while leaving its cached checkout on
the previous one. With that version, run this after fetching and before building
to select the commit just downloaded (replace `<helix-config>` with your actual
configuration directory):

```sh
git -C "<helix-config>/runtime/grammars/sources/resin" checkout --detach FETCH_HEAD
```

See Helix's [language configuration](https://docs.helix-editor.com/languages.html)
and [adding languages](https://docs.helix-editor.com/guides/adding_languages.html)
guides for configuration and runtime details.

## Develop from a checkout

Helix supports a [local grammar source](https://docs.helix-editor.com/guides/adding_languages.html#grammar-configuration).
Use it while developing Resin so uncommitted parser changes are available to the
editor. In [languages.toml](../editors/helix/languages.toml), comment out the Git `source` line and
uncomment the local `source = { path = ... }` line immediately below it.

In your Helix `languages.toml`, keep Resin's `[[language]]` entry and replace the
existing `resin-lsp` server and `resin` grammar entries with these local settings.
Replace `/absolute/path/to/checkout` with the checkout you are developing:

```toml
[language-server.resin-lsp]
command = "/absolute/path/to/checkout/target/debug/resin"
args = ["--lsp", "."]
environment = { RESIN_SERVER = "http://127.0.0.1:7412" }

[[grammar]]
name = "resin"
source = { path = "/absolute/path/to/checkout/crates/tree-sitter-resin" }
```

Replace the entire grammar `source` value, including its `git`, `rev`, and
`subpath` fields. These settings belong in your local configuration; the shared
configuration follows `main`. If you use `CARGO_TARGET_DIR`, point `command` at
that directory's debug executable.
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

From the checkout root in your configured development environment:

```sh
cargo build -p resin
hx --grammar build
hx --health resin
hx examples/eg009_imports.resin
```

The grammar build reads the local generated parser; no grammar fetch or commit
is needed. The `use-grammars` filter described above can restrict the
build to Resin. On Windows, run the Cargo and Helix commands in the developer
PowerShell described in the [guide](getting-started.md).

| Changed | Development loop |
| --- | --- |
| Compiler or LSP Rust code | `cargo build -p resin`, then `:lsp-restart` in Helix. |
| `grammar.js` | [Regenerate the parser](parser.md), run `cargo build -p resin` and `hx --grammar build`, then restart Helix. |
| Query `.scm` files | Restart Helix to reload the linked queries. |

When switching worktrees, update the three configured paths and the query link,
remove the compiled parser as above, then build the server and grammar and
restart Helix. This keeps the server, standard library, parser, and queries on
the same checkout. A second
`hx --grammar build` without parser changes reuses the compiled grammar.

## Library imports and formatting

Launch Helix from the configured development environment so the client inherits
any required native library paths:

```sh
hx examples/eg009_imports.resin
```

Standard-library and pinned-package selection belongs to the HTTP service.
Configure its `--library-root` / `RESIN_LIBRARY_ROOT`, then restart the service to
freeze that snapshot. The editor no longer accepts `libraryRoot`.

For local foreign header directories, optionally add:

```toml
[language-server.resin-lsp.config]
includeRoots = ["include", "vendor/include"]
```

Helix passes `config` as initialization options; `environment` sets the launched
process environment. See [Helix language-server configuration](https://docs.helix-editor.com/languages.html#language-server-configuration).
Relative roots resolve against the selected editor project directory. Restart Helix
after changing its configuration or environment. Use `:lsp-restart` after rebuilding
the local client. `RESIN_SERVER` must remain explicitly configured.

Use `:format` for LSP formatting. To format on save, add `auto-format = true`
to the `[[language]]` entry for Resin. The formatter uses hard tabs; the supplied
indentation settings match it. See the [formatting rules](editor-client.md#formatting).

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
cargo test -p resin --test editor_queries --test lsp
```

The query suite checks both editors against representative syntax, incomplete
code, examples, and standard-library sources.

The language server and parser are shared with Zed. The Helix queries use its
own highlight names (`variable.other.member`, `constant.numeric`) and indentation
captures (`indent`, `outdent`). Helix's Git source follows `main`; update its
parser and queries together when installing syntax changes. Zed retains its own
grammar pin, which must contain the parser required by its queries.
Compiler/LSP-only changes do not require a grammar update.
During development, use the
[local checkout workflow](#develop-from-a-checkout) above.

The integration is installed through user configuration; it is not bundled with
Helix. See the [language server documentation](editor-client.md)
for shared features and limitations.

Validated with Helix 25.07.1 on Linux using an isolated configuration: fetched and
built the pinned grammar, checked all query files with `hx --health resin`, and
verified hover, local/standard-library navigation, completion, unsaved diagnostic
recovery, formatting, function text objects, and tab indentation in the editor.
Both the `main` source and the commented local-path alternative build and pass
`hx --health resin`. The local configuration builds without fetching a grammar
and loads queries through a symlink to the checkout.
