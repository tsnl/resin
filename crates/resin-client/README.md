# Resin client and editor support

`resin` combines the command-line client, local formatter, and stdio LSP. Semantic
analysis and native compilation run on the explicit HTTP(S) service selected by
`RESIN_SERVER`. It downloads completed executables and runs them locally when no
`-o` is given. There is no automatic service startup or local compiler fallback.
The public async `Client` API and wire transport live in this crate; compiler phase
libraries remain independent of the client.

## Build and run

From the repository development shell:

```sh
cargo build -p resin -p resin-server -p resin-runtime
./target/debug/resin-server --listen 127.0.0.1:7412
```

In a second development shell:

```sh
export RESIN_SERVER=http://127.0.0.1:7412
cargo run -- examples/eg001.resin
cargo run -- --lsp /path/to/project
```
Configure the editor to launch `resin --lsp /path/to/project` with `RESIN_SERVER`
in its environment. Capability negotiation must succeed before LSP initialization.
The stdio client reserves stdout for protocol messages and stderr for logs.
See [Zed](../../editors/zed/README.md), [Helix](../../editors/helix/README.md), and
the [service guide](../../doc/compiler-service.md) for complete setup.

`$/` imports select the service's frozen standard library and pinned packages.
`RESIN_LIBRARY_ROOT` and `--library-root` configure the service; editor
`libraryRoot` is not supported. For local C header directories, use repeated
`-I DIR` / `--include-root DIR` flags or initialization options
`{ "includeRoots": ["include", "vendor/include"] }`. Relative initialization roots
resolve against the selected editor project directory. Complete selected header
directories are uploaded, so keep build outputs and changing logs elsewhere.

## Formatting

The client advertises `documentFormattingProvider` and handles
`textDocument/formatting`. Use your editor's **Format Document** command or enable
format-on-save. The editor synchronizes the buffer, requests formatting, and
applies the returned text edit through its normal undo/save workflow. The client
does not write the file. Rebuild/reinstall the client and restart it in your editor
to pick up formatting support; see the [Zed setup](../../editors/zed/README.md#formatting)
or [Helix setup](../../editors/helix/README.md#library-imports-and-formatting).

Formatting uses `resin_cst::format_source` on the latest accepted open
buffer, independently of background semantic analysis. Unresolved names, imports,
and type errors do not prevent formatting. Syntax errors return no edits, as does
formatting a document that is not open. An already formatted buffer returns an
empty edit list. A changed buffer returns one replacement trimmed to exclude its
unchanged prefix/suffix, with UTF-16 ranges that respect Unicode and CRLF boundaries.

Resin has a fixed source style:

- One hard tab per indentation level, regardless of LSP `tabSize`/`insertSpaces`.
  Your editor controls how wide tabs appear.
- Trailing commas are preserved and force one item per line in parameter lists,
  calls, tuples, arrays, records, struct fields, imports, and exports, except singleton
  tuples such as `(x,)` and `(int,)`. Singleton tuples and lists without trailing commas
  collapse unless comments or nested multiline constructs require breaks.
- Nonempty blocks and match arm lists use multiple lines. Operators, declarations, and separators
  receive consistent spacing; no automatic line-length wrapping is performed.
- Runs of blank lines collapse to at most one between items. Padding inside
  delimiters and at file boundaries is removed. Nonempty output ends with one LF;
  empty/whitespace-only input becomes empty. Whitespace outside comments and
  literals is normalized to tabs, spaces, and LF.
- Comments and literal spellings are preserved, including whitespace within block
  comments. Same-line trailing comments stay attached to their preceding token.

For example, `let mut xs = [1,2,3,];` becomes:

```resin
var xs = [
	1,
	2,
	3,
];
```

The CLI shares this formatter: `resin --format examples` formats files recursively,
and `resin --format --check examples` checks them without writing. See the
[CLI formatting guide](../../doc/guide.md#formatting) for exit codes and file selection.
Only whole-document LSP formatting is supported; range/on-type formatting is not
implemented.

## Captured buffers and shared analysis

The client owns open buffers, versions, URI/path mappings, and document epochs.
Every analysis captures the current user import graph again. Request-owned loaders
are seeded only by current supplied registrations and discard learned disk history
after capture. An open registration preserves its physical identity; close/reopen
normalizes the path again. Saves retain other unsaved buffers. Identical open aliases
share one registration until the final alias closes; differing text for the same
physical file is rejected with both URI names.

Uploaded names are entry-parent-relative logical identities. Local absolute paths,
URIs, requests, revisions, and presentation mappings do not enter compiler keys.
The server owns Source/CST/AST/HIR and downstream cache heads shared by all callers.
An editor analysis and an independent CLI build of the same saved bytes therefore
reuse completed compiler results. A different buffer version or dependency produces
a distinct graph. Input handles can select deltas; unavailable handles retry once
with retained full inputs. Managed snapshot changes refresh capabilities and recapture.

Analysis needs no exported runnable entry, native C build, shader optimizer, or GPU.
Incomplete declarations retain useful editor facts where possible. Hovers show
resolved types, including inferred results, nominal error unions, and match-arm
bindings. Managed definitions are materialized as immutable read-only local files
for ordinary file navigation; their server identities stay in the `$/` namespace.
The LSP retains those mirror files and their source text until the editor session
ends, including mirrors from earlier managed snapshots. This is a client navigation
limitation: published file URIs can be opened later, and the protocol provides no
notification that those links have been discarded. Compiler cache eviction does not
release these mirrors.

Formatting, acquisition, line indexing, and response preparation run off the stdio
receiver. Semantic work uses bounded HTTP requests. Superseded work sends explicit
cancellation, including retries if DELETE arrives before POST admission. Server
cancellation drains owned native processes; a synchronous parser already running
may finish before it releases its worker slot. Current results retain their own
captured sources, independently of later cache eviction.

## Build from the editor

The LSP advertises `resin.build` through `workspace/executeCommand`. Supply one
object argument:

```json
{
  "command": "resin.build",
  "arguments": [{
    "uri": "file:///path/to/main.resin",
    "entry": "main",
    "destination": "/path/to/program",
    "profile": "release"
  }]
}
```

`entry` defaults to `main`; `profile` accepts `debug` or `release` and defaults to
`release`. `destination` is required and names a file; a relative destination
resolves against the entry file's directory. A successful result contains its
`outputUri`. The client captures disk sources and uploads them to the selected service. The service
shares completed phase results with editor analysis and runs native tools; the client
verifies the downloaded artifact and atomically publishes the executable. It does
not save editor buffers or run the output. Source-overwriting destinations are
rejected, and cancellation propagates to native children. Native tools and build caches belong to the service. The destination and local
execution environment are never uploaded.

## Protocol behavior and limits

- Local `file://` documents, including new files and unsaved imported buffers.
- Full-document synchronization and UTF-16 positions, including CRLF and non-BMP
  characters. Older document versions are ignored.
- Each open file is an analysis entry. Diagnostics from its dependencies are
  published at the actual source URI and aggregated across entries.
- Syntax and semantic diagnostics from the compiler's normal frontend. HIR checking
  traverses expression/type holes and missing-field nodes, preserving facts from
  healthy statements, functions, and imported buffers. Unknown bindings appear as
  `?`; incomplete programs retain editor facts without publishing executable IR.
- Navigation includes locals, parameters, nominal types, explicit exports,
  re-exports, standard-library names, and import strings.
- Completion includes visible names, keywords, builtin types, and intrinsics,
  with identifier replacement ranges. Typing `.` offers fields from the receiver's
  record type, including nominal records, pointers, and nested access. Field
  suggestions use the cached recovered AST and semantic analysis without inserting
  synthetic identifiers. Unrecoverable declarations, unresolved imports, and
  unknown receiver types can still prevent suggestions. Automatic imports are
  not implemented.
- The editor is asked to watch workspace files and configured header include roots
  if it supports dynamic registration. Arbitrary header filenames, including
  extensionless files, must invalidate captured inputs when created, changed, or deleted.
  Configure an include root for headers outside the workspace so their directories
  receive file notifications too.
  Without file notifications, external changes to closed dependencies are
  discovered when a later edit or save schedules analysis.
- Complete accepted editor states are coalesced; each open/reopen has its own
  epoch. Queries and builds are admitted with a 64-request bound that includes
  queued, running, and completed-but-unconsumed responses. Overload returns an
  explicit error. Cancellation applies to each admitted request independently.
- Formatting responses check the captured document version. Semantic responses
  also check their dependency state. Obsolete results return `ContentModified` and
  cannot replace current diagnostics. Shutdown cancels and drains owned work.
- The stdio transport applies backpressure when its client stops reading; compiler
  concurrency does not remove that transport constraint.

## Validation

```sh
nix-shell --run 'cargo test -p resin-client -p resin-protocol -p resin-server'
nix-shell --run 'cargo test -p resin --test lsp --test formatting --test format_cli'
```

Tests cover source/header capture, relocation and aliases, HTTP negotiation and
cancellation races, strict artifact validation, stdio lifecycle and freshness, and
canonical formatting. Integration fixtures start their own explicitly configured
service; production clients always require an operator-provided URL.
