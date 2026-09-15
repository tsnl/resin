# Compiler service plan

Status: planned. This document records the intended architecture and implementation
sequence; it does not describe an already implemented service.

## Goal and sequence

Make compilation reusable across edits and concurrent requests, then expose the
same compiler through a persistent service for builds and editor analysis. Future
MCP tools and interactive development should use that same foundation.

1. Make compiler results immutable and reusable, with async orchestration and
   bounded parallel execution.
2. Exercise those contracts through the existing LSP mode as a long-running client.
3. Move build and analysis execution behind an HTTP server and make the CLI a client.

Each milestone should land as reviewable changes with useful behavior and tests.
The compiler libraries must remain usable and testable without starting a server.

## Organization and dependency boundaries

Keep three application-facing crates under `crates/`:

| Component | Responsibility |
| --- | --- |
| Root `resin` package | Thin executable wrapper around `resin-client`; remains the default workspace member. |
| `resin-client` | CLI dispatch, local workspace acquisition, HTTP transport, revision tracking, artifact placement, and local execution. Keep LSP and interpreter/run support together in private `lsp/` and `interp/` modules. |
| `resin-server` | HTTP application, sessions, retained revisions, scheduling, dependency acquisition, and invocation of compiler/build libraries. |
| `resin-protocol` | Versioned request/response data: logical paths, uploaded files, revision handles, targets, diagnostics, and artifact metadata. |
| Compiler/build libraries | Source snapshots, phase translations, reusable completed results, code generation, and native build execution. |

The final dependency direction is:

```text
resin -> resin-client -> resin-protocol

resin-server -> resin-protocol
resin-server -> reusable compiler/build libraries
```

Compiler/build libraries depend on none of the application crates or wire types.
The server translates compiler results into protocol responses. The protocol has
no compiler implementation or HTTP framework dependency. The client obtains build
and analysis results through the service.

Use a reusable build orchestration library (`resin-build` is the proposed name)
for sequencing passes and native builds. Keep native processes and build storage
in `resin-toolchain`, independent of compiler phase internals. Preserve the pipeline:

```text
syntax -> AST -> HIR -> LIR -> verified LIR -> C/SPIR-V -> native tools
```

**Compiler crate consolidation remains a separate decision.** Start from the
existing phase crates while changing their contracts. These three application
crates settle the application boundaries; they do not require merging compiler
phases or moving the compiler into the root CLI package. A later consolidation
would need an explicit way to preserve phase dependency rules.

Keep each crate's public contract discoverable in `lib.rs`, with cohesive private
modules for implementation. Avoid reverse dependencies and shared utility modules
that acquire compiler or application state. The current `resin-lsp` implementation
eventually moves into `resin-client`; separate LSP/interpreter client crates are
unnecessary. During milestones 1 and 2, local application code can call the reusable
libraries directly while the HTTP boundary is being prepared.

## Immutable inputs and reusable results

A workspace snapshot is an immutable mapping from unique logical paths to immutable
`Source` handles. Unchanged files share handles; changed text creates a new source
version. A source's logical identity, its version, and its content digest are
different concepts. A filename is a diagnostic/import name, not permission to read
that path from disk.

Retained compiler results should be almost entirely immutable. Private builders,
solvers, and traversal state may mutate during a computation, then return completed
data. Consumers must be able to retain old results while newer work proceeds.

### The completed result is the cache

Use the previous completed result as an optional input, with the same type as the
return value. For example, using illustrative names:

```rust
fn parse(sources: &Sources, previous: Option<&ParsedFiles>) -> ParsedFiles;
```

- `None` requests a cold computation. Supplying a previous result changes the work
  required, never the meaning or diagnostics of the current request.
- Ordinary maps containing shared immutable entries, such as `Arc` values, are
  sufficient initially. No HAMT or separate mutable compiler cache object is needed.
- Reuse an entry only when its source version and all relevant inputs match.
  An incremental reparse may use the previous version of the same logical source.
- Return only the currently requested entries and their required dependencies.
  Never carry obsolete entries forward merely to improve a possible future hit.
- Select the predecessor explicitly from the request's base revision. Do not infer
  a unique predecessor from a filesystem location when revisions can branch.

An edit list is not the complete requested set. First apply edits to the base
snapshot, then evaluate the resulting snapshot:

```text
next_sources = derive(base_sources, additions_and_edits_and_deletions)
next_result  = evaluate(next_sources, Some(base_result))
```

For `{a, b, c}` with an edit to `b`, the new request is `{a, b', c}` and can reuse
`a` and `c`. A request whose complete set is just `{b'}` retains neither `a` nor `c`.
Two computations can use the same predecessor and produce independent successors.
Avoid parent pointers that keep every historical result alive.

Start with coarse, correct reuse: per-file parsing, a completed HIR for an unchanged
resolved source graph, and LIR/verification for matching HIR, canonical target-entry
sets, and lowering options. Resolve the current import graph before deciding that
an unchanged entry file implies an unchanged program. Finer-grained semantic reuse
can follow when measurements justify it.

Workspace/session handles describe ownership and revision selection, not cache
identity. Each reuse layer must account for its relevant inputs: sources, dependencies,
libraries, compiler configuration and version, selected entries, targets, and native
toolchain inputs. Absolute checkout directories, usernames, and branch names should not
prevent reuse when they do not affect the result. Equal file contents alone do not
make distinct logical modules interchangeable: HIR identities and source origins
must remain correct. Arbitrary cross-session semantic reuse needs that identity
contract before it can be promised.

### Retention and artifacts

Applications retain completed graph roots for sessions and requests. A configurable
idle TTL and memory budget govern those roots; the compiler does not expire keys
inside an immutable result. Active requests and live session leases pin the results
they use. Expired/disconnected sessions eventually release ownership, and speculative
work must not keep abandoned sessions alive forever.

Dropping a root releases entries no other consumer retains. A later request may
recompute an evicted result without changing correctness. Persisted artifact storage
has its own retention policy, separate from workspace TTL.

Publish completed native artifacts immutably. Retaining or downloading artifact A
must not retain an exclusive mutable build-directory lock that prevents building B.
Content-based artifact reuse should account for the complete relevant native build
inputs. Global cache merging, request deduplication, and universal reuse across
unrelated sessions are later improvements; duplicate computation is acceptable first.

## Async execution, parallelism, and cancellation

Build these into the library contracts in milestone 1:

- Published source snapshots and retained results can be shared across threads
  (`Send + Sync`). Keep mutation local to the work that owns it.
- Keep CPU translations as ordinary synchronous passes where that is clearest.
  Run independent work on bounded workers; use async orchestration for file/network
  I/O, dependency acquisition, and native process output and completion.
- Share explicit execution resources rather than a mutable compiler cache. Keep
  bookkeeping locks short, with no global compiler lock across a long job or await.
- Coordinate worker limits with concurrent native builds and Ninja parallelism.
  Preserve semantic dependency ordering while parallelizing independent work.
- Pass process settings explicitly. Concurrent requests must not change the server's
  working directory or process-wide environment to configure a build.
- Support cooperative cancellation and bounded queues. Discard stale publication
  attempts; canceled incomplete work is not a completed result or a source diagnostic.
  Canceling one job must leave its predecessor and sibling results valid.

Moving CPU work to a blocking executor alone does not establish bounded parallelism
or cancellation. Choose execution and cancellation mechanisms deliberately, and
measure responsiveness as well as throughput.

## Workspace and service contracts

### Workspace acquisition

Use `resin.toml` as the project manifest and root marker. The planned selection order
is an explicit workspace directory, the nearest enclosing manifest for the selected
source, then the source's parent directory for standalone use. LSP project roots
must follow the same model. Pin down the manifest schema and input inclusion rules
before implementing upload behavior.

The client reads and uploads all user files belonging to the workspace, with paths
relative to that root and their contents. Normalize logical paths, support relative
imports within the snapshot, and reject paths that escape it. Map response paths
back to local paths or editor URIs on the client.

The server resolves user imports entirely from the supplied snapshot. Missing user
files produce diagnostics; there is no fallback to reading those paths from the
server's filesystem. The server fetches dependencies and supplies standard/builtin
libraries, so clients do not upload those trees. Dependency acquisition produces
pinned immutable inputs before compiler passes consume them.

Support both an initial full workspace POST and creation of a new immutable revision
from an explicit base revision plus additions, edits, and deletions. A server can
retain many workspaces, including multiple branches of one source tree.

### Editor revisions and disk

The LSP client tracks document versions and maps them to workspace revisions.
Open buffers override the client's disk snapshot when constructing an editor
snapshot. Edits do not automatically save files: the editor owns saving to disk.
A save updates that file's disk view while preserving other unsaved buffers.
An ordinary CLI build uses disk contents; editor requests use the selected editor
revision. Explicit handles keep these choices unambiguous.

Analysis may start eagerly after an edit, with speculative compilation of known
targets when resources permit. Coalesce rapid edits, cancel superseded work where
useful, and publish diagnostics only for the applicable document/workspace revision.
Queries must identify the revision they describe. Retained old snapshots remain
usable by requests already consuming them.

### Connection, build, and output

After milestone 3, build, run, and LSP requests always use the server URL in
`RESIN_SERVER`. Missing/invalid configuration, connection failure, or an incompatible
protocol produces an early, actionable error. No automatic server spawning, local
compilation fallback, workspace address files, or filesystem-based discovery.

Use a versioned protocol with an explicit build target contract, including relevant
OS, architecture, ABI/runtime requirements, entries, profile, and build options.
The server reports supported targets and rejects unsupported requests. A server
running in a Linux container does not implicitly provide macOS or Windows builds.

The server runs code generation, Ninja, compilers, and other required native tools.
Return structured diagnostics and artifact metadata, with a single output file
transferred over HTTP initially. The client writes a completed download to the
requested output location or executes it locally. Preserve the existing defaults:
debug build-and-run without `-o`, optimized output without execution with `-o`.
Run arguments, the runtime environment, working directory, and standard streams
belong to local execution and stay out of compilation/cache inputs. Build settings
are captured separately and sent through the explicit build contract.

Use one `resin-server` application in every environment. For development, launch it
locally and set `RESIN_SERVER`. Users can supervise it with Docker or systemd,
including user or service-account deployments. Initial setup is manual; an installer
and deployment packaging are outside the first implementation. Rocket or another
HTTP framework is an implementation choice. No runtime directory is needed to
discover a configured HTTP endpoint.

## Milestones and acceptance criteria

### 1. Immutable, reusable, concurrent compiler libraries

- [ ] Introduce complete immutable source snapshots and optional-prior-result APIs.
- [ ] Establish correct parse/HIR/LIR reuse and immutable native artifact publication.
- [ ] Add reusable build orchestration, async I/O/process handling, bounded workers,
      and cooperative cancellation.
- [ ] Test cold versus reused results, edits/deletions/import changes, option and
      dependency invalidation, retained old results, and forks from one predecessor.
- [ ] Test simultaneous requests, cancellation, deterministic diagnostics, and
      holding artifact A while building B, all without HTTP.
- [ ] Measure cold builds, unchanged builds, small edits, retained memory, and
      cancellation latency to establish a baseline.

### 2. Exercise the model through LSP

- [ ] Define `resin.toml` roots and local snapshot acquisition; combine disk changes
      and editor buffers into explicit revisions.
- [ ] Replace serial blocking analysis orchestration with bounded background work,
      revision-aware queries, cancellation, and stale-result filtering.
- [ ] Exercise eager analysis through LSP; do not add a separate watch-mode CLI.
- [ ] Test rapid edits, invalid code followed by repair, dependency changes,
      overlapping queries/builds, save behavior, cancellation, and shutdown.
- [ ] Check response latency, reuse, and memory retention during sustained editing.

This milestone proves that parallel computation and non-blocking protocol handling
work together before introducing a network boundary.

### 3. HTTP server and client

- [ ] Add `resin-client`, `resin-server`, and `resin-protocol`; reduce the root CLI
      to its wrapper and move LSP/run application code into the merged client.
- [ ] Implement strict `RESIN_SERVER` connection behavior, protocol/target negotiation,
      full workspace upload, and derivation from explicit base revisions.
- [ ] Integrate server-owned dependency/library acquisition and native builds.
- [ ] Add single-file artifact delivery, local output placement/execution, and
      configurable workspace retention with active-request/session ownership.
- [ ] Test compilation when client paths do not exist on the server; equivalent
      relative trees from different checkout roots; isolated workspaces/branches;
      target mismatch; output download; and LSP revision behavior over HTTP.
- [ ] Define expired handles and reconnect behavior: after a server restart or
      eviction, clients must be able to upload/resynchronize a snapshot.
- [ ] Document manual server startup, configuration, and required native tools.

## Details to settle during implementation

- Exact manifest schema, upload inclusion/exclusion rules, and dependency pinning.
- Protocol endpoint/schema details, version compatibility, target support, error
  responses, artifact metadata, and revision/session lease behavior.
- Executor, HTTP framework, cancellation granularity, and resource defaults.
- The native build helper arrangement: generated builds currently invoke `--embed`;
  a separate server executable must use an explicitly compatible helper.
- Authentication, workspace access, and execution isolation before deploying a
  shared service to mutually untrusted users.
- Whether compiler phase crates should eventually consolidate, and how source/IR
  identity permits broader reuse across unrelated sessions.

## Later extensions

MCP queries for documentation, diagnostics, symbols, and compiler facts; REPL/shell
sessions; additional watching clients; tar/tgz or multiple-file output bundles;
content-addressed uploads; finer-grained incremental analysis; shared cluster caches;
and installation tooling can build on these contracts. They are follow-on work,
not prerequisites for completing the first three milestones. Interactive runtime
state will need its own session lifecycle alongside immutable compilation results.
