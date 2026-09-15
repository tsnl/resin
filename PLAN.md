# Compiler service plan

Status: planned. Implement the three phases below in order. Each phase has its own
acceptance criteria; finish those criteria before moving to the next phase.

## Agreed direction

- Keep compiler inputs and completed outputs immutable and independently reusable.
- Let applications explicitly invoke each compiler pass and retain its outputs.
  Build handlers and LSP query handlers may duplicate straightforward sequencing.
  Do not introduce `resin-build`, a generic compilation driver, or a reusable
  orchestration/cache framework that hides those operations.
- Use one shared set of caches per process, keyed by the actual inputs to each pass.
  Results from any caller can contribute to another caller's compilation.
- Start each build or analysis from one source entry point and its resolved imports.
  There are no persistent compiler workspaces, workspace cache partitions, workspace
  leases, or `resin.toml`. A build also selects its exported function/target entries;
  analyzing a source for LSP does not require it to define a runnable `main`.
- Retain cache entries for a configurable **24 hours since last use**, sweeping
  expired entries periodically. Start with a configurable **one-minute sweep**.
- First establish the library contracts, then exercise them locally through LSP,
  then introduce the HTTP boundary. MCP and REPL support follow later.

### Final repository layout

The `resin` package stays in the repository root. All other members of Resin's Cargo
workspace live under `crates/`; the root remains the default member. The independent
editor extension workspace keeps its existing layout.

```text
Cargo.toml                       # root resin package and Cargo workspace
src/main.rs                      # thin wrapper around resin-client
crates/
  resin-client/src/               # CLI, HTTP, local files and execution
    lib.rs
    lsp/                         # private editor client code
    interp/                      # private run/interactive client code
  resin-server/src/               # HTTP handlers, explicit passes and cache tables
  resin-protocol/src/             # versioned wire data
  resin-source/src/               # immutable sources and import data
  resin-cst/src/
  resin-ast/src/
  resin-hir/src/
  resin-lir/src/
  resin-codegen/src/
  resin-toolchain/src/            # native tools and artifact ownership
  ...                            # other existing compiler/support crates
```

Final dependencies:

```text
root resin -> resin-client -> resin-protocol
resin-server -> resin-protocol
resin-server -> individual compiler crates and resin-toolchain
```

Compiler and toolchain crates depend on none of the client, server, or protocol
crates. `resin-protocol` contains wire data, without compiler IR or HTTP framework
dependencies. Keep one merged `resin-client`; its LSP adapter forwards analysis
requests to server handlers that know the compiler passes explicitly.

Preserve the existing compiler phase crates for this plan. Consolidating them is
separate follow-up work. Keep public contracts in `lib.rs`, substantial implementation
private, and dependencies in the existing direction:

```text
syntax -> AST -> HIR -> LIR -> verified LIR -> C/SPIR-V -> native tools
```

## Phase 1: Immutable, shareable compiler passes

### Deliverable

Compiler passes accept complete immutable inputs and return completed results that
can be retained, reused, and shared across threads. The current local applications
invoke these passes explicitly. This phase includes async I/O and parallel execution
support; no HTTP service is required yet.

### 1.1 Explicit pass contracts

- Expose the inputs, outputs, and diagnostics of each phase. An application must be
  able to reuse parsing without asking a HIR convenience method to parse again.
  Split any existing API that conceals sequencing needed by the application.
- Pass a completed source/import graph into semantic compilation. Source acquisition
  and dependency fetching happen outside CPU translations. Compiler passes never
  fall back to opening user source paths on the host filesystem.
- Keep private builders, inference solvers, and traversal state mutable only for
  the duration of their work. Published language data and retained results are
  immutable and `Send + Sync`; old versions remain usable during new computations.
- Preserve optional incremental predecessors where useful. The prior value has
  the same type as the return value, for example, with illustrative names:

  ```rust
  fn parse(source: &Source, previous: Option<&ParsedFile>) -> ParsedFile;
  ```

  `None` performs a cold computation. The predecessor is an optimization hint;
  choosing a different predecessor must not change meaning or diagnostics.
- A returned result contains only data needed for its current inputs. It does not
  accumulate unused historical entries or retain a chain of previous results.
  Ordinary maps and shared immutable values suffice; a HAMT is unnecessary.

### 1.2 Keys and source identity

Make reuse depend on complete inputs, independently of who requested them. Define
explicit key data beside each relevant phase's public input contract. Do not use
session IDs, absolute checkout roots, timestamps, or allocation addresses as semantic
cache keys. Each phase needs only inputs that can affect that phase:

| Cached value | Relevant key inputs |
| --- | --- |
| Source version | Explicit logical module identity/role and exact contents, including library/dependency namespace and any diagnostic name retained in the value. |
| CST / per-file AST | Source version and relevant parser/lowering configuration. |
| HIR and editor facts | Complete resolved source graph, import bindings, entry source, library/dependency versions, and semantic options. |
| LIR / verified LIR | HIR identity, canonical selected entry set, target profile, lowering limits/options, and verification contract. |
| Generated files / native artifact | Verified LIR, code generation options, target/ABI, compiler/runtime/library inputs, native toolchain inputs, and build settings. |

Use canonical, content-based identities for reuse across equivalent requests.
Persisted keys additionally include the relevant compiler/toolchain version. Check
the actual key data when interning; a digest must not silently equate different
inputs. Runtime arguments and runtime environment are not build inputs.

The current `Source` uses allocation-based version equality and process-local IDs.
Establish value-based source/version identity that remains correct after eviction
and reconstruction. Reusing a cached HIR with newly acquired equivalent sources must
preserve query lookup, spans, and declaration identity. Two distinct logical modules
with equal text must remain distinct within a program. A display name alone does
not define module identity; distinct modules may have identical diagnostic names.

Use logical source names in retained results. Applications map those names back to
their own local paths/URIs. Equivalent relative source graphs in different checkout
directories must be able to share the same completed result without leaking one
client's absolute paths into another client's diagnostics.

Start with per-file CST/AST reuse and whole resolved-graph HIR reuse. An unchanged
entry file with changed imports is a different HIR input. Arbitrary sharing of
sub-HIR fragments or incremental inference is not required in this plan.

### 1.3 Async work and native artifacts

- Keep CPU passes synchronous internally where ordinary control flow is clearest.
  Existing application code schedules independent work on bounded workers and uses
  async I/O/process operations. Do not create a shared orchestration library.
- Bound queues and CPU jobs, coordinate with Ninja/native-tool parallelism, and
  preserve true data dependencies. Do not put the whole compiler behind one lock.
- Add cooperative cancellation at useful pass boundaries/checkpoints and explicit
  native-child cancellation/reaping. Moving work to a blocking thread alone does
  not make it cancellable. Completed source errors and cancellation are distinct.
- Pass process settings explicitly; concurrent jobs never configure a build by
  changing process-wide environment or the application's working directory.
- Give `resin-toolchain` explicit async native-process and artifact operations.
  It continues to consume generated native projects and build settings, without
  sequencing compiler passes.
- Publish completed artifacts immutably. Retaining/downloading A must not hold an
  exclusive mutable staging-directory lock needed to build B. Artifact ownership
  must keep its files alive until all consumers finish, independently of cache rows.

### Phase 1 acceptance criteria

- [ ] **P1.1** Direct library tests run the explicit phase sequence without HTTP or
      an orchestration wrapper; CLI build/run behavior still works.
- [ ] **P1.2** Cold and incremental results agree for unchanged input, edits,
      deletions, changed imports, library changes, and relevant option changes.
- [ ] **P1.3** Completed inputs/results satisfy `Send + Sync`; concurrent work from
      the same predecessor preserves that predecessor and both successors.
- [ ] **P1.4** Independently constructed equivalent source graphs have compatible
      keys and query identities; changed import bindings and distinct equal-text
      modules cannot collide. Reconstructed sources work with retained HIR results.
- [ ] **P1.5** Tests exercise bounded parallel jobs, responsive async I/O, cancellation,
      and child cleanup. Cancellation cannot publish an incomplete successful result.
- [ ] **P1.6** Artifact A remains readable while B builds; concurrent publication and
      final-owner cleanup cannot overwrite/delete another live artifact generation.
- [ ] **P1.7** Record cold/unchanged/small-edit timing, concurrency, retained memory,
      and cancellation latency as a baseline, without imposing invented speed targets.

## Phase 2: Shared caches and retention exercised through LSP

### Deliverable

Use the existing LSP mode to exercise repeated and overlapping computations locally.
Implement explicit shared pass tables and time-based retention in application code.
There is no new watch-mode executable or reusable cache/orchestration crate.
Local application tests schedule analysis and builds against those tables. Separate
CLI and LSP processes begin sharing a running service only in Phase 3.

### 2.1 Shared completed outputs

Keep ordinary typed application tables for sources, parsed files, ASTs, HIR/editor
facts, and any later pass outputs the application requests. A row contains its full
key, an immutable shared value, and `last_used`. The process has one table per kind
of result, shared by all entries and callers. Paths never partition these tables.

For each pass, application code explicitly:

1. Forms the complete key from the current inputs.
2. Looks up a live cached output and takes a shared handle on a hit.
3. On a miss, invokes that pass outside the table lock, optionally supplying a
   compatible prior result, then publishes the completed output.

Different input files can hit results produced by different earlier requests. The
application assembles the current pass's inputs from those values. For example, a
third entry importing `a` and `b` can reuse their per-file results even when earlier
requests parsed them separately. Exact output hits do not need a single predecessor.

This updates the earlier predecessor-only cache plan: compiler results remain
immutable, while **application lookup tables and retention timestamps are mutable**.
Per-call results still retain only their required data; the application's tables
retain other completed results until TTL expiry. No mutable cache object enters a
compiler pass. A small amount of repeated lookup/sequencing code is acceptable.

Concurrent misses may compute twice initially. Recheck the table at publication:
select a compatible live winner or replace an expired row, and refresh the selected
row's `last_used`. Completed diagnostics may be cached for their exact inputs.
Cancellation, pending uploads, and transient acquisition/tool failures remain
retryable. Do not cache missing-file outcomes as complete source graphs that bypass
acquisition on later requests.

### 2.2 Time-based retention

Apply the same policy to cached sources and every kind of completed output:

1. Initialize `last_used` when a completed value is inserted. Refresh it when a
   request acquires that row. Use monotonic elapsed time, not wall-clock dates.
2. Default to **24 hours since last use**, configurable. A lookup at or beyond its
   expiry treats the row as a miss even if a sweep has not run yet.
3. Run a configurable periodic sweep, initially **every minute**, removing rows
   whose idle duration meets or exceeds the TTL. Sweeping does not renew entries.
4. Synchronize lookup, timestamp refresh, handle acquisition, and removal with
   short table locks. Compile and perform I/O outside those locks; release removed
   values outside the lock so destruction cannot stall unrelated lookups.
5. A sweep drops the table's ownership. Requests and downstream outputs retain
   their own shared handles; expiry never invalidates a source, diagnostic, or
   artifact they are already using.

Place timestamps beside values in application tables, not inside `Source` or compiler
IR. Ownership links between completed outputs must be acyclic; represent recursive
language relationships with IDs. Ordinary shared ownership releases an object when
its final owner drops, so no tracing garbage collector or allocator pools are needed.
See Rust's [Arc ownership contract](https://doc.rust-lang.org/std/sync/struct.Arc.html).

For example, a HIR may keep a source alive after its source-table row expires. A HIR
cache hit does not have to traverse and retimestamp every upstream row. Those owned
inputs remain usable; a later direct lookup of an expired row may reconstruct it
with the same value identity. Eviction affects reuse, not program semantics.

Apply the policy to retained input-graph handles and managed artifact handles too.
Artifact cleanup waits for downloads/build consumers to release ownership and must
only remove files belonging to that artifact generation. No ownership chain should
retain every old edit. Background maintenance must not repeatedly touch entries
and keep them warm without requests.

TTL bounds idle cache retention, not total memory: a high rate of distinct requests
within 24 hours or live consumers can retain substantial data. Measure table counts,
hits/misses, and retained memory so the TTL can be tuned. Fixed-size pools, LRU limits,
and a hard cache memory ceiling are outside this plan. Concurrency remains bounded.

### 2.3 Entry-driven editor inputs

Each analysis request selects an entry source and an immutable graph of exact source
versions and resolved import bindings. This graph is ordinary input data; it is not
a persistent workspace with its own cache or mutable head. Never use one global
`path -> latest contents` mapping, which would mix branches and unsaved revisions.

Remove manifest/root discovery from the plan. Resolve user imports relative to the
importer, including existing `../` imports. Use names relative to the entry file's
directory for transport/presentation; leading `../` components are valid logical
locators, never paths for the server to open. Library/dependency namespaces remain
distinct. The current LSP directory argument may guide editor file discovery, but
it does not establish a server workspace or cache identity.

The LSP application reads disk sources with open buffers taking precedence. It
tracks document versions and the exact input graph used by each request. An edit
creates a new graph sharing unchanged values; changed imports trigger discovery
again. Track import dependents so changing a helper also invalidates its unchanged
entry's analysis. A changed-file list first produces a complete new input selection.
Preserve existing client path normalization for aliases, symlinked ancestors, and
unsaved files; resolve aliases to one deterministic logical module before keying
the graph. Different import-discovery orders must produce compatible identities.

Editors own disk saves. A save changes that file's disk view and preserves other
dirty buffers. Ordinary CLI builds use disk contents. Analyze eagerly, allow
speculative compilation of known targets when resources permit, coalesce rapid
edits, and suppress stale diagnostics. Publish only results for the applicable
revision; requests already using older graphs may finish against those graphs.

Spell out the passes required by each query/build path. Share the completed phase
outputs while allowing straightforward orchestration to be duplicated.

### Phase 2 acceptance criteria

- [ ] **P2.1** Different callers/entries reuse the same shared output handles for
      identical inputs. A new compilation reuses per-file outputs originating in
      multiple earlier requests. Different contents/bindings never mix.
- [ ] **P2.2** A controllable clock verifies refresh past the original insertion
      deadline, the exact expiry boundary, lazy expiry, periodic sweep, and cold
      recomputation after eviction; tests do not wait real hours/minutes.
- [ ] **P2.3** Sweep during analysis, local builds, and artifact reads leaves retained
      results valid. After table eviction and final-consumer release, ownership is reclaimed.
      Upstream-row expiry while a downstream result lives preserves source identity.
- [ ] **P2.4** Lookup/publication/sweep races and concurrent misses preserve correct
      outputs and timestamps without holding locks through computation or I/O.
- [ ] **P2.5** LSP handles rapid edits, invalid code then repair, additions/deletions,
      changed dependencies, parent-directory imports, saves, and simultaneous queries.
      Alias imports select one module; discovery order does not change graph identity.
      Old revisions remain readable and stale results cannot replace current ones.
- [ ] **P2.6** Receive handling remains responsive during computation and sweeps;
      cancellation and shutdown drain/reap owned work with bounded queues/jobs.
- [ ] **P2.7** Sustained-edit measurements report latency, sharing, and retained memory;
      advancing beyond TTL releases idle history without workspace/session leases.

## Phase 3: HTTP server and thin client

### Deliverable

Introduce `resin-client`, `resin-server`, and `resin-protocol` under `crates/`, with the
root `resin` executable delegating to the merged client. The server owns explicit
per-pass tables and calls the passes directly for builds and analysis queries.
Move the local LSP compilation responsibilities to those server handlers.

### 3.1 Connection and requests

- Require a server URL in `RESIN_SERVER` for build, run, and LSP. Missing/invalid
  configuration, connection failure, or incompatible protocol produces an early
  actionable error. No discovery files, automatic spawning, or local build fallback.
- Use versioned wire types for source submissions, complete input references,
  diagnostics, query positions/revisions, targets, and artifact responses. Do not
  expose compiler implementation types on the wire.
- Each request names one entry source and its immutable inputs. There is no
  create-workspace API, workspace TTL, or mandatory long-lived compiler session.
  Optional cached graph/revision handles follow the same entry TTL as other values.
- Build and query handlers explicitly look up and invoke the required passes.
  They use the same process-wide tables; the wire request path does not partition
  reuse. Preserve Phase 2's TTL and ownership behavior for every retained output.

### 3.2 Acquire a source graph without a project manifest

Use an entry-driven upload exchange so the thin client need not parse imports or
guess a directory tree to upload:

1. The client reads/posts the selected entry file's logical name and contents.
2. The server parses available files and reports unresolved user imports with their
   importer and relative reference. Batch independent missing imports.
3. The client resolves those references locally, uses LSP buffers where present,
   and posts the corresponding names and contents, or reports a missing/unreadable
   file. Repeated references to the same local file select the same logical module
   within that request, using the alias normalization established in Phase 2.
4. Repeat until the user-source closure is complete. The server fetches dependencies
   and supplies standard/builtin libraries itself, with pinned versions/content.
5. Freeze the complete input graph before semantic compilation. On an edit, derive
   a new graph from explicit prior inputs and replacements/deletions, and resolve
   its imports again. Reuse all matching sources and outputs through the same tables.

Correlate every exchange/upload with its request revision. Capture the editor buffer
versions for that revision and select each disk file's contents once per exchange.
Observed relevant edits/deletions during discovery cancel or start a successor;
never merge uploads from different revisions. This defines an exact captured input
selection without claiming an atomic snapshot of the client's entire filesystem.

These are bounded acquisition exchanges, not workspace lifetimes. Abandoned exchanges
release request state. A missing/unreadable local source terminates acquisition with
an import diagnostic instead of repeated upload requests; partial editor facts may
still be returned. The server never opens the supplied client path as a fallback.
Missing cached upload references after eviction/restart prompt re-upload. Pending
exchanges do not produce completed semantic cache entries. Source-only parsing can
still be reused; future requests reacquire unavailable imports so creating a missing
file can repair the program.

Absolute paths remain client-side. Request-local logical names and import edges
disambiguate source roles; full semantic keys cover their contents and bindings.
Two clients using the same logical name can submit different versions concurrently.
Identical graphs can share outputs. Keep each client's local URI mapping outside
those shared outputs. No change is written to disk merely because an LSP request
submits different contents.

### 3.3 Native output and deployment

- Define an explicit supported target contract: OS, architecture, ABI/runtime needs,
  selected entries, profile, and relevant native build settings. Advertise support
  and reject unsupported requests; a Linux server does not implicitly cross-build
  for every client platform.
- Run code generation, Ninja, compilers, and other required native tools on the
  server. Resolve the native `--embed` helper explicitly for the server executable.
- Return structured diagnostics and artifact metadata, with one output file over
  HTTP initially. Complete downloads before publishing them at the client destination.
  Keep active downloads safe from TTL cleanup.
- Run downloaded executables locally, with local arguments, runtime environment,
  working directory, and standard streams. Preserve debug build-and-run without
  `-o` and optimized output without execution with `-o`.
- Use the same explicitly launched server for development and deployment. Document
  local startup and manual Docker/systemd supervision, including user/service-account
  operation. Initial installation and service setup are manual; no installer is needed.
- Pick and document an HTTP framework during this phase; Rocket is a candidate.
  Native toolchain and dependency settings are explicit server configuration.
  Define authentication/access and execution isolation before offering the service
  to mutually untrusted users; content keys alone do not grant access to an artifact.

### Phase 3 acceptance criteria

- [ ] **P3.1** Root `resin` is a thin wrapper; the three application crates live under
      `crates/`. Compiler/toolchain crates have no client/server/protocol dependencies,
      and no build orchestration library has been introduced.
- [ ] **P3.2** Build/run/LSP fail clearly without a usable `RESIN_SERVER`, negotiate
      compatibility, and reject unsupported targets before native compilation.
- [ ] **P3.3** An entry and nested/parent imports compile with no `resin.toml`, upload
      directory, or client semantic parser. Tests prove user sources are supplied
      over HTTP and cannot be loaded from coincidentally existing server paths.
      Missing imports terminate promptly and can be repaired by a later request.
- [ ] **P3.4** Two clients at different absolute roots share identical source/phase
      outputs. Different branches and unsaved edits coexist without diagnostic or
      declaration-identity leakage; combined inputs reuse multiple callers' results.
      Dependency edits during upload cannot mix revisions or publish stale results.
- [ ] **P3.5** Build and LSP query handlers explicitly invoke their needed passes
      while sharing the same tables. Phase 2 edit/concurrency/TTL tests also pass
      through the HTTP boundary.
- [ ] **P3.6** Expired/restarted-server references recover through re-upload; canceled
      or disconnected requests release acquisition state and owned native processes.
      A sweep cannot interrupt an active artifact download.
- [ ] **P3.7** Returned executables are placed/run locally with preserved build/run
      defaults and runtime arguments; native helper invocation works in the server.
- [ ] **P3.8** Document protocol/configuration defaults, dependency acquisition, target
      support, retention semantics, native tool requirements, and manual startup.

## Execution and follow-up scope

For each phase, implement the numbered work in order, add tests for its acceptance
criteria, and record validation before marking those criteria complete. Use the
repository's development environment and appropriate compiler/LSP/native integration
checks. Follow repository worktree and pull-request instructions; keep each change
reviewable. Passing a documentation check does not complete an implementation phase.

Phase-specific implementation details such as exact wire field names, executor,
framework, and dependency configuration should be selected and documented in the
phase that introduces them. They must preserve the contracts and acceptance cases
above rather than becoming prerequisites left to a future plan.

Later work: MCP queries for documentation/diagnostics/symbols, REPL/shell runtime
sessions, additional watching clients, tar/tgz or multiple-file artifacts, transport
deduplication/batching optimizations, finer-grained semantic reuse, broader deployment
support, and installers. Cross-caller reuse and time-based retention are required by
this plan; they are not deferred to a later shared-cluster project.
