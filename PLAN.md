# Compiler service plan

Status: planned. Implement the three phases below in order. Each phase has its own
acceptance criteria; finish those criteria before moving to the next phase.

## Agreed direction

- Keep compiler inputs and completed outputs immutable and independently reusable.
- Let applications explicitly invoke each compiler pass and retain its outputs.
  Build handlers and LSP query handlers may duplicate straightforward sequencing.
  Do not introduce `resin-build` or a generic compilation driver that hides passes.
- Represent each layer's retained outputs with an immutable `Cache<K, V>`.
  `update()` returns a new set containing reused/new results; `prune()` returns a
  new set with old entries removed. Published sets, entries, and values stay immutable.
- Start with one shared cache per output kind, keyed by complete inputs.
  Results from any caller can contribute to another caller's compilation. The same
  value operations must also work with separate folder/package sets; their placement
  is an application choice, independent of compiler pass contracts.
- Start each build or analysis from one source entry point and its resolved imports.
  There are no persistent compiler workspaces, required workspace cache partitions,
  workspace leases, or `resin.toml`. A build also selects its exported function/target entries;
  analyzing a source for LSP does not require it to define a runnable `main`.
- Retain entries for a configurable **24 hours since last use**. Prune as part of
  updating the active set. A timer may invoke the same operation for idle cleanup;
  background work is optional. Also support an explicit generation-based cutoff.
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
  resin-server/src/               # HTTP handlers, passes and active set pointers
  resin-protocol/src/             # versioned wire data
  resin-source/src/               # immutable sources and import data
  resin-cache/src/                # pure keyed update/prune collection
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
support; no HTTP service is required yet. Establish the small cache collection
here so local LSP and the later server use the same immutable data operations.

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
  This restriction applies to a compilation's output, not the cache retaining
  many outputs. Only explicit pruning discards unrelated cache entries.

### 1.2 Keys and source identity

Make reuse depend on complete inputs, independently of who requested them. Define
explicit key data beside each relevant phase's public input contract. Do not use
session IDs, absolute checkout roots, timestamps, or allocation addresses as semantic
cache keys. Each phase needs only inputs that can affect that phase:

| Cached value | Relevant key inputs |
| --- | --- |
| Source | Explicit logical module identity/role plus a hash of the complete file contents, including namespace and any diagnostic name retained in the value. |
| CST / per-file AST | Immutable Source content identity and relevant parser/lowering configuration. |
| HIR and editor facts | Complete resolved source graph, import bindings, entry source, library/dependency versions, and semantic options. |
| LIR / verified LIR | HIR identity, canonical selected entry set, target profile, lowering limits/options, and verification contract. |
| Generated files / native artifact | Verified LIR, code generation options, target/ABI, compiler/runtime/library inputs, native toolchain inputs, and build settings. |

Use a stable hash of the entire file's exact contents as its content identity; no
source revision counter is needed. Keep the contents themselves in the immutable
source. Check underlying identity/content when interning equal hashes so a collision
cannot silently equate different sources. Text storage may be shared by content
alone; a named module still carries its logical identity and source origins.

A global `path -> Source` table can select only one version at a path. Retaining
multiple edits/projects requires `(logical module identity, content hash) -> Source`
or an equivalent map of versions under each path. A request separately selects
`path -> Source` for its own input graph. Equal-content edits naturally reuse the
same content identity; unrelated callers never overwrite each other's selections.

A key identifies every input that can affect its value. A mapper must not secretly
read current files, environment, or changing dependencies absent from its key.
Persisted keys additionally include the relevant compiler/toolchain version.
Runtime arguments and runtime environment are not build inputs.

The current `Source` uses allocation-based version equality and process-local IDs.
Establish value-based source identity that remains correct after eviction
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

### 1.3 The cache collection

`Cache<K, V>` is an immutable collection of reusable results indexed by complete
input keys, with explicit retention metadata. Every update and prune returns a new
cache; existing instances remain valid snapshots. The API makes this value flow
visible. [Clojure's core.cache](https://clojure.github.io/core.cache/#clojure.core.cache.wrapped)
provides a precedent for immutable cache values with separate atomic publication.

Add a small `resin-cache` crate for this type. It owns only the keyed
collection and immutable retention metadata. It knows no compiler phases, dependency
scheduler, executor, clock-reading operation, filesystem, or global active pointer.
Keep `resin-common` minimal. Server and LSP code still choose and invoke passes.

The operations have these value semantics, using illustrative signatures:

```text
Cache<K,V>::update(&self, keys, used_at, fresh: K -> V) -> Self
Cache<K,V>::prune(&self, cutoff) -> Self
```

- An empty cache is the cold starting point. `update` deduplicates requested keys,
  reuses existing values, computes misses, and renews requested entries' metadata.
  It preserves unrequested entries. Only `prune` removes historical entries.
- Every timestamp is supplied by the application. Neither operation reads a clock
  or modifies old entries. Reading a snapshot alone does not secretly renew it;
  request code explicitly updates the keys it uses.
- The mapper computes one completed value from one complete key. It may capture
  already-resolved immutable inputs and an optional incremental predecessor. Those
  captures must not change the result for an otherwise equal key.
- Applications can evaluate independent ready keys in parallel with bounded jobs.
  Async/native work can be prepared before applying the collection update. Keep
  failures and cancellation explicit; never publish placeholder or partial values.
- Share values through `Arc` or their existing immutable handles. Keep collection
  representation private. Ordinary maps are a correct starting point, but cloning
  them copies their index. Measure large hit-only updates and rebasing costs; use
  an existing persistent map if needed to share map structure. Do not build a HAMT
  or make an unmeasured claim that map updates have constant cost.

This is a reusable collection operation, not a pass orchestration framework.
Compiler outputs never retain entire prior caches as their hidden cache.

### 1.4 Async work and native artifacts

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
  must keep its files alive until all consumers finish, independently of cache membership.

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
- [ ] **P1.8** Cache tests show that updating `{a,b}` with requested `{b,c}`
      returns `{a,b,c}`, shares `b`, computes only `c`, and renews only requested
      metadata. Update/prune leave old sets unchanged. Content hashes distinguish
      changed text and reuse identical text without source revision counters.

## Phase 2: Immutable caches exercised through LSP

### Deliverable

Use the existing LSP mode to exercise repeated and overlapping computations locally.
Applications explicitly maintain active pointers to immutable per-layer caches. Use the Phase 1 collection for updates and retention, with no new watch-mode
executable or orchestration library. Local tests schedule analysis and builds
against these sets. Separate CLI and LSP processes share a service only in Phase 3.

### 2.1 Shared completed outputs

Start with one active `Cache<K,V>` per output kind: sources, CSTs, per-file ASTs,
HIR/editor facts, and any later pass outputs requested. A set retains entries from
multiple callers, input graphs, and generations. Each entry shares its completed
value and records immutable last-use time and generation metadata.

For example, a new entry importing `a` and `b` can reuse per-file results produced
by separate earlier requests. The application explicitly assembles each pass's
current inputs from those values; it is not restricted to one predecessor graph.
Folder/package partitioning may be explored by placing the same kinds of sets in
separate application-owned scopes. Scope changes must not change compilation meaning;
complete keys allow selected results to be reused across those scopes.

Cache updates retain unrequested entries until pruning. Each individual compiler
result still refers only to its required data. The shared mutable state is the
active pointer, not the published map or its timestamps.

Completed diagnostics may be retained for their exact inputs. Cancellation, pending
uploads, and transient acquisition/tool failures remain retryable. Missing-file
outcomes cannot bypass source acquisition on a later request that may repair them.
Concurrent misses may compute twice initially; sharing identical completed values
is an optimization, while equal-key semantic correctness is mandatory.

### 2.2 Retention as a value operation

The normal application sequence is:

```text
updated = update(previous, requested_keys, use_stamp, fresh)
next    = prune(updated, retention_cutoff)
```

Requested entries are renewed before pruning. An old entry still present is valid
for its complete key and may be reused; reaching its age alone does not invalidate
its contents or force recomputation. `prune` establishes the retained membership.

Support two explicit alternative cutoff modes:

- **Time (default):** retain for 24 hours since last use, configurable. Applications
  pass monotonic timestamps and remove entries whose last use is at or before the
  computed cutoff. Timestamp renewal creates new metadata while sharing the value.
- **Generation:** remove entries at or before a supplied last-use generation cutoff.
  Generations count successfully published updates of that cache. An update
  candidate advances its predecessor's generation once and stamps requested keys;
  pruning alone preserves the generation. CAS retries do not count as extra updates.

Generations belong to one set's lineage; they are not comparable across phase or
package sets. Other callers' updates age an entry under a shared generation policy.
Time remains the service default. A policy selects one mode, avoiding ambiguous
combinations of time and generation conditions.

Invoke pruning when publishing updates, optionally batching/throttling it if measured
cost warrants that. A background sweep is not required for correctness. Opportunistic
pruning performs no cleanup while the server is idle; when idle reclamation is wanted,
a configurable timer can publish the same pure `prune` operation. Use one minute as
the initial optional timer interval. Neither pruning nor timer activity renews use.

Pruning a new set does not revoke older snapshots or handles. A retained HIR can
keep its sources alive after the source set prunes them. Ordinary shared ownership
reclaims values after their final owner drops; ownership links must be acyclic.
See Rust's [Arc ownership contract](https://doc.rust-lang.org/std/sync/struct.Arc.html).

Acquire needed result handles and release old whole-set snapshots promptly. A long
request retaining a whole snapshot also retains its unrelated entries. Do not add
parent links retaining every generation. A later request can reconstruct an evicted
source and still use retained HIR thanks to its content-based source identity.
Artifact ownership similarly protects live readers; cleanup removes only the files
belonging to that artifact generation after its final owner releases it.

TTL limits idle retention in the current set, not total process memory. Old snapshots,
live consumers, and many distinct results inside the retention window can retain
substantial data. Measure set sizes, sharing, hits/misses, and memory. Fixed-size
allocator pools, tracing GC, and a hard cache memory ceiling are outside this plan.

### 2.3 Publishing concurrent updates

Keep publication in application code, separate from the pure cache operations.
Use safe atomic shared-pointer publication, such as `ArcSwap`, for each active set.
Holding an owned snapshot protects it while the active pointer changes. Avoid a
hand-written raw-pointer reclamation scheme. Atomic publication does not make native
processes, allocations, and the rest of the compiler lock-free.

A plain load/compute/store can lose concurrent additions. A successful CAS publishes
one candidate atomically; a failed CAS requires rebasing against the current head:

1. Load a snapshot and capture needed hits and missing keys. Release the whole-table
   snapshot before running bounded jobs where possible. Keep a prepared change set
   of **all requested keys**, completed values, and use timestamps, including hits.
   Stamp newly computed values at completion. Keep this data across retries.
2. Load the current head and apply only those requested entries to it. Preserve its
   unrelated entries and prefer already-present compatible values. Renew timestamps
   with `max(current, requested)` so a slow request cannot move last use backward.
3. Form the candidate's next generation from that head. Retain the request's result
   handles, then prune using current metadata and an explicit cutoff. A long request
   remains valid even if its selected entries have aged out before publication.
4. Compare-and-swap the active pointer from that head to the candidate. If it loses,
   reload and repeat the merge/prune; retain completed computation results.

Do not replay a stale whole snapshot or a stale list of deletions. A prune racing
with a touch must re-evaluate age against the newer metadata. Reintroducing a pruned
key actually requested again is valid; reintroducing unrelated history is not.
Prune-only publication uses the same CAS/re-evaluation rule without advancing the
update generation.

Compiler/native work must remain outside the CAS retry operation. The atomic-Arc
API may retry its update closure; see the [ArcSwap update documentation](https://docs.rs/arc-swap/latest/arc_swap/struct.ArcSwapAny.html#method.rcu).
A repeated merge may copy map structure, but must not rerun an expensive completed
pass or repeat native side effects. Prefer a concurrent winner's compatible handle
when selecting downstream inputs; temporary duplicate computations are acceptable.

Each layer may publish independently: a pass consumes explicit immutable dependency
handles, never an accidental mixture of whatever several active pointers contain.
No multi-layer transaction is needed for correctness when keys cover those inputs.

### 2.4 Entry-driven editor inputs

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

- [ ] **P2.1** Different callers reuse matching completed values. A new compilation
      uses per-file outputs from multiple earlier requests; separately organized
      sets produce the same semantics. Same-path different contents coexist.
- [ ] **P2.2** With an explicit test clock, updates renew requested metadata without
      changing old sets. Time and generation pruning remove exact-cutoff entries;
      successful updates advance generations once, retries/pruning do not.
- [ ] **P2.3** Pruning during analysis, local builds, and artifact reads leaves held
      values valid. Releasing final roots/consumers reclaims them. Upstream pruning
      and source reconstruction preserve queries against retained HIR.
- [ ] **P2.4** Concurrent disjoint updates both survive. Update/prune races preserve
      renewed entries without restoring unrelated history; slow publication cannot
      regress timestamps. CAS contention never repeats completed compiler/native work.
- [ ] **P2.5** LSP handles rapid edits, invalid code then repair, additions/deletions,
      changed dependencies, parent imports, saves, and simultaneous queries. Aliases
      select one module; discovery order preserves identity; stale results cannot
      replace the current revision.
- [ ] **P2.6** Receive handling remains responsive through computation/publication;
      bounded queues/jobs and cancellation/shutdown release owned tasks/processes.
- [ ] **P2.7** Tests distinguish update-triggered pruning from optional idle cleanup.
      An old still-present exact-key result can be renewed before pruning; after
      removal it can be recomputed correctly.
- [ ] **P2.8** Measure sustained editing, hit-only updates, map copying/structural
      sharing, CAS retries, retained snapshots, and reclamation. Do not claim a
      constant-cost update or hard memory bound from pointer swapping alone.

## Phase 3: HTTP server and thin client

### Deliverable

Introduce `resin-client`, `resin-server`, and `resin-protocol` under `crates/`, with the
root `resin` executable delegating to the merged client. The server owns active
per-pass cache pointers and calls passes directly for builds and analysis queries.
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
- Build and query handlers explicitly select inputs, invoke passes, and publish
  immutable caches. They use the same process-wide sets initially; request
  origin does not partition reuse. Preserve Phase 2's update/prune, CAS, and ownership
  behavior for every retained output.

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
   its imports again. Reuse matching sources and outputs through the same caches.

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
  Keep active downloads safe from pruning and old-set reclamation.
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
      while sharing immutable caches. Phase 2 update/prune/concurrency tests pass
      through the HTTP boundary.
- [ ] **P3.6** Expired/restarted-server references recover through re-upload; canceled
      or disconnected requests release acquisition state and owned native processes.
      Pruning cannot interrupt an active artifact download.
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
support, and installers. Cross-caller reuse and immutable cache retention are
required by this plan. Broader folder/package partitioning can reuse these collection
operations without changing compiler contracts or requiring a workspace protocol.
