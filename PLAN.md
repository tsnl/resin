# Compiler service plan

Status: **Phases 0–1 implemented and validated; Phases 2–3 remain planned.**
Implement **Phase 0, Phase 1, Phase 2, and Phase 3** in order.
Each phase has separate acceptance criteria. Complete and validate a phase before
starting the next. Record evidence in [the validation log](doc/compiler-service-validation.md).

## Agreed architecture

- Compiler inputs and completed outputs are immutable, shareable, and independent
  of the application hosting them. Applications explicitly invoke each pass and
  retain its results. Build and LSP handlers may duplicate straightforward sequencing;
  do not add a build orchestration library or a generic compiler driver.
- Each output layer uses an immutable `Cache<K, V>`. Async `update()` reuses hits,
  builds misses concurrently, and evicts older unrequested entries while constructing
  the next cache. Capacity counts **entries**. Retain all requested results, even
  above capacity, and warn on overflow. There is no `prune()`, TTL, or periodic GC.
- Start with shared caches across callers, projects, and generations: a
  **mega-workspace**. Every request selects its own exact sources and import bindings.
  There is no mutable global path-to-latest-source table, persistent compiler
  workspace, workspace lease, or `resin.toml`.
- Compilation starts from a source entry point and its imports; builds additionally
  select exported function/target entries. Editor analysis needs no runnable `main`.
  Content and complete pass inputs determine reuse, independently of who requests it.
- Build async/await and bounded parallelism into the libraries in Phase 1, exercise
  them through local LSP in Phase 2, then move compilation behind HTTP in Phase 3.
- The client uses the existing full CST parser to discover imports and foreign
  headers. The server parses uploaded sources through its own caches. Use one grammar;
  do not introduce a scanner, regex import parser, or second header-only parser.
- The server supplies standard/builtin libraries, fetches configured dependencies,
  and runs native tools. Clients upload user sources and local header directory
  bundles, receive artifacts, and execute programs locally.

### Final repository layout

Keep the `resin` package and workspace manifest at the repository root, with the
root as the default Cargo member. Other members live under `crates/`; the independent
editor extension workspace keeps its existing layout.

```text
Cargo.toml                       # root resin package and Cargo workspace
src/main.rs                      # thin wrapper around resin-client
crates/
  resin-client/src/
    lib.rs                       # CLI, connection, source acquisition
    lsp/                         # private editor client implementation
    interp/                      # private run/interactive client implementation
  resin-server/src/              # service, explicit passes, active cache pointers
  resin-protocol/src/             # versioned wire data
  resin-cache/src/                # immutable cache, async update and eviction
  resin-source/src/               # immutable named sources and import graph data
  resin-cst/src/                  # parsing, formatting, preamble queries
  resin-ast/src/
  resin-hir/src/
  resin-lir/src/
  resin-codegen/src/
  resin-toolchain/src/            # native processes and artifact ownership
  ...                            # existing compiler/support crates
```

Final dependencies:

```text
root resin -> resin-client -> resin-protocol
                          -> resin-source / resin-cst (syntax and local inputs)
resin-server -> resin-protocol / resin-cache
             -> individual compiler crates / resin-toolchain

syntax -> AST -> HIR -> LIR -> verified LIR -> C/SPIR-V -> native tools
```

Keep one merged `resin-client`; migrate the existing `resin-lsp` implementation
into its private LSP module when introducing the server. Client-side parsing and
formatting are syntax operations; semantic compilation belongs to server handlers.
Compiler/toolchain/cache crates depend on none of the application or protocol crates.
`resin-protocol` contains wire data without compiler IR or HTTP framework types.
`resin-cache` knows no compiler phases. Keep public contracts in each crate's
`lib.rs`, implementation private, and `resin-common` minimal.

## Phase 0: Group foreign declarations in the source preamble

### Deliverable

Make the breaking foreign-function syntax change before changing compiler ownership.
The optional preamble is ordered `export`, `extern`, `import`, followed by declarations:

```resin
export { main };
extern {
  "resin_runtime.h": {
    def resin_image_write_png(path: Ptr<ubyte>, width: uint, height: uint, channels: uint, pixels: Ptr<ubyte>, stride: ulong) -> int;
  },
  "local/header.h": {
    def local_value() -> int;
  },
};
import { "$/span.resin" };

def main() -> int = { local_value() };
```

### 0.1 Language and migration

- Permit an omitted or empty top-level `extern` block. Header groups are separated
  by commas, with an optional trailing comma; definitions end with semicolons and
  the outer block ends with `};`. Allow empty header groups.
- Replace `extern "header.h" def ...;` with `def ...;` inside the corresponding
  header group. Preserve standalone `extern type Name;` for opaque foreign types.
- Grouping associates declarations with a header; it introduces no new namespace.
  Foreign functions remain module declarations with existing explicit export rules,
  duplicate-name checks, signature validation, and native ABI behavior.
- Imports remain available to resolve foreign signatures even though the import
  clause follows the extern block. Preserve source spans and useful recovery.
- Update the Tree-sitter grammar, generated parser/node types, corpus, CST consumers,
  AST lowering/printing, formatter, HIR diagnostics/completion, editor queries,
  standard libraries, examples, tests, and language documentation/AGENTS snippets.
  Keep the existing per-function header representation downstream if sufficient.
- Preserve declarations of header dependencies even for empty groups; do not lose
  a declared header merely because AST lowering flattens its functions.

### 0.2 Discover dependencies with the existing CST

Expose a small syntax-only query in `resin-cst` for imports and extern header groups,
including decoded strings, source locations, and preamble diagnostics. Reuse the
existing full parser and string handling; do not duplicate the grammar in a scanner.

Extract valid preamble information even when unrelated function bodies contain
syntax errors. Incomplete import/extern clauses produce recovery information or
diagnostics, rather than silently reporting a complete graph with no dependencies.
The client will run this query recursively in Phase 3. Parsing again on the server
is the accepted initial design; measure its cost without assuming it is negligible.

### Phase 0 acceptance criteria

- [x] **P0.1** Parser tests cover omitted/empty blocks, multiple and empty header
      groups, trailing commas, comments/escapes, preamble order, and malformed input.
      Old foreign-function syntax is rejected; standalone foreign types still work.
- [x] **P0.2** Grouped declarations retain module scope, exports, duplicate-name
      diagnostics, imported signature types, parameter/result ABI checks, and spans.
- [x] **P0.3** Formatting is stable; grammar-generated files, editor queries,
      standard libraries, examples, fixtures, and documentation use the new syntax.
- [x] **P0.4** The CST query finds imports and headers despite unrelated body errors,
      handles incomplete preambles, and preserves empty groups. There is one parser.
- [x] **P0.5** Native runtime/system/local-header examples build. Existing nested
      header change/deletion regression tests pass with the migrated syntax.

## Phase 1: Immutable caches and asynchronous compiler passes

### Deliverable

Compiler libraries expose complete immutable inputs and completed results that can
be shared across threads. Establish `resin-cache`, async pass interfaces, bounded
parallel computation, and explicit native artifact ownership. Current local
applications sequence these operations directly; HTTP is not required yet.

### 1.1 Pass contracts and immutable inputs

- Expose parsing, per-file AST construction, HIR/editor analysis, LIR construction,
  verification, generation, and native building separately. Split APIs that conceal
  earlier passes so applications can reuse each completed output explicitly.
- Source acquisition resolves an immutable graph before semantic compilation.
  Passes receive exact sources, import bindings, options, and dependency identities;
  they never open user source paths or consult mutable editor state.
- Completed inputs/results are immutable and `Send + Sync`. Private parsers,
  builders, inference solvers, and traversal state may mutate while doing their
  own work; that state is not shared as a completed result.
- Each result contains only the data required by its inputs. It does not retain a
  chain of predecessors or whole old caches. A cache may retain many independent
  results according to its capacity.
- Preserve optional incremental predecessors when useful. A predecessor has the
  same type as the output, and `None` means cold computation. For example:

  ```text
  async parse(source, previous: Option<&ParsedFile>, execution) -> ParsedFile
  ```

  A predecessor is an optimization hint. Cold and incremental computation must
  agree on meaning and diagnostics. Incremental semantic inference is not required.

### 1.2 Complete keys and source identity

Use explicit input/key types with **`Eq + Hash`** for hash maps, or **`Ord`** for
ordered maps; `PartialOrd` alone is insufficient. Do not require every IR node to be
hashable. Stable content digests use a specified encoding/hash, independently of
Rust's map hashing, whose encoding is not a portable persistence contract.

| Cached value | Key inputs |
| --- | --- |
| Source | Canonical logical module identity/role and exact complete text content identity. |
| CST / per-file AST | Source identity and relevant parser/lowering options; content alone is sufficient only for payloads independent of source origins. |
| HIR / editor facts | Entry source, complete resolved source graph and import edges, library/dependency identities, semantic options. |
| LIR / verified LIR | HIR identity, canonical selected entry set, Host/Shader profile, lowering limits/options, verification contract. |
| Generated files | Verified LIR, generation options, target/ABI, declared native header bindings. |
| Native artifact | Generated inputs, complete header bundle contents and include-search bindings/order, runtime/dependency/toolchain identities, target and build settings. |

Keys include every input affecting that value, including compiler version where
results survive a compiler change. Builders must not read changing files, environment,
or dependencies absent from their key. Runtime arguments/environment stay out of
build keys. Native settings that affect generated output belong in that earlier key.

Separate exact text identity, logical module identity, and client presentation.
Retain source text alongside its digest; check actual content when interning equal
digests so collisions cannot silently equate different inputs. Equal text can share
storage, but two distinct modules in one graph remain distinct declarations/origins.
Do not key semantic results by session, edit counter, absolute checkout root, or
allocation address. Canonical graph ordering must not depend on discovery order.

Each request selects `logical path -> Source`; shared source caches retain multiple
content identities at the same logical path. Logical names and import bindings
disambiguate modules; applications separately map those names to local URIs.
Equivalent relative graphs at different checkout roots can share results without
leaking one client's paths into another client's diagnostics.

Replace the current allocation-based Source version equality with compatible
content-based identities, including the module context that the value retains.
Source reconstruction after eviction must still support lookups into a retained
HIR's editor facts and origins. Begin with per-file CST/AST and whole-graph HIR reuse;
an unchanged entry with a changed imported file or binding is a different HIR key.

### 1.3 Full uploads and optional edits converge

Source construction accepts complete text or an optional predecessor plus edits:

```text
make_source(logical_identity, previous: Option<&Source>, edits) -> Source
```

`None` starts from empty text; a full upload is one insertion of all text. Validate
ranges/encodings and the predecessor identity before applying edits. Normalize to
the final text and its content digest **before** selecting the source/cache key.
A predecessor/edit recipe must not become semantic identity: different edit histories
and clean uploads producing identical named text must reuse the same entry.

On a parse miss, the builder may capture the corresponding prior CST and validated
edits to incrementally parse. Clone/edit private parser state, preserving the old
result. If the predecessor is absent, incompatible, or evicted, parse the complete
text. No wire-level edit protocol is required initially; LSP may upload full text.

### 1.4 `Cache<K, V>` and capacity

Add `resin-cache` with an immutable completed cache value and its configuration.
Keep the builder generic on **`update()`**, rather than storing `F` on `Cache`.
This allows per-request immutable inputs and optional predecessors without retaining
the closure, executor, or its captured history in every cache generation.

The conceptual contract is:

```text
Cache<K, V>::new(capacity: usize) -> Cache<K, V>
async Cache<K, V>::update(&self, requested_keys, build: F, execution) -> Result<Self, E>
    where F maps K to a future producing Result<V, E>
```

Exact ownership/future bounds belong to implementation; cheap owned keys/handles
are acceptable. The previous cache and successful return are the same type.
Independent builders run concurrently within explicit limits, optionally in groups.
The generic cache schedules item construction; applications still choose compiler
passes and their dependencies.

Every successful update:

1. Deduplicates requested keys and reuses matching values.
2. Computes missing values, at most once per distinct key within this update.
3. Marks every requested key as recently used in the **new** cache.
4. Retains all requested results; fills remaining capacity with the most recently
   used old, unrequested entries. Evicts least-recently-used unrequested entries.
5. Returns the new cache, sharing completed values while preserving the old cache.
   If requested distinct keys exceed capacity, retains them all, removes unrequested
   entries, and automatically logs a warning with capacity/requested/retained counts.

Capacity is an entry count: one large value counts as one entry. The returned size
is at most `max(capacity, distinct_requested_count)`. Zero capacity still permits
requested results with an overflow warning. An empty request retains only as many
old entries as fit. An oversized generation can shrink on a later update.

Recency is immutable bookkeeping, ordered by updates to that cache. Hits refresh
recency; reading a handle alone does not. Define deterministic tie handling independent
of task completion order. Update sequence numbers are not source identities or
age-expiration policies. There is no timer, TTL, separate `prune()`, or byte weigher.
A future byte-based configuration may estimate a per-layer element capacity outside
this type; it cannot claim precise byte accounting.

Completed source diagnostics may be cached as values for their complete inputs.
Cancellation/infrastructure failure returns an explicit failure, without publishing
partial successful entries. Do not retain retryable missing-file or native-process
failures as successful cache values.

Use ordinary maps and shared values initially; a HAMT is not required. Keep index
representation private and measure cloning/rebasing costs before adopting persistent
map storage. Capacity bounds new-cache membership, not total process memory:
old snapshots, dependency handles, and live consumers can keep evicted values alive.

### 1.5 Async execution and native artifacts

- Establish async pass interfaces and async `Cache::update()` in this phase.
  Permit bounded parallel per-file/per-key jobs and independent work within passes;
  keep dependent groups explicit and preserve deterministic completed output.
- Provide actual CPU scheduling, not just async syntax around blocking loops.
  Use bounded workers, cooperative yield/cancellation checkpoints, and async I/O.
  Isolate unavoidable synchronous foreign calls, including Tree-sitter parsing,
  on workers so protocol/I/O tasks remain responsive.
- Bound queues and coordinate compiler jobs with Ninja/native-tool concurrency.
  Cancellation must stop queued work and reap owned native children; a started
  blocking call may need to finish before cancellation completes.
- Keep execution/request context separate from semantic keys and completed caches.
  Pass process settings explicitly; do not change process-wide environment or
  working directory to configure concurrent builds.
- `resin-toolchain` supplies async native-process/artifact operations, consumes
  generated projects and explicit settings, and does not sequence compiler passes.
- Publish artifacts into immutable owned locations. Retaining/downloading artifact A
  must not hold an exclusive staging lock needed to build B. Files remain alive
  until their final consumer releases them, independently of cache membership.

### Phase 1 acceptance criteria

- [x] **P1.1** Direct library tests execute explicit async passes with no HTTP or
      orchestration wrapper. Existing local build/run behavior still works.
- [x] **P1.2** Full text and equivalent predecessor edits converge on source/parse
      keys. Cold and incremental results agree, including invalid input and repairs.
- [x] **P1.3** Completed inputs/results satisfy `Send + Sync`; concurrent successor
      computations preserve their predecessor and each other's results.
- [x] **P1.4** Equivalent independently acquired/reconstructed graphs reuse compatible
      results and editor identities. Changed imports/options miss affected caches;
      distinct equal-text modules remain distinct. Test digest collision handling.
- [x] **P1.5** Cache tests cover hits/misses, deduplication, refreshed recency,
      deterministic eviction, empty/zero/exact capacities, and overflowing requests.
      All requested values survive overflow and a warning is emitted; old snapshots
      remain unchanged. Later updates can shrink an oversized cache.
- [x] **P1.6** Independent builders overlap within configured bounds; async I/O stays
      responsive during CPU work. Cancellation/failure cannot publish partial success,
      and shutdown reaps child processes.
- [x] **P1.7** Artifact A remains readable while B builds; publication/final-owner
      cleanup cannot overwrite or remove another live artifact generation.
- [x] **P1.8** Record cold/unchanged/small-edit timings, concurrency, retained memory,
      map-copy costs, and cancellation latency. Document execution/capacity defaults.

## Phase 2: Exercise the caches and concurrency through local LSP

### Deliverable

Use existing LSP mode to exercise repeated, overlapping analysis and local build
requests against application-owned active cache pointers. No separate watch-mode
entry point or server is needed. Separate CLI/LSP processes share results in Phase 3.

### 2.1 Share completed results

Start with one active cache per output kind: sources, CSTs, per-file ASTs,
HIR/editor facts, and later outputs when requested. One compilation may reuse
per-file results produced by several earlier callers. The application assembles
explicit immutable pass inputs; it is not limited to one predecessor compilation.

The only shared mutable cache state is the application's active pointer. Published
maps, recency metadata, and values stay immutable. A later update can evict entries
used by an earlier request; that request retains its own result handles. There is
no global union of pinned request keys that grows the current cache indefinitely.

An in-flight update/CAS attempt may retain its base snapshot. After selecting and
publishing results, hold needed values rather than entire old maps during downstream
work. Avoid parent links and ownership cycles. Evicted values and artifacts are
released after their final owners, including downstream results, release them. An
idle cache needs no sweep: its membership changes on its next update.

### 2.2 Publish concurrent updates without losing contributions

Use safe atomic shared-pointer publication, such as `ArcSwap`, in application code.
Do not write raw-pointer reclamation. A plain load/compute/store can overwrite
concurrent additions, so use compare-and-swap (CAS) and rebase on failure:

1. Load an owned cache snapshot and run `update()` for this request's keys.
2. Retain **all requested** completed values, including hits, separately from the
   candidate cache so they survive eviction or a publication retry.
3. CAS the head from the loaded snapshot to the candidate.
4. On failure, load the latest head and call the same `update()` with the same keys
   and a cheap builder returning the already-computed values. Reuse compatible
   current-head values, renew recency against that head, and reapply eviction.
5. Retry publication without rerunning completed compiler/native work. Use handles
   from the successfully published candidate; do not reread a possibly newer head
   to discover this request's results.

Rebase only requested entries, never stale unrelated history or deletion lists.
Disjoint contributions survive when capacity permits; normal eviction may remove
older ones when it does not. Requested-key protection applies to each update,
not permanently to every overlapping request. Recency follows publication order;
CAS retries do not create extra published generations.

Each layer publishes independently. Downstream passes consume the selected immutable
dependency handles with complete keys, rather than mixing current global heads.
Concurrent identical misses may compute twice initially; retaining a compatible
winner is sufficient. Atomic cache publication does not make the whole compiler or
native tooling lock-free.

### 2.3 Editor revision handling

- Track LSP document versions and capture exact source selections per request.
  Open buffers take precedence over disk. Edits create new sources/graphs sharing
  unchanged values; older requests keep their own valid inputs.
- Discover imports with the CST query. Re-resolve the graph on requests, including
  when an entry is unchanged. Track dependents so helper edits refresh their callers.
  Handle additions, deletions, invalid code, aliases, and changed import bindings.
- Resolve user imports relative to the importer, including `../`. Normalize aliases
  to one logical module within a graph. Use entry-directory-relative logical names;
  parent components are logical locators, never server filesystem access.
- Keep absolute path/URI mappings in the application. The LSP directory argument
  can guide discovery but creates no compiler workspace or cache partition.
- Editors own saves; submitting an edit does not write it to disk. Saving one file
  preserves other dirty buffers. Ordinary build requests use the disk selection.
- Analyze eagerly, coalesce rapid changes, cancel superseded work where useful,
  and publish diagnostics only for the applicable revision. Allow resource-bounded
  speculative compilation of known targets without making it a prerequisite for queries.
- Query/build handlers explicitly invoke needed passes and share the same caches.
  An incomplete acquisition cannot masquerade as a complete semantic input graph.

### Phase 2 acceptance criteria

- [ ] **P2.1** A new compilation reuses per-file outputs from multiple earlier
      requests. Same-path different contents coexist; cache placement does not
      change semantics.
- [ ] **P2.2** CAS races retain disjoint contributions when capacity permits and
      evict according to current recency otherwise. Retries preserve requested keys
      without resurrecting unrelated history or repeating completed/native work.
- [ ] **P2.3** Eviction during analysis/build/artifact reads preserves held values;
      final-owner release reclaims them. Reconstructed sources still support queries
      against retained HIR. No cache predecessor chain keeps all generations alive.
- [ ] **P2.4** LSP tests cover rapid edits, invalid code/repair, dependency changes,
      missing files later created, aliases, parent imports, saves, and concurrent
      queries. Old results cannot replace current diagnostics.
- [ ] **P2.5** Receive handling stays responsive; bounded jobs/queues, cancellation,
      and shutdown release owned work. No periodic eviction task is required.
- [ ] **P2.6** Measure sustained editing, hit-only updates, map copies, CAS retries,
      retained snapshots, and memory reclamation. Record limitations of entry-count
      capacity; do not infer a hard memory bound from atomic publication.

## Phase 3: HTTP server and thin client

### Deliverable

Introduce `resin-client`, `resin-server`, and `resin-protocol` under `crates/`.
The root executable delegates to the merged client. Move compilation and active
per-pass caches from local LSP/application code into explicit server handlers.

### 3.1 Connection and source submission

- Require a server URL in `RESIN_SERVER` for build, run, and LSP. Missing/invalid
  configuration, connection failure, or incompatible protocol produces an early
  actionable error. No discovery files, automatic spawning, or local build fallback.
- Define versioned wire types for sources/import bindings, query positions/revisions,
  diagnostics, targets, header bundles, and artifact responses. Share no compiler
  IR types across this boundary.
- Each request selects one entry source and exact immutable inputs. Keep shared
  caches across callers; no create-workspace API or mandatory long-lived session.
  Revision tokens identify editor requests, not semantic cache entries.
- Client acquisition uses the full CST parser:
  1. Read the entry, with open-buffer precedence for LSP.
  2. Query imports/extern headers and recursively read/parse reachable user sources.
     Deduplicate aliases and terminate cycles; skip server-owned library/dependency roots.
  3. Capture source text once per request and retain its local-to-logical mapping.
     Observed relevant edits during acquisition trigger a successor request, rather
     than combining buffer generations. Do not claim an atomic filesystem snapshot.
  4. POST the complete captured user graph and required header bundles. The server
     parses/validates it, resolves pinned dependencies and standard/builtin libraries,
     and freezes the full graph before semantic passes.
- Client and server use the same grammar/query contract. Do not require a sequence
  of server round trips asking the client for each newly discovered import.
  Missing/unreadable imports yield diagnostics and can be repaired on a later request;
  malformed editor code may still return partial analysis.
- Server source loading is snapshot-only. Supplied client names are never opened on
  the server, even when a matching local path happens to exist. Application acquisition
  may fetch server-managed libraries/dependencies and represent them as Sources.
- Support deriving an input selection from explicit prior inputs plus changed or
  deleted files. Re-resolve imports and canonicalize complete inputs before lookup.
  Full-file replacement is sufficient initially. Missing references after eviction
  or restart request a complete resubmission, not loss of editor state.

### 3.2 Upload local C header directory bundles

An extern block names direct native headers; it need not enumerate their transitive
includes. Use complete user header directories/include roots as snapshot bundles,
and run the C preprocessor on the server.

- Resolve user headers relative to their declaring Resin source and explicit client
  include roots. Default to bundling a direct local header's containing directory;
  include all files and nested directories needed for preprocessing, including
  non-`.h` files. Do not attempt a regex `#include` closure.
- Allow explicit include roots in client/build settings, with no project manifest.
  Preserve their relative layout and ordered search bindings; coalesce overlapping
  directory uploads. Include layouts reaching outside the default directory require
  explicit wider/additional roots, rather than guessing an arbitrary ancestor tree.
- Distinguish uploaded user-header bindings from server-provided runtime, target
  system, and configured dependency headers. Prefer explicitly supplied local roots/
  local header matches, then configured server headers; report unresolved headers.
  Document the order, and test name collisions. Never resolve a client absolute path
  against the server filesystem; map supported local absolute paths into bundles.
- Preserve the declaring module's header binding even when two directories contain
  the same basename. Stage bundles into owned locations with validated relative paths,
  rewrite native include bindings as needed, and preserve nested C include semantics.
- Server-side preprocessing uses the selected target toolchain. Missing transitive
  includes outside the supplied bundles/server roots produce a useful diagnostic.
  The client needs neither a C compiler nor a preprocessor.
- Key native artifacts by all bundle file names/contents, declared header bindings,
  ordered include roots, runtime/dependency identities, and toolchain/build settings.
  Conservative whole-bundle invalidation is acceptable. Additions, changes, and
  deletions must invalidate affected artifacts even when Resin text is unchanged.

### 3.3 Build contract, output, and deployment

- Requests explicitly select OS, architecture, ABI/runtime needs, entry targets,
  profile, and relevant build settings. Advertise server capabilities and reject
  unsupported targets; remote compilation alone does not provide cross-compilation.
- Server handlers call compiler passes, publish caches, and invoke code generation,
  Ninja, C compilers, and SPIR-V tools. Keep pass order visible in build/query handlers.
  Preserve the existing toolchain library for native operations, without adding
  a reusable compiler orchestration wrapper.
- Resolve the native embedding helper explicitly when running as `resin-server`;
  do not assume its executable supports the root CLI's existing `--embed` mode.
- Return diagnostics, artifact metadata, and one output file over HTTP initially.
  Downloads retain artifact ownership and publish locally only after completion;
  failed downloads cannot replace a valid output.
- Run downloaded executables locally with local arguments, environment, working
  directory, and streams. Preserve debug build-and-run without `-o`, and optimized
  output without execution with `-o`. Execution arguments never enter build keys.
- Use the same explicitly launched service for development/deployment. Choose and
  document the HTTP framework in this phase; Rocket is a candidate. Document local
  startup, manual Docker/systemd setup, user/service-account operation, and explicit
  native/dependency settings. No installer is required.

### Phase 3 acceptance criteria

- [ ] **P3.1** Root `resin` is a thin wrapper. All other main workspace crates live
      under `crates/`. Client has syntax dependencies only; semantic/native work
      runs on the server. Compiler/cache/toolchain crates have no application/protocol
      dependencies, and handlers explicitly sequence passes.
- [ ] **P3.2** Build/run/LSP fail early without usable `RESIN_SERVER`, negotiate
      compatibility, and reject unsupported targets before native compilation.
- [ ] **P3.3** Entry, nested/parent imports, cycles, aliases, missing imports, and
      invalid editor bodies are handled through CST-based acquisition without a
      manifest or second parser. Server tests prove user files come from uploads.
- [ ] **P3.4** Two clients at different checkout roots share equivalent results;
      branches and dirty buffers remain distinct without leaking paths/identities.
      Dependencies changed during upload cannot mix editor revisions.
- [ ] **P3.5** Analyze an LSP edit, save it, then build from an independent CLI client:
      identical captured sources/import bindings hit the server's Source/CST/AST/HIR
      caches. Assert pass invocation counts or hit counters, not merely equal output.
      Native passes may run if LSP did not request them. Unsaved/disk differences
      and changed imported files miss only affected results.
- [ ] **P3.6** Native builds use uploaded directory bundles with nested/non-`.h`
      includes, duplicate basenames, explicit roots, and runtime/system headers.
      Test bundle edits/additions/deletions, relocation, missing transitive headers,
      precedence, and isolation from coincidentally existing server files.
- [ ] **P3.7** Phase 2 concurrency/eviction/ownership cases pass through HTTP.
      Eviction/restart recovers by resubmission; cancellation/disconnection releases
      owned request/native work, and eviction cannot invalidate active downloads.
- [ ] **P3.8** Artifacts are placed/run locally with existing defaults and arguments.
      Failed downloads preserve prior outputs; the server embedding helper works.
- [ ] **P3.9** Document wire/configuration contracts, capacities and overflow warnings,
      target support, dependency/header acquisition, required native tools, and
      manual local/Docker/systemd startup.

## Execution and follow-up scope

Implement each phase in reviewable changes, add meaningful tests for its acceptance
criteria, and record validation before checking a box. Use the repository development
environment and required parser/compiler/LSP/native checks. Update affected architecture
instructions alongside each implementation phase. Follow worktree/PR requirements.
Documentation validation alone completes none of the implementation phases.

Exact API field names, executor/framework choices, concurrency limits, per-layer
capacity defaults, and dependency configuration are implementation decisions in their
own phase. Select and document them there while preserving these contracts; do not
leave required behavior for an unspecified later phase.

Later work includes MCP documentation/diagnostic/symbol queries, REPL/shell sessions,
additional watching clients, archive/multiple-file artifacts, transport deduplication,
finer semantic reuse, and installers. Folder/package cache partitioning may reuse the
same immutable Cache operations without changing compiler contracts or introducing
a workspace protocol. Before exposing a service to mutually untrusted users, define
its authentication, artifact access, and native execution isolation.
