# `AGENTS.md`

Resin is a simple systems programming language targeting both host (CPU) and device (GPU). 
Think CUDA, but lowering to Vulkan and exposing fixed-function rendering functionality.

## Architecture and style

- Make the compiler educational to read. Prefer explicit data and direct control flow;
  a reader should be able to tell what a pass consumes, produces, and computes locally.
- Follow one direction: syntax → AST → HIR → LIR → verified LIR → C/SPIR-V → native tools.
  Put each phase's public language definitions and operations in `lib.rs`. Keep incoming
  translation in private `lower` modules and textual rendering in private `print` modules.
  A crate's complete public contract should be discoverable from its entry point.
- HIR is a typed, desugared tree with resolved bindings and explicit operation relations.
  Source scopes, inference variables, and recovery belong to HIR construction. Dependent
  operations retain their lexical overload candidate identities during specialization.
  HIR construction establishes definite initialization and checked moves while completing each body,
  including unused definitions. Follow runtime evaluation order and intersect branch
  states there; LIR storage lowering has no source initialization states or branch snapshots.
  LIR describes storage, cleanup, stack operations, and structured control-flow regions.
  If/Loop children form a tree; retain this structure through C and SPIR-V emission.
  Selection arms end in Merge, loop conditions in LoopTest, and loop bodies in Continue;
  keep these distinct and reject exits that do not match their region.
  Keep LIR verification in private `crates/resin-lir/src/verify/` modules, with the
  verification API beside the language in `lib.rs`; constructing LIR does not verify it.
- HIR owns type expressions and completed nominal declarations, including destruction and representation hooks;
  it has no concrete type interner. LIR construction creates the concrete catalog and
  translates each HIR function into a private concrete expression tree before assigning
  storage. Discard that concrete tree after lowering the function.
- HIR retains numeric text with its determined type, type-only layout queries, member
  names, and explicit conversion relations. Select numeric representations, field indices,
  and conversion operations during LIR specialization. `Type::Member` determines a field
  type from its substituted receiver; it does not infer the receiver from a desired field
  type. `Type::Operation` retains lexical candidates, completed arguments, and operand
  types; `FunctionParameter` and `FunctionResult` project its signature. Specialization
  substitutes these relations and selects one applicable signature without consulting
  source scopes or treating body failures as substitution failures. Keep literal parsing
  and concrete conversion rules in `resin-types`; LIR never
  chooses numeric defaults. Source-known failures are still diagnosed during HIR construction.
- HIR function signatures retain named type binders; function references retain completed
  type arguments. LIR keys instances by definition, normalized arguments, and Host/Shader
  profile; each profile counts toward the original function's allowance. LIR reserves IDs before
  translating bodies. Calls, shader references, native bridges, and drop hooks use those IDs.
  Keep substitution and concrete builtin selection in the incoming concrete-body translation;
  storage lowering receives no bound parameters. The default per-function allowance is 16,384,
  configurable through LIR `LoweringOptions`. Repeated requests
  cost nothing. Type depth/size guards and bounded application traces are separate from that cap.
- Generate selects exported host/shader entries. Canonicalize target sets for cache
  matching; lower only their transitive function/type dependencies. Shader artifacts and
  pipeline creation add shader roots while retaining host bridge functions. Concrete nominal
  discovery reserves recursive identities privately and installs real drop IDs before storage
  lowering. `Hir::build` produces HIR/editor facts without a LIR artifact. `resin_lir::build_lir`
  with an empty entry list requests every ordinary root for direct language clients and tests.
- Specialization selects concrete shader operations before storage lowering. Preserve
  place access separately from value reads: an opaque managed field may be addressed,
  while reading, copying, replacing, or destroying its value is host-only. LIR construction
  completes shader call graphs iteratively and rejects recursion/foreign calls; verification
  certifies the same rules and retains shader dependency order for code generation.
  Keep concrete shader type/operation rules in `resin-types`, with instruction and graph
  rules private to LIR. SPIR-V owns representation restrictions such as local-address escape.
- Keep resolved types and layout rules in `crates/resin-types`; they must not depend on a
  frontend, backend, or verifier. Keep the public contract in `lib.rs`, with substantial
  representation algorithms in private `types.rs` and concrete checking/conversion rules
  in private `typer.rs`. Server build/analyze handlers explicitly sequence passes and the toolchain
  owns external processes. Printers consume their own language, without reaching upstream.
- Each compiler phase is an unpublished workspace crate under `crates/`: `resin-cst`,
  `resin-ast`, `resin-hir`, `resin-lir`, and `resin-codegen`,
  with `resin-source` for immutable sources and loading, and `resin-types` for concrete types. Directory names match Cargo package names;
  keep external processes and build caches in
  `resin-toolchain`, source identities and file acquisition in `resin-source`, syntax-only
  preamble discovery in `resin-cst`, and the parser in
  `tree-sitter-resin`.
  Use `publish = false` and local path dependencies. Keep dependencies acyclic and explicit;
  do not work around a boundary with public implementation modules or reverse dev-dependencies.
  Language nodes are public data. Solvers, scopes, builders, and traversal state stay private;
  expose a small set of lowering, printing, and query operations instead.
- The root manifest is both the `resin` package and workspace; root `src/main.rs`
  calls `resin_client::main()` and has no library/compiler implementation. All other
  crates live under `crates/`. `resin-client` merges CLI, local execution, syntax
  acquisition, formatting, and LSP revision handling. It depends only on syntax/source/
  executor/protocol libraries, never AST/HIR/LIR/codegen/toolchain/runtime.
  `resin-server` owns HTTP handlers, compiler pass order, shared cache heads, dependencies,
  and native tools. `resin-protocol` contains only strict wire data and constants.
  Compiler/cache/toolchain crates never depend on client/server/protocol. Do not add
  reusable orchestration wrappers or a separate LSP crate.
- Keep the complete Tree-sitter package together. Editor integrations live under
  `editors/`; the Zed WASI extension has its own Cargo workspace in `editors/zed`.
  Crates own their isolated tests; root `tests/` exercises the complete executable and
  cross-crate behavior. Shared examples and documentation stay at the root. Resin
  libraries live directly under `resin/`, organized into modules or subdirectories.
  Keep the root package as the default member for root Cargo commands.
- Keep `resin-source` and `resin-types` independent of compiler phases and of each other.
  Define phase-specific errors in their producing crate; source locations and rendered
  source diagnostics are shared transport, not a common compiler-error enum.
  Keep `resin-common` minimal: it owns the shared `define_id!` macro, imported directly
  by consumers instead of re-exported through `resin-types`. Use `tempfile` for temporary
  files. Do not move domain types, source diagnostics, or phase state into `resin-common`
  to break a dependency cycle.
  C has a private target language, lowering, and printing inside `codegen`; SPIR-V lowering
  emits binary instructions directly through `rspirv`.
- Use canonical crate and module names; do not rename dependencies or language types
  for brevity. Prefer a qualified name when two phases use the same type name.
  Import needed vocabulary privately with `use resin_source::prelude::*;` and
  `use resin_types::prelude::*;`; do not
  re-export another crate's types merely because they appear in public signatures.
- Design interfaces for information hiding, not just public/private visibility. A caller
  should understand what an operation accepts, returns, and guarantees without learning
  how its module stores data or performs the work. Expose domain concepts and completed
  results; hide caches, builders, solvers, bookkeeping, and incidental dependencies.
- Hide complexity honestly: an abstraction may do substantial work behind a simple
  interface, but its descriptive names, types, and ownership must make that behavior
  predictable. Do not depend on callers reading documentation to discover intended
  usage, hidden side effects, or surprising restrictions. Make mistakes impossible
  by construction where practical, and make correct usage the natural path.
  Judge boundaries by whether they reduce cognitive load and help readers find their
  bearings, not by how many wrappers or private modules they introduce. Documentation
  explains rationale and detail; it must not repair a misleading interface.
- Keep each crate's complete public interface in `lib.rs`. Keep the interface small;
  the file itself may be large. Review the interface in isolation: names should explain responsibilities, and signatures
  should show the direction of data flow. Do not merely re-export implementation modules
  or add forwarding types that expose the same internals under another name.
- Prefer large, cohesive single-file modules when they make the interface and its
  implementation easier to follow. Put private state directly on the public type in
  `lib.rs`; avoid forwarding twins such as `Compilation { data: Data }` or a `Cache`
  wrapper around a compiler's own fields. Crate boundaries enforce information hiding;
  a separate file does not improve an abstraction by itself. Extract a private module
  only for substantial, coherent work, not to shorten a file or scatter one type's methods.
  Delineate major API sections with three-line comment blocks (`//`, a title, `//`).
- Prefer struct-style enum variants with descriptive named fields over tuple payloads,
  including single-field variants such as `Pack { args }`. Names should explain the
  payload at construction and at pattern matching sites.
- Language trees may expose their data directly; service objects and retained results
  should protect their invariants. Require the least information an operation needs.
  Code generation consumes verified LIR; the toolchain consumes an on-disk Ninja project
  and explicit build settings. Neither requires mutable compiler caches or rereads Resin sources.
- Treat `Source` as immutable named text, never implicitly as a file. Clones share a
  source version; changed text is a new value. `SourceId` is reconstructible logical
  identity. `Source::new` uses its name as identity; use `with_identity` when diagnostic
  names differ or repeat. Equality checks identity, display name, and exact text after
  digest comparison; never collapse equal text belonging to distinct modules.
  Applications acquire sources and resolve imports before every semantic cache lookup,
  then freeze versions and bindings in a `SourceGraph`. `resin_ast::build_program`
  receives that graph and completed file ASTs, never a loader. Keep filesystem/library
  lookup in `resin-source` and application acquisition; compiler passes have no overlays
  or notifications. Preserve earlier sources and results for their consumers.
- Primary compiler and native operations are async and take explicit `Execution` and
  `Cancellation`. Keep `resin-executor` limited to bounded work/cancellation and
  `resin-cache` to immutable capacity-bounded completed-value snapshots. Applications
  sequence passes directly; do not add a reusable compiler orchestration wrapper.
  Execution defaults to available logical CPUs, with a fallback of one. Each native
  build reserves one slot and invokes Ninja with `-j 1`. Started synchronous foreign
  calls retain their slot until they finish; queued work observes cancellation.
- Applications publish immutable cache generations through safe shared-pointer CAS.
  On a lost race, rebase every requested hit/miss handle on the current head with the
  same update/eviction operation; never repeat compiler work or replay unrelated old
  entries. Keep selected handles for downstream work and release whole cache snapshots.
  LSP acquisition uses request-owned loaders seeded with current supplied registrations.
  Preserve an open document's physical identity through edits and renew it on reopen.
  Bound admitted requests through response consumption, coalesce complete editor states,
  and gate result publication against accepted document/dependency revisions.
  Cancel and drain work on shutdown before releasing its ownership.
- Prefer small functions with descriptive names, ideally fewer than ten lines of logic.
  Split by a meaningful operation, not an arbitrary line count. Exhaustive language
  dispatch and simple data definitions may be longer when that keeps the cases together.
- Use concrete representations and a few explicit passes rather than callback frameworks,
  hidden cross-pass state, or speculative abstractions. Explain invariants and non-obvious
  ownership choices close to the code that needs them.
- Use [the Bitwise taste guide](doc/bitwise.md) when designing and refactoring code.
  It explains the motivation through concrete Ion examples: visible data, operations
  that establish complete guarantees, and control flow that teaches the problem.
  Adapt that taste to Resin's behavior, explicit phase boundaries, and idiomatic Rust.

### Nanopass design

- Follow the [Nanopass](https://docs.racket-lang.org/nanopass/index.html) approach by
  convention: explicit languages and translations whose outputs establish useful
  guarantees. Build on the existing phase boundaries and enum trees. A pass's signature
  should identify its inputs, completed outputs, and diagnostics.
- Write translations with ordinary Rust functions or methods and exhaustive matches.
  Keep simple cases inline and extract substantial cases into descriptive helpers.
  Each translation chooses its own context arguments and result types, including extra
  computed facts. Introduce visitor traits, generated traversal, or tree macros only
  when concrete reuse justifies them; adopting this style does not require changing
  every enum variant into a separate struct or changing its allocation strategy.
- Keep recursion under the translation's control. Scope extension, evaluation order,
  short-circuiting, and cleanup may require different treatment of children. Use shared
  walkers when traversal requirements actually agree. Dependency groups, constraint
  solving, and fixed-point worklists remain explicit algorithms inside their owning pass.
- Private mutable state is compatible with a translation that returns completed data.
  Give state the lifetime of the work it serves: per-function LIR lowering owns its
  builder, bindings, cleanup scopes, and source tracking, and returns a function and
  its origins to module assembly. Preserve source origins independently of any assumed
  one-to-one correspondence between function IDs in different languages.
- Preserve resolved decisions when they become available: declaration identities,
  chosen operations, field access, conversions, and declaration metadata. Completed
  outputs must carry the data needed by the next translation, without repeating source
  lookup or consulting mutable construction state. Keep inference and recovery inside
  HIR construction; editor facts must remain useful even when executable HIR cannot
  be completed.
- Add a private intermediate language when its distinct forms or guarantees simplify
  a meaningful translation. Prefer adapting existing data or constructing HIR directly
  when that establishes the same contract. Internal algorithm steps need not each
  become a separate language, public pass, or crate. Spend refactor churn on stronger
  data contracts and bounded ownership of state; retain existing representations and
  direct translations where they already express those contracts clearly.

## Cost control

- Run automatic CI for pull requests and pushes on Linux only. Keep Windows and macOS
  support and tests; reserve their hosted runners for manually requested checks.
- Before publishing a release, manually run the `Build` workflow against the release
  candidate ref and require Linux, Windows, and macOS jobs to pass. Use Actions > Build >
  Run workflow, or `gh workflow run build.yml --ref <release-candidate-ref>`.
- GitHub Free includes 2,000 Actions minutes per month for private repositories, shared
  across the repository owner's account. At the September 2026 standard Linux x64 rate
  of $0.006/minute, 2,000 Linux minutes have about $12 of compute value.
- Budget roughly 1x for Linux, 2x for Windows, and 10x for macOS. These are rounded cost
  comparisons: current standard rates are $0.006, $0.010, and $0.062/minute respectively
  (about 1x, 1.67x, and 10.33x), not exact billing multipliers. Recheck
  [GitHub Actions billing](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
  when changing CI coverage or budgets.

## Development Practices

- Target 64-bit Linux, macOS, and Windows (MSVC with LLVM Clang for emitted C).
  Keep host builds independent of a Vulkan SDK or GPU. GPU execution still requires the
  runtime's Vulkan features; MoltenVK discovery does not imply full GPU compatibility.
- Keep the native C ABI in `crates/resin-runtime/` and language-facing modules in `resin/`.
  Add other Resin libraries as modules or subdirectories under `resin/`.
  Resolve `$/` imports from that library root, configurable with `RESIN_LIBRARY_ROOT`.
  Examples import standard-library functionality through `$/` paths; imports without
  a leading `$` resolve relative to their importer. Each file has a private scope with
  explicit exports; do not reintroduce textual inclusion.
- Host entries take no arguments or three parameters `(i32, Ptr<Ptr<u8>>, Ptr<Ptr<u8>>)` for argc/argv/envp.
  Startup inputs are deep-copied before Resin entry and borrowed until process exit; treat
  them as read-only. Keep environment lookups on the supplied snapshot, not live OS state.
  `--` separates run arguments from compiler options; execution arguments stay out of build requests.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Build/run/LSP require explicit `RESIN_SERVER` and negotiate HTTP capabilities before
  source acquisition or successful LSP initialization; no autostart, discovery files,
  project manifest, or local compiler fallback. Local formatting/embedding need no server.
  The client fully parses CSTs for source/import/header acquisition and uploads immutable
  user snapshots with entry-relative logical names. `$/` libraries and pinned Git
  dependencies are frozen by the server. User source names are never opened as server paths.
  Delta handles are optional bounded transport state; cache identity excludes handles,
  revisions, local paths, and caller identity. An unavailable base retries complete inputs.
  LSP owns unsaved buffers without saving them; build requests capture disk independently.
  Managed definitions use immutable local mirror files and keep server source identities.
- Server handlers call `resin_hir::Hir::build` on assembled immutable `BuiltProgram`
  values, then select explicit host/shader entries for `resin_lir::build_lir`.
  `resin_codegen::generate` accepts verified LIR and immutable `NativeHeaders`, writes
  supplied header bundles, C, SPIR-V and Ninja edges into a unique owned temporary child.
  HIR and LIR retain source-scoped `ForeignHeader` values even for empty extern groups.
  Never bind includes by basename alone. Preserve ordered include roots and whole-bundle
  contents in native keys, and protect the compiler-injected runtime ABI include.
  `resin-toolchain` stages that directory, configures native tools, and invokes Ninja.
  The toolchain owns native command rules. The graph optimizes SPIR-V with `spirv-opt`,
  runs the configured service executable with `--embed` to write
  aligned byte-array headers, then compiles C. Ninja owns ordering and incremental builds.
  Inspect cached intermediates or immutable `Hir` results; do not add CLI inspection modes.
  Toolchain APIs consume explicit settings. Only staging/building holds the cache lock;
  retained `BuiltProject` and `Executable` handles share an immutable artifact generation
  that survives later builds and is removed when its final owner drops. Cancelled native
  work terminates its process tree before releasing staging ownership.
  Capture the Resin executable with `std::env::current_exe()` rather than resolving it on
  PATH. Do not run a blanket native-tool preflight: report failures when a build needs the
  tool. Document Ninja, a C compiler (`CC`/`--cc`), and `spirv-opt` (`SPIRV_OPT`/`--spirv-opt`) as installation
  requirements on the server; `NINJA` selects the build runner. Client `-I`/`--include-root`
  uploads complete header directory bundles. Server tool paths never come from build requests. Embedding preserves arbitrary bytes and
  their exact logical length, including empty inputs, without appending a NUL.
- Without `-o`, request a debug build, download to owned temporary storage, and run
  locally with local argv/env/cwd. With `-o`, request release and atomically publish the
  verified download without running it. Failed/cancelled downloads preserve prior output.
- Functions take a parenthesized sequence of arguments. Calls preserve callee-first,
  left-to-right evaluation; tuples are ordinary single values, never argument packs.
  Tuple members use decimal field indices (`pair.0`, `pair.1`).
  Every LIR function's first `parameter_count` locals are its initialized parameters,
  including foreign declarations. Zero-argument functions reserve no parameter local;
  the verifier rejects parameter counts larger than the local array.
- Functions use `fn`, nominal records use `struct`, transparent aliases use `type`, and local value bindings use `let`, including
  uninitialized locals. Identifier patterns are immutable unless marked `mut`, including
  parameters and match binders. Assignment uses `=` and returns unit. Struct fields use
  comma separators with an optional trailing comma; value initializers use `field = value`.
  Record initializers and parameters do not take these keywords. Records require named
  `struct` declarations; tuples provide anonymous aggregates.
  Foreign functions live in an optional top-level `extern` block between `export` and
  `import`: `extern { "header.h": { fn name(...) -> Type; }, };`. Header groups may
  be empty and retain their native include dependency. Opaque foreign types remain
  standalone `extern type Name;` declarations.
- Struct bodies contain only fields. Operations are ordinary free functions and are
  exported independently of types. `value:operation(args)` supplies the first argument;
  dots select fields, including callable field values. There is no implicit `self`.
  Resolve overloads using all arguments and expected results. Reject ambiguous signatures;
  body errors never provide overload fallback. Source-known calls become ordinary HIR calls;
  dependent calls retain lexical candidate IDs for specialization before storage lowering.
  Compiler-provided operations use the same argument checking and editor analysis path.
  HIR registers free `drop(RefMut<T>)` and `repr_bytes(Ref<T>)` hooks with their owning
  nominal declaration; direct calls remain ordinary calls.
- Libraries declare low-level compiler operations with `intrinsic "operation" fn name<T>(...) -> Type;`.
  Validate each signature against an explicit primitive contract during HIR construction.
  Intrinsic functions use ordinary module lookup and generic calls; retain their source
  identity. Specialize operations before storage lowering and verify concrete operands
  independently. Do not recognize library wrappers by their public type names.
- Structs copy by default when all stored fields copy and there is no custom drop
  hook. Noncopyable fields propagate move-only ownership through enclosing aggregates.
  Shared and weak handle copies retain ownership. `PhantomBox` in `$/ownership.resin`
  opts out through an empty drop hook. Ownership completion in HIR checks initialization,
  partial moves, immutable assignment, branch joins, and loop exits/backedges, including
  unused definitions. Moving an immutable owner is allowed. Generic value reuse infers
  copy requirements, retaining consumer types so dependent reference parameters borrow
  without imposing copying. Completed HIR retains these requirements and distinguishes
  definite moves from conditional owned reads. LIR specialization checks requirements
  and selects concrete copies/transfers without repeating flow analysis; body failures
  never supply overload fallback. Storage lowering emits transfer/cleanup operations.
  References and raw pointers retain unchecked lifetimes and aliasing. `Ref` is read-only,
  while `RefMut` permits writes without exclusivity; there is
  no borrow checker. `pointer:replace(replacement)` transfers a referent while leaving
  initialized storage. Shared owners copy by retaining their allocation; explicit `clone` operations remain available.
  `ICopy`/`IClone`, traits, effects, and general function CTFE remain deferred.
  See `doc/lifetimes.md` for lifecycle rules.
- Access permissions are explicit: `Ptr<T>` is read-only, `PtrMut<T>` is writable.
  Carry the permission through HIR, specialization, LIR verification, and projection.
  `PtrMut<T>` may weaken to `Ptr<T>` with the exact same pointee type; never strengthen
  or implicitly turn either reference kind into a pointer. Nested pointers are invariant.
  `Span`/`SpanMut` and `GpuPtr`/`GpuPtrMut`, `GpuSpan`/`GpuSpanMut` are source structs;
  use their explicit `read_only` methods. Allocation and host shared-owner `get` return
  writable views. Access is aliasable and independent of `let mut` on a binding.
  Literal bytes and process input snapshots expose read-only pointers. Host GPU element
  operations borrow their view; an offset operation does not itself retain its owner.
- Pointer families distinguish one value from a sequence: `Ptr<T>` / `Span<T>` are
  borrowed, `ArcPtr<T>` / `ArcSpan<T>` retain host ownership, `WeakPtr<T>` /
  `WeakSpan<T>` observe host ownership, and `GpuPtr<T>` / `GpuSpan<T>` retain GPU
  ownership. All are ordinary value types; there are no unsized payload types.
  `ArcPtr<Span<T>>` owns a descriptor, while `ArcSpan<T>` owns its elements.
  `arc_span_alloc::<T>(count, initial)` returns `(ArcSpan<T> | Err<OutOfMemory>)`, checks
  allocation arithmetic, and initializes every element using a copyable initializer.
  Final release destroys elements in reverse order. `get()` borrows a `SpanMut<T>`;
  `arc_ptr_alloc(initial)` moves one initialized value into its allocation. Both are ordinary
  source functions from `$/shared.resin`, backed by non-generic `StrongOwner` and
  `WeakOwner` primitives. Initialize native handles inside an inert shared payload.
  Borrowed views do not retain the owner. Numeric spans expose exact element bytes
  through `as_bytes()`. Image pixel writes accept bounded `Span<u8>` views.
- `Span`/`SpanMut`, shared/weak owners, GPU views, typed pipelines, and `String` are source structs.
  Keep only non-generic `StrongOwner`, `WeakOwner`, `GpuView`, and `GpuPipelineContract`
  handles in the compiler. Explicit intrinsic declarations register GPU wrapper projections;
  dispatch/draw consume completed projection plans checked again by the verifier.
  Pipeline tokens bind root, owner, and stage; validate them before projecting arguments.
  GPU owners expose borrowed `device()` pointers/spans and fallible `map()` host views.
  Both preserve permissions, offsets, and lengths; addresses carry no device/space tag.
  Reuse persistent coherent mappings for the allocation lifetime. The caller retains
  allocations and synchronizes raw accesses. Device-only storage cannot be host-mapped.
  GPU elements may contain plain pointers/spans; mapping never rewrites embedded addresses.
  Launch values with the exact shader root type preserve pointer bits and snapshot only
  the root; existing owner projection retains its explicit GPU owners. Keep the BDA ABI,
  without descriptor bundles, fat pointers, placed-map requirements, or implicit relocation.
  `gpu:create(initial)`, `gpu:alloc::<T>(count)`, and `gpu:alloc_in::<T>(count, memory)`
  are ordinary generic source functions. Shader-declaration factories keep explicit native bridges.
- String literals have primitive type `str`, distinct from `Span<u8>` and the owned
  nominal `String`. They expose `data` and `length` over static NUL-terminated bytes;
  length excludes the appended terminator. Literal storage may be shared; treat it as read-only.
  `bytes(text)` from `$/span.resin` explicitly borrows literal bytes; never implicitly convert a `str`
  to a span or construct a `str` from arbitrary bytes. `string_from_str(text)` copies a
  `str`, and `string_from_bytes(bytes)` copies a raw byte span. Both append a NUL outside
  their logical length. `fmt(format, arguments)` returns `String`, wrapping
  `ArcSpan<u8>`; import `$/string.resin` for `String`, `fmt`, and `repr`.
  Import `$/stdio.resin` for `print`, standard-stream writes, and byte/line input.
  Formatting and reference counting are host-only. Format tuple arguments use
  `value:bytes()` for source String and span wrappers; the primitive accepts an
  explicit `(Ptr<u8>, u64)` byte view and does not recognize nominal wrapper names.
  `print(text)` and standard-stream `write(stream, text)` operations accept `str`,
  `Span<u8>`, and `String` and write bytes verbatim. Bind `io_stdout()` or
  `io_stderr()` to a local before borrowing the stream for a write. Use `.data` when
  passing literal storage to C. Ordinary byte arrays contain exactly their declared
  elements, without a sentinel; nested array stride follows the packed shared layout.
  Empty C byte arrays reserve a placeholder byte that is outside the logical array.
  Device-backed byte arrays and spans use 8-bit storage; shader literals need an addressable
  constant-storage implementation and are currently rejected explicitly.
- Numeric primitives are `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`,
  `f32`, and `f64`. Numeric literals have no suffixes. They infer their type from
  context, defaulting to `i64` for integers and `f64` for floats. An annotation or
  explicit type application such as `f32(1.0)` selects a width; range checks still apply.
  One-armed `if` is equivalent to an explicit `else {}` and requires a unit-valued body.
- Function result annotations default to unit when omitted, including foreign functions.
  Named function parameters (`fn identity<T>(value: T) -> T`) bind rigid type variables.
  Each declaration reference deduces fresh type arguments from operands and expected results,
  or accepts explicit `identity::<i32>` arguments. Locals remain monomorphic. `_` introduces
  a weak monomorphic inference variable in local annotations, function results, and explicit
  applications, including nested positions; it may unify with a bound type variable. A caller
  cannot determine a definition's unresolved result hole. Keep parameters, type definitions,
  and foreign signatures fully explicit. Transparent aliases can bind named parameters, for
  example `type View<T> = Ptr<T>`. Their completed RHS and binders stay on source scopes;
  applications expand structurally without creating nominal types or method namespaces.
  Local aliases may capture enclosing type binders. Aliases resolve in declaration order;
  reject recursive expansion, and bound its depth and size independently of function instances.
  Structs may bind named type parameters; their fields retain those parameters in
  HIR and constructors use explicit applications such as `Cell<i32> { value = 1 }`.
  Local field-only structs capture enclosing type binders in their nominal identity.
  Free operations declare their whole type-parameter list. Colon-call turbofish arguments
  supply those same parameters; signatures may infer them from every operand and result.
  Generic drop hooks bind only owner parameters. Dependent field and operation relations
  remain explicit in HIR; signature queries determine parameter and result types before
  concrete storage lowering. Do not reopen source inference during specialization.
  Check source expressions into HIR, then lower that tree to LIR in a separate pass.
  Resolve dependency groups and all inference variables before handing the tree to lowering;
  retain named binders and determining member types in HIR. Scopes store these HIR schemes.
  HIR nominal declarations bind type parameters, and applications retain their source
  origin and arguments. LIR normalizes function and nominal instance keys before
  materializing layouts; an unused type argument must not force field or hook expansion.
  Materialized nominal instances close recursive fields and receive specialized drop hooks
  before storage lowering. Diagnostics render source type names instead of private type IDs.
  keep inference solvers and deferred emission callbacks out of the lowering pass.
- Shader entries use `@compute_shader`, `@vertex_shader`, or `@fragment_shader` decorators.
  Compute additionally permits a third `RefMut<State>`
  parameter, spelled `Workgroup<State>` through `$/workgroup.resin`. GPU wrappers
  provide zero-initialized Workgroup storage; ordinary host calls borrow a local
  and execute as one lane. Keep synchronization explicit and preserve per-thread
  entry/index/dispatch semantics. Check shared-reference origins in SPIR-V and
  reject checked-failure paths for these entries; uniform barrier participation
  and race freedom remain the programmer's responsibility. See `doc/workgroups.md`.
  Compute entries take `(u64, Ptr<T>)` or `(u64, PtrMut<T>)` and return unit; their index is the global X invocation
  index. Their signatures are checked at declaration; helpers need no decoration and remain host-callable.
  Pipeline creation accepts decorated shader declarations directly and requests their compiled
  representation internally. Shader functions have no bytecode property. Runtime shader aliases
  remain unsupported. Keep shader definitions inline in examples.
- Arrays, `Span<T>`, and `str` provide indexing with `items:at(index)`, returning `Ref<T>`
  (`Ref<u8>` for `str`);
  its index parameter is `u64`, with explicit conversions for other integer types.
  Use `items:at(index)` to read and `items:at_mut(index)` to obtain `RefMut<T>`
  for writes to writable array places or spans. String literals have no `at_mut`.
  Pointers to arrays, spans, and `str`
  also support `items:lea(index)` returning an element pointer. Local array values
  and references to arrays have no `lea` operation.
  Arrays retain the earlier `items(index)` spelling;
  source spans use `:at(index)`.
  Bounds checking is not part of the indexing contract. Host indexing diagnoses invalid indices;
  shader indexing is unchecked, and callers must stay within valid storage.
  Places remain a compiler expression category. `Ref<T>` exposes read-only access;
  `RefMut<T>` exposes writable access. Both are fixed, nonowning aliases to initialized
  storage. A writable borrow of a local or inline field requires `mut` on its binding. Unannotated locals and
  plain result holes infer value types; `let alias: Ref<T> = place;` retains the alias.
  Assignment through `RefMut` writes its referent even through an immutable reference
  binding. `RefMut` can weaken to `Ref`, never the reverse; permissions survive generic
  specialization and projections. Reading a pointer field starts that pointer's own access.
  Locals, their inline fields, and both reference kinds cannot have their addresses taken.
  A pointer dereference and its field projections remain addressable. Parameters and
  results retain their Ref/Ptr contract through generic specialization. Ref parameters require initialized places: locals, fields, pointer dereferences,
  or reference-valued results. Bind temporary values to explicit locals before borrowing. HIR retains reference use;
  specialization preserves distinct concrete Ref/Ptr types. LIR LocalRef produces a writable reference; Borrow preserves the pointer permission
  in a reference, never a pointer. ReadOnly weakens mutable references and pointers. Verification rejects reference-to-pointer
  conversions; only final target lowering chooses an address representation.
  Reject direct reference aggregate payloads, nested references, and reference-valued
  generic arguments. Shader-local references can cross helper calls using Function
  storage pointers plus projection paths; returning or merging distinct local
  references remains unsupported.
  Spans have `data` and `length` fields. Use `Span`/`SpanMut` for sequences in source
  APIs and application records; preserve lengths instead of passing naked pointers.
  Keep pointer/count and NUL-terminated forms at explicit native/compiler boundaries.
  Single-value `GpuPtr` views do not provide slicing; slice the originating `GpuSpan`.
  Direct pointer arithmetic is forbidden; use span indexing and `:lea(index)`.
  Explicit pointer/`u64` casts remain available for low-level host interop.
  Shader pointer casts (including pointer reinterpretation) are rejected during LIR construction
  and verification; use typed pointers and indexing. The current Vulkan C ABI retains
  pointer/length pairs; language-facing pipeline creation accepts shader declarations.
- Unions contain value types and use module-wide u32 type IDs, not variant positions.
  Type IDs index one canonical table of nominal, primitive, and structural definitions;
  host and shader emission share it. Union tags are the active payload's table index.
  `None` is a builtin singleton type and value; `T | None` expresses optionality. Postfix `!`
  removes `None` or traps, preserving the other members; it preserves `Err` members.
  `T | Err<E>` is an ordinary union; success values are plain and `Err(value)` wraps errors. Exhaustive `match` handles it, and postfix
  `?` returns early on error. Error holes collect the least union of propagated errors (`Never`
  if empty). Keep mutable pointers invariant; widening preserves ownership transfer for union/Err values.
- Initialized owners are destroyed in reverse scope order on normal exit and `?`;
  preserve returned values before cleanup. Chain expressions with no tail yield unit.
  Standard-library wrappers return error unions and keep integer-status C declarations private;
  public operations are exported free functions on resource types. `RuntimeError` is a union of named status errors.
  Standard-library resource handles now retain shared owners and clean up automatically;
  do not register manual native destruction for them. `commands:submit()` and `commands:cancel()`
  clear the shared native handle; presentation returns `(false)` for skipped frames.
  Reference counting and custom destruction are host-only; shader consumption of
  managed values is rejected.
- Always commit and push completed changes to a task branch and open a pull request,
  including in future sessions. Do not push changes directly to `main`.
  Preserve unrelated local changes.
- Use the development environment in `shell.nix` on Linux/macOS for builds, tests, parser generation, and
  examples. Enter it with `nix-shell` from the repository root, or run a command non-interactively
  with `nix-shell --run 'cargo test --workspace --all-features'`. On Windows, use a Visual Studio
  developer PowerShell with Rustup, LLVM Clang, and CMake on PATH, as described in `doc/getting-started.md`.
- The shell supplies Rustup, a C compiler, CMake, Ninja, GLFW's native build dependencies, SPIR-V Tools, `glslc` for handwritten test fixtures,
  and, on Linux, Vulkan tools and libraries. On macOS, GPU execution uses the Vulkan SDK's loader
  and MoltenVK. Cargo builds and statically links GLFW via `glfw-sys`. Rustup uses
  `rust-toolchain.toml`. Keep the shell's library paths; do not hardcode Nix store paths.
- Use `RESIN_REQUIRE_SPIRV_TOOLS=1 RESIN_REQUIRE_GLSLC=1 RESIN_REQUIRE_GPU=1` when validating the full GPU path so missing
  dependencies do not silently skip tests. A working Vulkan driver is still required.
- Window tests also need a display (desktop or Xvfb) and `RESIN_REQUIRE_WINDOW=1` to prevent
  skips. For Xvfb, set `DISPLAY` and `XDG_SESSION_TYPE=x11`; unsetting `WAYLAND_DISPLAY`
  alone does not prevent GLFW from finding a default Wayland socket. Test actual window
  operations in subprocesses so they run on the process main thread, as GLFW requires.
  Run a shared Xvfb test display with `-noreset` so its last client disconnecting does not
  reset the server while another window test connects.
- Install the matching parser CLI inside the shell with
  `cargo install --locked tree-sitter-cli --version 0.27.0`, then regenerate from
  `crates/tree-sitter-resin/` with `tree-sitter generate --js-runtime native`.
- The Tree-sitter grammar, generated parser, and Rust bindings live in `crates/tree-sitter-resin/`
  as ordinary files in this repository. Commit grammar changes and regenerated files together.
- Keep documentation current in the same PR as changes to language behavior, syntax,
  library APIs, tooling, or platform requirements. Update affected manual chapters,
  tutorials, library doc-comments, and executable examples together. Follow
  [Maintaining the manual](doc/manual-development.md) for documentation builds and validation.
- Documentation uses `///` / `/** ... */` for the following declaration or field,
  and leading `//!` / `/*! ... */` for a file module. CST associates Markdown with
  exact declaration-name spans; AST retains it for HIR/editor consumers. Diagnose
  misplaced comments. Documentation does not enter executable HIR/LIR. The local
  `resin --doc FILE [-o MARKDOWN]` command uses syntax-only exported signatures and
  needs no compiler service; do not add compiler-phase dependencies to the client.
