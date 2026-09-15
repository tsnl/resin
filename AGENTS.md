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
  Source scopes, inference variables, and recovery belong to HIR construction. Completed
  nominal declarations retain method identities for dependent lookup during specialization.
  HIR construction establishes definite initialization while completing each body,
  including unused definitions. Follow runtime evaluation order and intersect branch
  states there; LIR storage lowering has no source initialization states or branch snapshots.
  LIR describes storage, cleanup, stack operations, and structured control-flow regions.
  If/Loop children form a tree; retain this structure through C and SPIR-V emission.
  Selection arms end in Merge, loop conditions in LoopTest, and loop bodies in Continue;
  keep these distinct and reject exits that do not match their region.
  Keep LIR verification in private `crates/resin-lir/src/verify/` modules, with the
  verification API beside the language in `lib.rs`; constructing LIR does not verify it.
- HIR owns type expressions and completed nominal declarations, including method identities;
  it has no concrete type interner. LIR construction creates the concrete catalog and
  translates each HIR function into a private concrete expression tree before assigning
  storage. Discard that concrete tree after lowering the function.
- HIR retains numeric text with its determined type, type-only layout queries, member
  names, and explicit conversion relations. Select numeric representations, field indices,
  and conversion operations during LIR specialization. `Type::Member` determines a field
  type from its substituted receiver; it does not infer the receiver from a desired field
  type. `Type::Method` retains a receiver namespace and completed method arguments;
  `FunctionParameter` and `FunctionResult` project its caller-facing signature. Dependent
  calls and references select source nominal methods during specialization, without
  inferring receivers or method arguments. Keep literal parsing and concrete conversion rules in `resin-types`; LIR never
  chooses numeric defaults. Source-known failures are still diagnosed during HIR construction.
- HIR function signatures retain named type binders; function references retain completed
  type arguments. LIR keys instances by definition, normalized arguments, and Host/Shader
  profile; each profile counts toward the original function's allowance. LIR reserves IDs before
  translating bodies. Calls, shader references, native bridges, and drop hooks use those IDs.
  Keep substitution and concrete builtin selection in the incoming concrete-body translation;
  storage lowering receives no bound parameters. The default per-function allowance is 16,384,
  configurable through immutable `FrontendConfig` and LIR `LoweringOptions`. Repeated requests
  cost nothing. Type depth/size guards and bounded application traces are separate from that cap.
- Generate selects exported host/shader entries. Canonicalize target sets for cache
  matching; lower only their transitive function/type dependencies. Shader artifacts and
  pipeline creation add shader roots while retaining host bridge functions. Concrete nominal
  discovery reserves recursive identities privately and installs real drop IDs before storage
  lowering. `Frontend::build_hir` produces HIR/editor facts without a LIR artifact. The whole-module
  `build_lir_all` helpers remain explicit operations for direct language clients and tests.
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
  in private `typer.rs`. The compiler driver sequences passes and the toolchain
  owns external processes. Printers consume their own language, without reaching upstream.
- Each compiler phase is an unpublished workspace crate under `crates/`: `resin-cst`,
  `resin-ast`, `resin-hir`, `resin-lir`, and `resin-codegen`,
  with `resin-source` for immutable sources and loading, and `resin-types` for concrete types. Directory names match Cargo package names;
  keep compilation orchestration in `resin-frontend`, external processes and build caches in
  `resin-toolchain`, source identities and import discovery in `resin-source`, and the parser in
  `tree-sitter-resin`.
  Use `publish = false` and local path dependencies. Keep dependencies acyclic and explicit;
  do not work around a boundary with public implementation modules or reverse dev-dependencies.
  Language nodes are public data. Solvers, scopes, builders, and traversal state stay private;
  expose a small set of lowering, printing, and query operations instead.
- The root manifest is both the `resin` package and the workspace. Root `src/` contains
  only the CLI; library crates live under `crates/`. One `resin` executable builds/runs
  programs, formats source, and serves LSP with `--lsp DIR`. `resin-lsp` is a library,
  so editor and compilation services ship in the same binary. Do not re-export compiler
  implementation modules through the CLI crate.
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
  source version; a changed version is a new value. Logical identities are independent
  of diagnostic names. Loaders interpret imports and return cached instances when
  unchanged; filesystem paths, standard-library discovery, and editor buffers belong
  to `resin-source` and applications. `resin-frontend` depends on the concrete
  `resin_source::Loader`; do not add loader traits or callback adapters without a
  concrete need. The compiler has no overlays or file notifications.
  Resolve the import graph on every `compile()` before reusing a result, including when
  the entry is unchanged. Preserve old sources and compilations for their consumers.
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
- Host entries take no arguments or three parameters `(int, Ptr<Ptr<ubyte>>, Ptr<Ptr<ubyte>>)` for argc/argv/envp.
  Startup inputs are deep-copied before Resin entry and borrowed until process exit; treat
  them as read-only. Keep environment lookups on the supplied snapshot, not live OS state.
  `--` separates run arguments from compiler options; execution arguments stay out of build requests.
- Source files contain declarations only; keep runtime state inside functions and pass it
  explicitly. `FILE:ENTRY` selects an exported entry (default `main`); imports never run code.
- Keep `src/main.rs` as a wrapper around `resin::cli::main`; argument-to-`Mode` dispatch
  lives in `src/cli/`. Capture process settings through the platform toolchain's
  `Environment`, then choose CLI defaults and the build profile explicitly.
  Validate file selections and output destinations in the CLI. Interpreter mode
  owns host compilation from source to executable. `resin_frontend::Frontend::build_hir`
  consumes an immutable source and a loader to produce a `FrontendOutput`. Host and shader
  entries are selected when that output builds LIR for generate. The separate
  `resin_codegen::generate` operation takes verified LIR and
  writes C, unoptimized SPIR-V, and a Ninja
  dependency graph to disk in one call. Target representations and individual emitters stay private.
  `resin-toolchain` stages that directory, configures native tools, and invokes Ninja.
  The toolchain owns native command rules. The graph optimizes SPIR-V with `spirv-opt`,
  runs the same Resin binary with `--embed` to write
  aligned byte-array headers, then compiles C. Ninja owns ordering and incremental builds.
  Inspect cached intermediates or immutable `FrontendOutput` results; do not add CLI inspection modes.
  Toolchain APIs consume explicit settings; execution is separate and retains the build-cache lock.
  Capture the Resin executable with `std::env::current_exe()` rather than resolving it on
  PATH. Do not run a blanket native-tool preflight: report failures when a build needs the
  tool. Document Ninja, a C compiler (`CC`/`--cc`), and `spirv-opt` (`SPIRV_OPT`/`--spirv-opt`) as installation
  requirements; `NINJA` selects the build runner. Embedding preserves arbitrary bytes and
  their exact logical length, including empty inputs, without appending a NUL.
- Without `-o`, host compilation uses the debug cache and runs the program. With `-o`,
  build and copy the optimized executable without running it.
- Functions take a parenthesized sequence of arguments. Calls preserve callee-first,
  left-to-right evaluation; tuples are ordinary single values, never argument packs.
  Tuple members use decimal field indices (`pair.0`, `pair.1`).
  Every LIR function's first `parameter_count` locals are its initialized parameters,
  including foreign declarations. Zero-argument functions reserve no parameter local;
  the verifier rejects parameter counts larger than the local array.
- Functions use `def`, nominal records use `struct`, transparent aliases use `type`, and local value bindings use `var`, including
  uninitialized locals. Record initializers and parameters do not take these keywords. Foreign functions use
  `extern "header.h" def name(...) -> Type;`.
- Methods are declared inside their owning `struct`, after its fields. Aliases inherit
  the target namespace and cannot add methods. Local structs are field-only.
  `value.method(args)` supplies the receiver as the first argument,
  while `(value.field)(args)` calls a field value. Receiver parameter names are ordinary
  identifiers. Resolve source-known methods into ordinary HIR calls; dependent methods
  become ordinary calls during LIR specialization, before storage lowering. Compiler-provided methods use the
  same declaration lookup, argument checking, and editor analysis as source methods;
  register their signatures and intrinsic operations in `crates/resin-hir/src/lower/context.rs`.
  HIR construction recognizes `drop` as a hook; direct calls remain ordinary calls.
- Libraries declare low-level compiler operations with `intrinsic "operation" def name<T>(...) -> Type;`.
  Validate each signature against an explicit primitive contract during HIR construction.
  Intrinsic functions use ordinary module lookup and generic calls; retain their source
  identity. Specialize operations before storage lowering and verify concrete operands
  independently. Do not recognize library wrappers by their public type names.
- Reading existing values performs compiler-defined copying. Function and type
  applications consume their argument results; operators do the same, and aggregate
  constructors consume their field initializers. Structs define inherent methods and
  `drop(self: Ptr<T>)` hooks. There is no static move checking or borrow checker.
  Native wrappers must make their copying safe or expose ArcPtr-based ownership;
  `pointer.replace(replacement)` can disarm a native owner during deliberate transfer.
  See `doc/lifetimes.md` for lifecycle rules.
- Pointer families distinguish one value from a sequence: `Ptr<T>` / `Span<T>` are
  borrowed, `ArcPtr<T>` / `ArcSpan<T>` retain host ownership, `WeakPtr<T>` /
  `WeakSpan<T>` observe host ownership, and `GpuPtr<T>` / `GpuSpan<T>` retain GPU
  ownership. All are ordinary value types; there are no unsized payload types.
  `ArcPtr<Span<T>>` owns a descriptor, while `ArcSpan<T>` owns its elements.
  `ArcSpan<T>.alloc(count, initial)` returns `Result<ArcSpan<T>, OutOfMemory>`, checks
  allocation arithmetic, and initializes every element using ordinary copying.
  Final release destroys elements in reverse order. `get()` borrows a `Span<T>`;
  `ArcPtr<T>.alloc(initial)` allocates one initialized value. Both are ordinary
  source methods from `$/shared.resin`, backed by non-generic `StrongOwner` and
  `WeakOwner` primitives. Initialize native handles inside an inert shared payload.
  Borrowed views do not retain the owner. Numeric spans expose exact element bytes
  through `as_bytes()`. Image pixel writes accept bounded `Span<ubyte>` views.
- `Span`, shared/weak owners, GPU views, typed pipelines, and `String` are source structs.
  Keep only non-generic `StrongOwner`, `WeakOwner`, `GpuView`, and `GpuPipelineContract`
  handles in the compiler. Explicit intrinsic declarations register GPU wrapper projections;
  dispatch/draw consume completed projection plans checked again by the verifier.
  Pipeline tokens bind root, owner, and stage; validate them before projecting arguments.
  Host GPU access uses `load`, `store`, and `replace`, with no raw host pointer escape.
  `gpu.create(initial)`, `gpu.alloc::<T>(count)`, and `gpu.alloc_in::<T>(count, memory)`
  are ordinary generic source methods. Shader-declaration factories keep explicit native bridges.
- String literals have primitive type `str`, distinct from `Span<ubyte>` and the owned
  nominal `String`. They expose `data` and `length` over static NUL-terminated bytes;
  length excludes the appended terminator. Literal storage may be shared; treat it as read-only.
  `bytes(text)` from `$/span.resin` explicitly borrows literal bytes; never implicitly convert a `str`
  to a span or construct a `str` from arbitrary bytes. `String.from_str(text)` copies a
  `str`, and `String.from_bytes(bytes)` copies a raw byte span. Both append a NUL outside
  their logical length. `fmt(format, arguments)` returns `String`, wrapping
  `ArcSpan<ubyte>`; import `$/string.resin` for `String`, `fmt`, and `print`.
  Formatting and reference counting are host-only. Format tuple arguments use
  `value.bytes()` for source String and span wrappers; the primitive accepts an
  explicit structural byte view and does not recognize nominal wrapper names.
  `print(text)` and the ordinary `Io.stdout().write(text)` / `Io.stderr().write(text)` methods
  accept `str`, `Span<ubyte>`, and `String` and write bytes verbatim. Use `.data` when
  passing literal storage to C. Ordinary byte arrays contain exactly their declared
  elements, without a sentinel; nested array stride follows the packed shared layout.
  Empty C byte arrays reserve a placeholder byte that is outside the logical array.
  Device-backed byte arrays and spans use 8-bit storage; shader literals need an addressable
  constant-storage implementation and are currently rejected explicitly.
- Numeric suffixes are case insensitive: `b/h/i/l` select signed 8/16/32/64-bit integers,
  `ub/uh/ui/ul` select unsigned widths, and `f/d` select float32/float64. The formatter
  emits lowercase suffixes preceded by an underscore. Suffixes fix literal types and retain
  range checking. Hex accepts integer suffixes; signed `b` requires an underscore so bare
  `b/B` remains a digit, and hex `d/D/f/F` always remain digits. Unsuffixed numeric literals
  infer their type from context, defaulting to `long` for integers and `float64` for floats.
  One-armed `if` is equivalent to an explicit `else {}` and requires a unit-valued body.
- Function result annotations default to unit when omitted, including foreign functions.
  Named function parameters (`def identity<T>(value: T) -> T`) bind rigid type variables.
  Each declaration reference deduces fresh type arguments from operands and expected results,
  or accepts explicit `identity::<int>` arguments. Locals remain monomorphic. `_` introduces
  a weak monomorphic inference variable in local annotations, function results, and explicit
  applications, including nested positions; it may unify with a bound type variable. A caller
  cannot determine a definition's unresolved result hole. Keep parameters, type definitions,
  and foreign signatures fully explicit. Transparent aliases can bind named parameters, for
  example `type View<T> = Ptr<T>`. Their completed RHS and binders stay on source scopes;
  applications expand structurally without creating nominal types or method namespaces.
  Local aliases may capture enclosing type binders. Aliases resolve in declaration order;
  reject recursive expansion, and bound its depth and size independently of function instances.
  Structs may bind named type parameters; their fields retain those parameters in
  HIR and constructors use explicit applications such as `Cell<int> { value = 1 }`.
  Local field-only structs capture enclosing type binders in their nominal identity.
  Methods inherit their owner's type binders and may add their own named parameters.
  Method turbofish arguments supply only those additional parameters; owner arguments
  come from the receiver or applied type. An associated method reference includes its
  receiver parameter, with no implicit bound closure. Drop hooks bind only owner parameters.
  A bound receiver whose nominal origin is unknown retains dependent field/method lookup.
  Such method applications require explicit additional type arguments; a missing turbofish
  supplies none. This lookup selects source-declared nominal methods. Signature queries
  determine argument and result types without adding inference to LIR.
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
  Compute entries take `(ulong, Ptr<T>)` and return unit; their index is the global X invocation
  index. Their signatures are checked at declaration; helpers need no decoration and remain host-callable.
  `function.spirv` requests an embedded structural `{ data: Ptr<ubyte>, length: ulong }`
  view from a decorated declaration, never
  from a runtime function alias. Keep shader definitions inline in examples.
- Arrays, `Span<T>`, and `str` provide indexing with `items.at(index)`, returning `Ptr<T>`
  (`Ptr<ubyte>` for `str`);
  its index parameter is `ulong`, with explicit conversions for other integer types.
  Use `items.at(index).*` to read or write. Arrays retain the earlier `items(index)` spelling;
  source spans use `.at(index)`.
  Bounds checking is not part of the indexing contract. Host indexing diagnoses invalid indices;
  shader indexing is unchecked, and callers must stay within valid storage.
  `Place<T>` is a compiler expression category, not a source type; pointer-returning user
  functions support the same access rules. Spans have `data` and `length` fields. Pointer arithmetic
  is forbidden; explicit pointer/`ulong` casts permit low-level byte arithmetic. The C ABI retains
  pointer/length pairs; language-facing pipeline creation accepts spans.
- Unions contain value types and use module-wide u32 type IDs, not variant positions.
  Type IDs index one canonical table of nominal, primitive, and structural definitions;
  host and shader emission share it. Union tags are the active payload's table index.
  `None` is a builtin singleton type and value; `T | None` expresses optionality. Postfix `!`
  removes `None` or traps, preserving the other members; it does not unwrap Results.
  `Result<T, E>` is first-class; `ok`/`err` construct it, exhaustive `match` handles it, and postfix
  `?` returns early on error. Error holes collect the least union of propagated errors (`Never`
  if empty). Keep mutable pointers invariant; implicit widening only copies union/Result values.
- Initialized owners are destroyed in reverse scope order on normal exit and `?`;
  preserve returned values before cleanup. Chain expressions with no tail yield unit.
  Standard-library wrappers return Results and keep integer-status C declarations private;
  public operations use static and instance methods on resource types. `RuntimeError` is a union of named status errors.
  Standard-library resource handles now retain shared owners and clean up automatically;
  do not register manual native destruction for them. `commands.submit()` and `commands.cancel()`
  clear the shared native handle; presentation returns `ok(false)` for skipped frames.
  Reference counting and custom destruction are host-only; shader consumption of
  managed values is rejected.
- Always commit and push completed changes to a task branch and open a pull request,
  including in future sessions. Do not push changes directly to `main`.
  Preserve unrelated local changes.
- Use the development environment in `shell.nix` on Linux/macOS for builds, tests, parser generation, and
  examples. Enter it with `nix-shell` from the repository root, or run a command non-interactively
  with `nix-shell --run 'cargo test --workspace --all-features'`. On Windows, use a Visual Studio
  developer PowerShell with Rustup, LLVM Clang, and CMake on PATH, as described in `doc/guide.md`.
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
