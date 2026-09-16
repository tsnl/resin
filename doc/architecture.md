# Compiler architecture

Resin's compiler is a sequence of explicit languages and passes. Each phase is an
unpublished Cargo workspace crate. Public data describes its output; a small set
of public operations constructs, prints, or queries that data. Each crate lists
its full public interface in `lib.rs`; implementation modules stay private.

The root manifest is both a package and a workspace. Root `src/main.rs` forwards
to `resin_client::main()`. The client supplies build/run commands, local formatting,
and stdio LSP. `resin-server` is a separate HTTP application. Builds and LSP require
an explicit `RESIN_SERVER` URL and capability negotiation; the client has no local
semantic compilation or native-tool dependency.

Reusable libraries live under `crates/`, with directory names matching their Cargo
package names. Each language crate owns its representation and the translation that
produces it. Applications acquire imports through the concrete `resin_source::Loader`
and pass immutable `SourceGraph` values to `resin_ast::build_program`. `resin-source` owns
immutable text and standard-library resolution; `resin-types` owns concrete types and
representation rules. Neither depends on a compiler phase. `resin-toolchain` runs
generated Ninja projects and retains native artifacts without depending on compiler
or type crates. Server analysis/build handlers each sequence their compiler passes
explicitly and share immutable cache heads. `resin-protocol` contains only wire data;
`resin-client` acquires local sources/header bundles and renders remote diagnostics
and editor queries. The native C ABI lives in `resin-runtime`.

Crates own their isolated tests; root `tests/` exercises the complete executable and
cross-crate behavior. Examples and documentation stay at the repository root;
Resin libraries live under `resin/`. The root package is the default member, so
`cargo run -- examples/eg001.resin` selects that client after `RESIN_SERVER` is set. Use `--workspace` to build or test
all native packages.

`crates/tree-sitter-resin` keeps the grammar, generated parser, queries, JavaScript
tooling, and Rust bindings together as one package. `editors/zed` is an independent
Cargo workspace for the WASI extension and its language queries. Its manifest pins
a repository commit and the grammar's path; update both when relocating the grammar.

```mermaid
flowchart LR
    text[Source text] --> cst[CST]
    cst --> ast[AST]
    ast --> hir[HIR: resolved tree]
    hir --> lir[LIR: storage and structured regions]
    lir --> verified[Verified LIR]
    verified --> project[Codegen: C + SPIR-V + build.ninja]
    project --> ninja[Ninja]
    ninja --> spirv[spirv-opt: optimized SPIR-V]
    spirv --> headers[resin-server --embed: C headers]
    headers --> executable[C compiler: executable]
```

These arrows show data flow. Cargo dependencies point toward the languages a
pass consumes. Source and type vocabulary are independent foundations.

| Crate | Direct phase dependencies | Public purpose |
| --- | --- | --- |
| `resin-common` | none | Shared `define_id!` index-type macro |
| `resin-source` | none | Reconstructible sources, frozen import graphs, async file loading and library resolution |
| `resin-executor` | none | Bounded CPU/native work and cancellation |
| `resin-cache` | none | Immutable capacity-bounded snapshots of completed values |
| `resin-types` | none | Concrete types, values, conversions, layout and shader interfaces |
| `resin-cst` | generated `tree-sitter-resin` grammar | Syntax documents, `build_cst`, queries, formatting |
| `resin-ast` | `resin-cst` | Source AST, `build_ast` / `build_program`, parse diagnostics |
| `resin-hir` | `resin-ast`, `resin-cst` | Resolved tree, `build_hir` / `Hir::build`, editor analysis |
| `resin-lir` | `resin-hir` | Storage and control-flow lowering, `build_lir`, verification |
| `resin-codegen` | `resin-lir` | Generate a complete on-disk C/SPIR-V/Ninja project |
| `resin-toolchain` | none | Async native builds and independently owned output generations |
| `resin-protocol` | none | Strict versioned request, diagnostic, query, and artifact data |
| `resin-server` | Compiler crates, `resin-cache`, `resin-toolchain` | HTTP admission, explicit cached passes, managed inputs, native output streaming |
| `resin-client` | `resin-source`, `resin-cst`, `resin-executor` | Local capture/formatting, HTTP client, CLI, stdio editor rendering and downloaded execution |

The HIR dependency on CST supports editor queries at a syntax position. Its public
language owns its type expressions and nominal declarations. LIR lowering never
receives CST, AST, mutable source scopes, or an inference solver. Codegen consumes the
LIR certificate; it cannot reach the checker's private state.

All workspace packages have `publish = false`. Path dependencies are declared
centrally in the workspace manifest. Source diagnostics belong to `resin-source`;
phase-specific errors belong to the phase that rejects the input. Temporary directories
use `tempfile`. `resin-common` contains only the shared `define_id!` macro, imported
directly by types, HIR, and LIR; it owns no domain types or diagnostics.

## A consistent reading path

For each phase, start with `lib.rs`: language data appears beside the operations
that accept the preceding language and produce this one. Follow an operation into
private `lower` or `print` only when its implementation matters. Codegen's entry
point exposes one project-generation operation. Its C tree and SPIR-V builder are private to
the target modules. Source and type entry points contain their definitions directly;
language-specific builders and solver state stay private. In `resin-types`, the
public type model and operations remain in `lib.rs`; private `types.rs` implements
representation, table, and layout algorithms, while private `typer.rs` implements
concrete checks and conversions.

Keep related state and operations together. `Hir` keeps retained products and
editor queries beside HIR construction. A cohesive file can be substantial while
exposing few public concepts. Fields belong directly to the objects whose invariants
they serve; private helpers keep individual operations readable. Native building is
a separate operation connected by the [server build handler](../crates/resin-server/src/build.rs).

Use canonical crate and language names instead of renaming imports. Shared vocabulary
comes from private `use resin_source::prelude::*;` and
`use resin_types::prelude::*;` imports where needed; operations keep qualified paths. Public signatures can name another crate's public types without re-exporting
them. Phase crates therefore expose their own language and operations, and the
compiler returns `resin_hir::Hover` and `resin_hir::Completion` directly.

| Phase | Language definition | Incoming pass | Printing |
| --- | --- | --- | --- |
| CST | [Document](../crates/resin-cst/src/lib.rs) | [incremental parsing](../crates/resin-cst/src/lower.rs) | [source formatting](../crates/resin-cst/src/print.rs) |
| AST | [source nodes](../crates/resin-ast/src/lib.rs) | [CST → AST](../crates/resin-ast/src/lower.rs) | [S-expressions](../crates/resin-ast/src/print.rs) |
| HIR | [resolved nodes](../crates/resin-hir/src/lib.rs) | [AST → HIR](../crates/resin-hir/src/lower/mod.rs) | [typed S-expressions](../crates/resin-hir/src/print.rs) |
| LIR | [instructions and blocks](../crates/resin-lir/src/lib.rs) | [HIR → LIR](../crates/resin-lir/src/lower/mod.rs) | [S-expressions](../crates/resin-lir/src/print/mod.rs) |
| C | [private C tree](../crates/resin-codegen/src/c/mod.rs) | [verified LIR → C](../crates/resin-codegen/src/c/lower/mod.rs) | [C text](../crates/resin-codegen/src/c/print.rs) |
| SPIR-V | Private `rspirv` module | [verified LIR → SPIR-V](../crates/resin-codegen/src/spirv/mod.rs) | Binary assembly; inspect with `spirv-dis` |

Prefer small functions named for the operation they perform. The
[Bitwise taste guide](bitwise.md) explains the style reference through concrete
examples of visible data, direct construction, and code that teaches its implementation.
Resin uses idiomatic Rust and meaningful pass boundaries; an exhaustive language
dispatch may remain long when splitting it would obscure the cases.

## Library primitives

Libraries declare compiler operations through explicit intrinsic signatures:

```resin
intrinsic "pointer_index" def pointer_at<T>(
    data: Ptr<T>, length: ulong, index: ulong
) -> Ptr<T>;
```
The operation string selects a compiler contract; the function name belongs to the
source module. HIR checks the declared signature against that contract, preserving
its type binders and source identity. Calls use ordinary import resolution, generic
inference, and function specialization. An intrinsic declaration cannot supply an
arbitrary foreign signature or a replacement implementation.

LIR specializes the operation with its concrete types and verifies its operands
independently. Backends implement the verified primitive operation. Public library
wrapper names do not identify primitives, and the compiler does not locate library
modules by scanning type names.

## What each boundary guarantees

CST pairs source text with a Tree-sitter tree. `build_cst` may reuse a
previous document, and syntax-only queries and formatting need no semantic state.
AST lowering converts that syntax into source constructs. `build_ast` always
returns a file, inserting holes and retaining diagnostics for incomplete syntax.
Each AST `SourceModule` carries an immutable `Source`, its syntax, and resolved import
indices. Applications acquire imports before supplying a frozen `SourceGraph` and
completed `ModuleDocument` values to `build_program`. Assembly orders the modules
without source I/O; imports never execute code.

HIR construction has three internal steps:

1. Declare names and check expressions into a private tree with inference types.
2. Solve function dependency groups and retain the selected method declarations.
3. Complete schemes and solved expressions directly into public HIR.

Value lookup records declaration identities immediately. References, inference
dependencies, and signature/body records keep those identities through resolution;
names and spans remain diagnostic metadata. The private tree has no lexical cursors.
Source-order declaration records carry names, decorators, and foreign headers, so
function assembly does not rescan AST declarations. Persistent scopes remain with
source construction and editor queries. There is no second private expression tree
with concrete types: body completion reads the solved types and constructs public
HIR once. Its inputs include concrete type rules, function identities, shader
signatures, and method choices; it cannot access lexical scopes or frontend method
namespaces. The solver is read-only during completion and is dropped before the
completed file returns to module assembly. Inference retries restore method choices
alongside the solver so failed attempts cannot leak a stale selection.

Each body returns its HIR and referenced shader entries. Assembly marks those
entries for embedding only after that body completes successfully. Source-known method calls become ordinary calls
to resolved function IDs with explicit receiver adaptation and argument packing. Dependent
method calls retain their receiver, name, completed type arguments, and ordinary arguments.
Compiler-provided methods become intrinsic operations. Short-circuit operators
become `If` nodes. Field projections retain member names, numeric literals retain
text and their determined types, explicit conversions retain their source and destination
types, and layout queries retain only the queried type. Their operand effects cannot execute. Type aliases
expand into their targets. Completed nominal declarations retain names, bodies,
method identities, and destruction hooks in HIR. HIR has no concrete interner;
its constants and type expressions use its own language rather than concrete
storage values and a `resin_types::TypeTable`.

HIR remains a tree: `If`, `While`, blocks, matches, error propagation, places,
and values retain their structure. Completing each body also establishes definite
initialization, including unused definitions. Parameters and pattern bindings begin
initialized; local initializers cannot read or address their own binding. Assignment
and address acquisition do not read a whole local, while field projections require
an initialized base. Completion follows runtime evaluation order, intersects branch
states, and retains only the loop condition's guaranteed effects. Layout queries
retain no operand effects.

LIR lowering assigns storage to binding IDs and makes evaluation, ownership cleanup,
and control flow explicit. It has no source initialization states or branch snapshots;
runtime flags still protect partially initialized managed storage during cleanup. A LIR function's
`parameter_count` identifies its initial locals as parameters in declaration order. Zero-argument
functions reserve no parameter local; a tuple parameter occupies one local. Calls transfer separate
operands to those locals, and C and SPIR-V emission preserve the parameter list. Each `Instr`
documents its consumed operands and produced values.

Lowering records how many locals exist when each operand is produced, so cleanup can
interleave unfinished expressions with scoped owners. Structured regions preserve these
boundaries for carried operands; new results follow the region's local registrations.
This bookkeeping is private to lowering.

HIR signatures can bind named type parameters, and function references carry
completed type arguments. LIR construction owns a memoized worklist of concrete
applications, keyed by definition, normalized arguments, and semantic Host/Shader
profile. It reserves each ID before translating its body, so recursion reuses pending
requests. Generate supplies exported target roots; only their transitive function
and type dependencies enter the worklist. Decorated functions remain host-callable;
shader artifacts and pipeline creation request separate shader instances. C emits
host instances, while SPIR-V follows the requested shader graph.

Source functions bind named parameters with `fn identity<T>(value: T) -> T`.
Every declaration reference creates a fresh application, deduced from operands and
expected results or supplied with `identity::<int>`. Bound parameters remain rigid
inside the definition; local function values remain monomorphic. A `_` is a weak
monomorphic variable and can unify with an enclosing named parameter. Applications
retain substitutions around unresolved definition variables, so recursive dependency
groups can finish a result from its body without a caller determining it. Unseeded
cycles and undetermined arguments require annotations. Source structs also bind
named parameters. Their completed field schemes feed the same nominal application
representation used by LIR; constructors constrain field values against the applied
scheme. Local field-only structs retain enclosing type parameters as implicit
arguments before their explicitly declared parameters. Imports and aliases keep
the source nominal identity, and editor facts retain substituted fields without
materializing their layouts. Methods reuse the owner's binder identities and append
their own named parameters. Source method namespaces select a declaration by nominal
origin; HIR construction applies its signature once per call and emits an ordinary
function application, including receiver adaptation. Method dependencies participate
in result-inference groups. The completed HIR retains one method body; LIR specializes
it, and generic drop hooks, with the owner's arguments before the method's arguments.
When the nominal origin depends on substitution, HIR retains a `MethodLookup`.
Its `Type::Method` describes the caller-facing function signature: instance calls
omit the implicit receiver, while associated references retain every parameter.
`FunctionParameter` and `FunctionResult` project that signature; the same projections
support calls through dependent function-valued fields. These are determining type
expressions, not weak inference variables or constraints that infer a receiver.

LIR resolves these lookups against completed source nominal declarations. It
substitutes the owner's arguments and explicitly supplied method-local arguments,
checks the concrete receiver adaptation and arguments, and requests the selected
function through the existing instance worklist. It does not deduce additional
method arguments: an unknown namespace requires explicit arguments for those binders.
Known namespaces continue to support ordinary HIR deduction. Compiler primitive
method generators remain private to HIR construction; dependent lookup currently
selects source-declared methods. Lookup and signature expansion share the bounded
type normalization traversal. Repeated active signature queries report a cycle
requiring annotation; growing queries have a separate nesting limit of 32. These
guards bound work within an instance independently of the function-instance limit.
Failures retain their application trace. Concrete calls and aggregate constructors
check the substituted signature and shape before storage lowering; fixed argument
types cannot bypass those checks when a contextual type is a determining query.

Transparent aliases use the same lexical type binders, for example
`type View<T> = Ptr<T>`. Source scopes retain their completed RHS and named parameters;
applications expose the substituted structure to ordinary deduction. A local alias
can capture its enclosing function's parameters without capturing arguments to its
own binders. Aliases keep the target's nominal origin and method namespace and do
not generate functions. Definitions resolve in declaration order; references to an
alias while its RHS is being constructed report recursive expansion, even beneath
a pointer. Expansion has separate HIR limits of 256 levels and 65,536 nodes.

Persistent scopes store completed HIR schemes for imports, navigation, and hover.
HIR keeps one body per source definition, including symbolic literals, arithmetic,
fields, casts, layouts, and Err payloads. Error holes accumulate unions of symbolic
contributors; applications substitute the current contributors until the dependency
group reaches a fixed point. Only then do unseeded error sets become `Never`.
Concrete operation support and determining field types are resolved during LIR
construction, with application traces for failures.

HIR nominal declarations also bind named parameters. A nominal type expression
retains its declaration origin and arguments. LIR normalizes application keys with
these identities, without expanding field definitions just to compare arguments.
Only a signature, value, field, layout query, or other operation that uses the type
materializes its concrete representation. An unused type argument therefore does
not request its fields or drop hook. Member-derived arguments restore source
origins and canonical union order before memoization. Concrete names and diagnostic
arguments retain names such as `Node<int>` instead of private catalog indices.

`resin_lir::build_lir` accepts closed HIR applications and constructs their target
program. An empty entry list requests all ordinary functions and nongeneric nominal
declarations through the same machinery, for direct language clients. Requested programs discover nominal types lazily and translate every embedded
nominal application. Field and cast representation steps are selected against that concrete catalog.
Each requested nominal instance substitutes its owner arguments into fields and
drop hooks. Recursive identities remain private until their bodies and real hook
IDs are installed. Nominal expansion has its own depth guard across declarations
and growing nominal arguments.

The worklist translates each application into a private concrete expression tree,
substituting types and selecting supported builtin operations without inference.
It also selects numeric representations, computes layout constants, resolves field indices,
and chooses explicit conversion operations. A determining `Type::Member` resolves the type
of a named field from a substituted receiver; it never deduces a receiver from the field.
Dependent methods and function-signature projections use the same direction of resolution.
Numeric parsing and range checks use the same `resin-types` operation as source construction,
with an explicit type and no defaults. Concrete conversion rules reject unwrapping custom
owners before their destruction can be bypassed. Source-known errors remain HIR diagnostics;
errors that depend on an application carry the LIR worklist's application trace.
Shader scalar/type rules live in `resin-types` and are also used by certification and
emission. Specialization distinguishes value reads from place access, so taking the
address of an opaque managed field remains valid while copying its value is rejected.
Completed shader instances cannot call foreign functions or perform host ownership
operations. An iterative graph traversal rejects shader recursion before publishing LIR.
Verification independently certifies these guarantees for direct language clients and
retains dependency order, exposed by `Verified::shader_functions`. SPIR-V consumes that
order instead of discovering the target call graph. Its remaining local-address escape
and merge restrictions concern SPIR-V representation and remain in target lowering.
It lowers that tree against completed concrete nominal definitions using fresh
per-function state, then discards the tree. Foreign ABI parameters come from the
concrete signature. Function references, shader artifacts, pipeline bridges, exports,
and drop hooks use reserved LIR identities. Module assembly combines completed
functions and source origins only after every request succeeds. Independent failures
retain their own source locations and a bounded application trace; diagnostics never
assume HIR and LIR indices agree. Foreign declarations retain one local per parameter and
no blocks; they do not create function-body lowering state.

`LoweringOptions.max_monomorphs_per_function` defaults to 16,384 and cannot be zero.
LIR receives that limit for one construction run. Different semantic profiles count separately toward the
same source function's allowance. Existing canonical requests cost nothing, including
pending and failed requests. A new request consumes its allowance before body
translation. Independent type-depth and type-size guards bound structural expansion
before substituted trees are cloned. These are resource diagnostics, not a claim
that every program has a finite successful set of instances.

LIR retains a structured tree of blocks. Each block contains straight-line stack
instructions and an `If`, `Loop`, `Merge`, `LoopTest`, `Continue`, or `Return`
terminator. `If` owns its two arms, which finish with `Merge`; `Loop` owns a
condition region ending in `LoopTest` and a body region ending in `Continue`.
Both may own a `next` continuation that consumes their result operands. A nested
tail selection may omit its continuation and forward to the enclosing selection's
merge. Completing a region at function scope or inside a loop condition or body
requires an explicit continuation ending in the appropriate terminator. Returns
leave the function after explicit cleanup.
Blocks live in a flat arena to preserve stable instruction and source identities,
but references express unique tree ownership rather than jump destinations.

A loop's `LoopTest` consumes a bool above the carried operands. False exits with
those operands; true runs the body, whose `Continue` must restore the condition's
input types. Conditions can contain nested branches, loops, and early returns;
their branch arms still end in `Merge`, followed by the explicit `LoopTest`.
The verifier checks these contracts with one tree traversal, rejecting reused,
cyclic, orphaned, and invalid block references. Function paths must return;
branch arms that return do not contribute operands to a subsequent join.
Continuation traversal is iterative, so sequential conditionals and `?` expressions
do not consume nesting depth in verification, printing, or target lowering.

LIR's private [verify module](../crates/resin-lir/src/verify/mod.rs) checks arbitrary
LIR, including instruction operands, region nesting and exit kinds, returns, nominal layouts,
shader signatures, and drop hooks. Its public contract stays in LIR's `lib.rs`.
`resin_lir::VerifiedModule` owns the module and analysis behind private fields. `view()`
borrows an immutable certificate for that exact module. `into_module()` consumes
the certificate and returns ordinary LIR; editing it requires verification again.
Target lowering accepts a certificate for the module being lowered. Instruction
effects stay private to verification; backends obtain checked operand counts through
`resin_lir::FunctionTypes::operand_count(block, index)`.

C lowering chooses the ABI, runtime operations, and native entry wrapper. It builds
an owned tree of translation units, functions, conditionals, loops, and returns;
leaf strings contain target syntax for declarations, expressions, and operand transfers.
The C printer formats only that completed target tree, without consulting LIR or
verification facts.

SPIR-V lowering resolves reachable functions, checks device restrictions, and emits
binary instructions with `rspirv`. It preserves structured selection and loop regions
using explicit merge and continue targets. Function storage holds addressable locals;
physical buffer accesses use Resin's shared layout and Vulkan device addresses.
Direct function identities and local addresses remain private lowering information.
The optimizer runs separately in the toolchain; codegen invokes no external processes.

## Calling the passes

The application resolves its source inputs and calls each async translation explicitly.
This complete one-file example creates a generated project without running native tools:

```rust
use resin_executor::{Cancellation, Execution};
use resin_source::{Source, SourceGraph};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::main]
async fn main() -> (() | Err<Box<dyn std::error::Error + Send + Sync>>) {
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let source = Source::new("example.resin", "export { main }; def main() -> int = { 42 };");
    let syntax = Arc::new(resin_cst::build_cst(
        source.text().to_owned(), None, &execution, &cancellation,
    ).await?);
    let parsed = resin_ast::build_ast(syntax.clone(), &execution, &cancellation).await?;
    let document = Arc::new(resin_ast::ModuleDocument {
        source: source.clone(), syntax, file: Arc::new(parsed.file), errors: parsed.errors,
    });
    let graph = SourceGraph::new(source.clone(), [source.clone()], [])?;
    let program = resin_ast::build_program(
        graph, BTreeMap::from([(source, document)]), &execution, &cancellation,
    ).await?;
    let hir = resin_hir::Hir::build(Arc::new(program), &execution, &cancellation).await?;
    let lir = resin_lir::build_lir(
        hir.hir()?.clone(), vec![], resin_lir::LoweringOptions::default(),
        &execution, &cancellation,
    ).await?;
    let checked = resin_lir::VerifiedModule::build(lir, &execution, &cancellation).await?;
    let temporary_parent = tempfile::TempDir::new()?;
    let project = resin_codegen::generate(
        Arc::new(checked), Some("main".into()),
        Arc::new(resin_codegen::NativeHeaders::default()), temporary_parent.path(),
        &execution, &cancellation,
    ).await?;
    let text = tokio::fs::read_to_string(project.c_source().unwrap()).await?;
    assert!(text.contains("main") && project.build_file().is_file());
    Ok(())
}
```

For imports, applications obtain sources from async `Loader::load_file_async` and
`load_import_async`, supplied editor text, or explicit logical bindings. They read
`Document::preamble()` and freeze one version per identity into a `SourceGraph` with
`ImportBinding` edges. A graph validates its identities and bindings; AST assembly
reports missing imports, cycles, and syntax errors while preserving recovered files.
The [client acquisition pass](../crates/resin-client/src/inputs.rs) shows filesystem
traversal; the [server analysis handler](../crates/resin-server/src/analyze.rs) shows
explicit cached phase sequencing over uploaded and frozen managed inputs.
Neither operation belongs to a reusable compiler orchestration library.

For file-backed inputs, applications call `Loader::logical_sources` before looking
up CST or AST caches. With the entry's parent as the explicit root, this maps files
to root-relative names and library files to `$/` names; reserved characters and
non-UTF-8 path bytes are escaped without losing identity. The loader retains physical
sources for import acquisition. Each request separately records logical IDs to local
sources or paths for diagnostics and editor navigation. Equivalent checkout graphs
can therefore share compiler results while each editor keeps its own locations.

`Loader::source_from_text(path, text)` registers authoritative editor text, and
`remove_source(path)` restores disk loading on close. `set_import(importer, reference,
source)` binds logical references explicitly. Filesystem imports resolve relative to
their importer; `$/` selects the configured library root. Compiler passes receive no
loader, path policy, overlays, or file notifications.

Direct clients can construct an AST `Program` in dependency order and call async
`build_hir(Arc<Program>, execution, cancellation)`. Its `CheckedProgram` preserves
editor analysis on source failures. `Hir::build` additionally retains the assembled
program and syntax documents for queries. LIR construction selects host/shader roots
and their dependencies; an empty entry list requests every ordinary root. Verification
is a separate async `VerifiedModule::build` operation.

Codegen receives `Arc<VerifiedModule>`, an optional host entry, immutable
`Arc<NativeHeaders>`, and an existing temporary parent directory. `generate` creates a unique child directory and returns a
`GeneratedProject` that owns it. C/SPIR-V lowering completes before files are written.
`None` selects a shader-only project. Optimized binaries, embedded headers, and native
executables remain planned outputs for Ninja. Retained generated inputs stay immutable;
native tools stage their own copies. The final project owner removes its directory.

`NativeHeaders` owns canonical staged file bytes, ordered include roots, and bindings
from each LIR `ForeignHeader { source, spelling }` to a staged or target-system include.
HIR and LIR preserve the declaring source separately from spelling, including empty
extern groups. Different modules can therefore bind equal basenames to different
headers. An explicit runtime include prevents user roots from replacing the compiler
ABI. Codegen validates and writes these supplied bytes without loading any source or
header path. The server validates complete bindings and keys generation by every
bundle byte and ordered root, including currently unused files. Direct compiler callers
may supply default empty bindings to keep their literal native includes.

The toolchain takes any compatible on-disk Ninja project and captured `Environment`
settings. Async `Toolchain::build(project, name, entry, profile, execution, cancellation)`
stages inputs, supplies rules/settings, runs Ninja, and publishes an owned output
generation. `BuiltProject` and `Executable` clones share that generation. The staging
lock ends with the build; retaining, copying, or running A does not lock out build B.
Generations live under `build/.artifacts`, separately from incremental cache slots.
Unix retains immutable published file inodes through hard links; Windows copies files
so executing an image cannot block replacing the cache file. Final-owner cleanup
removes only that generation. The inspectable debug/release cache remains available
for Ninja reuse after artifact handles are released.

C projects declare `native-inputs.json`, described by `resin_toolchain::NativeInputs`:

```json
{
  "translation_units": [{ "source": "main.c", "preprocessed": "main.i" }],
  "restrict_header_paths": false,
  "c_flags": [],
  "preprocessing_flags": ["-I", "."],
  "generated_prerequisites": ["shader_1.h"]
}
```

The toolchain first configures stable `toolchain.state`, builds declared prerequisite
Ninja targets, then preprocesses each original C unit into its declared `.i` file.
It preserves line markers, including system-header provenance needed by compiler
diagnostics. Exact captured bytes are recorded in `native-inputs.state` with explicit
length framing; unchanged content preserves its timestamp. The
`compile_preprocessed_program` Ninja rule compiles `main.i`, so headers changed after
capture cannot alter that build. Generated `main.c` remains available for inspection.

Common compiler options belong in `c_flags`; include roots and macro definitions
belong in `preprocessing_flags`. Both stages use the configured compiler, target
options, and captured environment, while only preprocessing applies include/define
arguments. C edges depend on `native-inputs.state`; shader/embedding edges depend
only on `toolchain.state`. This detects changed header contents with preserved mtimes,
new shadowing headers, and conditional includes without hashing unrelated workspace
files. Shader-only graphs need no manifest or C preprocessing. Handwritten C projects
use this same captured-input contract; `compile_program` remains available for raw C
graphs that do not declare input capture.

Server-generated projects enable `restrict_header_paths`. The same preprocessing
invocation writes a compiler dependency report; validation permits canonical staged
paths, the configured runtime, and compiler-discovered system include roots before
publishing captured C. Discovery excludes ambient CPATH-style overlays from that allow
list. Dependency reports are independent of C `#line` display names. This enforces
native input ownership, not an operating-system sandbox; native tools run under the
service account and operators control isolation. Arbitrary direct native graphs may
leave this opt-in restriction disabled.

Persistent tool/runtime settings use BLAKE3 with explicitly framed input bytes,
including executable contents. They do not use Rust's unspecified `DefaultHasher`
encoding. Native cache directory labels also use a specified BLAKE3 encoding.

Ninja orders unoptimized SPIR-V → `spirv-opt -O` → the configured executable's `--embed` headers → C linking.
The embed command uses the Resin executable captured with `current_exe`, never a PATH
lookup. The server needs Ninja, a C compiler (`CC`/server `--cc`), and SPIR-V Tools
(`SPIRV_OPT`/server `--spirv-opt`); a tool is checked when its graph command runs. Host-only
projects never invoke the shader optimizer. Native settings use per-command cwd and
environment rather than process-wide mutations.

## Source identity and reuse

`SourceId::new(logical_name)` identifies the same module across acquisitions.
`Source::new(name, text)` uses its name as that identity; `Source::with_identity`
separates logical identity from diagnostic display name. Callers must give distinct
logical modules distinct IDs even when their text or displayed names match. Loader
acquisition handles use canonical paths in a separate identity namespace.
`Loader::logical_sources` maps those handles to entry-parent-relative module names
and `$/` library roles before compiler cache lookup. Applications retain original
sources/paths separately for client presentation. Equivalent checkouts therefore
share compiler facts without leaking one checkout's paths into another's diagnostics.

Source equality uses logical identity, display name, and exact text. A BLAKE3 digest
speeds comparisons but never replaces the exact-text check. Full-text reconstruction
and equivalent `Source::from_edits` requests therefore address the same keys. Cloning
shares a version, and `with_text` leaves the previous version intact. Names and paths
have no implicit filesystem meaning inside compiler passes.

Applications resolve imports on every request before attempting semantic reuse,
including requests whose entry text is unchanged. Immutable `SourceGraph` keys include
the selected versions and actual resolved import edges. Equal text from a distinct
logical module never reuses another module's origins or nominal identities.
`Hir::build` accepts a completed `BuiltProgram`; it has no previous-HIR or loader input.
Its retained diagnostics and editor queries refer to the exact source versions used.

`resin-cache` provides immutable `Cache<K, V>` snapshots of shared completed values.
Async `update` deduplicates requested keys, builds misses within the execution bound,
refreshes requested recency, and evicts unrequested entries deterministically. If a
request exceeds capacity, all its values survive with a warning; a later request can
shrink the snapshot. Failure or cancellation produces no successor cache. Retained
values survive cache eviction and keep earlier compilations usable.

The HTTP server owns one `ArcSwap` head per cache layer: sources, CST, and per-file AST
have capacities of 4,096 entries each, HIR and verified LIR have 64, and generated
projects have 32; complete input handles have 64. Its private publication operation computes requested values, then
compares and swaps the head. A lost race rebases all requested handles, including
hits, against the latest cache and reapplies its recency and capacity policy. Only
requested entries are replayed, so old unrelated history cannot return. Completed
compiler/native work never repeats merely because publication lost a race.

Each request keeps its selected handles after publication. Subsequent updates may
evict those keys while the request remains valid. Published generations have no
parent links. HIR reuse applies to complete source graphs; an edited dependency
rebuilds HIR while unchanged per-file results remain reusable.

Acquisition loaders belong to requests. The editor coordinator retains only current
supplied registrations; `Loader::supplied_snapshot()` copies their physical identities
and configured bindings without keeping learned disk history. Close/reopen epochs
prevent coalesced editor updates from confusing old and new registrations. Completed
analysis retains source handles and presentation maps for active roots. Old outputs,
open buffers, and allocator overhead can outlive current cache membership, so entry
capacities are not a hard byte or process-memory bound.

## Execution and cancellation

`resin-executor::Execution` defaults to `available_parallelism()` logical CPUs,
falling back to one; callers can supply an explicit nonzero job count. CPU passes run
on bounded workers, including synchronous Tree-sitter calls. Async orchestration does
not consume a worker slot. One native build reserves one slot and invokes Ninja with
`-j 1`, coordinating compiler and native concurrency through the same bound.

A shared `Cancellation` stops queued work and is checked during longer passes. A
started synchronous foreign call may have to finish; its permit remains occupied
until it does, even if the waiting future is dropped. Shutdown stops submissions,
cancels requests, and waits for occupied execution slots to drain. Context and
cancellation do not enter semantic cache keys.

Native operations retain a supervisor task on caller-future abandonment, cancel its
process tree, and keep staging ownership through cleanup. Awaited cancellation waits
for that cleanup. Unix commands run in private sessions containing Ninja's separate
compiler process groups; Windows assigns suspended children to kill-on-close jobs
before resuming them. Runtime shutdown also terminates owned trees and reaps the
direct children before releasing native ownership. A cancelled build returns no
partial artifact handle. Atomic executable copying leaves no partially copied target.

The stdio LSP receiver accepts complete editor snapshots and uses a replaceable
mailbox to coalesce edits. A bounded request permit follows a query/build through
execution and response consumption. Path normalization, line indexing, formatting,
semantic queries, and diagnostic preparation use workers. Before sending results,
the protocol checks their captured document epochs, versions, and dependency state.
Diagnostics are aggregated from current roots; the protocol tracks actually published
URIs so coalescing cannot lose a required diagnostic clear.

`workspace/executeCommand` with `resin.build` captures disk contents and uploads
complete inputs. Server build handlers sequence AST/HIR, LIR, verification, codegen,
and native compilation, sharing heads with remote editor analysis. The client validates
and atomically publishes the download; dirty buffers remain under editor control.
HTTP admission follows both successful and failed bodies through consumption. Request
and stream guards cancel on disconnect; service shutdown stops admission and drains
owned work. The client retains its POST while retrying a cancellation DELETE that
arrived before admission. [The service guide](compiler-service.md) defines wire,
deployment, target negotiation, and trust boundaries.
