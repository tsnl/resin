# Compiler architecture

Resin's compiler is a sequence of explicit languages and passes. Each phase is an
unpublished Cargo workspace crate. Public data describes its output; a small set
of public operations constructs, prints, or queries that data. Each crate lists
its full public interface in `lib.rs`; implementation modules stay private.

The root manifest is both a package and a workspace. Root `src/` contains the
`resin` CLI: `resin FILE` builds and runs, `resin FILE --output PATH` builds an
executable, `resin --format DIR` formats source, and `resin --lsp DIR` serves the
Language Server Protocol. There is one executable to distribute.

Reusable libraries live under `crates/`, with directory names matching their Cargo
package names. `resin-compiler` sequences passes over immutable sources and retains
analysis caches. It uses the concrete `resin_source::Loader` to obtain imports from
files, supplied text, or explicit bindings. `resin-source` owns immutable text and
standard-library resolution; `resin-types` owns concrete types and representation
rules. Neither depends on a compiler phase. `resin-toolchain` runs generated Ninja
projects and retains native artifacts without depending on compiler or type crates.
The CLI connects compilation, code generation, and native building. `resin-lsp`
adapts compiler queries to the protocol; the native C ABI lives in `resin-runtime`.

Crates own their isolated tests; root `tests/` exercises the complete executable and
cross-crate behavior. Examples and documentation stay at the repository root;
Resin libraries live under `resin/`. The root package is the default member, so
`cargo run -- examples/eg001.resin` works there. Use `--workspace` to build or test
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
    spirv --> headers[resin --embed: C headers]
    headers --> executable[C compiler: executable]
```

These arrows show data flow. Cargo dependencies point toward the languages a
pass consumes. Source and type vocabulary are independent foundations.

| Crate | Direct phase dependencies | Public purpose |
| --- | --- | --- |
| `resin-common` | none | Shared `define_id!` index-type macro |
| `resin-source` | none | Immutable sources, locations, import loading and standard-library resolution |
| `resin-types` | none | Concrete types, values, conversions, layout and shader interfaces |
| `resin-cst` | generated `tree-sitter-resin` grammar | Syntax documents, reparsing, queries, formatting |
| `resin-ast` | `resin-cst` | Source AST, recovery, parsing diagnostics |
| `resin-hir` | `resin-ast`, `resin-cst` | Resolved tree, checking and elaboration, editor analysis |
| `resin-lir` | `resin-hir` | Storage and control-flow lowering, verification certificates |
| `resin-codegen` | `resin-lir` | Generate a complete on-disk C/SPIR-V/Ninja project |
| `resin-toolchain` | none | Captured process settings, Ninja builds, locked output files |
| `resin-compiler` | CST, AST, HIR, LIR | Import traversal, pass sequencing, immutable compilations |
| `resin-lsp` | `resin-compiler`, `resin-hir`, `resin-cst` | Compiler queries and formatting over LSP |

The HIR dependency on CST supports editor queries at a syntax position. Its public
language uses only shared source identities and concrete types. LIR lowering never
receives CST, AST, source namespaces, or an inference solver. Codegen consumes the
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

Keep related state and operations together. The compiler's `lib.rs` contains
`Compiler` with its private caches, `Compilation` with its retained products, and
the operations that use them. A cohesive file can be substantial while exposing
few public concepts. Fields belong directly to the objects whose invariants they
serve; private helpers keep individual operations readable. Native building is a separate
operation connected by the [CLI](../src/cli/mod.rs).

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

## What each boundary guarantees

CST pairs source text with a Tree-sitter tree. `Document::reparse` may reuse a
previous document, and syntax-only queries and formatting need no semantic state.
AST lowering converts that syntax into source constructs. The recovering API keeps
holes and diagnostics; the strict API returns a file only when syntax is valid.
Each AST `SourceModule` carries an immutable `Source`, its syntax, and resolved import
indices. The compiler traverses imports through `resin_source::Loader::load_import`,
then orders the modules into a `Program`. The loader resolves explicit bindings or
file references; AST generation performs no source I/O and imports never execute code.

HIR construction has three internal steps:

1. Declare names and check expressions into a private tree with inference types.
2. Solve function dependency groups and resolve every annotation and expression type.
3. Elaborate into public HIR, removing source-only lookup and syntax.

The private checking tree retains lexical context for declaration visibility;
those contexts never cross into public HIR. Method calls become ordinary calls
to resolved function IDs with explicit receiver adaptation and argument packing.
Compiler-provided methods become intrinsic operations. Short-circuit operators
become `If` nodes, field projections are resolved, numeric literals become values,
and layout queries become constants whose operands cannot execute. Type aliases
and method namespaces disappear. Nominal identities and destruction hooks remain
in the shared type table because later phases need them.

HIR remains a tree: `If`, `While`, blocks, matches, Result propagation, places,
and values retain their structure. LIR lowering assigns storage to binding IDs,
checks definite initialization, and makes evaluation, ownership cleanup, and
control flow explicit. Thus a HIR module can still fail storage/initialization
checks during LIR lowering. Every LIR function reserves local zero for its unary
parameter, including unit, tuples, and foreign declarations. Each `Instr` documents
its consumed operands and produced values.

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

The smallest host pipeline uses just the public phase APIs:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let syntax = resin_cst::Document::reparse(
        "export { main }; def main() -> int = { 42 };".into(), None,
    );
    let ast = resin_ast::generate(&syntax)?;
    let hir = resin_hir::generate(&ast)?;
    let lir = resin_lir::generate(&hir)?;
    let checked = resin_lir::VerifiedModule::new(lir)?;
    let directory = tempfile::TempDir::new()?;
    let project = resin_codegen::generate(checked.view(), Some("main"), directory.path())?;
    let source = std::fs::read_to_string(project.c_source().unwrap())?;
    assert!(source.contains("main") && project.build_file().is_file());
    Ok(())
}
```

For imports, call `Compiler::compile(entry, &mut loader)` with an immutable `Source`
and a concrete `resin_source::Loader`. Explicit bindings let the same loader work
with generated or in-memory sources:

```rust
use resin_source::prelude::*;

fn main() {
    let math = Source::new(
        "generated math", "export { answer }; def answer() -> int = { 42 };",
    );
    let entry = Source::new("editor buffer", r#"
        export { main }; import { "math" };
        def main() -> int = { answer() };
    "#);
    let mut loader = resin_source::Loader::new(resin_source::library_root());
    loader.set_import(&entry, "math", math).unwrap();
    let mut compiler = resin_compiler::Compiler::new();
    let compilation = compiler.compile(entry.clone(), &mut loader);
    assert!(compilation.diagnostics().is_empty(), "{:?}", compilation.diagnostics());
    assert_eq!(compilation.entry(), &entry);
}
```

`load_file(path)` reads a file; `source_from_text(path, text)` registers supplied text
as authoritative for that path's imports. `remove_source(path)` restores disk loading
for a closed buffer. `set_import(importer, reference, source)` binds a reference
explicitly, allowing names that have no filesystem origin. Otherwise imports resolve
relative to their importing file, with `$/` selecting the configured library root.
Unchanged text reuses its source version. The compiler needs no buffer or path policy.

A caller may also construct a `resin_ast::Program` in dependency order and call
`resin_hir::generate_program`. `resin_hir::analyze_program` returns a `CheckedProgram` with
diagnostics and opaque editor analysis even on failure. `Analysis` owns its private
query state directly. Its queries take a source handle, byte offset, and shared CST
documents in a `BTreeMap<Source, Arc<resin_cst::Document>>`; no document-provider trait
or forwarding object is needed. HIR functions carry optional source locations directly.
Source handles retain their text, so later phases need no separate path-to-text table.
`resin_lir::analyze` collects errors across functions; `resin_lir::generate` returns the first.
Codegen accepts only verified LIR. `generate(verified, Some(entry), directory)` writes
host C, the SPIR-V requested by `.spirv`, and `build.ninja`; `None` generates a shader-only
project containing all declared shaders. It returns paths, never target ASTs or per-target
emission operations. C and SPIR-V lowering finish before any generated files are written.

The compiler's [lib.rs](../crates/resin-compiler/src/lib.rs) contains source traversal,
syntax caching, HIR/LIR generation, verification, and retained query access.
`Compilation::verified()` supplies the certificate for codegen without reloading sources.
The CLI's private [Request](../src/cli/request.rs) resolves source and destination choices
against the captured working directory, including output naming and ancestor validation.
Argument parsing passes the original paths to this boundary. Native compiler search paths
retain their meaning relative to that captured directory; Ninja resolves discovered header
dependencies in the directory where it runs the compiler.

The [toolchain](../crates/resin-toolchain/src/lib.rs) captures explicit `Environment`
inputs and builds any compatible Ninja source directory. `Toolchain::build` stages the
project, supplies the native command rules and settings in `toolchain.ninja`, and lets
Ninja execute the generated dependency edges: unoptimized SPIR-V → `spirv-opt -O` →
C headers, followed by C compilation and linking. `BuiltProject` exposes
retained output paths; an `Executable` keeps the cache lock through copying and execution.
The embedding command uses the running Resin executable obtained through `current_exe`,
so it uses the same version without looking up `resin` on PATH.

Install Ninja, SPIR-V Tools (`spirv-opt`), and a C compiler (`CC` or `--cc` overrides
the default). `SPIRV_OPT` or `--spirv-opt` selects the optimizer. No tool
preflight is performed: a required command reports failure when executed. A host project
without embedded shaders never invokes `spirv-opt`. No language crate invokes native tools;
`resin-toolchain` has no dependency on compiler internals or concrete Resin types.

## Source identity and incrementality

`Source::new(name, text)` creates immutable text with a fresh logical `SourceId`.
Cloning shares that exact source version. `with_text(text)` creates a new version
with the same logical ID, leaving the original intact. Equality identifies versions;
equal text or equal diagnostic names do not make independently created sources equal.
Names have no filesystem meaning inside the compiler.

`Compiler::compile(entry, loader)` returns an `Arc<Compilation>`. Each call resolves
the complete import graph before considering cached analysis, so changed resolutions
and newly available dependencies are observed. The loader reuses unchanged source
handles; supplied text and explicit bindings determine the versions returned for imports.
The compiler diagnoses cycles and inconsistent versions instead of mixing their facts.

The compiler caches CST/AST documents by logical source ID. An unchanged version
reuses its document; a changed version can reuse the previous tree for incremental
CST parsing. A matching successfully loaded graph can reuse its compilation. When
that graph changes, semantic checking reruns the entry's import closure. Native
artifact caching is separate. This is not a per-function incremental solver or backend.

A `Compilation` retains diagnostics, recovered AST, completed phase products, and
opaque editor facts for one entry and its imports. A failed later pass preserves
earlier products. `definition`, `hover`, and `completions` accept a retained source
handle and byte offset; their results refer to that exact source version. Old
compilations remain usable while the caller creates new sources and compiles again.

The name distinguishes retained work across phases from a HIR or LIR `Module`.
The CLI uses a compiler and filesystem loader for one invocation. The LSP library
keeps a compiler and loader across edits, registering changed buffers and removing
closed ones. It schedules analysis in response to document and file notifications.
The loader owns source lookup, while protocol versions and scheduling remain in the LSP library.
