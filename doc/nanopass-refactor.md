# Scope: Nanopass refactoring before polymorphism

Status: proposed behavior-preserving refactor, reviewed against main at
`a7e3e8b7`. This scopes implementation work; it does not implement it.
The [polymorphism draft](https://github.com/tsnl/resin/pull/159) remains separate.

## Assessment

Resin already follows the central Nanopass idea at its major boundaries:
explicit languages, translations between them, and downstream consumers that
receive completed data. The [Nanopass framework](https://nanopass.org/) emphasizes
small passes and multiple intermediate representations. We can adopt that style
more consistently inside the frontend without replacing the compiler's crate
structure or introducing a framework.

The main prerequisite is a focused frontend refactor: preserve binding and
operation decisions in the intermediate data, and separate completed products
from the context used to construct them. A smaller per-function LIR translation
refactor also gives us a useful preparation for specialization. Adding more arrows
to the public pipeline would not by itself improve those contracts.

| Existing boundary | Evidence and assessment |
| --- | --- |
| CST → AST | [AST](../crates/resin-ast/src/lib.rs) describes source constructs and has its own translation and printer. Recovery syntax is explicit. Keep it. |
| AST → HIR | Public [HIR](../crates/resin-hir/src/lib.rs) has concrete types, binding/function identities, resolved operations, and explicit conversions. The internal translation still relies on lexical cursors and repeated resolution. This is the main refactor target. |
| HIR → LIR | [LIR construction](../crates/resin-lir/src/lower/mod.rs) consumes HIR alone and introduces storage, initialization state, cleanup, and structured regions. This is a real language change already. |
| LIR → verified LIR | [VerifiedModule](../crates/resin-lir/src/lib.rs) owns analysis for its exact immutable module; mutation discards the certificate. This already retains a useful refinement result. |
| Verified LIR → C / SPIR-V | C has a [private target tree](../crates/resin-codegen/src/c/mod.rs) and independent printer. SPIR-V translation constructs `rspirv` instructions directly. Neither needs frontend state. |
| Generated project → native artifacts | [Codegen](../crates/resin-codegen/src/lib.rs) finishes target translation before writing files; the [toolchain](../crates/resin-toolchain/src/lib.rs) owns native processes. Keep this separation. |

The [phase-boundary tests](../tests/phase_boundaries.rs) already exercise lowering
after source construction state is dropped, generated files outliving their input,
and edits invalidating the LIR certificate. These are concrete reasons to preserve
the existing architecture rather than restart it.

## Where the data stops short

The private [typed tree](../crates/resin-hir/src/lower/typed.rs) is parameterized by
inference types and then concrete types. Both forms still contain `Cursor`,
`Var { name }`, method/member names, and raw numeric text. The completed type also
permits error nodes, although successful resolution already rejects them.
Substituting `Ty` for an inference type therefore does not make the body independent
of its construction context or encode every guarantee of successful resolution.

[Elaboration](../crates/resin-hir/src/lower/elaborate.rs) selects the cursor stored
on each term, looks up variable names again, and uses the shared generator's type
and method context to finish operations. [CheckedFile](../crates/resin-hir/src/lower/check/mod.rs)
returns concrete bodies together with a `ContextView`; the next translation needs
both. Inference has already computed useful facts that the output could retain.
This is more than a naming issue with `check` and `elaborate`.

The [coordinating frontend](../crates/resin-hir/src/lower/mod.rs) also rereads AST
declarations to recover names, decorators, and foreign headers when declaring and
elaborating functions. A self-contained handoff must carry this declaration
metadata together with its signatures and bodies.

Editor [Analysis](../crates/resin-hir/src/lib.rs) retains scope data and the
frontend's `Context`, including type/method construction vocabulary. Its queries
are immutable today, and it does not retain the inference solver. The opportunity
is to give completed editor facts their own representation, so their contract no
longer depends on the source-construction implementation. Recovering an unfinished
receiver still needs useful facts even when there is no complete executable HIR.

Within LIR construction, the [module generator](../crates/resin-lir/src/lower/mod.rs)
owns both module output and per-function mutable state. Its
[function translation](../crates/resin-lir/src/lower/functions.rs) clears bindings
and owners, installs an optional function builder, mutates the module, and resets
that state for the next function. A function-to-function translation can return
its completed function and source map directly, leaving module assembly outside.
This needs no new intermediate language.

The [compiler's LIR diagnostic adapter](../crates/resin-compiler/src/lib.rs) also
indexes HIR functions using a LIR error's function index. That matches today's
one-to-one translation but encodes an unnecessary relationship between languages.
An error can carry its source origin directly. This is a small useful preparation
for later one-to-many specialization, independent of changing function-ID types.

## Refactor scope

Keep the public sequence `AST → HIR → LIR → verified LIR → target`. Each pass
produces the representation it promises, or diagnostics. Smaller private passes
must have explicit input/output data with an identifiable difference. Ordinary
Rust structs, enums, and direct functions are sufficient.

For LIR, make the unit of translation one HIR function with the concrete
definitions it needs. A private per-function state object always contains its
builder, bindings, ownership scopes, and origin tracking. Its output is a LIR
function plus instruction origins, or a failure with an optional source location.
Module assembly collects these outputs in the existing deterministic order and
preserves signatures, entries, shader metadata, and type tables. Standalone
constructed HIR with no source location remains supported. Retain current ID
assignments; neither a new function-ID system nor specialization is required.

The frontend work should strengthen its existing internal handoff before adding
another whole-program representation. Binding IDs can be recorded when lexical
lookup first succeeds. Inference can continue using local equations, dependency
groups, and recovery state. Its outgoing translation must establish a completed
body whose subsequent translation no longer selects lexical cursors or resolves
source names.

The completed private body may preserve surface constructs whose expansion is a
separate useful translation, such as short-circuiting and receiver argument
packing. Its language must describe those constructs with the information needed
to expand them: concrete operand/result types, chosen declaration or intrinsic,
receiver adaptation, field identity, literal value, and conversion descriptions.
Inference variables and failed source nodes belong to the incoming inference and
recovery data. They are not variants of this completed output language.

This need not create a second copy of every tree. Adapt the existing private
representation, sharing unchanged payloads where appropriate. If a construct can
be expressed directly in HIR when its facts become known, emit that HIR form.
Retain a private intermediate only for the source constructs whose distinct
translation makes the implementation easier to understand. A separate full bound
AST is not a prerequisite merely because it appeared in the polymorphism sketch.

Each completed body must own or explicitly reference an immutable declaration/type
catalog sufficient for its translation, including source names and locations,
foreign headers, decorators, and other function metadata. Reserve identities for recursive functions,
nominal types, and drop hooks before resolving uses. Do not require a callee's body
to finish before referring to its completed signature. Keep dependency solving and
numeric defaulting at their current boundaries.

Editor output is a separate completed product of the same source translation:
definition identities and locations, visibility scopes, resolved use sites,
determined types, and member/signature facts. Preserve source-version identity and
the visibility point in parent scopes. CST may still provide token ranges and
identify incomplete syntax; it must not trigger inference or code generation.
Failed definitions can produce diagnostics and partial editor facts without
fabricating a completed body or erasing independent healthy facts.

## Suggested implementation PRs

1. **Isolate function-to-function LIR translation.** Separate module assembly from
   per-function storage and control-flow state. Return a completed function and its
   origins, and attach source locations directly to failures. Remove the driver's
   dependency on matching HIR/LIR function indices. This can land independently
   and establishes the first bounded implementation step.
2. **Retain resolved identities at the first lookup.** Replace name-only references
   in the private tree with declaration identities and retain the corresponding
   function/binding mapping. Key inference dependencies and signature/body records
   by those identities as well. Carry resolved operation decisions when available;
   defer type-dependent choices until inference completes. Keep diagnostics and
   source names as metadata. Remove lexical re-lookup from the outgoing translation.
3. **Make the frontend handoff a completed language.** Separate the unfinished
   inference/recovery representation from completed bodies and catalogs. Translate
   completed bodies to HIR without a solver, `ContextView`, or mutable frontend
   `Context`. Include function metadata so this translation no longer rereads AST
   declarations. Make the coordinating function visibly compose these translations.
4. **Retain editor facts independently of construction.** Freeze the declarations,
   scopes, type/member information, and use-site facts consumed by queries. Drop
   the construction context after producing HIR and analysis. Preserve useful
   output after malformed syntax, failed inference, and unrelated errors.

These are review boundaries, not an estimate of equal-sized patches. The first is
small relative to the frontend work. Steps two and three touch inference and
elaboration closely; the fourth has substantial recovery and editor coverage.
Overall this is a medium refactor spread across several focused PRs, with most
work in the frontend. Exact patch sizes depend on how much of the existing private
tree can become direct HIR.

## Stack order and methods inside structs

Deliver the implementation as a stack managed with `gh stack`: the refactor PRs
above first, then the language migration below, then incremental polymorphism.
Each PR must compile and pass its relevant tests on its parent. Land from the
bottom and update descendants through the stack workflow. These documentation
drafts describe the work; they are not the implementation stack itself.

The polymorphism portion starts by moving ordinary programs onto the new HIR and
LIR ownership contracts, then adds bounded specialization, named function type
parameters with contextual deduction, and finally generic structs, transparent
generic aliases, and methods with their own type parameters.
Keep editor support, imports, diagnostics, and lifecycle behavior working in each
slice. Do not publish parser syntax whose basic lowering path is absent.

Before generic structs, replace source `impl` blocks with methods declared inside
their owning `struct`. This is a separate language-migration PR after the
behavior-preserving refactor, rather than a hidden syntax change within it:

```resin
struct Counter {
    value: int,

    def read(self: Counter) -> int = { self.value };
    def increment(self: Ptr<Counter>) = { self.value := self.value + 1; };
};
```

The initial grammar keeps comma-separated fields followed by method declarations;
methods use ordinary `def` syntax and occupy no runtime field storage. Their
owner is the enclosing struct's declaration identity, not a type name looked up
from an `impl` header. Keep the explicit receiver parameter and current static
calls, receiver adaptation, and `drop(self: Ptr<Owner>)` rules. Do not introduce
implicit `self`, unqualified field access, or captured runtime variables.

Current [method ownership](../crates/resin-hir/src/lower/mod.rs) already restricts
`impl` to nominal structs from the defining module; its header accepts one type
name. There are no general blanket implementations today. The simplification is
to make the owner and closed source method set structural, and to avoid designing
a future `impl<T>` binder and owner-pattern matching system. For `struct Cell<T>`,
its methods inherit that same bound `T`; later method parameters such as `<U>`
add their own binders. One source member set applies to every substitution, with
operation support determined when a method is specialized.

Aliases retain their underlying type's methods but cannot declare another method
set. A bare `extern type` cannot gain methods today either; nominal wrappers keep
their methods inside their structs. Compiler-provided methods on `Ptr`, `Span`,
`Arc`, GPU pointers, and other builtins remain registered declarations with their
existing operations. Receiver type deduction, defining-namespace lookup,
pointer/Arc adaptation, and duplicate-name diagnostics still have to work.

Preserve method signature/body visibility when moving declarations. For example,
[GPU owners](../resin/gpu.resin) currently have an `Arc<Owner>` alias between the
struct and its `impl`, and use that alias in method signatures. Collect and resolve
module type declarations before method signatures and bodies; placing a method
textually inside the owner must not make those existing aliases disappear.
Reserve all method declarations before their bodies so recursion and sibling
method references keep their existing behavior. Field declaration visibility
continues to follow the current type rules.

The syntax migration initially covers module-level methods, matching today's
`impl` placement. Keep local field-only structs working; methods on local structs
need a separate decision about declaration scheduling and remain outside this
migration. Merely nesting a method in source must not give it closure captures.

Update the Tree-sitter grammar and generated parser, AST ownership, formatter,
queries/editor recovery, standard library, examples, tests, and language guidance
together. Remove the old source `impl` form rather than retaining two method
declaration mechanisms. Preserve layout, method-versus-field call behavior,
destruction, alias navigation, and GPU method decorators. Test incomplete method
bodies without losing the surrounding struct's fields or later declarations.

## Work that can wait

Today's HIR is already a concrete expression language. Keep using it as the input
to storage translation. A new specialization worklist, concrete-body language for
polymorphic HIR, HIR-owned schemes, moving concrete type interning into LIR, and
the configurable monomorph allowance belong to the polymorphism implementation.
Adding them now would enlarge a behavior-preserving prerequisite into that feature.

Do not split initialization, cleanup, and region formation solely because they
are different activities. Their existing local state cooperates to produce LIR.
A further pass would need an output representation that preserves initialization
and ownership decisions across branches and early returns. No concrete need for
that extra language has been established by this scope.

Keep the LIR certificate: arbitrary constructed LIR still needs its invariant
computations, and code generation consumes their retained result. Likewise, the
backend's platform requirements are legitimate conditions of translating to its
target language. A separate shader-program representation may become useful with
explicit target requests, but moving that boundary is not required for the
frontend refactor.

Keep the C tree, direct SPIR-V lowering, printers, build graph, and crate layout.
There is no need for a generic visitor framework, a language-definition DSL,
an IR per helper, or renamed public APIs as a prerequisite.

## Acceptance and risks

The refactor preserves source syntax, accepted programs, runtime behavior, numeric
defaults, inference dependency groups, module visibility, and useful diagnostics.
In particular, `_` retains its current semantics; this work does not introduce
template parameters or change which declarations allow inference.

Use existing coverage as the baseline and add tests for the new guarantees:

- [Inference](../tests/inference.rs) and [modules](../tests/modules.rs): recursive
  definitions, local type declarations, shadowed function names, imports and
  aliases, contextual literals, layout-operand non-evaluation, and identity despite
  equal source spans.
- [HIR boundaries](../crates/resin-hir/tests/boundary.rs) and
  [phase boundaries](../tests/phase_boundaries.rs): translate completed private
  bodies with lexical builders and inference state dropped; confirm HIR still
  lowers independently and codegen accepts only verified LIR.
- [LIR public API](../crates/resin-lir/tests/public_api.rs) and
  [structured flow](../crates/resin-lir/tests/structured_flow.rs): preserve local
  zero, foreign signatures, initialization/cleanup, and region structure; ensure
  one function's failure cannot contaminate the next function's state. Exercise
  both absent source locations and imported-function diagnostic origins.
- [Editor analysis](../tests/analysis.rs) and
  [compiler results](../crates/resin-compiler/tests/public_api.rs): incomplete
  members, unknown bindings shadowing outer names, failed recursive groups,
  unsaved imports, unchanged cache reuse, and old queries after source edits.
- [Results](../tests/results.rs), [GPU pointers](../tests/gpu_pointers.rs), and
  [GPU HIR](../crates/resin-hir/tests/gpu.rs): preserve one-time argument evaluation,
  receiver adaptation, pipeline signatures, source locations, and ownership.

The largest risk is losing recovery facts while making completed bodies stricter.
Other risks are resolving a name under the wrong visibility point, treating a
pending recursive declaration as missing, and moving literal defaulting earlier.
Exercise those cases directly. Keep diagnostics attached to immutable sources;
preserve the driver contract that a later failure leaves earlier products usable.

Run relevant crate and integration tests in `nix-shell` for each implementation
PR, followed by the required workspace checks. Only expand GPU/window execution
coverage when a change affects those paths, using the repository's required flags.
This scoping PR itself changes documentation only.

The prerequisite is complete when downstream translations and editor queries can
consume their declared data without retaining frontend construction capabilities.
At that point polymorphism can change the HIR type vocabulary and introduce
specialization without simultaneously repairing these existing handoffs.
