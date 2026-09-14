# Remaining template implementation

Status: proposed follow-up to the landed template foundation. This document scopes
the remaining implementation of the [language design](templates.md): generic
source structs and methods, ordinary generic library wrappers, and the GPU API
migration. It does not propose implementing the compiler architecture again.

## What has landed

The [architecture guide](architecture.md) describes the current compiler. The
preparatory work and the first template layers are now on `main`:

The [landed-work inventory](templates.md#what-has-landed) links each implementation
PR, from isolated per-function lowering through function/alias templates and
nominal application normalization. The superseded Nanopass scope is no longer a
prerequisite.

Public HIR already represents generic nominal declarations and applications.
LIR already substitutes their arguments, closes recursive fields, and specializes
their destruction hooks. Source construction still rejects struct and method
binders in [declaration preparation](../crates/resin-hir/src/lower/mod.rs).
Removing those guards requires completing that construction path and its callers;
removing the guards alone would expose concrete-only assumptions.

## Preserve the existing translations

Follow the [Nanopass conventions](../AGENTS.md#nanopass-design) with explicit data,
ordinary Rust functions, and exhaustive matches. Extend these existing boundaries:

| Translation | Completed result |
| --- | --- |
| AST → HIR | Resolved declarations, completed schemes, one typed body per definition, and determining type expressions; no free inference variables. |
| HIR application → private concrete body | Closed types, selected operations and conversions, concrete field identities, and reserved function identities. |
| Concrete body → LIR | Explicit storage, copies, cleanup, and structured control flow. |
| LIR → verified LIR | Certified instruction, layout, region, and shader-graph invariants. |
| Verified LIR → C/SPIR-V | The corresponding target representation and ABI. |

There is no additional template checking pass. HIR construction performs deduction,
unification, numeric defaulting, and definite initialization. LIR substitutes
already chosen arguments, evaluates determining expressions, and reports unsupported
concrete operations. It never guesses an owner, chooses a method argument, or assigns
a type to an unresolved weak variable. The private concrete tree is discarded after
each function's storage translation; HIR acquires neither a concrete interner nor
a list of monomorphs.

## Construct source nominal schemes

Extend the existing [scope machinery](../crates/resin-hir/src/lower/scope.rs) to
reserve a struct's source identity and bind its named parameters before translating
its fields. Reuse the binder identities and annotation decoding used by functions
and aliases. Store the completed field type expressions on the HIR nominal
declaration instead of requiring `Evaluator::ty` to produce a concrete `Ty`.

Applications retain the source nominal identity and their ordered arguments.
`Pair<int>` and `Pair<float32>` are applications of one declaration; two unrelated
`Pair` declarations remain distinct even when their layouts match. Transparent
aliases expand through the existing scoped schemes and preserve that identity.
Use declaration identities in deduction and method lookup, never rendered names.

Struct construction must match initializer fields against the applied field
scheme. Start with explicit type arguments on constructors, then use the existing
aggregate inference machinery to constrain field values and contextual literals.
Constructor argument deduction remains separate work. Preserve missing/duplicate
field diagnostics and source evaluation order. Do not derive nominal identity
by guessing a record type with the requested fields.

Known nominal fields can substitute owner arguments during HIR construction.
A genuinely dependent receiver retains a determining `Type::Member`; its result
cannot be used backwards to infer the receiver. Numeric constraints, Result
propagation, and explicit conversions should pass through these same symbolic
types, without temporarily materializing a concrete nominal catalog in HIR.

Recursive pointer fields use reserved nominal identities; inline layout cycles
remain errors when materialization needs their layout. If local structs capture
enclosing type binders, represent those captures explicitly in their application
identity. Different enclosing applications must not share an accidentally concrete
local type. Keep local structs field-only and preserve declaration-order rules.

Retain recovery facts and completed schemes for imports, hover, and navigation.
Editor analysis must display a generic declaration and its applied uses without
running LIR or replacing a definition's symbolic facts with its last application.

## Apply owner and method binders together

A method has its owner's bound parameters and, optionally, additional parameters
declared on the method. Give each binder a distinct identity, retaining the owner
namespace established by methods-inside-structs. A method reference identifies one
source function plus the complete substitution needed by that function.

For `container.method::<U>(arguments)`, the applied receiver determines the owner's
arguments. The turbofish supplies only the method's own named arguments. An
associated call on an applied owner supplies the same owner substitution without
an implicit receiver. Infer omitted method arguments from operands and expected
results through the ordinary scheme application path. Keep receiver adaptation and
argument packing explicit when completing the call into HIR.

Flatten owner arguments followed by method arguments in the completed function
application, with a documented binder order. Scope metadata keeps the distinction
needed by source syntax and editor queries; LIR receives a complete ordered list.
The [instance worklist](../crates/resin-lir/src/lower/instances.rs) then normalizes
that list using the existing [substitution](../crates/resin-lir/src/lower/substitute.rs)
and keys the request by original function identity, arguments, and Host/Shader
profile. Owners and method applications share the source function's allowance.

The default allowance remains 16,384 distinct requests per function, configured
through immutable `CompilerConfig` and `LoweringOptions`. Repeated normalized
requests cost nothing. Keep the independent type-depth, type-size, alias-expansion,
and nominal-expansion guards; memoization cannot stop an unbounded sequence of
distinct growing arguments by itself.

Start with known nominal owner schemes, including owners containing enclosing
bound arguments. HIR can select their declaration and complete deduction without
materializing the owner. Fully dependent method lookup remains a subsequent layer;
the existing `Type::Member` field relation does not represent a method signature
or an unresolved method call.

That layer must add explicit determining signature relations and a dependent call
form to public HIR. Preserve the receiver type, member name, explicit method
arguments, and completed argument/result relationships without capturing a solver
or lexical lookup cursor. HIR must settle every deduction variable to a concrete,
bound, or determining type expression. If completing a call would require guessing
an owner or rerunning inference after selecting a method, require explicit
arguments or sufficient receiver annotations.

LIR substitutes the receiver, selects the method in its defining nominal namespace,
evaluates the completed relations, and requests its concrete application. It then
emits an ordinary resolved call before storage lowering. Keep source origins for
missing methods, arity/type conflicts, and unsupported concrete operations. This
is concrete lookup and consistency enforcement, with no overload search or new
type inference in LIR.

A generic struct's `drop` uses the owner's arguments and cannot require additional
undeduced method parameters. Reserve its real specialized function identity before
publishing concrete lifecycle metadata. A pending body never temporarily makes the
owner unmanaged. Exercise method recursion, aliases, private helpers, imported
owners, and drop hooks through the same function machinery.

## Move library policies out of special type constructors

The target split is ordinary generic library definitions over compiler primitives.
`Ptr`, arrays, `Result`, primitive scalars, and representation-level operations
remain compiler concepts. Public collection/ownership/GPU wrappers become nominal
source definitions using the new owner and method binders.

| Current compiler-provided family | Intended library responsibility | Primitive responsibility to retain |
| --- | --- | --- |
| `Span<T>` | Pointer/count representation, indexing, slicing, and explicit byte/literal borrowing APIs. | Raw pointer access, layout, array storage, and narrow indexing/address operations. |
| `Arc<T>`, `Weak<T>` | Typed shared-owner APIs, construction, payload access, downgrade, and upgrade. | Safe implicit retain/release and allocation destruction under ordinary copying. |
| `GpuPtr<T>`, `GpuSpan<T>` | Typed views, lengths, allocation methods, slicing, and permission-narrowing APIs. | Checked mapped access, allocation retention, and recording-time projection. |
| `GpuComputePipeline<Root, Owner>`, `GpuGraphicsPipeline<Root, Owner>` | Typed pipeline wrappers and ordinary resource methods. | Decorated shader identity, stage/root compatibility, and trusted pipeline construction/recording. |

This inventory includes the typed pipeline formers, not only the five originally
named wrappers. Internal `GpuArguments` may remain an opaque representation for
recording if needed; it is not a public reusable host-to-device conversion API.
Inventory constructor recognition, builtin signatures, grammar reservations,
editor labels, HIR/concrete type variants, layout rules, copying, verification,
and both emitters before deleting each old family.

Use registered declarations with explicit intrinsic implementations for necessary
low-level operations. Source wrappers use ordinary scheme application, imports,
method lookup, and diagnostics. Compiler recognition must describe the primitive
contract and provenance, not depend on a user-visible wrapper's spelling or the
accidental order of its fields. Keep primitive registration in
[HIR context construction](../crates/resin-hir/src/lower/context.rs); add no public
solver or backend callbacks to make library declarations work.

### Ownership is a semantic prerequisite

An ordinary record containing a raw allocation pointer and a `drop` method does
not implement `Arc`: implicit record copies would duplicate that pointer without
retaining it. Source `clone` methods alone cannot repair reads, argument passing,
aggregate construction, replacement, or early-return cleanup.

Choose a narrow primitive managed-ownership representation whose ordinary copies
retain and whose destruction releases. An `Arc<T>` wrapper can then compose that
ownership with typed payload access. `Weak<T>` must retain the control allocation
without keeping the payload alive, and upgrade must acquire a live strong owner
atomically where the runtime contract requires it. Preserve destruction order,
custom payload hooks, and empty/expired weak behavior. The exact primitive fields,
control-block layout, and runtime ABI remain implementation decisions to review
before this layer; a new allocation-header scheme is not assumed.

Apply the resulting copy/drop rules recursively to nominal records and retain the
existing host-only rules for managed values. Shader code cannot gain permission to
copy an owner merely because its former builtin variant became a nominal wrapper.
Avoid introducing general user-defined copy or operator overloading solely to
complete this migration.

Current aliases such as `Gpu = Arc<GpuOwner>` also rely on compiler receiver and
field adaptation through the shared owner. An ordinary nominal `Arc<T>` does not
automatically inherit `T`'s namespace. The migration must choose explicit resource
wrapper methods or a narrow payload-access/receiver protocol; aliases still cannot
add methods. Preserve or deliberately replace these call sites in the same layer.

### GPU access and projection keep their guarantees

The existing [GPU contract](gpu-buffers.md) includes owning field addresses,
checked indexing/slicing, permission narrowing, mapping/range/alignment checks,
and rejection of host access while recording holds an allocation. Those guarantees
must survive removal of `GpuPointer` and `GpuSpan` type variants.

Additional library data members can describe a view's bounds and permissions, but
the checked access operation must consume and enforce them. Copies and derived
views preserve the owner and cannot restore permissions that were removed. An
unchecked public `host: Ptr<T>` escape would bypass mapping and recording checks;
it is not sufficient to hide checks inside `.at()` while dereference or field
addressing can recover an unrestricted pointer. Specify the primitive access/place
contract before choosing the wrapper's representation. OS page protection is not
required by this design.

Ordinary methods can implement indexing and slicing, while any retained shorthand
for dereference or field addresses needs a narrow compiler protocol preserving
ownership and access checks. The exact protocol is still to be designed. It must
operate on a checked capability, avoid general operator overloading, and preserve
the distinction between acquiring an address, reading, and writing. Backend and
verifier rules must express the same contract.

Only compiler-builtin projection used during pipeline dispatch/draw converts host
GPU views into shader pointers. Pipeline creation continues to require decorated
declaration identities, and shader entry parameters remain pointer-based. Projection
uses the declared root shape, keeps view offsets, validates GPU/context and access
compatibility, and retains every referenced allocation through recording completion.
There is no public device-address query, raw-pointer cast, or reusable projected
root. Ordinary generic pipeline wrappers must retain enough trusted provenance to
prevent fabricated owners or stage/root substitutions from bypassing those checks.

## API and library migration

The target result shapes are `gpu.create::<T>() -> Result<GpuPtr<T>, RuntimeError>`
and `gpu.alloc::<T>(count) -> Result<GpuSpan<T>, RuntimeError>`. Expected Result or
post-`?` context can determine `T`; `count` cannot. These are ordinary generic
methods, including when `Gpu` itself is an alias of an applied owner type.
The zero-argument `create` spelling still needs an explicit initialization policy;
this plan does not promise zeroing or silently replace today's initialized
`gpu.new(value)` semantics. Settle that policy in the API layer and test it.

Migrate all modules under [resin/](../resin/README.md), including strings,
formatting, I/O, images, windows, and graphics, to the source wrappers. Update
explicit exports/imports, examples, and documentation together. `str` stays the
distinct primitive literal type; ordinary byte spans must not gain implicit string
conversion. Preserve native pointer/count ABI requirements and document any
intentional source API changes. Library module paths and default availability of
the wrappers remain concrete choices for the corresponding migration PR.

## Reviewable implementation sequence

Manage the following implementation branches with `gh stack`. Each layer must
leave ordinary programs usable and remove the replaced path when its replacement
is complete; the numbers below are an order, not assigned PR numbers.

1. **Source generic structs.** Construct nominal schemes, applied field/constructor
   types, recursive identities, and retained editor/import facts. Accept when source
   examples exercise the public HIR/LIR nominal support, distinct applications keep
   distinct origins/layouts, and missing arguments and invalid fields diagnose at
   the proper boundary. Include generic payload destruction and local captures.
2. **Owner and method applications.** Complete separate owner/method binders through
   receiver and associated calls, explicit turbofish, and contextual results. Accept
   when expected types infer a method-only parameter on a nongeneric owner, aliases
   preserve lookup, and HIR records every argument before LIR. Test ambiguity,
   conflicting context, recursive reuse, and the per-source-function budget.
3. **Dependent method relations.** Add the HIR signature/call forms needed when
   a receiver's nominal origin is known only after substitution. Accept when calls
   retain complete determining types, LIR resolves them without inference, and
   unsupported or underdetermined cases report the source/application responsible.
   Test explicit method arguments, dependent numeric contexts, and absent methods.
4. **Ordinary spans and primitive operations.** Define `Span<T>` in Resin and migrate
   its indexing, slicing, literal borrowing, and ABI uses. Accept when host/shader
   callers and byte layout agree, source imports/editor queries see the declaration,
   and no compiler polymorphic span representation or constructor dispatch remains.
5. **Managed ownership and source `Arc`/`Weak`.** Review the primitive ownership
   contract, then replace typed compiler wrappers and migrate dependent owners.
   Accept when copies across calls/aggregates/replacement retain correctly, weak
   expiry and custom payload destruction work, and shader rejection remains intact.
6. **GPU views and typed allocation.** Review checked access and initialization,
   implement source `GpuPtr`/`GpuSpan`, and expose `create`/`alloc`. Accept when
   inferred/explicit element types use the ordinary method path, offsets and field
   addresses retain ownership, and permission/mapping/recording checks cannot escape.
7. **Pipeline wrappers and dispatch projection.** Replace typed pipeline formers
   with library wrappers over narrow trusted operations. Accept when stage/root
   mismatches and raw host pointers are rejected, projection runs only while
   recording, and all referenced allocations survive submit/cancel and aliases.
8. **Complete library/API migration.** Remove remaining special registrations,
   grammar reservations, representations, and stale examples after migrating every
   library client. Accept when the inventory above is exhausted and the docs describe
   ordinary source APIs with only the intended compiler primitives remaining.

For each layer, add behavioral regressions at its language boundary and relevant
end-to-end cases. Preserve immutable editor results after edits and independent
HIR diagnostics in unused definitions. Use the workspace checks in the development
shell; GPU and window validation must require their dependencies rather than skip
silently. Full migration includes host C, SPIR-V, and resource-lifetime examples.
