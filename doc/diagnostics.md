# Compiler diagnostics and phase guarantees

A diagnostic describes the failed operation, its source range, and any related
locations. The terminal prints a source excerpt and notes. The editor displays a
compact message, its diagnostic code when available, and related locations.
Messages contain source language concepts; internal function/block numbers belong
to compiler failure reports.

## Which phase establishes which guarantee?

| Phase | Completed guarantee |
| --- | --- |
| HIR construction | Resolved bindings and source-known types; definite initialization, checked moves, and reference/address-taking rules |
| Specialization | Concrete signatures and operations satisfy the same rules after substituting generic arguments |
| LIR verification | Calls, returns, projections, casts, stack operations, and control-flow regions obey their concrete contracts |
| Target lowering | Operations have a supported C or SPIR-V representation; representation failures retain their source origin |
| Native tools | Generated code has compiled and shaders have passed the configured optimizer |

`Ref<T>`, `RefMut<T>`, and `Ptr<T>` stay distinct through verification.
Writes require writable access; read-only references cannot regain that permission. Neither specialization
nor verification has a conversion from a reference to a pointer. See
[References](references.md) for the language contract and unchecked lifetimes.

## Failures after specialization

Later failures retain their diagnostic data through cache publication and the
compiler service. Generic application notes identify the chain that requested an
instance. The following codes are available:

| Code | Meaning |
| --- | --- |
| `invalid-specialization` | Substituted types make a source operation invalid |
| `unsupported-profile` | An operation or value is unavailable in the selected CPU/GPU profile |
| `specialization-limit` | A function exceeded its configured instance allowance |
| `type-expansion-limit` | A type exceeded the compiler's depth or size bound |
| `unsupported-target-feature` | A valid intermediate operation has no supported target representation |
| `invalid-entry` | The selected native entry has an unsupported signature |
| `invalid-hir`, `invalid-target-input` | A compiler invariant failed; report a compiler bug |

Frontend diagnostics do not yet all have individual codes. Editor analysis currently
stops at HIR; specialization and target diagnostics are reported when building the
selected entry. Target restrictions include returning shader-local references and
joining distinct shader-local referents.

Client and server negotiate protocol version 2, which carries diagnostic codes,
notes, and help text. Upgrade them together. External compiler and shader-tool
failures retain their tool output; they are separate from source diagnostics.
