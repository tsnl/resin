# Language reference

The reference describes the current language. Each topic owns its semantic rules;
the tutorial links here when a worked example depends on them.

- [Functions and control flow](syntax.md): declarations, bindings, expressions, loops.
- [Types and generics](generics.md): numeric inference, holes, aliases, specialization.
- [Unions and errors](errors.md) and [optional values](options.md): `None`, `Err<E>`, `match`, `?`, and `!`.
- [References and pointers](references.md): places, `Ref<T>`, `Ptr<T>`, `:at`, `:lea`.
- [Ownership](lifetimes.md): moves, copying, explicit cloning, cleanup, and unchecked lifetimes.
- [Operations](methods.md) and [operators](operators.md): free functions, colon calls, overloads.
- [Modules](modules.md), [foreign functions](ffi.md), and [representation](representation.md).

## Guarantees and responsibilities

References expose access without granting pointer creation. This distinction is
preserved and checked through specialization and LIR. A reference does not retain
its owner or track its lifetime. Callers must keep its referent alive.

CPU and GPU functions share source semantics for the supported shader subset.
Device storage has a narrower representation contract than shader-local values.
Host indexing diagnoses invalid indices; shader indexing is unchecked. A program
must stay within valid storage on either target.

The [diagnostics chapter](diagnostics.md) explains where the compiler establishes
its guarantees. Target limitations are recorded with the relevant feature, rather
than being silently generalized into language rules.
