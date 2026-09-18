# Language reference

The reference describes the current language. Each topic owns its semantic rules;
the tutorial links here when a worked example depends on them.

- [Functions and control flow](syntax.md): bindings, UFCS, expressions, loops, assertions, and returns.
- [Types and generics](generics.md): primitives, structural and nominal types, aliases, and type parameters.
- [Overloads and SFINAE](overloads.md): candidate selection and substitution failure.
- [Inference](inference.md): contextual types, generic arguments, and explicit holes.
- [Unions, optionals, and errors](errors.md): `None`, `Err<E>`, `match`, `?`, and `!`.
- [References and pointers](references.md): places, `Ref<T>`, `RefMut<T>`, `Ptr<T>`, `PtrMut<T>`, and indexing.
- [Ownership](lifetimes.md): moves, copying, cloning, cleanup, and unchecked lifetimes.
- [Modules](modules.md), [documentation comments](documentation-comments.md),
  [foreign functions](ffi.md), and [representation](representation.md).

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
