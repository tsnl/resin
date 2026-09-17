# Documentation comments

Documentation is Markdown attached to a declaration. The compiler retains it for
editor hovers, and the local `resin --doc` command renders exported API documentation
from the same parsed source. It has no effect on executable code.

## Declarations and fields

Use `///` immediately before a function, struct, alias, constant, foreign declaration,
intrinsic declaration, or struct field:

```resin
//! A small counter module.
export { Counter, increment };

/// An integer counter.
struct Counter {
    /// The current count.
    value: i32,
}

/// Increase the count by one.
///
/// Borrows access without granting the caller a pointer.
fn increment(counter: RefMut<Counter>) {
    counter.value = counter.value + 1;
}
```

Consecutive comments join with newlines; a bare `///` makes a blank Markdown line.
Keep list indentation and code fences after the marker. Documentation before shader
decorators attaches to the decorated function. Documentation before a grouped
`const` declaration applies to each binding; comments on individual specifications
append to that group documentation.

`//!` documents the containing file module. Put it at the start of the file, before
`export`, `extern`, `import`, or any declaration. Ordinary license comments can
precede it. Inner documentation inside a function or struct is not supported.

## Block form

`/** ... */` documents the following declaration; `/*! ... */` documents the module.
A conventional leading `*` on each continuation line is stripped. Common indentation
is removed while preserving relative indentation, so Markdown lists and code remain
readable. Ordinary Resin block comments, including documentation blocks, do not nest.

```resin
/** Read a counter.
 *
 * Returns its current value without changing it.
 */
fn read(counter: Ref<Counter>) -> i32 {
    counter.value
}
```

`//` and `/* ... */` are ordinary comments. `////`, `/*** ... */`, `/**/`, and
`/***/` are also ordinary comments, matching Rust's marker distinction. Misplaced
or trailing doc comments are diagnosed at the comment; they are never silently
attached to an expression, local value binding, or unrelated module clause.

## Editor and generated output

Hovering a declaration, a resolved use, a colon call, or a struct field shows its
signature followed by its Markdown, including declarations imported from another
module. Unsaved edits update that text with the rest of the editor analysis.

Generate Markdown locally, without a compiler service:

```sh
cargo run -- --doc resin/math.resin
cargo run -- --doc resin/math.resin -o math-api.md
```

The output contains module documentation, exported declaration signatures, and
fields of exported structs. Overloads retain their separate signatures and text.
It uses source signatures rather than inferred signatures; it does not resolve
imports, document re-exports from another file, or type-check example code blocks.
The book incorporates this output into its static library reference.

Markdown links are ordinary links. Rust-style automatic symbol links, documentation
attributes, and executable documentation tests are not implemented. Use the
repository's compiled tutorial tests for examples that need execution coverage.
