# Coding taste: what Resin takes from Bitwise

Code communicates a model of the problem. A reader should be able to discover the
important objects, follow what happens to them, and predict where a change belongs.
That is the quality Resin takes from Per Vognsen's
[Bitwise](https://github.com/pervognsen/bitwise) project. Its educational purpose is
useful to us: the implementation has to explain the machinery while making it work.
A compiler that teaches its own structure is easier to debug and extend.

The most relevant reference is Ion, Bitwise's C implementation of a programming
language compiler. The examples here link to commit
[`5a261e9`](https://github.com/pervognsen/bitwise/tree/5a261e99efea080e1111a312d897f8d794f061a7/ion),
so each observation has a concrete source. This essay describes Resin's reading of
that code. The Rust sketches are illustrative adaptations; the linked C is the
original implementation.

The attraction is the small amount of machinery between an idea and its expression.
A binary expression has an operator and two operands. Parsing an operator constructs
that expression. A type constructor returns a type whose required properties are
already established. Useful abstractions let the reader reason at this level while
keeping their implementation close enough to inspect.

## The representation explains the operations

Ion's [`Expr`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/ast.h#L218-L298)
is a tagged union. Its binary case contains `op`, `left`, and `right`.
[`new_expr_binary`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/ast.c#L333-L339)
allocates that case, fills those fields, and returns the node. The shape of the
constructor follows directly from the shape of the data.

An analogous Rust construction is immediately legible:

```rust
Expr::Binary {
    op,
    left: Box::new(left),
    right: Box::new(right),
}
```

A reader can understand the result by inspecting the expression variant. The
construction does not require knowledge of how the node will later be visited or
emitted. This gives syntax a useful independence from the operations performed on it.

Ion's shared [`new_expr`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/ast.c#L195-L200)
handles allocation, the tag, and source position. There is a reason for that helper:
those details must agree across every expression constructor. In Rust, an enum
already keeps the tag and payload consistent. A direct enum literal can therefore
be the clearest construction; an extra constructor earns its place when it
establishes an additional invariant or names a meaningful conversion.

The same quality matters in Resin's
[HIR language](../crates/hir/src/language.rs). `Term` exposes its source span,
concrete type, and expression kind. A call or conditional is recognizable as data.
Reading that definition should be enough to understand what the next pass receives.
If understanding a node requires consulting the checker's current scope or invoking
an inference callback, the representation has left part of its meaning behind.

## Control flow follows the subject

Ion's [`parse_expr_add` and `parse_expr_mul`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/parse.c#L319-L343)
make precedence visible in their call graph. Addition first parses a complete
multiplicative expression. It then loops over addition-level operators, obtaining
another multiplicative expression for each right operand and replacing the
accumulated left expression with a binary node.

Two consequences can be worked out directly from that loop:

```text
a + b * c   becomes   Add(a, Multiply(b, c))
a - b - c   becomes   Subtract(Subtract(a, b), c)
```

These are explanatory tree shapes. The first follows from the call to the tighter
precedence level. The second follows from carrying the accumulated expression
through the loop. The code itself supplies the explanation of precedence and
left associativity.

The neighboring parsing functions repeat some structure. That repetition makes the
grammar easy to survey: each level has a name, and each function shows which level
it consumes. A shared implementation would be worthwhile if it made the grammar
and its exceptions easier to see. Saving a handful of loop statements alone does
not establish that benefit.

For Resin, the transferable quality is the correspondence between an operation
and the problem it expresses. Tree-sitter owns our parsing grammar; there is no
reason to replace it to reproduce Ion's parser. In lowering, the analogous goal
is for a reader to trace a conditional into branches and a result propagation into
an error test, cleanup, and an early return. The important decisions should be
visible in the ordinary control flow of the pass.

## A helper owns a complete decision

Consider Ion's [`type_ptr`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/type.c#L267-L279).
It looks up a pointer type by its base type. On a miss, it allocates the type, sets
its size, alignment, and base, and caches it. Callers receive the canonical object
for that base without coordinating allocation and registration themselves.

A Rust sketch of the interning decision makes the owning context explicit:

```rust
fn pointer_type(&mut self, base: TypeId) -> TypeId {
    if let Some(&id) = self.pointer_types.get(&base) {
        return id;
    }
    let id = self.insert(Type::Pointer { base });
    self.pointer_types.insert(base, id);
    id
}
```

The useful boundary is the whole operation of obtaining a canonical pointer type.
Every caller benefits from the same guarantee. Splitting allocation, initialization,
and registration into independently required public calls would transfer that
coordination back to each caller. The resulting API could look more flexible while
being easier to misuse.

This is also a useful way to judge reuse. A good shared function answers the same
question for each caller. Two blocks with similar syntax may implement different
rules; two blocks with different syntax may be making the same ownership decision.
The latter are often the more valuable extraction. A future edit should change the
meaning in one place and have all affected callers receive that change.

A helper's name should promise exactly what it establishes. An operation called
`validate` that also chooses defaults and allocates persistent state makes the
reader learn a second, hidden contract. Separate operations can express those
steps directly, with a small coordinating function showing their order.

## Small pieces can have substantial reach

Ion's [common utilities](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/common.c#L95-L212)
include growing buffers and an arena. Their data makes their behavior explainable:
the buffer tracks length and capacity; the arena tracks a current position, an end,
and the blocks it owns. Growing a buffer and advancing an arena allocation are
operations a reader can trace through a small implementation.

These pieces are useful across many compiler tasks because the underlying needs
are common. Collecting arguments, declarations, or fields all needs a sequence.
Allocating syntax nodes with one shared lifetime needs a storage owner. Reuse comes
from these recurring needs, with the compiler's language rules remaining in the
code that uses the utilities.

The arena example also shows how a short constructor can establish ownership.
[`new_expr_call`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/ast.c#L304-L310)
copies the argument-pointer array into AST storage through
[`ast_dup`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/ast.c#L13-L22).
That shallow copy gives the array the AST's lifetime. It does not recursively clone
the expressions. Understanding that distinction is enough to explain why the
parser's temporary argument collection can have a separate lifetime.

Resin already has `Vec`, `Box`, maps, and explicit Rust ownership. Those are usually
the appropriate pieces for expressing the same relationships. Reimplementing a
buffer would add code we must explain and maintain. A custom utility earns its
place through a concrete need that the existing vocabulary expresses poorly.

## Small functions preserve a coherent thought

Ion's [`parse_expr_base`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/parse.c#L255-L286)
is longer than its precedence helpers. It starts with an operand and repeatedly
attaches calls, indexes, fields, or postfix modifications. For `f(x)[i].field`, a
single local expression accumulates each layer. Keeping those alternatives together
makes the complete postfix grammar visible.

This is why Resin's preference for functions with fewer than ten lines of logic
is a prompt to examine a boundary. A longer function may express one coherent
operation. A short function can still be difficult if it forwards through several
layers of vaguely named helpers. The relevant cost is how much a reader must hold
in mind or go elsewhere to recover.

Exhaustive dispatch often benefits from staying together. Ion's
[`gen_expr`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/gen.c#L613)
dispatches over expression kinds. Such a dispatch serves as an inventory of the
language handled by an operation. In Rust, an exhaustive `match` additionally makes
new variants identify the operations that need attention. Extract a case when it
has a useful name and independent reasoning; keep a simple case visible when an
extra jump would obscure the inventory.

## Comments explain what the code cannot show

Descriptive data and direct control flow reduce the need for narration. Explaining
that an assignment stores a field adds little to a constructor whose field names
already say so. Explaining why an argument array must outlive the parser, however,
communicates a relationship that a memory copy alone cannot establish for the reader.

Assertions can make some of that reasoning executable. Ion's
[`arena_alloc`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/common.c#L192-L203)
checks available space and alignment around the allocation. These checks expose
assumptions the implementation relies on. They help a reader connect the compact
operation to the conditions that make it valid.

Resin's [LIR definition](../crates/lir/src/language.rs) has a useful comment explaining
that local zero is always the parameter, including for unit and foreign functions.
A vector of locals cannot express that convention by itself. Keeping the explanation
beside the representation and checking it in the verifier is more useful than
scattering reminders through its callers.

Low comment density should emerge from clear code. Removing explanations of
invariants, ownership, or deliberate compromises would make the implementation
harder to understand. Conversely, when a comment needs to describe a long sequence
of bookkeeping steps, there may be a missing named operation or a representation
that requires too much coordination.

## Carry the taste into Resin's architecture

Ion makes tradeoffs specific to its implementation. Its
[`main.c`](https://github.com/pervognsen/bitwise/blob/5a261e99efea080e1111a312d897f8d794f061a7/ion/main.c#L1-L25)
includes the implementation files into one translation unit. Its parser uses global
token state, and the type cache above is global too. Those choices make some local
code compact, but their state and lifetime assumptions still exist. They matter
when adapting the style to a reusable compiler library or a long-lived editor.

Resin's [phase architecture](architecture.md) gives those assumptions explicit
homes. HIR construction owns scopes and inference. Resolved HIR is data that LIR
lowering can consume after construction state has been discarded. The driver owns
pass sequencing; language crates describe and transform their own languages.
This lets each component be understood in terms of a bounded input and output.

The [verification certificate](../crates/lir-verifier/src/lib.rs) is a concrete
example of an abstraction earning its place:

```rust
let checked = lir_verifier::VerifiedModule::new(module)?;
let target = c::lower::generate(checked.view(), "main", &[])?;
let text = c::print::module(&target);
```

The wrapper keeps a module with the analysis that certifies it. The caller can
borrow the certificate for generation; obtaining editable LIR consumes the wrapper
and discards the certificate. This prevents a verified flag from surviving an edit
to the data it describes. The target printer then receives a completed target tree,
so it needs neither the verifier nor the frontend's construction state.

The same discipline should guide an edit after finding a bug. First trace the
operation and the facts it needs. Then ask where each fact is established, how it
travels, and which callers must remember to update it. A bug caused by two paths
making the same decision suggests one shared operation. A bug caused by state
getting lost suggests representing that state at the boundary where it is needed.
A new callback or a parallel bookkeeping map is useful only if it makes that
relationship clearer and gives it an identifiable owner.

A successful cleanup leaves the next reader with fewer facts to coordinate.
They can see what a value means, which operation establishes its guarantees, and
where to make a change. That is the practical standard behind Resin's Bitwise
reference, and the reason to keep returning to the actual code when discussing taste.
