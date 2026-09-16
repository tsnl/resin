# Inherent methods

A struct owns its methods directly. Declare fields first, followed by `def`
statements inside the same braces. There are no separate implementation blocks.
Transparent aliases inherit the underlying nominal type's namespace and origin;
they cannot add methods. Local structs currently contain fields only.

All module types and aliases are available when method signatures are resolved,
including an alias written after its target struct. An alias of `ArcPtr<Owner>`
inherits the wrapper namespace; payload methods require `.get()`. Methods are declared
before bodies, so sibling methods and recursive calls can refer to each other.

Functions accompany the type when it is exported and need no separate exports.
There are no user-defined traits, interfaces, or dynamic dispatch.

```resin
struct Counter { value: int,
    def new(value: int) -> Counter = { Counter { value = value } };
    def increment(counter: Ptr<Counter>) = { counter.value := counter.value + 1; };
    def read(counter: Counter) -> int = { counter.value };
};

def example() -> int = {
    var counter = Counter.new(41);
    counter.increment();
    Counter.read(counter)
};
```

`counter.read()` supplies `counter` as the first argument to `Counter.read`.
The first parameter may have any name, including `self`; it is checked at the
call. A pointer receiver takes the address of an initialized place when needed,
and a value receiver loads a matching pointer. `Counter.read(counter)` supplies
all arguments explicitly, like any other function call.

The parser distinguishes method invocation from calling a function-valued field:

```resin
counter.read(args)    // invoke the function in the receiver type's namespace
(counter.read)(args)  // read the field, then call its value
```

The distinction also applies when a field and method have the same name. Methods
can return `Ref<T>` to expose a place, including wrappers around array indexing. Parameter
and result types are explicit; an omitted result means unit.

Arrays provide a compiler-defined `.at(index)` method; the source `Span<T>`
wrapper exposes the same signature through an ordinary method.
It accepts a `ulong` index and returns `Ref<T>`, so a field can be indexed as
`root.particles.at(i)`. Use `items.at(i) := value` to update an element.
The receiver and index are evaluated once; indexing an array place keeps its
storage address instead of copying the array. Arrays also support the existing
`items(i)` spelling. Host indexing checks the declared length and diagnoses invalid
indices; shader indexing is unchecked, so callers must stay within valid storage.

Source-known method resolution happens during HIR construction. Dependent lookup
uses the completed HIR nominal declarations during specialization. Both produce
ordinary function values and calls in LIR; LIR types have no method tables, and the
verifier checks these functions and calls using its existing rules.

A generic function can invoke a method before its receiver's nominal type is known:

```resin
def read<T>(value: T) -> _ = { value.read() };
def replace<T, U>(value: T, next: U) -> _ = {
    value.replace_with::<U>(next)
};
```

The concrete application selects the source-declared method and checks its signature.
Field lookup follows the same rule, so method and field accesses can be chained.
Associated calls and references such as `T.make::<U>()` and `T.make::<U>` also
retain their lookup until substitution. The receiver and arguments are evaluated
once, in source order, with the ordinary pointer receiver adaptation rules.

When the receiver's namespace is unknown, extra method parameters must be supplied
explicitly; omitting `::<...>` supplies zero extra arguments. HIR completes inference
before specialization, which performs no deduction. Primitive compiler-provided
methods still require a source-known receiver shape. Operations requiring a known
pointer or Result shape may need a result annotation before `.*` or `?` can be used.

## Operator overloading

Structs implement operators with Python-style dunder methods:

```resin
struct Vec2<T> { x: T, y: T,
    def __add__(left: Vec2<T>, right: Vec2<T>) -> Vec2<T> = {
        Vec2<T> { x = left.x + right.x, y = left.y + right.y }
    };
    def __neg__(value: Vec2<T>) -> Vec2<T> = {
        Vec2<T> { x = -value.x, y = -value.y }
    };
};

def add<T>(left: T, right: T) -> _ = { left + right };
```

`left + right` selects `__add__` on the left operand's nominal type; `-value`
selects `__neg__` on its operand's type. The same generic `add` function works
with primitive numbers and structs. Transparent aliases inherit the original
struct's operators. Exporting a struct makes its operator methods available to
importers.

| Expression | Method |
| --- | --- |
| `+value` | `__pos__` |
| `-value` | `__neg__` |
| `~value` | `__invert__` |
| `!value` | `__not__` |
| `left + right` | `__add__` |
| `left - right` | `__sub__` |
| `left * right` | `__mul__` |
| `left / right` | `__truediv__` |
| `left % right` | `__mod__` |
| `left << right`, `left >> right` | `__lshift__`, `__rshift__` |
| `left & right`, `left \| right`, `left ^ right` | `__and__`, `__or__`, `__xor__` |
| `left == right`, `left != right` | `__eq__`, `__ne__` |
| `left < right`, `left <= right` | `__lt__`, `__le__` |
| `left > right`, `left >= right` | `__gt__`, `__ge__` |

A unary method takes one parameter; a binary method takes two. The first
parameter must be the owning struct by value, with its declared type parameters.
The second parameter and result may have different types; for example,
`__mul__(value: Vec2<T>, scale: T) -> Vec2<T>` supports vector scaling.
Comparisons and `__not__` must return `bool`. Each method name has one declaration,
and its signature is checked even if unused. Operators inherit their struct's type
parameters and cannot add method-local type parameters.

Dunder methods also support ordinary calls and function references:
`left.__add__(right)`, `Vec2<int>.__add__(left, right)`, and
`var add_vectors = Vec2<int>.__add__;`. These forms share the same function
specializations as operator expressions. Hover and go-to-definition on an operator
symbol show its selected method.

Operands are evaluated once, left to right, with ordinary copying and cleanup.
Operator lookup reads `Ref<T>` operands as values and does not automatically
dereference pointers or take addresses. Precedence remains unchanged. Source-known
operators resolve during HIR construction; generic operators retain signature
queries until specialization selects a primitive operation or source method.
Both CPU and GPU code use the resulting ordinary operations and calls.

`__not__` is a Resin-specific extension to Python's naming convention. Conditions
still require `bool`; Resin does not implicitly call `__bool__`. Primitive `/`
retains Resin's existing numeric semantics, including integer division. Every
comparison selects its own method: `__eq__` does not supply `__ne__` automatically.
Reflected methods such as `__radd__` and in-place methods such as `__iadd__` have
no special dispatch. Assignment, address-of, dereference, indexing, postfix `!`
and `?`, and short-circuit `&&` / `||` retain their existing behavior. Constant
expressions do not execute operator methods.

Run `cargo run -- examples/operators.resin` for a generic vector example.

## Shader and ownership rules

Methods cannot be shader entry points, but shader helpers can call them.
Pointer receivers follow the existing GPU address restrictions: a shader-local
address cannot escape into a callee.

Shared wrappers have their own method namespaces. Use `owner.get().method()`
to call a payload method through its borrowed address. The compiler invokes the reserved
`drop(self: Ptr<T>)` hook during cleanup; see [lifetimes](lifetimes.md).
