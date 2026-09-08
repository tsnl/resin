# Inherent methods

An `impl` block adds functions to a nominal type's namespace. Each nominal type
records its defining module; only that module can add functions. Transparent
aliases use the underlying nominal type's namespace and origin. An alias of a
primitive or anonymous record does not create a new nominal namespace.

Functions accompany the type when it is exported and need no separate exports.
There are no user-defined traits, interfaces, or dynamic dispatch.

```resin
struct Counter { value: int };
impl Counter {
    def new(value: int) -> Counter = { Counter { value = value } };
    def increment(counter: Ptr<Counter>) = { counter.value := counter.value + 1; };
    def read(counter: Counter) -> int = { counter.value };
}
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
can return ordinary pointers, including wrappers around array indexing. Parameter
and result types are explicit; an omitted result means unit.

Method resolution and module origins belong to the frontend. Lowering emits
ordinary function values and calls; IR types have no method tables, and the IR
verifier checks these functions and calls using its existing rules.

Methods cannot be shader entry points, but shader helpers can call them.
Pointer receivers follow the existing GPU address restrictions: a shader-local
address cannot escape into a callee.
