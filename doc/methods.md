# Inherent methods

An `impl` block adds functions to a nominal struct defined in the same module.
Methods accompany the type when it is exported; they do not need separate exports.
There are no user-defined traits or interfaces.

```resin
struct Counter { value: int };
impl Counter {
    def new(value: int) -> Counter = { Counter { value = value } };
    def increment(self: Ptr<Counter>) = { self.value := self.value + 1; };
    def read(self: Counter) -> int = { self.value };
}
def example() -> int = {
    var counter = Counter.new(41);
    counter.increment();
    counter.read()
};
```

The first parameter named `self` makes an instance method. It can take the
struct by value or `Ptr<T>`; a call takes the address of an initialized place
when a pointer receiver is required, or loads a pointer for a value receiver.
Functions without `self` are called on the type. Method parameters and result
types are explicit; an omitted result means unit.

Methods can return ordinary pointers, including wrappers around array indexing.
Calling a function-valued record field continues to work. Methods cannot be
shader entry points, but shader helpers can call them.
Pointer receivers follow the existing GPU address restrictions: a shader-local
address cannot escape into a callee.
