# Operator overloading

Declare free functions with Python-style operator names:

```resin
struct Vec2 { x: int, y: int }
fn __add__(left: Ref<Vec2>, right: Ref<Vec2>) -> Vec2 {
	Vec2 { x = left.x + right.x, y = left.y + right.y }
}
fn __mul__(scale: int, value: Ref<Vec2>) -> Vec2 {
	Vec2 { x = scale * value.x, y = scale * value.y }
}
```

Both operands participate in resolution. These signatures borrow vectors;
value parameters would move them. Operands evaluate once, left to right.
`__add__(left, right)` and `left:__add__(right)` select the same overload as
`left + right`. Export the function to make it visible in another module.
See the [operator mapping](methods.md#operator-overloading) and
[vector example](../examples/operators.resin).
