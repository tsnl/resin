- [ ] How to use proof tactics? `by rfl`, `by simpa using ...`, etc.
- [ ] How to explain this thing:

  ```lean
  inductive Node : {rank : Nat} → Shape rank → Type where
    | const
        {rank : Nat}
        {shape : Shape rank}
        (literal : TensorLiteral shape)
        : Node shape
    | param
        {rank : Nat}
        {shape : Shape rank}
        (id : Nat)
        : Node shape
    | view
        {rank : Nat}
        {shape : Shape rank}
        {rank' : Nat}
        {shape' : Shape rank'}
        (src : Node shape')
        : Node shape
    | mul
        {rank : Nat}
        {shape : Shape rank}
        (lhs rhs : Node shape)
        : Node shape
  ```
