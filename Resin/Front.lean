import Resin.AddressMath
import Resin.TensorLiteral

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
      {srcRank : Nat}
      {srcShape : Shape srcRank}
      (src : Node srcShape)
      (view : View shape)
      : Node shape
  | mul
      {rank : Nat}
      {shape : Shape rank}
      (lhs rhs : Node shape)
      : Node shape
