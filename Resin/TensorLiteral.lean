import Resin.ListVector
import Resin.AddressMath

/-! TensorLiteral is a constant tensor value to embed within IR nodes. -/

/-- Recurse on the raw shape list. `List Nat` is an inductive type, so this is structural
    recursion; recursing on `Shape rank` directly fails because the subtype's `rank` parameter
    is not fixed across the recursive call. -/
def TensorLiteralOfList : List Nat → Type
  | [] => Float
  | s :: ss => ListVector (TensorLiteralOfList ss) s

def TensorLiteral {rank : Nat} (shape : Shape rank) : Type :=
  TensorLiteralOfList shape.val
