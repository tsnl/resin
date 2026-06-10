def PNat: Type := { v: Nat // v > 0 }

def pNat? (n : Nat) : Option PNat :=
  if h : n > 0 then some ⟨n, h⟩ else none

example : pNat? 1 = some ⟨1, by decide⟩ := by rfl
example : pNat? 0 = none := by rfl

/-- Vector helpers
    -/

def Vector.zip3 {α β γ : Type _} {n : Nat}
  (v1 : Vector α n) (v2 : Vector β n) (v3 : Vector γ n) : Vector (α × β × γ) n :=
  -- Combine v1 and v2 into tuples, then combine the result with v3
  Vector.zipWith (fun a (bc : β × γ) => (a, bc.1, bc.2)) v1 (Vector.zipWith Prod.mk v2 v3)

def makeVector {α : Type} {n : Nat} (l : List α) (h : l.length = n := by rfl) : Vector α n :=
  ⟨l.toArray, h⟩
