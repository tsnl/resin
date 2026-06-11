/-! PNat = Positive Nat
    -/

def PNat: Type := { v: Nat // v > 0 }

def pNat? (n : Nat) : Option PNat :=
  if h : n > 0 then some ⟨n, h⟩ else none

example : pNat? 1 = some ⟨1, by decide⟩ := by rfl
example : pNat? 0 = none := by rfl
