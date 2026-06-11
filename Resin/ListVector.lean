/-! ListVector T n = Vector T n but with lists, so no need for native_decide. -/

def ListVector (T : Type) (n : Nat) := { lst: List T // lst.length = n }

def ListVector.mk
    {T : Type}
    {n : Nat}
    (lst : List T)
    (h : lst.length = n := by rfl)
    : ListVector T n :=
  ⟨lst, h⟩

def ListVector.zip
    {T U : Type}
    {n : Nat}
    (v1 : ListVector T n)
    (v2 : ListVector U n)
    : ListVector (T × U) n :=
  ListVector.mk (List.zip v1.val v2.val) (by simp [v1.property, v2.property])

def ListVector.all
    {T : Type}
    {n : Nat}
    (v : ListVector T n)
    (p : T → Bool)
    : Bool :=
  List.all v.val p

def ListVector.map
    {T U : Type}
    {n : Nat}
    (f : T → U)
    (v : ListVector T n)
    : ListVector U n :=
  ListVector.mk (List.map f v.val) (by simp [v.property])

def ListVector.foldl
    {T U : Type}
    {n : Nat}
    (f : U → T → U)
    (init : U)
    (v : ListVector T n)
    : U :=
  List.foldl f init v.val
