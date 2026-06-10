/-- Scalar datatypes supported by GPUs. -/
inductive DType where
  | f32 : DType
  | f64 : DType
  | i32 : DType
  | i64 : DType

/-- Buffer encapsulates memory written to by an operation. -/
structure Buffer where
  count : Nat
  dType : DType
  args : List Buffer

def maxViewOffset (offset : Nat) (shape pitch : List Nat)
    (rank_eq : pitch.length = shape.length) : Nat :=
  let rec loop (acc : Nat) (shape pitch : List Nat)
      (h : pitch.length = shape.length) : Nat :=
    match shape, pitch, h with
    | [], [], _ => acc
    | s :: ss, p :: pp, h => loop (acc + (s - 1) * p) ss pp (by simpa using h)
  loop offset shape pitch rank_eq

structure BufferView (shape : List Nat) where
  buffer : Buffer
  offset : Nat
  pitch : List Nat
  rank_eq : pitch.length = shape.length := by rfl
  max_offset_lt : maxViewOffset offset shape pitch rank_eq < buffer.count := by rfl

def hello : IO Unit :=
  IO.println "Hello, world"
