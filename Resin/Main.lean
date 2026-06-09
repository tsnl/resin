inductive DType where
  | f32 : DType
  | f64 : DType
  | i32 : DType
  | i64 : DType

structure Buffer where
  shape : List Nat
  dtype : DType
  args  : List Buffer

structure View where
  name : String
  dtype : DType
  offset : Nat
  shape : List Nat
  pitch : List Nat
  /-- `pitch` holds one stride per axis, so it must match `shape`'s rank.
      `by rfl` discharges this automatically whenever both are concrete. -/
  pitch_len : pitch.length = shape.length := by rfl


structure Expr (shape : List Nat) (dtype : DType) where
  node : Buffer
  view : View
  shape_eq : view.shape = shape := by rfl
  dtype_eq : view.dtype = dtype := by rfl
  /- TODO: need to ensure view is node-compatible -/

/-- Elementwise add. Both operands must share the same `shape` and `dtype`;
    a mismatch is rejected by the type checker, not at runtime. The result
    carries the same shape. -/
def add {shape : List Nat} {dtype : DType}
    (lt rt : Expr shape dtype) : Expr shape dtype :=
  { node := { shape := shape, dtype := dtype, args := [lt.node, rt.node] },
    view := lt.view,
    shape_eq := lt.shape_eq,
    dtype_eq := lt.dtype_eq }

def main : IO Unit := do
  IO.println "Hello, world!"
