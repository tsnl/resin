import Resin.PNat
import Resin.ListVector

/-! Index contains math to translate integer tuples into address offsets (and vice-versa) when
    dealing with multi-dimensional arrays. -/

/-- Offset is a single integer identifying the position of an element in a flattened array.
    Every [`Index`] can be mapped to a unique Offset in a contiguous array, and vice-versa.

    Offsets may be negative by convention. I.e., our address space is ℤ by convention.
    Callers can restrict this to ℤ+ when mapping to specific devices. -/
def Offset := Int

/-- Shape is a tuple of integers representing the number of elements per-dimension of a
    multi-dimensional array.
    -/
def Shape (n : Nat) := ListVector Nat n

/-- Pitch is a tuple of integers representing the strides needed to move between elements in a
    multi-dimensional array. May be negative to allow for reverse traversal.
    - E.g. in a dense 1D array, pitch is (1,)
    - E.g. in a dense 2D array with row-major order, pitch is (num_cols, 1)
    Note that pitch, like shape, always counts in elements, not bytes. -/
def Pitch (n : Nat) := ListVector Int n

/-- Index is a tuple of integers identifying a single element in a multi-dimensional array. -/
def Index (n : Nat) := ListVector Nat n

def indexBoundsCheck {n : Nat} (index : Index n) (shape : Shape n) : Bool :=
  ListVector.all (ListVector.zip index shape) (fun (i, s) => i < s)

/-- Extent is the minimum and maximum offset of all elements in a [View]. -/

def computeMinOrMaxOffsetOfView
    {n : Nat}
    (baseOffset : Int)
    (zippedShapePitch : ListVector (PNat × Int) n)
    (pickDimIndex : PNat × Int → Int)
    : Int :=
  ListVector.foldl
    (fun acc sp => let (_, p) := sp; acc + (pickDimIndex sp * p))
    baseOffset
    zippedShapePitch

def minOffsetOfView
    {n : Nat}
    (baseOffset : Int)
    (zippedShapePitch : ListVector (PNat × Int) n)
    : Int :=
  let pickDimIndex sp : Int :=
    let (s, p) := sp
    if p < 0
    then s.val - 1  /- If pitch is negative, highest dimension index => lowest address -/
    else 0          /- If pitch is positive, lowest dimension index => lowest address -/
  computeMinOrMaxOffsetOfView baseOffset zippedShapePitch pickDimIndex

def maxOffsetOfView
    {n : Nat}
    (baseOffset : Int)
    (zippedShapePitch : ListVector (PNat × Int) n)
    : Int :=
  let pickDimIndex sp : Int :=
    let (s, p) := sp
    if p < 0
    then 0          /- If pitch is negative, lowest dimension index => highest address -/
    else s.val - 1  /- If pitch is positive, highest dimension index => highest address -/
  computeMinOrMaxOffsetOfView baseOffset zippedShapePitch pickDimIndex

/-- View is a subset of a dense multi-dimensional array's elements, identified by arithmetic
    progressions along each dimension.

    ## `baseOffset` is NOT `minOffsetOfView`

    Conceptually, we have an arithmetic progression along each dimension:
        base_d + i * pitch_d for i in [0, shape_d]
    where
        base_d is the address offset of the first element in this dimension's subarray
        shape_d is the number of elements in this dimension's subarray
        pitch_d is the stride (in elements) between consecutive elements

    The `base_d` term lets us express descending offsets with negative pitches while guaranteeing
    the subarray offsets are non-negative.

    We can compute the element offset for some index ⟨i_d⟩ as:
        def offset(index) = sum_d base_d + (i_d * pitch_d)

    By associativity of addition, we can group together the base_d offsets into a single
    `baseOffset`:
        def offset(index) = (baseOffset := sum_d base_d) + (sum_d (i_d * pitch_d))

    Thus, even though we have a single `baseOffset` field, we still have the full expressiveness of
    a separate base_d for each dimension, especially when it comes to negative pitches.

    This means the `baseOffset` field is NOT the minimum offset of the view.

    This also means the `minOffsetOfView` may be negative. This is a feature, not a bug: callers
    can map offsets into nonnegative address spaces post-hoc.
    -/

structure View
  {rank : Nat}
  (shape : Shape rank)
where
  baseOffset : Nat
  pitch : Pitch rank

def rankOfView
    {rank : Nat}
    {shape : Shape rank}
    (_ : View shape)
    : Nat :=
  rank

def offsetOfIndexInView
    {rank: Nat}
    {shape : Shape rank}
    (v : View shape)
    (i : Index rank)
    (_ : (indexBoundsCheck i shape) := by decide)
    : Int :=
  let ip := ListVector.zip i v.pitch
  ListVector.foldl
    (fun acc (i, p) => acc + i * p)
    v.baseOffset
    ip

example : 12 = (
    let view : View (ListVector.mk [3]) := { baseOffset := 10, pitch := ListVector.mk [2] }
    let index : Index _ := ListVector.mk [1]
    offsetOfIndexInView view index
  ) := by decide

example : 8 = (
    let view : View (ListVector.mk [3]) := { baseOffset := 10, pitch := ListVector.mk [-2] }
    let index : Index _ := ListVector.mk [1]
    offsetOfIndexInView view index
  ) := by decide

example : 114 = (
  let view : View (ListVector.mk [3, 4]) := { baseOffset := 100, pitch := ListVector.mk [10, 2] }
  let index : Index _ := (ListVector.mk [1, 2])
  offsetOfIndexInView view index
) := by decide
