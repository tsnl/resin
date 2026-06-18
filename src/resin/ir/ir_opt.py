"""
IR->IR optimization passes.

---

## Ideas for optimizations

### Kernel fusion and scheduling

-   **Elementwise RPN fusion** <br/>
    If we have a sequence of elementwise operations, we can fuse them into a single
    kernel.

-   **Matmul + RPN epilogue fused kernel** <br/>
    Support an RPN scalar epilogue after each matmul. Possibly also RPN prologue for
    each argument. But epilogue first.

-   **IrElementType = ScalarType | TiledScalarType** <br/>
    Replace ScalarType with IrElementType, which could be a scalar OR a tiled scalar
    type.
    E.g. mat4x4_f4
    When we perform a matmul on block matrices i.e. matrices with tile elements, we can
    use cooperative matrix operations.
    Tile types should be hidden from the end user. We can convert regular tensors into
    tiled tensors by default.

    **IrElementType => No explicit high-level matmul** <br/>
    Matmul can be expressed using elementwise tiled matmul operations, reduction, and
    stride-tricks. This makes scheduling and kernel fusion much easier. Instead of
    Matmul + epilogue, we can have reduction + epilogue or prologue + reduction +
    epilogue with matmul an elementwise operation on matrix tiles.

### Constant folding

-   Constant folding
-   Constant inlining into RPN expressions, matmul, scatter, gather
-   Dead code elimination

### Big Picture Ideas

-   Big symbolic scalar graph <br/>
    A totally different approach: imagine each value in a matrix is a single scalar
    symbolic variable. We effectively get a huge scalar formula. How can we group,
    schedule, and allot memory for these scalar formulae, using the fewest kernels
    possible? This throws out the user's conception of what is a tensor. Pretty radical,
    needs the most thought.

### Emitter-side optimizations

-   **Use SPV extensions** like `SPV_NV_tensor_addressing` to more efficiently access
    tensor elements upto 5D.
    https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_tensor_addressing.html
"""

from .ir import IrProgram


def optimize(program: IrProgram) -> IrProgram:
    # TODO: kernel fusion, dead-buffer elimination, etc.
    return program
