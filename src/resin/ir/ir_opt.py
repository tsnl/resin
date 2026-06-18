"""
IR->IR optimization passes.

---

Ideas for optimizations

-   **Matmul + RPN expression fused kernel** <br/>
    elementwise RPN expressions for each matmul argument, then matmul, then elementwise
    RPN expressions for the matrix product.

-   **IrElementType = ScalarType | TiledScalarType** <br/>
    Replace ScalarType with IrElementType, which could be a scalar OR a tiled scalar
    type.
    E.g. mat4x4_f4
    When we perform a matmul on block matrices i.e. matrices with tile elements, we can
    use cooperative matrix operations.
    Tile types should be hidden from the end user. We can convert regular tensors into
    tiled tensors by default.

    **IrElementType => No explicit high-level matmul** <br/>
    If we don't have a dedicated Matmul kernel, then ReductionNode becomes the sole sync
    point in our graph. Matmul can be expressed using elementwise tiled matmul
    operations, reduction, and stride-tricks (need to plan this out and confirm).
    This could make scheduling and kernel fusion much easier. We don't need an ugly
    Matmul+RPN fused kernel. Instead, we can generate kernels with elementwise chains,
    reductions, and more elementwise chains.

-   Big symbolic scalar graph <br/>
    A totally different approach: imagine each value in a matrix is a single scalar
    symbolic variable. We effectively get a huge scalar formula. How can we group,
    schedule, and allot memory for these scalar formulae, using the fewest kernels
    possible? This throws out the user's conception of what is a tensor. Pretty radical,
    needs the most thought.

Other emitter-side optimizations:

-   **Use SPV extensions** like `SPV_NV_tensor_addressing` to more efficiently access
    tensor elements upto 5D.
    https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_tensor_addressing.html
"""

from .ir import IrProgram


def optimize(program: IrProgram) -> IrProgram:
    # TODO: kernel fusion, dead-buffer elimination, etc.
    return program
