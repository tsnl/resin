import math
from dataclasses import dataclass


@dataclass
class ShapeJoin:
    shape: tuple[int, ...]
    pitch1: tuple[int, ...]
    pitch2: tuple[int, ...]


def shape_join(
    shape1: tuple[int, ...],
    pitch1: tuple[int, ...],
    shape2: tuple[int, ...],
    pitch2: tuple[int, ...],
    report_shape1: tuple[int, ...] | None = None,
    report_shape2: tuple[int, ...] | None = None,
) -> ShapeJoin:
    # Swap args and re-enter if needed to ensure len(shape1) <= len(shape2)
    if len(shape1) > len(shape2):
        join = shape_join(shape2, pitch2, shape1, pitch1)
        return ShapeJoin(shape=join.shape, pitch1=join.pitch2, pitch2=join.pitch1)

    # Broadcast self up to other's dim if needed and re-enter.
    if len(shape1) < len(shape2):
        p_ndim = len(shape2) - len(shape1)
        shape1 = (1,) * p_ndim + shape1
        pitch1 = (0,) * p_ndim + pitch1
        assert len(pitch1) == len(pitch2)
        return shape_join(shape1=shape1, pitch1=pitch1, shape2=shape2, pitch2=pitch2)

    # From here on, both have same ndim.
    assert len(shape1) == len(shape2)

    # If both have same ndim, check if shapes are compatible for broadcasting.
    # Each dimension must either be the same or one of them must be unity.
    # We also build the new pitch for each operand tensor at the same time.
    new_shape_list = []
    new_pitch1_list = []
    new_pitch2_list = []
    for i_dim, (s, o) in enumerate(zip(shape1, shape2)):
        if s != o and s != 1 and o != 1:
            report_shape1 = report_shape1 or shape1
            report_shape2 = report_shape2 or shape2
            raise ValueError(
                f"Shapes {report_shape1} and {report_shape2} are not compatible for "
                "broadcasting."
            )

        new_shape_list.append(max(s, o))
        new_pitch1_list.append(0 if s == 1 else pitch1[i_dim])
        new_pitch2_list.append(0 if o == 1 else pitch2[i_dim])

    out_shape = tuple(new_shape_list)
    new_pitch1 = tuple(new_pitch1_list)
    new_pitch2 = tuple(new_pitch2_list)

    # Finalize:
    return ShapeJoin(shape=out_shape, pitch1=new_pitch1, pitch2=new_pitch2)


# Contiguity and permutation:
# - Contiguous: no gaps between elements in memory (i.e. no "holes" in the tensor).
# - Permutation: a reordering of the dimensions of a tensor. E.g. transpose of matrix.
# - Permutation only involves reordering shape and pitch, no data copy needed.
# - C-contiguous: contiguous and has monotonically decreasing strides (like C arrays).
# - Lemma: tensor is contiguous if and only if it can be made C-contiguous by permuting
#   dimensions.


def c_contiguous_pitch_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """
    Returns the C-contiguous strides for a given shape.

    An array is C-contiguous if it both...
    -   Contains no gaps between elements
    -   Has monotonically decreasing strides (like C arrays).

    Note that it is possible for a tensor to be contiguous without being C-contiguous.
    E.g. a permutation of a C-contiguous tensor.
    """
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def is_c_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    """
    Check if a tensor with the given shape and pitch is C-contiguous.
    """
    return pitch == c_contiguous_pitch_for_shape(shape)


def is_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    new_shape, new_pitch = c_permuted(shape, pitch)
    return is_c_contiguous(new_shape, new_pitch)


def c_permuted(
    shape: tuple[int, ...],
    pitch: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """
    Permute the dimensions by decreasing pitch, breaking ties by decreasing shape, and
    return the new shape and pitch.

    Note that this makes contiguous shapes C-contiguous.

    This normalization is useful even for comparing non-contiguous shapes.
    """

    permutation = c_permutation(shape, pitch)
    new_shape = permute(shape, permutation)
    new_pitch = permute(pitch, permutation)
    return new_shape, new_pitch


def c_permutation(shape: tuple[int, ...], pitch: tuple[int, ...]) -> tuple[int, ...]:
    """
    Permute the dimensions by decreasing pitch, breaking ties by decreasing
    shape, and return the new shape and pitch.

    Note that this makes contiguous shapes C-contiguous.

    This normalization is useful even for comparing non-contiguous shapes.
    """

    # argsort the dimensions by decreasing pitch, breaking ties by decreasing
    # shape.
    return tuple(
        sorted(
            range(len(pitch)),
            key=lambda i: (pitch[i], shape[i]),
            reverse=True,
        )
    )


def invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    """
    Computes the inverse of a permutation, i.e. a permutation `inv` such that
        forall i: inv[permutation[i]] == i
        forall i: permutation[inv[i]] == i
    """

    n = len(permutation)
    assert set(permutation) == set(range(n)), "Invalid permutation"
    return tuple(permutation.index(i) for i in range(n))


def permute[T](seq: tuple[T, ...], perm: tuple[int, ...]) -> tuple[T, ...]:
    """
    Applies a permutation to a tuple, returning the permuted tuple.
    """

    assert len(seq) == len(perm), "Permutation length must match sequence length"
    return tuple(seq[i] for i in perm)
