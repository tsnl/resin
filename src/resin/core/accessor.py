import math
from dataclasses import dataclass
from typing import assert_never, cast

#
# Accessor
#


@dataclass(frozen=True, kw_only=True)
class Accessor:
    offset: int
    shape: tuple[int, ...]
    pitch: tuple[int, ...]

    def __post_init__(self):
        assert self.offset >= 0
        assert len(self.shape) == len(self.pitch), "Inconsistent rank"

    @classmethod
    def dense(cls, shape: tuple[int, ...], *, offset: int = 0) -> "Accessor":
        """A C-contiguous accessor addressing `shape` from `offset`."""
        return cls(
            offset=offset,
            shape=shape,
            pitch=c_contiguous_pitch_for_shape(shape),
        )

    @property
    def rank(self) -> int:
        return len(self.shape)

    def broadcast(self, leading_shape: tuple[int, ...]) -> "Accessor":
        zs = (0,) * len(leading_shape)
        return Accessor(
            offset=self.offset,
            shape=leading_shape + self.shape,
            pitch=zs + self.pitch,
        )

    def narrow(self, key: int | slice | tuple[int | slice, ...]) -> "Accessor":
        key = (key,) if isinstance(key, (int, slice)) else key

        old_offset = self.offset
        old_shape = self.shape
        old_pitch = self.pitch

        def bounded_index(k: int, dim: int) -> int:
            d = old_shape[dim]
            if not (-d <= k < d):
                raise IndexError(
                    f"Index {k} out of bounds for dimension {dim} of size {d}"
                )
            return k % d

        def bounded_end(k: int, dim: int) -> int:
            d = old_shape[dim]
            if not (0 <= k <= d):
                raise IndexError(
                    f"Slice end {k} out of bounds for dimension {dim} of size {d}"
                )
            return k

        assert len(key) <= len(old_shape)

        new_offset = old_offset
        new_pitch: list[int] = []
        new_shape: list[int] = []

        for dim, k in enumerate(key):
            match k:
                case int():
                    new_offset += bounded_index(k, dim) * old_pitch[dim]
                case slice():
                    start = cast(int | None, k.start)
                    stop = cast(int | None, k.stop)
                    step = cast(int | None, k.step)
                    b = bounded_index(start, dim) if isinstance(start, int) else 0
                    e = (
                        bounded_end(stop, dim)
                        if isinstance(stop, int)
                        else old_shape[dim]
                    )
                    s = step if isinstance(step, int) else 1
                    a = abs(s)

                    new_offset += b * old_pitch[dim]
                    new_pitch.append(old_pitch[dim] * s)
                    new_shape.append(max(0, (e - b + (a - 1)) // a))
                case _:
                    assert_never(k)

        for dim in range(len(key), len(old_shape)):
            new_pitch.append(old_pitch[dim])
            new_shape.append(old_shape[dim])

        return Accessor(
            offset=new_offset,
            shape=tuple(new_shape),
            pitch=tuple(new_pitch),
        )

    def permute(self, permutation: tuple[int, ...]) -> "Accessor":
        if sorted(permutation) != list(range(self.rank)):
            raise ValueError(f"Bad permutation {permutation} for shape {self.shape}")

        return Accessor(
            offset=self.offset,
            shape=tuple(self.shape[i] for i in permutation),
            pitch=tuple(self.pitch[i] for i in permutation),
        )

    def transpose(self) -> "Accessor":
        identity = tuple(range(self.rank))
        permutation = identity[:-2] + (identity[-1], identity[-2])
        return self.permute(permutation)

    def squeeze(self, axes: tuple[int, ...]) -> "Accessor":
        for axis in axes:
            if axis < 0 or axis >= self.rank:
                raise IndexError(f"Axis {axis} out of bounds for shape {self.shape}")
            if self.shape[axis] != 1:
                raise ValueError(f"Cannot squeeze axis {axis} with {self.shape[axis]=}")

        return Accessor(
            offset=self.offset,
            shape=tuple(s for i, s in enumerate(self.shape) if i not in axes),
            pitch=tuple(p for i, p in enumerate(self.pitch) if i not in axes),
        )

    def is_dense_c_contiguous(self, backing_shape: tuple[int, ...]) -> bool:
        return (
            self.offset == 0
            and self.shape == backing_shape
            and is_c_contiguous(self.shape, self.pitch)
        )

    def max_flat_address(self) -> int:
        return self.offset + sum(
            (s - 1) * p for s, p in zip(self.shape, self.pitch) if p > 0
        )

    def raise_if_addresses_out_of_bounds(self, capacity: int) -> None:
        if self.shape and math.prod(self.shape) > 0:
            max_address = self.max_flat_address()
            if max_address >= capacity:
                raise ValueError(
                    f"Accessor (offset={self.offset}, shape={self.shape}, "
                    + f"pitch={self.pitch}) addresses element {max_address} outside "
                    + f"backing buffer of {capacity} elements"
                )


#
# Shape utilities
#


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
    if len(shape1) > len(shape2):
        join = shape_join(shape2, pitch2, shape1, pitch1)
        return ShapeJoin(shape=join.shape, pitch1=join.pitch2, pitch2=join.pitch1)

    if len(shape1) < len(shape2):
        p_ndim = len(shape2) - len(shape1)
        shape1 = (1,) * p_ndim + shape1
        pitch1 = (0,) * p_ndim + pitch1
        assert len(pitch1) == len(pitch2)
        return shape_join(shape1=shape1, pitch1=pitch1, shape2=shape2, pitch2=pitch2)

    assert len(shape1) == len(shape2)

    new_shape_list: list[int] = []
    new_pitch1_list: list[int] = []
    new_pitch2_list: list[int] = []
    for i_dim, (s, o) in enumerate(zip(shape1, shape2)):
        if s != o and s != 1 and o != 1:
            report_shape1 = report_shape1 or shape1
            report_shape2 = report_shape2 or shape2
            raise ValueError(
                f"Shapes {report_shape1} and {report_shape2} are not compatible for "
                + "broadcasting."
            )

        new_shape_list.append(max(s, o))
        new_pitch1_list.append(0 if s == 1 else pitch1[i_dim])
        new_pitch2_list.append(0 if o == 1 else pitch2[i_dim])

    out_shape = tuple(new_shape_list)
    new_pitch1 = tuple(new_pitch1_list)
    new_pitch2 = tuple(new_pitch2_list)

    return ShapeJoin(shape=out_shape, pitch1=new_pitch1, pitch2=new_pitch2)


def c_contiguous_pitch_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[i + 1 :]) for i in range(len(shape)))


def is_c_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    return pitch == c_contiguous_pitch_for_shape(shape)


def is_contiguous(shape: tuple[int, ...], pitch: tuple[int, ...]) -> bool:
    new_shape, new_pitch = c_permuted(shape, pitch)
    return is_c_contiguous(new_shape, new_pitch)


def c_permuted(
    shape: tuple[int, ...],
    pitch: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    permutation = c_permutation(shape, pitch)
    new_shape = permute(shape, permutation)
    new_pitch = permute(pitch, permutation)
    return new_shape, new_pitch


def c_permutation(shape: tuple[int, ...], pitch: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(pitch)),
            key=lambda i: (pitch[i], shape[i]),
            reverse=True,
        )
    )


def invert_permutation(permutation: tuple[int, ...]) -> tuple[int, ...]:
    n = len(permutation)
    assert set(permutation) == set(range(n)), "Invalid permutation"
    return tuple(permutation.index(i) for i in range(n))


def permute[T](seq: tuple[T, ...], perm: tuple[int, ...]) -> tuple[T, ...]:
    assert len(seq) == len(perm), "Permutation length must match sequence length"
    return tuple(seq[i] for i in perm)
