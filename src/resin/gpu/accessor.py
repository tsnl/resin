import math
from dataclasses import dataclass

from .shape import c_contiguous_pitch_for_shape, is_c_contiguous


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
        """
        Returns the accessor for the subview selected by `key`, in backing-buffer
        coordinates.
        """
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
                    b = bounded_index(k.start, dim) if k.start is not None else 0
                    e = (
                        bounded_end(k.stop, dim)
                        if k.stop is not None
                        else old_shape[dim]
                    )
                    s = k.step if k.step is not None else 1
                    a = abs(s)

                    new_offset += b * old_pitch[dim]
                    new_pitch.append(old_pitch[dim] * s)
                    new_shape.append(max(0, (e - b + (a - 1)) // a))
                case _:
                    raise TypeError(f"Invalid index {k} for dimension {dim}")

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
        """Permute dimensions so the last two are swapped, the rest left unaltered."""
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
        """True if this accessor addresses all of `backing_shape` densely."""
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
                    f"pitch={self.pitch}) addresses element {max_address} outside "
                    f"backing buffer of {capacity} elements"
                )
