__all__ = ["Tree"]

type Tree[T] = "dict[str, Tree[T]] | list[Tree[T]] | T"
"""Similar to PyTree in JAX."""
