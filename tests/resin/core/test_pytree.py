from resin.core.etype import F4
from resin.core.pytree import (
    PyTree,
    flatten_pytree,
    flatten_pytree_paths,
    map_pytree,
    map_pytree_paths,
)
from resin.dsl.view import View, param


class TestPyTreeTyping:
    def test_leaf_value_is_pytree(self) -> None:
        leaf = param(shape=(2,), etype=F4)
        tree: PyTree[View] = leaf
        assert tree is leaf


class TestFlattenPytree:
    def test_nested_dict_list_and_tuple(self) -> None:
        tree = {"a": [1, ("b", 2)], "c": 3}
        assert list(flatten_pytree(tree)) == [1, "b", 2, 3]


class TestMapPytree:
    def test_maps_every_leaf(self) -> None:
        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def double(x: int) -> int:
            return x * 2

        assert map_pytree(tree, double) == {"a": [2, 4], "b": 6}

    def test_maps_tuple_nodes(self) -> None:
        tree: PyTree[int] = (1, {"a": 2})

        def increment(x: int) -> int:
            return x + 1

        assert map_pytree(tree, increment) == (2, {"a": 3})

    def test_callback_inspects_leaf_type(self) -> None:
        tree: PyTree[int | str] = {"a": [1, 2], "b": "skip"}

        def maybe_double(x: int | str) -> int | str:
            if isinstance(x, int):
                return x * 2
            return x

        assert map_pytree(tree, maybe_double) == {"a": [2, 4], "b": "skip"}


class TestFlattenPytreePaths:
    def test_nested_dict_paths(self) -> None:
        tree = {"l1": {"weight": 1, "bias": 2}, "l2": 3}
        assert list(flatten_pytree_paths(tree)) == [
            ("l1.weight", 1),
            ("l1.bias", 2),
            ("l2", 3),
        ]


class TestMapPytreePaths:
    def test_prefixes_paths_at_leaves(self) -> None:
        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def label(path: str, value: int) -> str:
            return f"{path}:{value}"

        assert map_pytree_paths(tree, label) == {
            "a": ["a.0:1", "a.1:2"],
            "b": "b:3",
        }
