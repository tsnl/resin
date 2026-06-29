from typing import cast

from resin.core.etype import F4
from resin.core.pytree import (
    PyTree,
    flatten_pytree,
    flatten_pytree_paths,
    map_pytree,
    map_pytree_paths,
    tree_map,
)
from resin.dsl.view import View, param
from resin.nn import Linear


class TestPyTreeTyping:
    def test_leaf_value_is_pytree(self) -> None:
        leaf = param(shape=(2,), etype=F4)
        tree: PyTree[View] = leaf
        assert tree is leaf

    def test_module_repr_subtypes_pytree(self) -> None:
        layer: Linear[View] = Linear.new(2, 3, bias=True)
        model: list[Linear[View]] = [layer]
        tree: PyTree[View] = model
        assert tree is model


class TestFlattenPytree:
    def test_nested_dict_list_and_tuple_leaf(self) -> None:
        tree = {"a": [1, ("b", 2)], "c": 3}
        assert list(flatten_pytree(tree)) == [1, ("b", 2), 3]


class TestMapPytree:
    def test_maps_every_leaf(self) -> None:
        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def double(x: int) -> int:
            return x * 2

        assert map_pytree(tree, double) == {"a": [2, 4], "b": 6}

    def test_callback_inspects_leaf_type(self) -> None:
        # cast (not annotation) so the tree keeps type PyTree[int | str] rather than
        # narrowing to the literal dict, whose ambiguous list leaf confuses T inference.
        tree = cast(PyTree[int | str], {"a": [1, 2], "b": "skip"})

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


class TestTreeMap:
    def test_combines_leaves_across_trees(self) -> None:
        params: PyTree[int] = {"w": 10, "b": 2}
        grads: PyTree[int] = {"w": 3, "b": 1}

        def sub(p: int, g: int) -> int:
            return p - g

        assert tree_map(sub, params, grads) == {"w": 7, "b": 1}


class TestModule:
    def test_params_flattens_leaves(self) -> None:
        layer = Linear.new(3, 2, bias=True)
        ps = layer.params()
        assert set(ps) == {"weight", "bias"}
        assert ps["weight"] is layer.weight
        assert ps["bias"] is layer.bias

    def test_params_skips_none_bias(self) -> None:
        layer = Linear.new(3, 2, bias=False)
        assert set(layer.params()) == {"weight"}

    def test_module_in_pytree_flattens_with_field_paths(self) -> None:
        model = [Linear.new(2, 3, bias=True)]
        assert [p for p, _ in flatten_pytree_paths(model)] == ["0.weight", "0.bias"]


class TestMapPytreePaths:
    def test_prefixes_paths_at_leaves(self) -> None:
        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def label(path: str, value: int) -> str:
            return f"{path}:{value}"

        assert map_pytree_paths(tree, label) == {
            "a": ["a.0:1", "a.1:2"],
            "b": "b:3",
        }
