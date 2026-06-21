from resin.core.etype import F4
from resin.core.pytree import PyTree, flatten_pytree, map_pytree, tree_map_leaves
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

        def double(x: object) -> int:
            assert isinstance(x, int)
            return x * 2

        assert map_pytree(tree, double) == {"a": [2, 4], "b": 6}

    def test_maps_tuple_nodes(self) -> None:
        tree: PyTree[int] = (1, {"a": 2})

        def increment(x: object) -> int:
            assert isinstance(x, int)
            return x + 1

        assert map_pytree(tree, increment) == (2, {"a": 3})


class TestTreeMapLeaves:
    def test_only_maps_matching_leaves(self) -> None:
        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def is_int(value: object) -> bool:
            return isinstance(value, int)

        def double(x: object) -> int:
            assert isinstance(x, int)
            return x * 2

        assert tree_map_leaves(tree, is_int, double) == {
            "a": [2, 4],
            "b": 6,
        }

    def test_uses_map_pytree_map_leaves_only(self) -> None:
        tree: PyTree[int] = (1, [2, 3])

        def is_int(value: object) -> bool:
            return isinstance(value, int)

        def increment(x: object) -> int:
            assert isinstance(x, int)
            return x + 1

        mapped = tree_map_leaves(tree, is_int, increment)
        assert mapped == (2, [3, 4])
        assert map_pytree(
            tree,
            increment,
            is_leaf=is_int,
            map_leaves_only=True,
        ) == mapped