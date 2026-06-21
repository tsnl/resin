from resin.core.pytree import flatten_pytree, map_pytree, tree_map_leaves


class TestFlattenPytree:
    def test_nested_dict_list_and_tuple(self) -> None:
        tree = {"a": [1, ("b", 2)], "c": 3}
        assert list(flatten_pytree(tree)) == [1, "b", 2, 3]


class TestMapPytree:
    def test_maps_every_leaf(self) -> None:
        tree = {"a": [1, 2], "b": 3}
        assert map_pytree(tree, lambda x: x * 2) == {"a": [2, 4], "b": 6}

    def test_maps_tuple_nodes(self) -> None:
        tree = (1, {"a": 2})
        assert map_pytree(tree, lambda x: x + 1) == (2, {"a": 3})


class TestTreeMapLeaves:
    def test_only_maps_matching_leaves(self) -> None:
        tree = {"a": [1, 2], "b": 3}

        def is_int(value: object) -> bool:
            return isinstance(value, int)

        assert tree_map_leaves(tree, is_int, lambda x: x * 2) == {
            "a": [2, 4],
            "b": 6,
        }

    def test_uses_map_pytree_map_leaves_only(self) -> None:
        tree = (1, [2, 3])

        def is_int(value: object) -> bool:
            return isinstance(value, int)

        mapped = tree_map_leaves(tree, is_int, lambda x: x + 1)
        assert mapped == (2, [3, 4])
        assert map_pytree(
            tree,
            lambda x: x + 1,
            is_leaf=is_int,
            map_leaves_only=True,
        ) == mapped