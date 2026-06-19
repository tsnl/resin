from resin.dsl.dsl import const


class TestViewMath:
    def test_chained_slice_offsets(self) -> None:
        t = const(list(range(10)), etype="f4")
        v = t[2:8][::2]

        assert v.node is t.node
        assert v.offset == 2
        assert v.shape == (3,)
        assert v.pitch == (2,)

    def test_permute_after_slice(self) -> None:
        t = const([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], etype="f4")
        v = t[1:].permute((1, 0))

        assert v.node is t.node
        assert v.offset == 4
        assert v.shape == (4, 2)
        assert v.pitch == (1, 4)
