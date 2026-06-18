from resin.core.accessor import invert_permutation, permute


class TestPermutationMath:
    def test_permute_inverse(self) -> None:
        permutation = (3, 2, 0, 1)
        inverse_permutation = invert_permutation(permutation)

        xs = (100, 101, 102, 103)
        ys = permute(xs, permutation)
        xs_ref = permute(ys, inverse_permutation)

        assert xs == xs_ref
