from resin.gpu.accessor import Accessor
from resin.gpu.kernel import ElementwiseRpnKernel, MatmulKernel
from resin.gpu.rpn import ScalarRpnExpr


class TestDispatchSize:
    def test_exact_fit(self) -> None:
        # 64 elements, 8 items/thread => 8 threads, 8 threads/workgroup => 1 wg
        kernel = MatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            stype="f4",
            shape=(8, 8),
            lg2_items_per_thread=3,
            workgroup_size=8,
        )
        assert kernel.dispatch_size() == (1, 1, 1)

    def test_partial_last_workgroup(self) -> None:
        # 65 elements => 9 threads => 2 workgroups of size 8
        kernel = ElementwiseRpnKernel(
            arg_accessors=(Accessor.dense((65,)),),
            stype="f4",
            shape=(65,),
            rpn_expr=ScalarRpnExpr(string=(0,)),
            lg2_items_per_thread=3,
            workgroup_size=8,
        )
        assert kernel.dispatch_size() == (2, 1, 1)

    def test_empty_output(self) -> None:
        kernel = MatmulKernel(
            arg_accessors=(
                Accessor.dense((0, 8)),
                Accessor.dense((8, 4)),
            ),
            stype="f4",
            shape=(0, 4),
        )
        assert kernel.dispatch_size() == (0, 1, 1)

    def test_workgroup_size_affects_dispatch(self) -> None:
        kernel = MatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            stype="f4",
            shape=(8, 8),
            lg2_items_per_thread=3,
            workgroup_size=4,
        )
        # 64 elems / 8 per thread = 8 threads / 4 per wg = 2 workgroups
        assert kernel.dispatch_size() == (2, 1, 1)