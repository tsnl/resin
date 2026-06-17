from resin.accessor import Accessor
from resin.gpu import WgslBuilder
from resin.ir import IrElementwiseRpnKernel, IrMatmulKernel
from resin.rpn import ScalarRpnExpr


class TestDispatchSize:
    def test_exact_fit(self) -> None:
        builder = WgslBuilder(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            stype="f4",
            shape=(8, 8),
        )
        assert builder.dispatch_size(kernel) == (1, 1, 1)

    def test_partial_last_workgroup(self) -> None:
        builder = WgslBuilder(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrElementwiseRpnKernel(
            arg_accessors=(Accessor.dense((65,)),),
            stype="f4",
            shape=(65,),
            rpn_expr=ScalarRpnExpr(string=(0,)),
        )
        assert builder.dispatch_size(kernel) == (2, 1, 1)

    def test_empty_output(self) -> None:
        builder = WgslBuilder()
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((0, 8)),
                Accessor.dense((8, 4)),
            ),
            stype="f4",
            shape=(0, 4),
        )
        assert builder.dispatch_size(kernel) == (0, 1, 1)

    def test_workgroup_size_affects_dispatch(self) -> None:
        builder = WgslBuilder(lg2_items_per_thread=3, workgroup_size=4)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            stype="f4",
            shape=(8, 8),
        )
        assert builder.dispatch_size(kernel) == (2, 1, 1)
