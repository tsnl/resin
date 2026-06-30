from resin.core.accessor import Accessor
from resin.ir import IrElementwiseRpnKernel, IrMatmulKernel
from resin.ir import ElementRpnExpr
from resin.wgpu import WgslKernelConfig, dispatch_size_for_kernel
from resin.core.etype import F4


class TestDispatchSize:
    def test_exact_fit(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            etype=F4,
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (1, 1, 1)

    def test_partial_last_workgroup(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrElementwiseRpnKernel(
            arg_accessors=(Accessor.dense((65,)),),
            etype=F4,
            shape=(65,),
            rpn_expr=ElementRpnExpr(string=(0,)),
            arg_etypes=(F4,),
        )
        assert dispatch_size_for_kernel(kernel, config) == (2, 1, 1)

    def test_empty_output(self) -> None:
        config = WgslKernelConfig()
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((0, 8)),
                Accessor.dense((8, 4)),
            ),
            etype=F4,
            shape=(0, 4),
        )
        assert dispatch_size_for_kernel(kernel, config) == (0, 1, 1)

    def test_workgroup_size_affects_dispatch(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=4)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            etype=F4,
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (2, 1, 1)
