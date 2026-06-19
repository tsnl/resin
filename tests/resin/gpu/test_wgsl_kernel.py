from resin.core.accessor import Accessor
from resin.ir import (
    ElementRpnExpr,
    IrElementwiseRpnKernel,
    IrMatmulKernel,
    IrReindexKernel,
    IrReindexViaAccessor,
)
from resin.wgpu import WgslKernelConfig, dispatch_size_for_kernel, emit_wgsl_for_kernel


class TestDispatchSize:
    def test_exact_fit(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            etype="f4",
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (1, 1, 1)

    def test_partial_last_workgroup(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrElementwiseRpnKernel(
            arg_accessors=(Accessor.dense((65,)),),
            etype="f4",
            shape=(65,),
            rpn_expr=ElementRpnExpr(string=(0,)),
        )
        assert dispatch_size_for_kernel(kernel, config) == (2, 1, 1)

    def test_empty_output(self) -> None:
        config = WgslKernelConfig()
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((0, 8)),
                Accessor.dense((8, 4)),
            ),
            etype="f4",
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
            etype="f4",
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (2, 1, 1)


class TestScatterCodegen:
    def _scatter_kernel(
        self,
        *,
        operator: str | None,
        etype: str = "f4",
    ) -> IrReindexKernel:
        return IrReindexKernel(
            arg_accessors=(Accessor.dense((2,)),),
            etype=etype,  # type: ignore[arg-type]
            shape=(3,),
            direction="scatter",
            addressing=IrReindexViaAccessor(woffset=0, wpitch=(1,)),
            operator=operator,  # type: ignore[arg-type]
            clear_output_before_dispatch=True,
        )

    def test_clobber_uses_plain_output_binding(self) -> None:
        wgsl = emit_wgsl_for_kernel(self._scatter_kernel(operator=None), WgslKernelConfig())
        assert "array<f32>" in wgsl
        assert "atomic<" not in wgsl
        assert "output[scatter_out_address(src_index)] = " in wgsl

    def test_accumulate_uses_u32_atomic_and_cas(self) -> None:
        wgsl = emit_wgsl_for_kernel(self._scatter_kernel(operator="add"), WgslKernelConfig())
        assert "array<atomic<u32>>" in wgsl
        assert "atomicCompareExchangeWeak" in wgsl
        assert "bitcast<f32>" in wgsl