from resin.core.accessor import Accessor
from resin.ir import (
    ElementRpnExpr,
    IrElementwiseRpnKernel,
    IrMatmulKernel,
    IrScatterAccumulateKernel,
    IrScatterClobberKernel,
)
from resin.wgpu import (
    WgslKernelConfig,
    WgslTargetFeatures,
    dispatch_size_for_kernel,
    emit_wgsl_for_kernel,
)


class TestDispatchSize:
    def test_exact_fit(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrMatmulKernel(
            arg_accessors=(
                Accessor.dense((8, 8)),
                Accessor.dense((8, 8)),
            ),
            dtype="f4",
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (1, 1, 1)

    def test_partial_last_workgroup(self) -> None:
        config = WgslKernelConfig(lg2_items_per_thread=3, workgroup_size=8)
        kernel = IrElementwiseRpnKernel(
            arg_accessors=(Accessor.dense((65,)),),
            dtype="f4",
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
            dtype="f4",
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
            dtype="f4",
            shape=(8, 8),
        )
        assert dispatch_size_for_kernel(kernel, config) == (2, 1, 1)


class TestScatterCodegen:
    def _scatter_clobber_kernel(self, *, dtype: str = "f4") -> IrScatterClobberKernel:
        return IrScatterClobberKernel(
            arg_accessors=(Accessor.dense((2,)),),
            dtype=dtype,  # type: ignore[arg-type]
            shape=(3,),
            woffset=0,
            wpitch=(1,),
        )

    def _scatter_accumulate_kernel(
        self,
        *,
        operator: str,
        dtype: str = "f4",
    ) -> IrScatterAccumulateKernel:
        return IrScatterAccumulateKernel(
            arg_accessors=(Accessor.dense((2,)),),
            dtype=dtype,  # type: ignore[arg-type]
            shape=(3,),
            operator=operator,  # type: ignore[arg-type]
            woffset=0,
            wpitch=(1,),
        )

    def test_clobber_uses_plain_output_binding(self) -> None:
        wgsl = emit_wgsl_for_kernel(self._scatter_clobber_kernel(), WgslKernelConfig())
        assert "array<f32>" in wgsl
        assert "atomic<" not in wgsl
        assert "output[out_address] = " in wgsl

    def test_accumulate_default_uses_u32_atomic_and_cas(self) -> None:
        wgsl = emit_wgsl_for_kernel(
            self._scatter_accumulate_kernel(operator="add"), WgslKernelConfig()
        )
        assert "array<atomic<u32>>" in wgsl
        assert "atomicCompareExchangeWeak" in wgsl
        assert "bitcast<f32>" in wgsl
        assert "atomicAdd" not in wgsl

    def test_accumulate_mul_uses_cas_even_with_float32_atomic(self) -> None:
        config = WgslKernelConfig(
            target_features=WgslTargetFeatures(shader_float32_atomic=True),
        )
        wgsl = emit_wgsl_for_kernel(self._scatter_accumulate_kernel(operator="mul"), config)
        assert "array<atomic<u32>>" in wgsl
        assert "atomicCompareExchangeWeak" in wgsl
        assert "atomicAdd" not in wgsl

    def test_accumulate_add_uses_native_f32_atomic_when_enabled(self) -> None:
        config = WgslKernelConfig(
            target_features=WgslTargetFeatures(shader_float32_atomic=True),
        )
        wgsl = emit_wgsl_for_kernel(self._scatter_accumulate_kernel(operator="add"), config)
        assert "array<atomic<f32>>" in wgsl
        assert "atomicAdd(&output[out_address]" in wgsl
        assert "atomicCompareExchangeWeak" not in wgsl

    def test_accumulate_add_uint_uses_native_atomic(self) -> None:
        wgsl = emit_wgsl_for_kernel(
            self._scatter_accumulate_kernel(operator="add", dtype="u4"),
            WgslKernelConfig(),
        )
        assert "array<atomic<u32>>" in wgsl
        assert "atomicAdd(&output[out_address]" in wgsl
        assert "atomicCompareExchangeWeak" not in wgsl