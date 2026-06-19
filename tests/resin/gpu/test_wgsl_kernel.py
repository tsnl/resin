from resin.core.accessor import Accessor
from resin.ir import ElementRpnExpr, IrElementwiseRpnKernel, IrMatmulKernel, IrScatterKernel
from resin.wgpu import WgslKernelConfig, dispatch_size_for_kernel, emit_wgsl_for_kernel


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


class TestEmitBindings:
    def test_scatter_assign_uses_plain_output_binding(self) -> None:
        kernel = IrScatterKernel(
            arg_accessors=(Accessor.dense((3,)),),
            dtype="f4",
            shape=(3,),
            operator=None,
            woffset=0,
            wpitch=(1,),
        )
        wgsl = emit_wgsl_for_kernel(kernel, WgslKernelConfig())
        assert "var<storage, read_write> output: array<f32>;" in wgsl
        assert "atomic<" not in wgsl

    def test_scatter_add_uses_atomic_output_binding_with_output_dtype(self) -> None:
        kernel = IrScatterKernel(
            arg_accessors=(Accessor.dense((3,)),),
            dtype="f4",
            shape=(3,),
            operator="add",
            woffset=0,
            wpitch=(1,),
        )
        wgsl = emit_wgsl_for_kernel(kernel, WgslKernelConfig())
        assert "var<storage, read_write> output: array<atomic<u32>>;" in wgsl
        assert "bitcast<f32>(old_bits)" in wgsl
