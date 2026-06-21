from resin.core.accessor import Accessor
from resin.dsl import const
from resin.ir import IrProgramBuilder, IrReductionKernel
from resin.wgpu import WgslKernelConfig, emit_wgsl_for_kernel
from resin.core.etype import F4


class TestIrReductionKernel:
    def test_post_init_validates_shapes(self) -> None:
        kernel = IrReductionKernel(
            arg_accessors=(Accessor.dense((2, 3)),),
            etype=F4,
            shape=(2, 1),
            operator="add",
            axes=(1,),
        )
        assert kernel.reduced_count == 3


class TestEmitReductionWgsl:
    def test_sum_axis_1(self) -> None:
        kernel = IrReductionKernel(
            arg_accessors=(Accessor.dense((2, 3)),),
            etype=F4,
            shape=(2, 1),
            operator="add",
            axes=(1,),
        )
        wgsl = emit_wgsl_for_kernel(kernel, WgslKernelConfig())

        assert "fn input_index" in wgsl
        assert "for (var ri: u32 = 1u; ri < 3u; ri += 1u)" in wgsl
        assert "acc += v;" in wgsl

    def test_max_multi_axis(self) -> None:
        kernel = IrReductionKernel(
            arg_accessors=(Accessor.dense((2, 3)),),
            etype=F4,
            shape=(1, 1),
            operator="max",
            axes=(0, 1),
        )
        wgsl = emit_wgsl_for_kernel(kernel, WgslKernelConfig())

        assert "for (var ri: u32 = 1u; ri < 6u; ri += 1u)" in wgsl
        assert "acc = max(acc, v);" in wgsl

    def test_mul_single_element_axis(self) -> None:
        kernel = IrReductionKernel(
            arg_accessors=(Accessor.dense((4, 1)),),
            etype=F4,
            shape=(4, 1),
            operator="mul",
            axes=(1,),
        )
        wgsl = emit_wgsl_for_kernel(kernel, WgslKernelConfig())

        assert "for (var ri: u32 = 1u; ri < 1u; ri += 1u)" not in wgsl
        assert "output[out_address] = acc;" in wgsl


class TestProgramBuilderReduction:
    def test_lowers_reduction_node(self) -> None:
        t = const([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], etype=F4)
        n = t.reduce(axes=(1,), operator="add")

        builder = IrProgramBuilder()
        builder.build_sink("out", n)
        program = builder.finish()

        assert len(program.queue) == 1
        kernel = program.queue[0].kernel
        assert isinstance(kernel, IrReductionKernel)
        assert kernel.operator == "add"
        assert kernel.axes == (1,)
        assert kernel.shape == (2, 1)
