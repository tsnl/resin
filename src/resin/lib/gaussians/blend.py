"""Untiled GaussianBlendNode with forward tape and df_do gradient node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from resin.core.etype import F4, U4, ElementType
from resin.dsl.node import CustomNode
from resin.dsl.view import View, zeros
from resin.lib.gaussians.kernels import gaussian_blend_grad_wgsl, gaussian_blend_wgsl

if TYPE_CHECKING:
    from resin.ir.ir import IrKernel


@dataclass(kw_only=True, frozen=True, eq=False)
class GaussianBlendNode(CustomNode):
    """Untiled global-sort alpha blend; ports: image, final_T, n_contrib."""

    width: int
    height: int

    @override
    def output_ports(self) -> tuple[str, ...]:
        return ("image", "final_T", "n_contrib")

    @override
    def port_shape(self, port: str) -> tuple[int, ...]:
        match port:
            case "image":
                return (self.height, self.width, 3)
            case "final_T" | "n_contrib":
                return (self.height, self.width)
            case _:
                raise KeyError(port)

    @override
    def port_etype(self, port: str) -> ElementType:
        match port:
            case "image" | "final_T":
                return F4
            case "n_contrib":
                return U4
            case _:
                raise KeyError(port)

    @override
    def build_kernel(self, *, used_ports: frozenset[str]) -> IrKernel:
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports  # always write full tape for backward compatibility
        count = self.args[0].shape[0]
        wg = (self.width * self.height + 63) // 64
        return IrWgslMultiOutputKernel(
            arg_accessors=tuple(a.accessor for a in self.args),
            etype=F4,
            shape=(self.height, self.width, 3),
            wgsl=gaussian_blend_wgsl(
                width=self.width, height=self.height, count=count
            ),
            entry_point="main",
            dispatch_size=(wg, 1, 1),
            arg_etypes=tuple(a.etype for a in self.args),
            output_etypes=(F4, F4, U4),
            num_outputs=3,
            clear_output_before_dispatch=True,
        )

    @override
    def df_do_ports(self, df_douts: dict[str, View]) -> tuple[View, ...]:
        df_dimage = df_douts.get("image")
        if df_dimage is None:
            z_m = zeros(self.args[0].shape, etype=F4)
            z_c = zeros(self.args[1].shape, etype=F4)
            z_col = zeros(self.args[2].shape, etype=F4)
            z_o = zeros(self.args[3].shape, etype=F4)
            z_ord = zeros(self.args[4].shape, etype=U4)
            return (z_m, z_c, z_col, z_o, z_ord)

        means2d, conics, colors, opacities, order = self.args
        count = means2d.shape[0]
        grad_node = GradGaussianBlendNode(
            shape=(count, 3),
            etype=F4,
            args=(df_dimage, means2d, conics, colors, opacities, order),
            width=self.width,
            height=self.height,
            count=count,
        )
        grad_colors = View.identity(grad_node, port="grad_colors")
        grad_opacities = View.identity(grad_node, port="grad_opacities")
        # Means/conics grads not yet implemented in the smoke backward kernel.
        grad_means = zeros(means2d.shape, etype=F4)
        grad_conics = zeros(conics.shape, etype=F4)
        # Order (u4 indices) is not differentiable.
        grad_order = zeros(order.shape, etype=U4)
        return (grad_means, grad_conics, grad_colors, grad_opacities, grad_order)


@dataclass(kw_only=True, frozen=True, eq=False)
class GradGaussianBlendNode(CustomNode):
    """Gradient kernel for colors and opacities from image loss."""

    width: int
    height: int
    count: int

    @override
    def output_ports(self) -> tuple[str, ...]:
        return ("grad_colors", "grad_opacities")

    @override
    def port_shape(self, port: str) -> tuple[int, ...]:
        match port:
            case "grad_colors":
                return (self.count, 3)
            case "grad_opacities":
                return (self.count,)
            case _:
                raise KeyError(port)

    @override
    def port_etype(self, port: str) -> ElementType:
        return F4

    @override
    def build_kernel(self, *, used_ports: frozenset[str]) -> IrKernel:
        from resin.ir.ir import IrWgslMultiOutputKernel

        _ = used_ports
        return IrWgslMultiOutputKernel(
            arg_accessors=tuple(a.accessor for a in self.args),
            etype=F4,
            shape=(self.count, 3),
            wgsl=gaussian_blend_grad_wgsl(
                width=self.width, height=self.height, count=self.count
            ),
            entry_point="main",
            dispatch_size=(1, 1, 1),
            arg_etypes=tuple(a.etype for a in self.args),
            output_etypes=(F4, F4),
            num_outputs=2,
            clear_output_before_dispatch=True,
        )


def gaussian_blend(
    *,
    width: int,
    height: int,
    means2d: View,
    conics: View,
    colors: View,
    opacities: View,
    order: View,
) -> View:
    """Return the ``image`` port of an untiled GaussianBlendNode."""
    count = means2d.shape[0]
    node = GaussianBlendNode(
        shape=(height, width, 3),
        etype=F4,
        args=(means2d, conics, colors, opacities, order),
        width=width,
        height=height,
    )
    _ = count
    return View.identity(node, port="image")
