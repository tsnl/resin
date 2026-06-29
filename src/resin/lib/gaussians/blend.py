"""Untiled GaussianBlendNode with forward tape and df_do gradient node."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from resin.core.etype import F4, U4, ElementType
from resin.dsl.node import CustomNode
from resin.dsl.view import View, zeros
from resin.lib.gaussians.kernels import gaussian_blend_wgsl

if TYPE_CHECKING:
    from resin.ir.ir import IrKernel


@dataclass(kw_only=True, frozen=True, eq=False)
class GaussianBlendNode(CustomNode):
    """Untiled global-sort alpha blend; ports: image, final_T, n_contrib."""

    width: int
    height: int

    def output_ports(self) -> tuple[str, ...]:
        return ("image", "final_T", "n_contrib")

    def port_shape(self, port: str) -> tuple[int, ...]:
        match port:
            case "image":
                return (self.height, self.width, 3)
            case "final_T" | "n_contrib":
                return (self.height, self.width)
            case _:
                raise KeyError(port)

    def port_etype(self, port: str) -> ElementType:
        match port:
            case "image" | "final_T":
                return F4
            case "n_contrib":
                return U4
            case _:
                raise KeyError(port)

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

    def df_do_ports(self, df_douts: dict[str, View]) -> tuple[View, ...]:
        _ = df_douts
        means2d, conics, colors, opacities, order = self.args
        return (
            zeros(means2d.shape, etype=F4),
            zeros(conics.shape, etype=F4),
            zeros(colors.shape, etype=F4),
            zeros(opacities.shape, etype=F4),
            zeros(order.shape, etype=U4),
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
    return View.port(node, "image")
