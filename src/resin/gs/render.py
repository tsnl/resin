"""DSL graph builders for 3DGS forward rendering."""


from resin.core.etype import F4, U4
from resin.dsl import View
from resin.dsl.node import WgslKernelNode
from resin.gs.kernels import argsort_depths_wgsl, gaussian_blend_wgsl


def argsort_depths(depths: View) -> View:
    count = depths.shape[0]
    wgsl = argsort_depths_wgsl(count=count)
    return View.identity(
        WgslKernelNode(
            shape=(count,),
            etype=U4,
            args=(depths,),
            wgsl=wgsl,
            entry_point="main",
            dispatch_size=(1, 1, 1),
            arg_etypes=(F4,),
        )
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
    count = means2d.shape[0]
    wgsl = gaussian_blend_wgsl(width=width, height=height, count=count)
    wg = (width * height + 63) // 64
    return View.identity(
        WgslKernelNode(
            shape=(height, width, 3),
            etype=F4,
            args=(means2d, conics, colors, opacities, order),
            wgsl=wgsl,
            entry_point="main",
            dispatch_size=(wg, 1, 1),
            arg_etypes=(F4, F4, F4, F4, U4),
        )
    )


def pack_means2d(means2d: tuple[tuple[float, float], ...]) -> list[list[float]]:
    return [[mx, my] for mx, my in means2d]


def pack_triplets(items: tuple[tuple[float, float, float], ...]) -> list[list[float]]:
    return [[a, b, c] for a, b, c in items]


def pack_means2d_flat(means2d: tuple[tuple[float, float], ...]) -> list[float]:
    flat: list[float] = []
    for mx, my in means2d:
        flat.extend([mx, my])
    return flat


def pack_triplets_flat(items: tuple[tuple[float, float, float], ...]) -> list[float]:
    flat: list[float] = []
    for a, b, c in items:
        flat.extend([a, b, c])
    return flat