"""
Draw2dExt offers a higher-level interface for 2D drawing compared to Draw2d.

It helps generate Draw2dQuad instances in physical pixel coordinates from higher-level
primitives specified in device-independent pixel coordinates and text primitives.

It provides:
-   HiDPI support: you specify logical (device-independent) pixel coordinates, it
    handles conversion to physical pixels using a configured scale factor.
-   Efficient text rendering with pre-generated bitmap glyph atlases.
"""

__all__ = [
    "Draw2dExtBasePrimitive",
    "Draw2dExtCanvas",
    "Draw2dExtQuadPrimitive",
    "Draw2dExtTextPrimitive",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import wgpu

from .basic import (
    BaseDisposable,
    Font,
    FontSize,
    FontWeight,
    HorizontalAlignment,
    VerticalAlignment,
    logger,
)
from .bundled_data import BUNDLED_DATA_PATH
from .resources import COOKED_ATLAS_PATH_SUFFIX, CookedAtlas
from .draw_2d import Draw2dQuad
from . import typed_freetype as ft


#
# API:
#


class Draw2dExtCanvas(BaseDisposable):
    _glyph_atlas: "GlyphAtlas"

    def __init__(self, *, device: wgpu.GPUDevice, queue: wgpu.GPUQueue):
        super().__init__()

        self._glyph_atlas = GlyphAtlas(canvas=self, device=device, queue=queue)

    def quads(
        self,
        *,
        primitives: list["Draw2dExtBasePrimitive"],
        scale: float = 1.0,
    ) -> list[Draw2dQuad]:
        quads: list[Draw2dQuad] = []
        for primitive in primitives:
            quads += primitive._generate_primitive_quads(canvas=self, scale=scale)
        return quads

    def _on_dispose(self) -> None:
        self._glyph_atlas.dispose()


@dataclass(frozen=True, kw_only=True)
class Draw2dExtBasePrimitive(ABC):
    @abstractmethod
    def _generate_primitive_quads(
        self, *, canvas: Draw2dExtCanvas, scale: float
    ) -> list[Draw2dQuad]: ...


@dataclass(frozen=True, kw_only=True)
class Draw2dExtQuadPrimitive(Draw2dExtBasePrimitive):
    """
    A quad in device-independent pixel coordinates.

    Coordinates use the `_dip` suffix to indicate device-independent pixels.
    These are converted to physical pixels using the canvas scale factor.
    """

    dst_xywh_dip: tuple[int, int, int, int] | None = None  # None = full target
    src_xy_px: tuple[int, int] | None = None  # None = full image
    fill_texture: wgpu.GPUTexture | None = None  # None = white 1x1 texture
    fill_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    border_thickness_dip: tuple[int, int, int, int] = (0, 0, 0, 0)  # TRBL
    border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def _generate_primitive_quads(
        self,
        *,
        canvas: Draw2dExtCanvas,
        scale: float,
    ) -> list[Draw2dQuad]:
        """Convert to a Draw2dQuad with physical pixel coordinates."""
        dst_xy_px: tuple[int, int] = (0, 0)
        dst_wh_px: tuple[int, int] | None = None
        if self.dst_xywh_dip is not None:
            dst_xy_px = (
                int(self.dst_xywh_dip[0] * scale),
                int(self.dst_xywh_dip[1] * scale),
            )
            dst_wh_px = (
                int(self.dst_xywh_dip[2] * scale),
                int(self.dst_xywh_dip[3] * scale),
            )

        border_thickness_px = (
            int(self.border_thickness_dip[0] * scale),
            int(self.border_thickness_dip[1] * scale),
            int(self.border_thickness_dip[2] * scale),
            int(self.border_thickness_dip[3] * scale),
        )

        return [
            Draw2dQuad(
                dst_xy_px=dst_xy_px,
                dst_wh_px=dst_wh_px,
                src_xy_px=self.src_xy_px,
                src_wh_px=None,
                fill_texture=self.fill_texture,
                fill_color_rgba=self.fill_color,
                border_thickness_px=border_thickness_px,
                border_color_rgba=self.border_color,
            )
        ]


@dataclass(frozen=True, kw_only=True)
class Draw2dExtTextPrimitive(Draw2dExtBasePrimitive):  #
    """Internal representation of a text primitive."""

    text: str
    font: Font
    dst_xy_dip: tuple[int, int]
    dst_wh_dip: tuple[int, int]
    font_size: FontSize
    font_weight: FontWeight
    color: tuple[float, float, float, float]
    wrap: bool
    horizontal_alignment: HorizontalAlignment
    vertical_alignment: VerticalAlignment

    def _generate_primitive_quads(
        self,
        *,
        canvas: Draw2dExtCanvas,
        scale: float,
    ) -> list[Draw2dQuad]:
        """Convert text to a list of glyph quads."""
        if not self.text:
            return []

        atlas = canvas._glyph_atlas
        font_size_px = FONT_SIZE_PX[self.font, self.font_size]
        weight_value = FONT_WEIGHT_VALUE[self.font_weight]

        # Get text shaper for this configuration:
        shaper = atlas.get_shaper(self.font)

        # Get shaped glyph infos and positions:
        effective_size_px = int(font_size_px * scale)
        infos, positions = shaper.shape_text(
            text=self.text,
            font_size_px=effective_size_px,
            font_weight=weight_value,
        )

        # Get font metrics from pre-cooked glyph cache:
        metrics = atlas.get_font_metrics(
            font=self.font,
            font_size=self.font_size,
            font_weight=self.font_weight,
            scale=scale,
        )
        if metrics is None:
            # Fallback: no pre-cooked metrics for this configuration
            return []

        # Convert destination rect to physical 26.6 fixed-point:
        dst_x_26_6 = int(self.dst_xy_dip[0] * scale * 64)
        dst_y_26_6 = int(self.dst_xy_dip[1] * scale * 64)
        dst_w_26_6 = int(self.dst_wh_dip[0] * scale * 64)
        dst_h_26_6 = int(self.dst_wh_dip[1] * scale * 64)

        # Physical pixel versions for clipping:
        dst_x_phys = int(self.dst_xy_dip[0] * scale)
        dst_y_phys = int(self.dst_xy_dip[1] * scale)
        dst_w_phys = int(self.dst_wh_dip[0] * scale)
        dst_h_phys = int(self.dst_wh_dip[1] * scale)

        # Layout lines:
        lines = self._layout_text_lines(
            infos=infos,
            positions=positions,
            text=self.text,
            wrap=self.wrap,
            max_width_26_6=dst_w_26_6,
        )

        total_text_height_26_6 = len(lines) * metrics.height_26_6

        # Vertical alignment:
        start_y_26_6 = dst_y_26_6
        if self.vertical_alignment == "middle":
            start_y_26_6 += (dst_h_26_6 - total_text_height_26_6) // 2
        elif self.vertical_alignment == "bottom":
            start_y_26_6 += dst_h_26_6 - total_text_height_26_6

        pen_y_26_6 = start_y_26_6 + metrics.ascender_26_6

        result: list[Draw2dQuad] = []

        for start_idx, end_idx, line_width_26_6 in lines:
            # Horizontal alignment:
            pen_x_26_6 = dst_x_26_6

            min_ink, max_ink = self._get_line_optical_bounds(
                shaper=shaper,
                infos=infos,
                positions=positions,
                start_idx=start_idx,
                end_idx=end_idx,
            )
            optical_width = max_ink - min_ink

            if self.horizontal_alignment == "center":
                pen_x_26_6 += (dst_w_26_6 - optical_width) // 2 - min_ink
            elif self.horizontal_alignment == "right":
                pen_x_26_6 += dst_w_26_6 - max_ink
            elif self.horizontal_alignment == "left":
                pen_x_26_6 -= min_ink

            for i in range(start_idx, end_idx):
                info = infos[i]
                pos = positions[i]

                codepoint = info.codepoint
                x_advance_26_6 = pos.x_advance
                y_advance_26_6 = pos.y_advance
                x_offset_26_6 = pos.x_offset
                y_offset_26_6 = pos.y_offset

                glyph_entry = atlas.get_glyph(
                    font=self.font,
                    glyph_index=codepoint,
                    font_size=self.font_size,
                    font_weight=self.font_weight,
                    scale=scale,
                )

                if glyph_entry is not None:
                    # Convert pen position to physical pixels:
                    pen_x_phys = (pen_x_26_6 + 32) >> 6
                    pen_y_phys = (pen_y_26_6 + 32) >> 6
                    x_offset_phys = (x_offset_26_6 + 32) >> 6
                    y_offset_phys = (y_offset_26_6 + 32) >> 6

                    # Glyph quad position in physical pixels:
                    qx_phys = pen_x_phys + x_offset_phys + glyph_entry.bitmap_left
                    qy_phys = pen_y_phys - glyph_entry.bitmap_top - y_offset_phys

                    # Glyph dimensions in physical pixels:
                    qw_phys = glyph_entry.width
                    qh_phys = glyph_entry.height

                    # Intersection with dst rect (clipping):
                    ix_phys = max(qx_phys, dst_x_phys)
                    iy_phys = max(qy_phys, dst_y_phys)
                    ir_phys = min(qx_phys + qw_phys, dst_x_phys + dst_w_phys)
                    ib_phys = min(qy_phys + qh_phys, dst_y_phys + dst_h_phys)

                    if ir_phys > ix_phys and ib_phys > iy_phys:
                        # Clipping offset:
                        src_off_x = ix_phys - qx_phys
                        src_off_y = iy_phys - qy_phys
                        src_w = min(ir_phys - ix_phys, qw_phys - src_off_x)
                        src_h = min(ib_phys - iy_phys, qh_phys - src_off_y)

                        # Source rect in atlas (physical pixels):
                        src_xy_px = (
                            glyph_entry.atlas_x + src_off_x,
                            glyph_entry.atlas_y + src_off_y,
                        )

                        result.append(
                            Draw2dQuad(
                                dst_xy_px=(ix_phys, iy_phys),
                                dst_wh_px=(src_w, src_h),
                                src_xy_px=src_xy_px,
                                src_wh_px=(src_w, src_h),
                                fill_texture=atlas.get_gpu_texture(self.font),
                                fill_color_rgba=self.color,
                                border_thickness_px=(0, 0, 0, 0),
                                border_color_rgba=(0.0, 0.0, 0.0, 0.0),
                            )
                        )

                # Accumulate pen position:
                pen_x_26_6 += x_advance_26_6
                pen_y_26_6 += y_advance_26_6

            pen_y_26_6 += metrics.height_26_6

        return result

    def _layout_text_lines(
        self,
        *,
        infos: list["GlyphInfo"],
        positions: list["GlyphPosition"],
        text: str,
        wrap: bool,
        max_width_26_6: int,
    ) -> list[tuple[int, int, int]]:
        """
        Layout text into lines.

        Returns list of (start_idx, end_idx, line_width_26_6) tuples.
        """
        lines: list[tuple[int, int, int]] = []
        line_start_index = 0
        current_line_width_26_6 = 0

        i = 0
        while i < len(infos):
            info = infos[i]
            pos = positions[i]
            cluster = info.cluster
            char = text[cluster] if cluster < len(text) else " "

            if char == "\n":
                lines.append((line_start_index, i, current_line_width_26_6))
                line_start_index = i + 1
                current_line_width_26_6 = 0
                i += 1
                continue

            if wrap and not char.isspace():
                is_word_start = False
                if i == line_start_index:
                    is_word_start = True
                else:
                    prev_cluster = infos[i - 1].cluster
                    prev_char = text[prev_cluster] if prev_cluster < len(text) else " "
                    if prev_char.isspace():
                        is_word_start = True

                if is_word_start:
                    word_width_26_6 = 0
                    for j in range(i, len(infos)):
                        c = infos[j].cluster
                        c_char = text[c] if c < len(text) else " "
                        if c_char.isspace() or c_char == "\n":
                            break
                        word_width_26_6 += positions[j].x_advance

                    if (
                        current_line_width_26_6 + word_width_26_6 > max_width_26_6
                    ) and (current_line_width_26_6 > 0):
                        lines.append((line_start_index, i, current_line_width_26_6))
                        line_start_index = i
                        current_line_width_26_6 = 0

            current_line_width_26_6 += pos.x_advance
            i += 1

        lines.append((line_start_index, len(infos), current_line_width_26_6))
        return lines

    def _get_line_optical_bounds(
        self,
        *,
        shaper: "TextShaper",
        infos: list["GlyphInfo"],
        positions: list["GlyphPosition"],
        start_idx: int,
        end_idx: int,
    ) -> tuple[int, int]:
        """
        Returns (min_x, max_x) of the ink bounds for the given range of glyphs,
        relative to the start of the line (pen_x = 0).
        Values are in 26.6 fixed point.
        """
        min_x = 2147483647
        max_x = -2147483648
        pen_x = 0
        has_ink = False

        for i in range(start_idx, end_idx):
            info = infos[i]
            pos = positions[i]

            extents = shaper.get_glyph_extents(info.codepoint)

            if extents.width != 0 and extents.height != 0:
                left = pen_x + pos.x_offset + extents.x_bearing
                right = left + extents.width

                if left < min_x:
                    min_x = left
                if right > max_x:
                    max_x = right
                has_ink = True

            pen_x += pos.x_advance

        if not has_ink:
            return 0, 0

        return min_x, max_x


#
# Configuration
#


# Pixel sizes for each FontSize (at scale=1.0)
FONT_SIZE_PX: dict[tuple[Font, FontSize], int] = {
    ("sans-serif", "regular"): 14,
    ("sans-serif", "large"): 24,
    ("sans-serif", "extra-large"): 32,
    ("serif", "regular"): 16,
    ("serif", "large"): 24,
    ("serif", "extra-large"): 32,
    ("monospaced", "regular"): 14,
    ("monospaced", "large"): 24,
    ("monospaced", "extra-large"): 32,
}

# Weight values for variable fonts
FONT_WEIGHT_VALUE: dict[FontWeight, int] = {
    "light": 200,
    "regular": 400,
    "bold": 700,
}

# Characters to pre-rasterize for the glyph atlas
# ASCII printable characters plus some common punctuation
GLYPH_CHARSET = "".join(chr(c) for c in range(32, 127))  # ASCII printable


#
# Implementation: Glyph Atlas
#


@dataclass(frozen=True)
class GlyphEntry:
    """Entry for a single glyph in the atlas."""

    atlas_x: int
    atlas_y: int
    width: int
    height: int
    bitmap_left: int
    bitmap_top: int


@dataclass(frozen=True)
class FontMetrics:
    """Font metrics for a specific size."""

    ascender_26_6: int
    height_26_6: int


class GlyphAtlas(BaseDisposable):
    """
    Pre-generated glyph atlas containing all glyphs needed for text rendering.

    Glyphs are loaded from pre-cooked bitmap font atlases at context creation time.
    The atlases are created offline for all combinations of:
    - Font families (sans-serif, serif, monospaced)
    - Font weights (light, regular, bold)
    - Font sizes (regular, large, extra-large)
    - Common ASCII characters
    - Scale factors (1.0, 2.0)
    """

    _device: wgpu.GPUDevice
    _queue: wgpu.GPUQueue

    # Glyph cache: string key -> GlyphEntry
    # Key format: "glyph_index,font_size,font_weight,scale_num/scale_den"
    _glyph_cache: dict[Font, dict[str, GlyphEntry | None]]

    # GPU textures per font
    _gpu_textures: dict[Font, wgpu.GPUTexture]

    # Text shapers per font:
    _shapers: dict[Font, "TextShaper"]

    # Pre-computed font metrics: (font, font_size, font_weight, scale) -> FontMetrics
    _font_metrics: dict[tuple[Font, FontSize, FontWeight, Fraction], FontMetrics]

    def __init__(
        self,
        *,
        canvas: Draw2dExtCanvas,
        device: wgpu.GPUDevice,
        queue: wgpu.GPUQueue,
    ):
        super().__init__()

        self._device = device
        self._queue = queue
        self._glyph_cache = {}
        self._gpu_textures = {}
        self._font_metrics = {}

        # Create shapers:
        self._shapers = {}
        all_fonts: list[Font] = ["sans-serif", "serif", "monospaced"]
        for font in all_fonts:
            self._shapers[font] = TextShaper(font=font)

        # Load pre-cooked atlases:
        self._load_cooked_atlases()

    def _on_dispose(self) -> None:
        pass  # WebGPU textures are automatically cleaned up

    def _load_cooked_atlases(self) -> None:
        """Load pre-cooked glyph atlases from disk."""
        all_fonts: list[Font] = ["sans-serif", "serif", "monospaced"]

        for font in all_fonts:
            atlas_dir = (BUNDLED_DATA_PATH / "fonts" / font).with_suffix(
                COOKED_ATLAS_PATH_SUFFIX
            )

            if not atlas_dir.exists():
                raise FileNotFoundError(
                    f"Cooked atlas not found for font '{font}' at {atlas_dir}. "
                    f"Run 'make build-fonts' to generate bitmap font atlases."
                )

            # Load the cooked atlas
            cooked = CookedAtlas.load(
                path=atlas_dir,
                image_format="r32float",
                load_readme_text=False,
                load_license_text=False,
            )

            # Convert to uint8 if needed
            atlas_data = cooked.atlas_data
            if atlas_data.dtype != np.uint8:
                # Assume linear float [0, 1], convert to uint8
                atlas_data = (np.clip(atlas_data, 0.0, 1.0) * 255).astype(np.uint8)

            # Create texture and upload data
            rgba_texture = self._device.create_texture(
                label=f"GlyphAtlas.{font}",
                size=(atlas_data.shape[1], atlas_data.shape[0], 1),
                format="rgba8unorm",
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
            )

            self._queue.write_texture(
                destination=wgpu.TexelCopyTextureInfo(
                    texture=rgba_texture,
                    mip_level=0,
                    origin=(0, 0, 0),
                ),
                data=atlas_data.tobytes(),
                data_layout=wgpu.TexelCopyBufferLayout(
                    offset=0,
                    bytes_per_row=atlas_data.shape[1] * 4,
                    rows_per_image=atlas_data.shape[0],
                ),
                size=(atlas_data.shape[1], atlas_data.shape[0], 1),
            )

            self._gpu_textures[font] = rgba_texture

            # Build glyph cache from the cooked atlas and extract metrics
            glyph_cache: dict[str, GlyphEntry | None] = {}
            if cooked.as_glyph_cache:
                for key, info in cooked.as_glyph_cache.items():
                    # Get xywh from image list using image_id
                    xywh = cooked.image_xywh_list[info.image_id]

                    # Create string key for internal cache (matching get_glyph format)
                    cache_key = f"{key.glyph_index},{key.font_size},{key.font_weight},{key.scale.numerator}/{key.scale.denominator}"

                    entry = GlyphEntry(
                        atlas_x=xywh[0],
                        atlas_y=xywh[1],
                        width=xywh[2],
                        height=xywh[3],
                        bitmap_left=info.bitmap_left,
                        bitmap_top=info.bitmap_top,
                    )
                    glyph_cache[cache_key] = entry

                    # Store font metrics (one per configuration)
                    metrics_key = (
                        key.font_name,
                        key.font_size,
                        key.font_weight,
                        key.scale,
                    )
                    if metrics_key not in self._font_metrics:
                        self._font_metrics[metrics_key] = FontMetrics(
                            ascender_26_6=info.ascender_26_6,
                            height_26_6=info.height_26_6,
                        )

            self._glyph_cache[font] = glyph_cache

            LOG.info(f"Loaded glyph atlas for font '{font}' from {atlas_dir}")

    def get_gpu_texture(self, font: Font) -> wgpu.GPUTexture:
        """Get GPU texture for a specific font."""
        return self._gpu_textures[font]

    def get_shaper(self, font: Font) -> "TextShaper":
        return self._shapers[font]

    def get_glyph(
        self,
        *,
        font: Font,
        glyph_index: int,
        font_size: FontSize,
        font_weight: FontWeight,
        scale: float,
    ) -> GlyphEntry | None:
        """
        Get a pre-generated glyph entry.
        """

        # Convert scale to rational representation
        scale_frac = Fraction(scale).limit_denominator(16)
        cache_key = f"{glyph_index},{font_size},{font_weight},{scale_frac.numerator}/{scale_frac.denominator}"

        font_cache = self._glyph_cache.get(font)
        if font_cache is None:
            return None

        return font_cache.get(cache_key)

    def get_font_metrics(
        self,
        *,
        font: Font,
        font_size: FontSize,
        font_weight: FontWeight,
        scale: float,
    ) -> FontMetrics | None:
        """Get pre-computed font metrics from glyph cache."""
        scale_frac = Fraction(scale).limit_denominator(1000)
        metrics_key = (font, font_size, font_weight, scale_frac)
        return self._font_metrics.get(metrics_key)


#
# Implementation: Text Shaper
#


@dataclass(frozen=True)
class GlyphInfo:
    """Glyph info compatible with HarfBuzz interface."""

    codepoint: int
    cluster: int


@dataclass(frozen=True)
class GlyphPosition:
    """Glyph position compatible with HarfBuzz interface."""

    x_advance: int
    y_advance: int
    x_offset: int
    y_offset: int


@dataclass(frozen=True)
class GlyphExtents:
    """Glyph extents compatible with HarfBuzz interface."""

    x_bearing: int
    y_bearing: int
    width: int
    height: int


class TextShaper:
    """
    Text shaper for a specific font family.

    Uses FreeType for simple left-to-right text layout. Metrics come from pre-cooked 
    glyph atlas.
    """

    _font: Font
    _ft_face: ft.Face

    def __init__(self, *, font: Font):
        self._font = font

        # Load font for FreeType:
        file_path = self._get_font_file_path(font)
        self._ft_face = ft.Face(str(file_path))

    @staticmethod
    def _get_font_file_path(font: Font) -> Path:
        return {
            "sans-serif": BUNDLED_DATA_PATH / "fonts/Inter.ttf",
            "serif": BUNDLED_DATA_PATH / "fonts/Lora.ttf",
            "monospaced": BUNDLED_DATA_PATH / "fonts/SourceCodePro.ttf",
        }[font]

    def shape_text(
        self,
        *,
        text: str,
        font_size_px: int,
        font_weight: int,
    ) -> tuple[list[GlyphInfo], list[GlyphPosition]]:
        """Shape text and return glyph infos and positions (simple LTR layout)."""
        self.set_font_size(font_size_px)
        self.set_font_weight(font_weight)
        
        infos: list[GlyphInfo] = []
        positions: list[GlyphPosition] = []

        # Simple left-to-right layout
        for cluster, char in enumerate(text):
            glyph_index = self._ft_face.get_char_index(ord(char))
            self._ft_face.load_glyph(glyph_index, ft.FT_LOAD_TARGET_NORMAL)

            # Get advance in 26.6 fixed point
            x_advance = self._ft_face.glyph.advance.x

            infos.append(GlyphInfo(codepoint=glyph_index, cluster=cluster))
            positions.append(
                GlyphPosition(
                    x_advance=x_advance,
                    y_advance=0,
                    x_offset=0,
                    y_offset=0,
                )
            )

        return infos, positions

    def set_font_size(self, size_px: int) -> None:
        """Set font size for shaping."""
        self._ft_face.set_pixel_sizes(0, size_px)

    def set_font_weight(self, weight: int) -> None:
        """Set font weight for shaping (variable fonts only)."""
        var_info = self._ft_face.get_variation_info()
        if not var_info.axes:
            return
        coords = list(self._ft_face.get_var_design_coords())
        for i, axis in enumerate(var_info.axes):
            if axis.tag == "wght":
                coords[i] = float(weight)
        self._ft_face.set_var_design_coords(coords)

    def get_glyph_extents(self, glyph: int) -> GlyphExtents:
        """Get glyph extents for optical bounds calculation."""
        self._ft_face.load_glyph(glyph, ft.FT_LOAD_TARGET_NORMAL)

        # Get glyph metrics in 26.6 fixed point
        metrics = self._ft_face.glyph.metrics  # type: ignore
        return GlyphExtents(
            x_bearing=metrics.horiBearingX,
            y_bearing=metrics.horiBearingY,
            width=metrics.width,
            height=metrics.height,
        )


#
# Logging
#

LOG = logger(__name__)
