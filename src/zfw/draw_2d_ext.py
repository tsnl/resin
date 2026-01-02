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
from .cook import CookedAtlas
from .draw_2d import Draw2dQuad, Draw2dRenderer
from . import typed_uharfbuzz as hb


#
# API:
#


class Draw2dExtCanvas(BaseDisposable):
    _glyph_atlas: "GlyphAtlas"

    def __init__(self, *, renderer: Draw2dRenderer):
        super().__init__()

        self._glyph_atlas = GlyphAtlas(canvas=self, gpu_device=renderer.gpu_device)

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
    src_xywh_px: tuple[int, int, int, int] | None = None  # None = full image
    fill_image: GpuImage | None = None  # None = white 1x1 texture
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
        dst_xywh_px: tuple[int, int, int, int] | None = None
        if self.dst_xywh_dip is not None:
            dst_xywh_px = (
                int(self.dst_xywh_dip[0] * scale),
                int(self.dst_xywh_dip[1] * scale),
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
                dst_xywh_px=dst_xywh_px,
                src_xywh_px=self.src_xywh_px,
                fill_image=self.fill_image,
                fill_color=self.fill_color,
                border_thickness_px=border_thickness_px,
                border_color=self.border_color,
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
                        src_xywh_px = (
                            glyph_entry.atlas_x + src_off_x,
                            glyph_entry.atlas_y + src_off_y,
                            src_w,
                            src_h,
                        )

                        result.append(
                            Draw2dQuad(
                                dst_xywh_px=(ix_phys, iy_phys, src_w, src_h),
                                src_xywh_px=src_xywh_px,
                                fill_image=atlas.get_gpu_image(self.font),
                                fill_color=self.color,
                                border_thickness_px=(0, 0, 0, 0),
                                border_color=(0.0, 0.0, 0.0, 0.0),
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
        infos: list[hb.GlyphInfo],
        positions: list[hb.GlyphPosition],
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
        infos: list[hb.GlyphInfo],
        positions: list[hb.GlyphPosition],
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

    _renderer: Draw2dExtCanvas
    _gpu_device: GpuDevice

    # Glyph cache: string key -> GlyphEntry
    # Key format: "glyph_index,font_size,font_weight,scale_num/scale_den"
    _glyph_cache: dict[Font, dict[str, GlyphEntry | None]]

    # GPU images per font
    _gpu_images: dict[Font, GpuImage]

    # Text shapers per font:
    _shapers: dict[Font, "TextShaper"]

    # Pre-computed font metrics: (font, font_size, font_weight, scale) -> FontMetrics
    _font_metrics: dict[tuple[Font, FontSize, FontWeight, Fraction], FontMetrics]

    def __init__(
        self,
        *,
        canvas: Draw2dExtCanvas,
        gpu_device: GpuDevice,
    ):
        super().__init__(parent_resource=canvas)

        self._renderer = canvas
        self._gpu_device = gpu_device
        self._glyph_cache = {}
        self._gpu_images = {}
        self._font_metrics = {}

        # Create shapers:
        self._shapers = {}
        all_fonts: list[Font] = ["sans-serif", "serif", "monospaced"]
        for font in all_fonts:
            self._shapers[font] = TextShaper(font=font)

        # Load pre-cooked atlases:
        self._load_cooked_atlases()

    def _on_dispose(self) -> None:
        for gpu_image in self._gpu_images.values():
            gpu_image.dispose()

    def _load_cooked_atlases(self) -> None:
        """Load pre-cooked glyph atlases from disk."""
        all_fonts: list[Font] = ["sans-serif", "serif", "monospaced"]

        for font in all_fonts:
            atlas_dir = (BUNDLED_DATA_PATH / "fonts" / font).with_suffix(
                CookedAtlas.PATH_SUFFIX
            )

            if not atlas_dir.exists():
                raise FileNotFoundError(
                    f"Cooked atlas not found for font '{font}' at {atlas_dir}. "
                    f"Run 'make build-fonts' to generate bitmap font atlases."
                )

            # Load the cooked atlas
            cooked = CookedAtlas.load(
                path=atlas_dir,
                color_space="linear",
                load_readme_text=False,
                load_license_text=False,
            )

            # Convert to uint8 if needed
            atlas_data = cooked.atlas_data
            if atlas_data.dtype != np.uint8:
                # Assume linear float [0, 1], convert to uint8
                atlas_data = (np.clip(atlas_data, 0.0, 1.0) * 255).astype(np.uint8)

            # Upload to GPU
            gpu_image = GpuImage(
                device=self._gpu_device,
                usages=["texture-binding"],
                data=atlas_data,
            )
            self._gpu_images[font] = gpu_image

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

    def get_gpu_image(self, font: Font) -> GpuImage:
        """Get GPU image for a specific font."""
        return self._gpu_images[font]

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


class TextShaper:
    """
    Text shaper for a specific font family.

    Wraps HarfBuzz for shaping. Metrics come from pre-cooked glyph atlas.
    """

    _font: Font
    _hb_font: hb.Font

    def __init__(self, *, font: Font):
        self._font = font

        # Load font for HarfBuzz:
        file_path = self._get_font_file_path(font)

        with open(file_path, "rb") as f:
            hb_blob = f.read()
        hb_face = hb.Face(hb_blob)
        self._hb_font = hb.Font(hb_face)

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
    ) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition]]:
        """Shape text and return glyph infos and positions."""
        scale = font_size_px * 64  # HarfBuzz uses 26.6 fixed point
        self._hb_font.scale = (scale, scale)
        self._hb_font.set_variations({"wght": font_weight})

        hb_buffer = hb.Buffer()
        hb_buffer.add_str(text)
        hb_buffer.guess_segment_properties()

        hb.shape(self._hb_font, hb_buffer)

        return hb_buffer.glyph_infos, hb_buffer.glyph_positions

    def get_glyph_extents(self, glyph: int) -> hb.GlyphExtents:
        """Get glyph extents for optical bounds calculation."""
        return self._hb_font.get_glyph_extents(glyph)


#
# Logging
#

LOG = logger(__name__)
