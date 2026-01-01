"""
Bitmap font cooker for ZFW SDK.

Pre-generates glyph atlases for all font configurations.
"""

__all__ = ["main"]

from fractions import Fraction
from pathlib import Path
import json

import numpy as np
import PIL.Image

from zfw import logger, Font, FontSize, FontWeight, CookedAtlas, BUNDLED_DATA_PATH
import zfw.typed_freetype as ft
import zfw.typed_uharfbuzz as hb


LOG = logger(__name__)


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
GLYPH_CHARSET = "".join(chr(c) for c in range(32, 127))  # ASCII printable


# Font definitions: (font_name, font_file_path, license_file_path)
FONT_DEFINITIONS: list[tuple[Font, Path, Path]] = [
    (
        "sans-serif",
        Path("res/fonts/Inter/Inter.ttf"),
        Path("res/fonts/Inter/LICENSE.txt"),
    ),
    ("serif", Path("res/fonts/Lora/Lora.ttf"), Path("res/fonts/Lora/LICENSE.txt")),
    (
        "monospaced",
        Path("res/fonts/SourceCodePro/SourceCodePro.ttf"),
        Path("res/fonts/SourceCodePro/LICENSE.txt"),
    ),
]


class GlyphEntry:
    """Entry for a single glyph in the atlas."""

    def __init__(
        self,
        atlas_x: int,
        atlas_y: int,
        width: int,
        height: int,
        bitmap_left: int,
        bitmap_top: int,
    ):
        self.atlas_x = atlas_x
        self.atlas_y = atlas_y
        self.width = width
        self.height = height
        self.bitmap_left = bitmap_left
        self.bitmap_top = bitmap_top


class GlyphAtlasCooker:
    """Cooks bitmap font atlases."""

    font: Font
    font_path: Path
    output_dir: Path

    def __init__(self, *, font: Font, font_path: Path, output_dir: Path):
        self.font = font
        self.font_path = font_path
        self.output_dir = output_dir

        self._page_width = 2048  # 2048x2048 pixel atlas
        self._page_height = 1024
        self._pixel_data = np.zeros(
            (
                self._page_height,
                self._page_width,
                4,
            ),
            dtype=np.uint8,
        )

        self._glyph_cache: dict[str, GlyphEntry | None] = {}
        self._glyph_entries: dict[str, tuple[int, int, int, int]] = {}
        self._glyph_metadata: dict[str, dict] = {}  # Bitmap offsets

        # Packing state
        self._cursor_x = 0
        self._cursor_y = 0
        self._row_height = 0

        # Load font with FreeType and HarfBuzz
        with open(font_path, "rb") as f:
            hb_blob = f.read()
        hb_face = hb.Face(hb_blob)
        self._hb_font = hb.Font(hb_face)

        self._ft_face = ft.Face(str(font_path))

        # Find weight axis
        self._ft_weight_axis_index = None
        info = self._ft_face.get_variation_info()
        for i, axis in enumerate(info.axes):
            if axis.tag == "wght":
                self._ft_weight_axis_index = i
                break

    def cook(self) -> None:
        """Generate all glyph atlases and save to disk."""
        all_sizes: list[FontSize] = ["regular", "large", "extra-large"]
        all_weights: list[FontWeight] = ["light", "regular", "bold"]

        # Standard scale factors to pre-generate
        scales = [Fraction(1, 1), Fraction(2, 1)]

        for font_size in all_sizes:
            for font_weight in all_weights:
                for scale in scales:
                    self._generate_glyphs_for_config(
                        font_size=font_size,
                        font_weight=font_weight,
                        scale=scale,
                    )

        # Save the atlas
        self._save_atlas()

    def _generate_glyphs_for_config(
        self,
        *,
        font_size: FontSize,
        font_weight: FontWeight,
        scale: Fraction,
    ) -> None:
        """Generate glyphs for a specific font configuration."""
        font_size_px = FONT_SIZE_PX[self.font, font_size]
        weight_value = FONT_WEIGHT_VALUE[font_weight]
        effective_size_px = int(font_size_px * float(scale))

        # Shape the charset to get glyph indices
        self._set_hb_scale(effective_size_px, weight_value)
        hb_buffer = hb.Buffer()
        hb_buffer.add_str(GLYPH_CHARSET)
        hb_buffer.guess_segment_properties()
        hb.shape(self._hb_font, hb_buffer)

        infos = hb_buffer.glyph_infos

        # Get unique glyph indices
        glyph_indices = set(info.codepoint for info in infos)

        for glyph_index in glyph_indices:
            cache_key = f"{glyph_index},{font_size},{font_weight},{scale.numerator}/{scale.denominator}"

            if cache_key in self._glyph_cache:
                continue

            # Rasterize the glyph
            entry = self._rasterize_glyph(
                glyph_index=glyph_index,
                font_size_px=effective_size_px,
                font_weight=weight_value,
            )
            self._glyph_cache[cache_key] = entry

            if entry is not None:
                self._glyph_entries[cache_key] = (
                    entry.atlas_x,
                    entry.atlas_y,
                    entry.width,
                    entry.height,
                )
                self._glyph_metadata[cache_key] = {
                    "bitmap_left": entry.bitmap_left,
                    "bitmap_top": entry.bitmap_top,
                }

    def _set_hb_scale(self, font_size_px: int, font_weight: int) -> None:
        """Set HarfBuzz scale and weight."""
        scale = font_size_px * 64  # HarfBuzz uses 26.6 fixed point
        self._hb_font.scale = (scale, scale)
        self._hb_font.set_variations({"wght": font_weight})

    def _set_freetype_weight(self, weight: int) -> None:
        """Set FreeType weight."""
        if self._ft_weight_axis_index is None:
            return
        coords = list(self._ft_face.get_var_design_coords())
        coords[self._ft_weight_axis_index] = float(weight)
        self._ft_face.set_var_design_coords(coords)

    def _rasterize_glyph(
        self,
        *,
        glyph_index: int,
        font_size_px: int,
        font_weight: int,
    ) -> GlyphEntry | None:
        """Rasterize a single glyph and add to atlas."""
        self._ft_face.set_pixel_sizes(0, font_size_px)
        self._set_freetype_weight(font_weight)

        self._ft_face.load_glyph(
            glyph_index, ft.FT_LOAD_RENDER | ft.FT_LOAD_TARGET_NORMAL
        )

        bitmap_left = self._ft_face.glyph.bitmap_left
        bitmap_top = self._ft_face.glyph.bitmap_top
        bitmap = self._ft_face.glyph.bitmap

        if not bitmap.buffer or bitmap.width == 0 or bitmap.rows == 0:
            return None

        h, w = bitmap.rows, bitmap.width
        pitch = bitmap.pitch

        # Load buffer
        buffer_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(h, pitch)
        if pitch != w:
            buffer_array = buffer_array[:, :w]

        # Allocate space in atlas
        entry = self._allocate_glyph(
            width=w,
            height=h,
            bitmap_left=bitmap_left,
            bitmap_top=bitmap_top,
            data=buffer_array,
        )

        return entry

    def _allocate_glyph(
        self,
        *,
        width: int,
        height: int,
        bitmap_left: int,
        bitmap_top: int,
        data: np.ndarray,
    ) -> GlyphEntry | None:
        """Allocate space for a glyph and copy its data."""
        pad = 1

        # Try to fit on current row
        if self._cursor_x + width + pad > self._page_width:
            # Move to next row
            self._cursor_x = 0
            self._cursor_y += self._row_height + pad
            self._row_height = 0

        # Check if we have space
        if self._cursor_y + height + pad > self._page_height:
            LOG.error("Glyph atlas full, cannot allocate more glyphs")
            return None

        # Allocate
        x = self._cursor_x
        y = self._cursor_y

        # Copy data (grayscale to RGBA white with alpha)
        self._pixel_data[y : y + height, x : x + width, 0] = 0xFF  # R
        self._pixel_data[y : y + height, x : x + width, 1] = 0xFF  # G
        self._pixel_data[y : y + height, x : x + width, 2] = 0xFF  # B
        self._pixel_data[y : y + height, x : x + width, 3] = data  # A

        # Update cursor
        self._cursor_x += width + pad
        self._row_height = max(self._row_height, height)

        return GlyphEntry(
            atlas_x=x,
            atlas_y=y,
            width=width,
            height=height,
            bitmap_left=bitmap_left,
            bitmap_top=bitmap_top,
        )

    def _save_atlas(self) -> None:
        """Save the atlas to disk using CookedAtlas."""
        # Load license text
        license_path = self.font_path.parent / "LICENSE.txt"
        with open(license_path, "r") as f:
            license_text = f.read()

        # Create a simple README
        readme_text = (
            f"Bitmap font atlas for {self.font}.\n\nGenerated by zfw_bmfont_cooker."
        )

        # Create CookedAtlas
        atlas = CookedAtlas(
            atlas_data=self._pixel_data,
            image_xywh=self._glyph_entries,
            image_metadata=self._glyph_metadata,
            color_space="linear",
            readme_text=readme_text,
            license_text=license_text,
        )

        # Save to output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atlas.save(self.output_dir)

        LOG.info(f"Saved font atlas for {self.font} to {self.output_dir}")


def main() -> None:
    """Main entry point."""

    project_root = Path.cwd()

    for font_name, font_rel_path, license_rel_path in FONT_DEFINITIONS:
        font_path = project_root / font_rel_path
        license_path = project_root / license_rel_path

        if not font_path.exists():
            LOG.error(f"Font file not found: {font_path}")
            continue

        if not license_path.exists():
            LOG.error(f"License file not found: {license_path}")
            continue

        output_dir = BUNDLED_DATA_PATH / "fonts" / font_name

        cooker = GlyphAtlasCooker(
            font=font_name,
            font_path=font_path,
            output_dir=output_dir,
        )
        cooker.cook()

    LOG.info("Bitmap font cooking complete!")


if __name__ == "__main__":
    main()
