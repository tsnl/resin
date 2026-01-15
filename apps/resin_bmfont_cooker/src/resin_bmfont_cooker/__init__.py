"""
Bitmap font cooker for Resin SDK.

Pre-generates glyph atlases for all font configurations.
"""

__all__ = ["main"]

import argparse
from fractions import Fraction
import logging
from pathlib import Path
import sys
from typing import cast

import numpy as np

from resin import (
    setup_logging,
    Font,
    FontSize,
    FontWeight,
    CookedAtlas,
    CookedAtlasGlyphInfo,
    CookedAtlasGlyphCacheKey,
    COOKED_ATLAS_PATH_SUFFIX,
    typed_freetype as ft,
)


LOG = logging.getLogger(__name__)


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

FONT_TO_FONT_NAME_DICT: dict[Font, str] = {
    cast(Font, "sans-serif"): "Inter",
    cast(Font, "serif"): "Lora",
    cast(Font, "monospaced"): "SourceCodePro",
}

# Weight values for variable fonts
FONT_WEIGHT_VALUE: dict[FontWeight, int] = {
    "light": 200,
    "regular": 400,
    "bold": 700,
}

# Characters to pre-rasterize for the glyph atlas
GLYPH_CODE_POINTS = set(range(32, 127))  # Basic ASCII


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

    def __init__(
        self,
        *,
        font: Font,
        font_path: Path,
        license_path: Path,
        output_dir: Path,
    ):
        self.font = font
        self.font_path = font_path
        self.license_path = license_path
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
        self._image_xywh_list: list[tuple[int, int, int, int]] = []
        self._as_glyph_cache: dict[CookedAtlasGlyphCacheKey, CookedAtlasGlyphInfo] = {}

        # Packing state
        self._cursor_x = 0
        self._cursor_y = 0
        self._row_height = 0

        # Load font with FreeType
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

        # Get unique glyph indices
        glyph_indices = {self._ft_face.get_char_index(c) for c in GLYPH_CODE_POINTS}

        # Compute and store metrics for this configuration
        self._ft_face.set_pixel_sizes(0, effective_size_px)
        self._set_freetype_weight(weight_value)
        metrics = self._ft_face.size

        for glyph_index in glyph_indices:
            cache_key = f"{glyph_index},{font_size},{font_weight},{scale.numerator}/{scale.denominator}"
            atlas_key = CookedAtlasGlyphCacheKey(
                glyph_index=glyph_index,
                font_name=self.font,
                font_size=font_size,
                font_weight=font_weight,
                scale=scale,
            )

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
                # Add to image list and get image_id
                image_id = len(self._image_xywh_list)
                self._image_xywh_list.append(
                    (
                        entry.atlas_x,
                        entry.atlas_y,
                        entry.width,
                        entry.height,
                    )
                )
                # Store glyph info with image_id
                self._as_glyph_cache[atlas_key] = CookedAtlasGlyphInfo(
                    image_id=image_id,
                    ascender_26_6=metrics.ascender,
                    descender_26_6=metrics.descender,
                    height_26_6=metrics.height,
                    bitmap_left=entry.bitmap_left,
                    bitmap_top=entry.bitmap_top,
                )

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
        with open(self.license_path, "r") as f:
            license_text = f.read()

        # Create a simple README
        readme_text = (
            f"Bitmap font atlas for {self.font}.\n\nGenerated by resin_bmfont_cooker."
        )

        # Convert pixel data to float32 [0, 1] range
        atlas_data = self._pixel_data.astype(np.float32) / 255.0

        # Create CookedAtlas
        atlas = CookedAtlas(
            atlas_type="glyph-cache",
            atlas_data=atlas_data,
            image_xywh_list=self._image_xywh_list,
            image_format="r32float",
            readme_text=readme_text,
            license_text=license_text,
            as_glyph_cache=self._as_glyph_cache,
        )

        # Save to output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        atlas.save(self.output_dir)


def main_impl() -> int:
    """Main entry point."""

    setup_logging()

    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    args = ap.parse_args()

    project_root = Path.cwd()

    output_path: Path = args.output

    if output_path.suffix != COOKED_ATLAS_PATH_SUFFIX:
        LOG.error(f"Output path must have suffix {COOKED_ATLAS_PATH_SUFFIX!r}")
        return 1

    if output_path.is_dir():
        LOG.error(f"Bitmap font already exists: {str(output_path)!r}")
        return 1

    font = cast(Font, output_path.stem.removesuffix(COOKED_ATLAS_PATH_SUFFIX))
    font_name = FONT_TO_FONT_NAME_DICT.get(font)
    if font_name is None:
        LOG.error(f"Unknown font: {font!r}")
        return 1

    font_dir = project_root / "res/fonts" / font_name
    if not font_dir.is_dir():
        LOG.error(f"Font folder not found: {font_dir}")
        return 1

    font_ttf_path = font_dir / f"{font_name}.ttf"
    if not font_ttf_path.exists():
        LOG.error(f"Font file not found: {font_ttf_path}")
        return 1

    license_path = font_dir / "LICENSE.txt"
    if not license_path.exists():
        LOG.error(f"License file not found: {license_path}")
        return 1

    cooker = GlyphAtlasCooker(
        font=font,
        font_path=font_ttf_path,
        output_dir=output_path,
        license_path=license_path,
    )
    cooker.cook()

    return 0


def main():
    sys.exit(main_impl())


if __name__ == "__main__":
    main()
