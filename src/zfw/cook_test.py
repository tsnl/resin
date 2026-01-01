"""
Round-trip tests for CookedAtlas.
"""

from fractions import Fraction
from pathlib import Path
import shutil

import numpy as np

from .cook import (
    CookedAtlas,
    CookedAtlasGlyphCacheKey,
    CookedAtlasGlyphInfo,
)


def test_cooked_atlas_roundtrip_glyph_cache():
    """Test round-trip save/load of a glyph cache atlas."""
    # Create a simple test atlas
    atlas_data = np.random.rand(64, 64, 4).astype(np.float32)

    # Create image list
    image_xywh_list = [
        (0, 0, 16, 16),
        (16, 0, 16, 16),
        (32, 0, 16, 16),
    ]

    # Create glyph cache
    as_glyph_cache = {
        CookedAtlasGlyphCacheKey(
            glyph_index=65,  # 'A'
            font_name="sans-serif",
            font_size="regular",
            font_weight="regular",
            scale=Fraction(1, 1),
        ): CookedAtlasGlyphInfo(
            image_id=0,
            ascender_26_6=1000,
            descender_26_6=-200,
            height_26_6=1200,
            bitmap_left=2,
            bitmap_top=12,
        ),
        CookedAtlasGlyphCacheKey(
            glyph_index=66,  # 'B'
            font_name="sans-serif",
            font_size="regular",
            font_weight="regular",
            scale=Fraction(1, 1),
        ): CookedAtlasGlyphInfo(
            image_id=1,
            ascender_26_6=1000,
            descender_26_6=-200,
            height_26_6=1200,
            bitmap_left=2,
            bitmap_top=12,
        ),
        CookedAtlasGlyphCacheKey(
            glyph_index=67,  # 'C'
            font_name="sans-serif",
            font_size="large",
            font_weight="bold",
            scale=Fraction(2, 1),
        ): CookedAtlasGlyphInfo(
            image_id=2,
            ascender_26_6=2000,
            descender_26_6=-400,
            height_26_6=2400,
            bitmap_left=4,
            bitmap_top=24,
        ),
    }

    # Create atlas
    original = CookedAtlas(
        atlas_data=atlas_data,
        image_xywh_list=image_xywh_list,
        color_space="linear",
        readme_text="Test atlas",
        license_text="MIT License",
        as_glyph_cache=as_glyph_cache,
    )
    assert original.as_glyph_cache is not None

    # Save and load
    save_path = Path(
        "output/zfw/cook_test/test_cooked_atlas_roundtrip_glyph_cache.zfw_atlas"
    )
    if save_path.exists():
        shutil.rmtree(save_path)

    original.save(save_path)

    # Load it back
    loaded = CookedAtlas.load(
        path=save_path,
        color_space="linear",
        load_readme_text=True,
        load_license_text=True,
        load_glyph_metrics=True,
    )

    # Verify atlas data (with tolerance for PNG compression)
    assert loaded.atlas_data.shape == original.atlas_data.shape
    np.testing.assert_allclose(
        loaded.atlas_data, original.atlas_data, rtol=0.01, atol=0.01
    )

    # Verify image list
    assert loaded.image_xywh_list == original.image_xywh_list

    # Verify metadata
    assert loaded.color_space == original.color_space
    assert loaded.readme_text == original.readme_text
    assert loaded.license_text == original.license_text

    # Verify glyph cache
    assert loaded.as_glyph_cache is not None
    assert len(loaded.as_glyph_cache) == len(original.as_glyph_cache)

    for key, original_info in original.as_glyph_cache.items():
        assert key in loaded.as_glyph_cache
        loaded_info = loaded.as_glyph_cache[key]
        assert loaded_info.image_id == original_info.image_id
        assert loaded_info.ascender_26_6 == original_info.ascender_26_6
        assert loaded_info.descender_26_6 == original_info.descender_26_6
        assert loaded_info.height_26_6 == original_info.height_26_6
        assert loaded_info.bitmap_left == original_info.bitmap_left
        assert loaded_info.bitmap_top == original_info.bitmap_top


def test_cooked_atlas_roundtrip_without_glyph_cache():
    """Test round-trip save/load of an atlas without glyph cache data."""
    # Create a simple test atlas
    atlas_data = np.random.rand(128, 128, 4).astype(np.float32)

    # Create image list
    image_xywh_list = [
        (0, 0, 32, 32),
        (32, 0, 32, 32),
    ]

    # Create atlas without glyph cache
    original = CookedAtlas(
        atlas_data=atlas_data,
        image_xywh_list=image_xywh_list,
        color_space="srgb",
        readme_text=None,
        license_text=None,
        as_glyph_cache=None,
    )

    # Save and load
    save_path = Path(
        "output/zfw/cook_test/test_cooked_atlas_roundtrip_no_glyph_cache.zfw_atlas"
    )
    if save_path.exists():
        shutil.rmtree(save_path)
    original.save(save_path)

    # Load it back
    loaded = CookedAtlas.load(
        path=save_path,
        color_space="srgb",
        load_readme_text=False,
        load_license_text=False,
        load_glyph_metrics=False,
    )

    # Verify basic data
    assert loaded.atlas_data.shape == original.atlas_data.shape
    np.testing.assert_allclose(
        loaded.atlas_data, original.atlas_data, rtol=0.01, atol=0.01
    )
    assert loaded.image_xywh_list == original.image_xywh_list
    assert loaded.color_space == original.color_space
    assert loaded.readme_text is None
    assert loaded.license_text is None
    assert loaded.as_glyph_cache is None


def test_cooked_atlas_mono_channel():
    """Test round-trip save/load of a mono-channel atlas."""
    # Create a mono-channel test atlas
    atlas_data = np.random.rand(64, 64, 1).astype(np.float32)

    # Create image list
    image_xywh_list = [
        (0, 0, 16, 16),
        (16, 16, 16, 16),
    ]

    # Create atlas
    original = CookedAtlas(
        atlas_data=atlas_data,
        image_xywh_list=image_xywh_list,
        color_space="linear",
        readme_text="Mono atlas",
        license_text=None,
        as_glyph_cache=None,
    )

    # Save and load

    save_path = Path(
        "output/zfw/cook_test/test_cooked_atlas_roundtrip_mono_channel.zfw_atlas"
    )
    if save_path.exists():
        shutil.rmtree(save_path)

    original.save(save_path)

    # Load it back
    loaded = CookedAtlas.load(
        path=save_path,
        color_space="linear",
        load_readme_text=True,
        load_license_text=False,
        load_glyph_metrics=False,
    )

    # Verify shape is preserved
    assert loaded.atlas_data.shape == original.atlas_data.shape
    np.testing.assert_allclose(
        loaded.atlas_data, original.atlas_data, rtol=0.01, atol=0.01
    )
    assert loaded.image_xywh_list == original.image_xywh_list
    assert loaded.readme_text == original.readme_text
