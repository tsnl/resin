__all__ = [
    "CookedAtlas",
]

from abc import ABC
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import orjson
import pydantic
from pydantic import Field
import PIL.Image

from .excepts import LogicError
from .basic import ColorSpace, Font, FontSize, FontWeight, JsonObject
from .loader import load_image


#
# CookedAtlas:
#


CookedAtlasType = Literal["glyph_cache"]


@dataclass(kw_only=True, frozen=True)
class CookedAtlas:
    PATH_SUFFIX: str = ".zfw_atlas"

    atlas_type: CookedAtlasType = "glyph_cache"
    atlas_data: np.ndarray  # (h, w, 4) RGBA f32 image or (h, w, 1) mono f32 image
    image_xywh_list: list[tuple[int, int, int, int]]
    color_space: ColorSpace = "linear"
    readme_text: str | None = None
    license_text: str | None = None
    as_glyph_cache: dict[CookedAtlasGlyphCacheKey, "CookedAtlasGlyphInfo"] | None = None

    @staticmethod
    def load(
        *,
        path: Path,
        color_space: ColorSpace = "linear",
        load_readme_text: bool = False,
        load_license_text: bool = False,
        load_glyph_metrics: bool = True,
    ) -> "CookedAtlas":
        if path.suffix != CookedAtlas.PATH_SUFFIX:
            raise ValueError(
                f"CookedAtlas load path must have suffix {CookedAtlas.PATH_SUFFIX!r}: "
                f"{path=}"
            )
        if not path.is_dir():
            raise FileNotFoundError(f"Cooked atlas path not found: {path}")

        # Load index.json
        with open(path / "index.json", "rb") as f:
            index = CookedAtlasIndexFile(**orjson.loads(f.read()))
        image_xywh_list = index.image_xywh_list

        # Load atlas.png
        atlas_path = path / "atlas.png"
        atlas_data = load_image(
            file_path=atlas_path,
            expected_channel_count=index.channel_count,
            input_color_space=index.color_space,
            output_color_space=color_space,
        )

        # (Optional) Load README.md
        if load_readme_text:
            with open(path / "README.md", "r", encoding="utf-8") as f:
                readme_text = f.read()
        else:
            readme_text = None

        # (Optional) Load LICENSE.txt
        if load_license_text:
            with open(path / "LICENSE.txt", "r", encoding="utf-8") as f:
                license_text = f.read()
        else:
            license_text = None

        # (Optional) Load glyph metrics
        if load_glyph_metrics:
            with open(path / "glyph_metrics.json", "rb") as f:
                as_glyph_cache = {
                    key: val
                    for key, val in CookedAtlasGlyphCacheExtFile(
                        **orjson.loads(f.read())
                    ).data
                }
        else:
            as_glyph_cache = None

        return CookedAtlas(
            atlas_data=atlas_data,
            image_xywh_list=image_xywh_list,
            color_space=color_space,
            readme_text=readme_text,
            license_text=license_text,
            as_glyph_cache=as_glyph_cache,
        )

    def save(self, path: Path) -> None:
        if path.suffix != CookedAtlas.PATH_SUFFIX:
            raise ValueError(
                f"CookedAtlas save path must have {CookedAtlas.PATH_SUFFIX!r} suffix: "
                f"{path=}"
            )

        path.mkdir(parents=True, exist_ok=True)

        # Save index.json
        with open(path / "index.json", "wb") as f:
            index = CookedAtlasIndexFile(
                cooked_atlas_type=self.atlas_type,
                image_xywh_list=self.image_xywh_list,
                channel_count=self.atlas_data.shape[2],
                color_space=self.color_space,
            )
            f.write(orjson.dumps(index.model_dump()))

        # Save atlas.png
        pil_atlas = (np.clip(self.atlas_data, 0.0, 1.0) * 255).astype(np.uint8)
        pil_atlas = pil_atlas.squeeze()
        pil_mode = "RGBA" if self.atlas_data.shape[2] == 4 else "L"
        pil_image = PIL.Image.fromarray(pil_atlas, mode=pil_mode)
        pil_image.save(path / "atlas.png", format="PNG")

        # (Optional) Save README.md
        if self.readme_text:
            with open(path / "README.md", "w", encoding="utf-8") as f:
                f.write(self.readme_text)

        # (Optional) Save LICENSE.txt
        if self.license_text:
            with open(path / "LICENSE.txt", "w", encoding="utf-8") as f:
                f.write(self.license_text)

        # (Optional) Save glyph cache extension data:
        if self.as_glyph_cache:
            if self.atlas_type != "glyph_cache":
                raise LogicError(
                    "Inconsistent CookedAtlas instance: if 'as_glyph_cache' is set, "
                    "'atlas_type' must be 'glyph_cache'."
                )
            with open(path / "glyph_metrics.json", "wb") as f:
                glyph_metrics_file = CookedAtlasGlyphCacheExtFile(
                    data=list(self.as_glyph_cache.items())
                )
                f.write(orjson.dumps(glyph_metrics_file.model_dump()))


class CookedAtlasIndexFile(pydantic.BaseModel):
    cooked_atlas_type: CookedAtlasType
    image_xywh_list: list[tuple[int, int, int, int]]
    channel_count: Literal[1, 4]
    color_space: ColorSpace


#
# CookedAtlas: GlyphCacheExt
#


class CookedAtlasGlyphCacheExtFile(pydantic.BaseModel):
    data: list[tuple[CookedAtlasGlyphCacheKey, "CookedAtlasGlyphInfo"]]


@dataclass(frozen=True, kw_only=True)
class CookedAtlasGlyphCacheKey:
    glyph_index: int
    font_name: Font
    font_size: FontSize
    font_weight: FontWeight
    scale: Fraction


class CookedAtlasGlyphInfo(pydantic.BaseModel):
    image_id: int
    ascender_26_6: int
    descender_26_6: int
    height_26_6: int
    bitmap_left: int
    bitmap_top: int
