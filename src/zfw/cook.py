__all__ = [
    "CookedAtlas",
]

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import orjson
import pydantic
import PIL.Image

from .basic import ColorSpace
from .loader import load_image


#
# CookedAtlas:
#


@dataclass(kw_only=True, frozen=True)
class CookedAtlas:
    atlas_data: np.ndarray  # (h, w, 4) RGBA f32 image or (h, w, 1) mono f32 image
    image_xywh: dict[str, tuple[int, int, int, int]]
    color_space: ColorSpace
    readme_text: str | None
    license_text: str | None

    @staticmethod
    def load(
        *,
        path: Path,
        color_space: ColorSpace = "linear",
        load_readme_text: bool = False,
        load_license_text: bool = False,
    ) -> "CookedAtlas":
        if not path.is_dir():
            raise FileNotFoundError(f"Cooked atlas path not found: {path}")

        # Load index.json
        with open(path / "index.json", "rb") as f:
            index = CookedAtlasIndexFile(**orjson.loads(f.read()))

        # Load README.md
        if load_readme_text:
            with open(path / "README.md", "r", encoding="utf-8") as f:
                readme_text = f.read()
        else:
            readme_text = None

        # Load LICENSE.txt
        if load_license_text:
            with open(path / "LICENSE.txt", "r", encoding="utf-8") as f:
                license_text = f.read()
        else:
            license_text = None

        # Load atlas.png
        atlas_path = path / "atlas.png"
        atlas_data = load_image(
            file_path=atlas_path,
            expected_channel_count=index.channel_count,
            input_color_space=index.color_space,
            output_color_space=color_space,
        )

        return CookedAtlas(
            atlas_data=atlas_data,
            image_xywh=index.images,
            color_space=color_space,
            readme_text=readme_text,
            license_text=license_text,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

        # Save atlas.png
        with open(path / "index.json", "wb") as f:
            index = CookedAtlasIndexFile(
                images=self.image_xywh,
                channel_count=self.atlas_data.shape[2],
                color_space=self.color_space,
            )
            f.write(orjson.dumps(index.model_dump()))

        # Save README.md
        if self.readme_text:
            with open(path / "README.md", "w", encoding="utf-8") as f:
                f.write(self.readme_text)

        # Save LICENSE.txt
        if self.license_text:
            with open(path / "LICENSE.txt", "w", encoding="utf-8") as f:
                f.write(self.license_text)

        # Save atlas.png
        PIL.Image.open(path / "atlas.png").save(path / "atlas.png", format="PNG")


class CookedAtlasIndexFile(pydantic.BaseModel):
    images: dict[str, tuple[int, int, int, int]]
    channel_count: Literal[1, 4]
    color_space: ColorSpace
