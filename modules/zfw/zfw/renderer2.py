__all__ = [
    "Renderer",
    "RendererContext",
    "RendererImageInfo",
    "RendererImageKey",
]

from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from itertools import product

import numpy as np
from zfw.bundled_data import BUNDLED_DATA_PATH

from .basic import BaseResource, Font, expect, logger
from .gpu import (
    GpuBuffer,
    GpuCommandEncoder,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuImage,
    GpuBufferMeta,
    GpuImageMeta,
    GpuBufferImageCopyRegion,
    GpuDescriptorSet,
    GpuPipeline,
    GpuPipelineLayout,
    GpuSampler,
)
from . import typed_freetype as ft  # Must import before uharfbuzz
from . import typed_uharfbuzz as hb


#
# Renderer Context
#


class RendererContext(BaseResource):
    def __init__(self):
        super().__init__(parent_resource=None)

        self._glyph_image_dict = RendererContext._build_glyph_image_dict()

    @staticmethod
    def _build_glyph_image_dict() -> dict["RendererImageKey", "RendererImageInfo"]:
        glyph_atlas_builder = GlyphAtlasBuilder()

        res = {}

        chars = "".join(chr(i) for i in range(32, 127))

        for font_name, size_px, weight in product(
            glyph_atlas_builder.all_font_names,
            [18, 28],
            [100, 400, 700],
        ):
            glyph_images = glyph_atlas_builder.gen_glyph_images(
                font=font_name,
                size_px=size_px,
                weight=weight,
                chars=chars,
            )
            res.update(glyph_images)

        return res

    def _augment_image_dict(
        self,
        image_dict: dict["RendererImageKey", "RendererImageInfo"],
    ) -> dict["RendererImageKey", "RendererImageInfo"]:
        """
        Augments a user-supplied `image_dict` to include internal resources for the
        renderer, e.g. glyph images for fonts to support text rendering.
        """
        augmented_image_dict = image_dict.copy()
        augmented_image_dict |= self._glyph_image_dict
        return augmented_image_dict


#
# Renderer
#


class Renderer(BaseResource):
    """
    A renderer that can draw scenes using GPU resources.

    ## Resource Binding

    Bound resources are static (immutable) after creation. This allows the renderer to
    read from them without worrying about synchronization issues. If you want to use
    different resources, create a new renderer instance. Creating a renderer is not
    cheap, so design your application to avoid changing renderers frequently.

    Resource names are designed to be stable across different renderer instances. This
    allows you to reuse resource names when creating new renderers.

    Example model for streaming (e.g. an open-world 3D game):
    -   Maintain a global resource dictionary for images, geometries, and materials.
    -   Break the game world into chunks (e.g. 100m x 100m areas).
    -   For each chunk, maintain a list of resource names it needs.
    -   When the player moves, determine which chunks are now needed. You can have
        multiple "Renderer" instances ready to hot-swap as soon as the player views a
        new chunk set.
    """

    _context: "RendererContext"
    _gpu_device: GpuDevice

    _image_heap: "ImageHeap"

    _gpu_pipeline_layout: "GpuPipelineLayout"
    _gpu_pipeline: "GpuPipeline"

    def __init__(
        self,
        *,
        context: "RendererContext",
        gpu_device: GpuDevice,
        image_dict: dict["RendererImageKey", "RendererImageInfo"] | None = None,
        geometry_dict: dict["GeometryKey", "GeometryInfo"] | None = None,
        material_dict: dict["MaterialName", "MaterialInfo"] | None = None,
    ) -> None:
        super().__init__(parent_resource=context)
        self._context = context
        self._gpu_device = gpu_device

        # Build the image heap: add glyph images from the context to the provided image
        # dictionary
        self._image_heap = ImageHeap(
            parent_resource=self,
            gpu_device=gpu_device,
            input_dict=context._augment_image_dict(image_dict or {}),
        )

        # TODO: Implement geometry and material heaps.


class Canvas:
    pass


#
# Renderer Resources Common:
#


@dataclass(eq=True, frozen=True)
class BaseRendererResourceKey(ABC):
    pass


#
# Image Resources:
#


@dataclass(eq=True, frozen=True)
class RendererImageKey(BaseRendererResourceKey, ABC):
    pass


type RendererImageChannelCount = Literal[1, 4]


@dataclass
class RendererImageInfo:
    data: np.ndarray

    def __post_init__(self) -> None:
        assert self.data.ndim == 3
        assert self.data.shape[2] in (1, 4)
        assert self.data.dtype == np.float32

    @property
    def channel_count(self) -> RendererImageChannelCount:
        return self.data.shape[2]

    @property
    def area_px(self) -> int:
        h, w = self.data.shape[0:2]
        return h * w


class ImageHeap(BaseResource):
    # Constant parameters:
    page_size_px: int

    # Homogeneous sub-heaps: each sub-heap contains images with the same channel count.
    sub_heaps: dict[RendererImageChannelCount, "HomogeneousImageHeap"]

    # Samplers, used across sub-heaps:
    gpu_nearest_sampler: "GpuSampler"
    gpu_linear_sampler: "GpuSampler"

    # Single GPU descriptor set for accessing all sub-heaps' images (bindless style):
    gpu_descriptor_set_layout: "GpuDescriptorSetLayout"
    gpu_descriptor_set: "GpuDescriptorSet"

    def __init__(
        self,
        *,
        parent_resource: BaseResource,
        gpu_device: GpuDevice,
        input_dict: dict[RendererImageKey, "RendererImageInfo"],
        page_size_px: int = 4096,
    ):
        super().__init__(parent_resource=parent_resource)

        self.page_size_px = page_size_px

        self.sub_heaps = {
            1: HomogeneousImageHeap(
                parent_resource=self,
                gpu_device=gpu_device,
                channel_count=1,
                page_size_px=page_size_px,
                input_dict={
                    k: v  #
                    for k, v in input_dict.items()
                    if v.channel_count == 1
                },
            ),
            4: HomogeneousImageHeap(
                parent_resource=self,
                gpu_device=gpu_device,
                channel_count=4,
                page_size_px=page_size_px,
                input_dict={
                    k: v  #
                    for k, v in input_dict.items()
                    if v.channel_count == 4
                },
            ),
        }

        self.gpu_descriptor_set_layout = GpuDescriptorSetLayout(
            device=gpu_device,
            bindings=OrderedDict(
                {
                    "atlasMono": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        count=self.sub_heaps[1].page_count,
                    ),
                    "atlasRgba": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        count=self.sub_heaps[4].page_count,
                    ),
                    "rectsMono": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "rectsRgba": GpuDescriptorSetLayoutBinding(type="storage-buffer"),
                    "nearestSampler": GpuDescriptorSetLayoutBinding(type="sampler"),
                    "linearSampler": GpuDescriptorSetLayoutBinding(type="sampler"),
                }.items()
            ),
        )
        self.gpu_descriptor_set = GpuDescriptorSet(
            device=gpu_device,
            layout=self.gpu_descriptor_set_layout,
            bindings={
                "atlasMono": self.sub_heaps[1].gpu_page_images,
                "atlasRgba": self.sub_heaps[4].gpu_page_images,
                "rectsMono": self.sub_heaps[1].gpu_rect_buffer,
                "rectsRgba": self.sub_heaps[4].gpu_rect_buffer,
                "nearestSampler": self.gpu_nearest_sampler,
                "linearSampler": self.gpu_linear_sampler,
            },
        )

    def _on_dispose(self) -> None:
        self.gpu_descriptor_set.dispose()
        self.gpu_descriptor_set_layout.dispose()

        self.gpu_nearest_sampler.dispose()
        self.gpu_linear_sampler.dispose()

        for _, sub_heap in self.sub_heaps.items():
            sub_heap.dispose()


class HomogeneousImageHeap(BaseResource):
    # Constant parameters:
    channel_count: RendererImageChannelCount
    page_size_px: int

    # Book-keeping:
    rects: np.ndarray
    index: dict[RendererImageKey, int]
    page_count: int

    # GPU resources:
    gpu_rect_buffer: GpuBuffer
    gpu_page_images: list[GpuImage]

    def __init__(
        self,
        *,
        parent_resource: BaseResource | None,
        gpu_device: GpuDevice,
        channel_count: RendererImageChannelCount,
        page_size_px: int,
        input_dict: dict[RendererImageKey, "RendererImageInfo"],
    ):
        super().__init__(parent_resource=parent_resource)

        self.channel_count = channel_count
        self.page_size_px = page_size_px

        self.rects, self.index, self.page_count = HomogeneousImageHeap._alloc_uv_rects(
            input_dict=input_dict,
            page_size_px=page_size_px,
            channel_count=channel_count,
        )
        self.gpu_rect_buffer = HomogeneousImageHeap._emplace_rects_buffer(
            gpu_device=gpu_device,
            rects=self.rects,
        )
        self.gpu_page_images = HomogeneousImageHeap._emplace_pages_images(
            gpu_device=gpu_device,
            channel_count=channel_count,
            page_size_px=page_size_px,
            input_dict=input_dict,
            name_to_index_dict=self.index,
            rects=self.rects,
            page_count=self.page_count,
        )

    @staticmethod
    def _alloc_uv_rects(
        *,
        input_dict: dict["RendererImageKey", "RendererImageInfo"],
        page_size_px: int,
        channel_count: RendererImageChannelCount,
    ) -> tuple[np.ndarray, dict["RendererImageKey", int], int]:
        """
        Lays out images into fixed-size pages using a simple scanline packing algorithm.
        Returns:
        -   rects: np.ndarray of shape (N, 4) where N is the number of images, and each
            row is (x, y, w, h) in UV coordinates. The 'y' component's integral part
            indicates the page index, and the fractional part indicates the vertical
            position within that page in page UV space.
        -   name_to_index_dict: dict mapping image names to their corresponding index in
            the rects array.
        -   page_count: the number of pages used to store all images.
        """

        # Turn the input dict into a list of items i.e. key-value pairs.
        # Sort by descending height (tallest first)
        input_items = list(input_dict.items())
        input_items.sort(key=lambda item: item[1].data.shape[0], reverse=True)

        # Allocate rectangles per-item using a simple scanline packing algorithm:
        # - For a given row, keep inserting images until the row is full.
        # - When the row is full, move to the next row.
        # - When the page is full, move to the next page.
        # We sort by descending height to minimize wasted space.
        rects = np.zeros((len(input_items), 4), dtype=np.float32)  # x, y, w, h as UV
        name_to_index_dict = {}
        insert_cursor_z = 0  # page index
        insert_cursor_x = 0
        insert_cursor_y = 0
        insert_cursor_row_height = 0
        for image_index, (image_name, image_info) in enumerate(input_items):
            img_height, img_width = image_info.data.shape[0:2]

            # Check if the image is too large to fit in a page:
            if img_width > page_size_px or img_height > page_size_px:
                raise ValueError(
                    f"Image '{image_name}' is too large to fit in a page "
                    f"({img_width}x{img_height}px vs {page_size_px}px)"
                )

            # Check if the image fits in the current row. If not, move to next row:
            if insert_cursor_x + img_width > page_size_px:
                insert_cursor_x = 0
                insert_cursor_y += insert_cursor_row_height
                insert_cursor_row_height = 0

            # Check if the image fits in the current page. If not, move to next page:
            if insert_cursor_y + img_height > page_size_px:
                insert_cursor_z += 1
                insert_cursor_x = 0
                insert_cursor_y = 0
                insert_cursor_row_height = 0

            # Store this allocation.
            # NOTE: page index is stored in the 'y' component as the integral part.
            rects[image_index, :] = [
                insert_cursor_x / page_size_px,
                insert_cursor_y / page_size_px + insert_cursor_z,
                img_width / page_size_px,
                img_height / page_size_px,
            ]
            name_to_index_dict[image_name] = image_index

            # Update cursor:
            insert_cursor_x += img_width
            insert_cursor_row_height = max(insert_cursor_row_height, img_height)

        # Compute total page count:
        page_count = 1 + insert_cursor_z

        # Log metrics:
        image_count = len(input_items)
        alloc_area = page_count * (page_size_px**2)
        used_area = sum(image_info.area_px for _, image_info in input_items)
        utilization = used_area / alloc_area
        alloc_mib = (alloc_area * channel_count * 4) / (1024 * 1024)  # float32
        LOG.debug(
            "HomogeneousImageHeap: allocation complete: "
            f"{channel_count=}: "
            f"{page_count=}, {image_count=}, {utilization=:.6%}, {alloc_mib=:.2f}",
        )

        # Done:
        return rects, name_to_index_dict, page_count

    @staticmethod
    def _emplace_rects_buffer(*, gpu_device: GpuDevice, rects: np.ndarray) -> GpuBuffer:
        """
        Creates and populates a GPU buffer containing the given rects data.
        """
        # Create staging and device buffers:
        rects_buf_meta = GpuBufferMeta.from_array(rects)
        rects_device_buf = GpuBuffer(
            device=gpu_device,
            usages=["storage", "copy-dst"],
            meta=rects_buf_meta,
        )
        rects_staging_buf = GpuBuffer(
            device=gpu_device,
            usages=["staging", "copy-src"],
            meta=rects_buf_meta,
        )

        # Write to staging buffer:
        rects_staging_buf.write(data=rects)

        # Submit a one-time command buffer to copy from staging to device buffer.
        # NOTE: Wait for completion so we can dispose the staging buffer immediately.
        command_encoder = GpuCommandEncoder(device=gpu_device, queue_type="transfer")
        command_encoder.copy_buffer_to_buffer(
            src=rects_staging_buf,
            dst=rects_device_buf,
            size=rects.nbytes,
        )
        command_encoder.submit().wait()

        # Clean up:
        rects_staging_buf.dispose()
        command_encoder.dispose()

        # Done:
        return rects_device_buf

    @staticmethod
    def _emplace_pages_images(
        *,
        gpu_device: GpuDevice,
        channel_count: RendererImageChannelCount,
        page_size_px: int,
        input_dict: dict[RendererImageKey, "RendererImageInfo"],
        name_to_index_dict: dict[RendererImageKey, int],
        rects: np.ndarray,
        page_count: int,
    ) -> list[GpuImage]:
        """
        Creates and populates GPU images (pages) containing the input images at the
        allocated locations.
        """

        # Create images for each page:
        page_image_meta = GpuImageMeta(
            shape=(page_size_px, page_size_px, channel_count),
            dtype=np.float32,
            color_space="linear",
        )
        page_images = [
            GpuImage(
                device=gpu_device,
                usages=["texture-binding", "transfer-dst"],
                meta=page_image_meta,
            )
            for _ in range(page_count)
        ]

        # Create staging buffers for each page.
        # Memory-map them for CPU access:
        page_buffer_meta = page_image_meta.into_buffer_meta()
        page_staging_buffers = [
            GpuBuffer(
                device=gpu_device,
                usages=["staging", "copy-src"],
                meta=page_buffer_meta,
            )
        ]
        page_staging_buffers_mmap = [
            page_staging_buffer.memory.map()
            for page_staging_buffer in page_staging_buffers
        ]
        page_staging_buffers_data = [
            np.frombuffer(
                mm.__enter__(),
                dtype=page_image_meta.dtype,
            ).reshape(page_image_meta.shape)
            for mm in page_staging_buffers_mmap
        ]

        # Get a list of image names by index:
        name_list: list[RendererImageKey | None] = [None] * len(name_to_index_dict)
        for image_key, _ in input_dict.items():
            image_index = name_to_index_dict[image_key]
            name_list[image_index] = image_key

        # For each input image, copy its data into the appropriate page staging buffer:
        for image_index, image_key in enumerate(name_list):
            image_key = expect(image_key, "Image key should not be None")
            image_info = input_dict[image_key]
            image_data = image_info.data
            h, w = image_data.shape[0:2]

            # Get allocated rectangle:
            rect = rects[image_index, :]
            page_index = int(rect[1])  # integral part of 'uv_y' component
            page_u = rect[0]
            page_v = rect[1] - page_index  # fractional part of 'uv_y' component
            x = int(page_u * page_size_px)
            y = int(page_v * page_size_px)

            # Copy image data into the appropriate location in the staging buffer:
            page_data = page_staging_buffers_data[page_index]
            page_data[y : y + h, x : x + w, :] = image_data

        # Unmap staging buffers:
        for mm in page_staging_buffers_mmap:
            mm.__exit__(None, None, None)

        # Submit a one-time command buffer to copy from staging buffers to page images.
        # NOTE: Wait for completion so we can dispose the staging buffers immediately.
        command_encoder = GpuCommandEncoder(device=gpu_device, queue_type="transfer")
        for page_index, page_image in enumerate(page_images):
            command_encoder.copy_buffer_to_image(
                src=page_staging_buffers[page_index],
                dst=page_image,
                regions=[
                    GpuBufferImageCopyRegion(
                        buffer_offset=0,
                        image_offset=(0, 0, 0),
                        image_extent=(page_size_px, page_size_px, 1),
                    )
                ],
            )
        command_encoder.submit().wait()

        # Clean up:
        for page_staging_buffer in page_staging_buffers:
            page_staging_buffer.dispose()
        command_encoder.dispose()

        # Done:
        return page_images

    def _on_dispose(self) -> None:
        # Dispose GPU images per-page:
        for page_images in self.gpu_page_images:
            page_images.dispose()

        # Dispose rects buffer:
        self.gpu_rect_buffer.dispose()


#
# Font Image Resources:
#


@dataclass(eq=True, frozen=True)
class GlyphImageKey(RendererImageKey):
    font: Font
    glyph: int
    size_px: int
    weight: int


class GlyphAtlasBuilder:
    """
    Helper class for generating glyph images for fonts.

    Every `Renderer` instance's image heap includes glyph images generated by this
    class for ASCII characters at a handful of known sizes. This is sufficient for
    rendering basic UI.
    """

    all_font_names: list[Font]
    glyph_atlas_font_map: dict[Font, "GlyphAtlasFont"]

    def __init__(self, font_list: list[Font] | None = None):
        self.all_font_names = font_list or ["sans-serif", "serif", "monospaced"]
        self.glyph_atlas_font_map = {
            font: GlyphAtlasFont(font)  #
            for font in self.all_font_names
        }

    def gen_glyph_images(
        self,
        font: Font,
        size_px: int,
        weight: int,
        chars: str,
    ) -> dict[RendererImageKey, RendererImageInfo]:
        # Use HarfBuzz to get glyphs for each character, assuming laid out horizontally
        # on a single line with no complex shaping. This assumes the glyphs don't depend
        # on shaping. This is true for basic Latin characters.

        ga_font = self.glyph_atlas_font_map[font]

        hb_font = ga_font.hb_font
        hb_font.scale = (size_px * 64, size_px * 64)

        hb_buffer = hb.Buffer()
        hb_buffer.add_str(chars)
        hb_buffer.guess_segment_properties()

        hb.shape(hb_font, hb_buffer)

        glyph_infos = hb_buffer.glyph_infos

        # De-duplicate glyphs:
        glyph_infos_dict = {}
        for glyph_info in glyph_infos:
            if glyph_info.codepoint in glyph_infos_dict:
                continue
            glyph_infos_dict[glyph_info.codepoint] = glyph_info

        # Render each glyph using FreeType:
        glyph_images: dict[RendererImageKey, RendererImageInfo] = {}
        for glyph_index, glyph_info in glyph_infos_dict.items():
            # Get FreeType face:
            ft_face = self.glyph_atlas_font_map[font].ft_face
            ft_weight_axis_index = ga_font.ft_weight_axis_index

            # Set font size:
            ft_face.set_pixel_sizes(0, size_px)

            # Set font weight:
            coords = list(ga_font.ft_face.get_var_design_coords())
            coords[ft_weight_axis_index] = float(weight)
            ft_face.set_var_design_coords(coords)

            # Render:
            ft_face.load_glyph(
                glyph_index,
                ft.FT_LOAD_RENDER | ft.FT_LOAD_TARGET_NORMAL,
            )

            # Get bitmap:
            h, w = ft_face.glyph.bitmap.rows, ft_face.glyph.bitmap.width
            pitch = ft_face.glyph.bitmap.pitch
            buffer = ft_face.glyph.bitmap.buffer
            raw_bitmap = np.array(buffer, dtype=np.uint8).reshape((h, pitch))
            bitmap = np.ones((h, w, 4), dtype=np.float32)
            bitmap[..., 3] = raw_bitmap[:, :w] / 255.0

            # Create image key and info:
            image_key = GlyphImageKey(
                font=font,
                glyph=glyph_index,
                size_px=size_px,
                weight=weight,
            )
            image_info = RendererImageInfo(data=bitmap)

            # Store:
            glyph_images[image_key] = image_info

        # Done:
        return glyph_images


class GlyphAtlasFont:
    font: Font
    ft_face: ft.Face
    hb_font: hb.Font
    ft_weight_axis_index: int

    def __init__(self, font: Font):
        self.font = font
        self.ft_face = GlyphAtlasFont._new_freetype_font(font)
        self.hb_font = GlyphAtlasFont._new_harfbuzz_font(font)
        self.ft_weight_axis_index = GlyphAtlasFont._ft_wght_ax_idx(font, self.ft_face)

    @staticmethod
    def _get_font_file_path(font: Font) -> Path:
        return {
            "sans-serif": (BUNDLED_DATA_PATH / "data/font-Inter_4_1/InterVariable.ttf"),
            "serif": (BUNDLED_DATA_PATH / "data/font-Lora/Lora-VariableFont_wght.ttf"),
            "monospaced": (
                BUNDLED_DATA_PATH
                / "data/font-SourceCodePro/SourceCodePro-VariableFont_wght.ttf"
            ),
        }[font]

    @staticmethod
    def _new_harfbuzz_font(font: Font) -> hb.Font:
        file_path = GlyphAtlasFont._get_font_file_path(font)

        with open(file_path, "rb") as f:
            hb_blob = f.read()

        hb_face = hb.Face(hb_blob)
        hb_font = hb.Font(hb_face)
        return hb_font

    @staticmethod
    def _new_freetype_font(font: Font) -> ft.Face:
        file_path = GlyphAtlasFont._get_font_file_path(font)
        ft_face = ft.Face(str(file_path))
        return ft_face

    @staticmethod
    def _ft_wght_ax_idx(font: Font, ft_face: ft.Face) -> int:
        for i, axis in enumerate(ft_face.get_variation_info().axes):
            if axis.name.lower() == "wght":
                return i
        raise ValueError(f"Font {font!r} does not have a weight axis")


#
# Geometry Resources:
#


@dataclass(eq=True, frozen=True)
class GeometryKey(BaseRendererResourceKey, ABC):
    pass


@dataclass
class GeometryInfo:
    v_data: np.ndarray
    i_data: np.ndarray

    def __post_init__(self) -> None:
        assert self.v_data.ndim == 2
        assert self.v_data.dtype == VERTEX_DTYPE
        assert self.i_data.ndim == 1
        assert self.i_data.dtype == INDEX_DTYPE


VERTEX_DTYPE = np.dtype(
    [
        ("position", np.float32, 3),
        ("normal", np.float32, 3),
        ("texcoord", np.float32, 2),
    ]
)
INDEX_DTYPE = np.dtype(np.uint16)


#
# Material Resources:
#


type MaterialName = str


@dataclass
class MaterialInfo:
    color_map: RendererImageKey | None
    color_tint: tuple[float, float, float, float]
    normal_map: RendererImageKey | None
    normal_tint: tuple[float, float, float, float]
    metalness_map: RendererImageKey | None
    metalness_tint: float
    roughness_map: RendererImageKey | None
    roughness_tint: float


#
# Logger:
#

LOG = logger(__name__)
