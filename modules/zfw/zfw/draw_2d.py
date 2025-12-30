__all__ = [
    "Draw2dContext",
    "Draw2dRenderer",
]

from collections import OrderedDict
from pathlib import Path
from typing import Literal

import numpy as np

from .basic import (
    BaseResource,
    Font,
    HorizontalAlignment,
    VerticalAlignment,
    logger,
    round_up_to_po2,
)
from .bundled_data import BUNDLED_DATA_PATH
from .gpu import (
    GpuBuffer,
    GpuBufferImageCopyRegion,
    GpuBufferMeta,
    GpuCommandEncoder,
    GpuContext,
    GpuDescriptorSet,
    GpuDescriptorSetLayout,
    GpuDescriptorSetLayoutBinding,
    GpuDevice,
    GpuEzBuffer,
    GpuImage,
    GpuImageMeta,
    GpuPipeline,
    GpuPipelineLayout,
    GpuRenderPassCommandEncoder,
    GpuSampler,
    GpuShader,
)
from . import typed_freetype as ft  # Must import before uharfbuzz
from . import typed_uharfbuzz as hb


LOG = logger(__name__)


##--------------------------------------------------------------------------------------
## Context
##--------------------------------------------------------------------------------------


class Draw2dContext(BaseResource):
    """
    Context for 2D drawing operations. This is the top-level resource that owns all
    shared state for 2D rendering.
    """

    gpu_context: GpuContext

    def __init__(self, *, gpu_context: GpuContext):
        super().__init__(parent_resource=None)
        self.gpu_context = gpu_context


##--------------------------------------------------------------------------------------
## Draw2dRenderer
##--------------------------------------------------------------------------------------


class Draw2dRenderer(BaseResource):
    """
    A GPU-accelerated 2D renderer that draws textured quads.

    Unlike the bindless renderer, this uses separate batches per texture with
    dynamic glyph atlas for text rendering.

    All coordinates are in PHYSICAL pixels unless otherwise noted.
    The `scale` property can be used to configure HiDPI/LoDPI scaling.
    """

    context: Draw2dContext
    gpu_device: GpuDevice
    scale: float

    # Pipelines:
    _pipeline_layout: GpuPipelineLayout
    _descriptor_set_layout: GpuDescriptorSetLayout
    _text_vertex_shader: GpuShader
    _text_fragment_shader: GpuShader
    _rgba_vertex_shader: GpuShader
    _rgba_fragment_shader: GpuShader
    _cached_text_pipeline: GpuPipeline | None
    _cached_rgba_pipeline: GpuPipeline | None

    # Sampler:
    _nearest_sampler: GpuSampler
    _linear_sampler: GpuSampler

    # Quad batches:
    _text_quad_batches: list["QuadBatch"]
    _rgba_quad_batches: dict[int, "QuadBatch"]
    _quad_count: int

    # Glyph atlas:
    _glyph_atlas: "GlyphAtlas"

    # Font support:
    _all_fonts: list[Font]
    _hb_font_map: dict[Font, hb.Font]
    _ft_face_map: dict[Font, ft.Face]
    _ft_weight_axis_index: dict[Font, int]

    # Default 1x1 white image for solid color quads:
    _default_white_image: GpuImage
    _default_white_batch: "QuadBatch | None"

    def __init__(
        self,
        *,
        context: Draw2dContext,
        device: GpuDevice,
        scale: float = 1.0,
    ):
        super().__init__(parent_resource=context)

        self.context = context
        self.gpu_device = device
        self.scale = scale

        # Create samplers:
        self._nearest_sampler = GpuSampler(
            device=self.gpu_device,
            min_filter="nearest",
            mag_filter="nearest",
        )
        self._linear_sampler = GpuSampler(
            device=self.gpu_device,
            min_filter="linear",
            mag_filter="linear",
        )

        # Create descriptor set layout (same for both text and rgba):
        self._descriptor_set_layout = GpuDescriptorSetLayout(
            device=self.gpu_device,
            bindings=OrderedDict(
                {
                    "uniform": GpuDescriptorSetLayoutBinding(
                        type="uniform-buffer",
                        stages=["vertex", "fragment"],
                    ),
                    "quads": GpuDescriptorSetLayoutBinding(
                        type="storage-buffer",
                        stages=["vertex", "fragment"],
                    ),
                    "atlasTexture": GpuDescriptorSetLayoutBinding(
                        type="sampled-image",
                        stages=["fragment"],
                    ),
                    "atlasSampler": GpuDescriptorSetLayoutBinding(
                        type="sampler",
                        stages=["fragment"],
                    ),
                }.items()
            ),
        )

        # Create pipeline layout:
        self._pipeline_layout = GpuPipelineLayout(
            device=self.gpu_device,
            descriptor_set_layouts=[self._descriptor_set_layout],
        )

        # Load shaders:
        self._text_vertex_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_2d_quad_text.vert.spv",
            stage="vertex",
        )
        self._text_fragment_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_2d_quad_text.frag.spv",
            stage="fragment",
        )
        self._rgba_vertex_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_2d_quad_rgba.vert.spv",
            stage="vertex",
        )
        self._rgba_fragment_shader = GpuShader(
            device=self.gpu_device,
            spirv_path=BUNDLED_DATA_PATH / "shaders/draw_2d_quad_rgba.frag.spv",
            stage="fragment",
        )

        # Pipeline caches:
        self._cached_text_pipeline = None
        self._cached_rgba_pipeline = None
        self._cached_depth_image = None

        # Initialize quad batches:
        self._text_quad_batches = []
        self._rgba_quad_batches = {}
        self._quad_count = 0

        # Create glyph atlas:
        self._glyph_atlas = GlyphAtlas(renderer=self)

        # Initialize font support:
        self._all_fonts = ["sans-serif", "serif", "monospaced"]
        self._hb_font_map = {
            font: self._load_harfbuzz_font(font) for font in self._all_fonts
        }
        self._ft_face_map = {
            font: self._load_freetype_font(font) for font in self._all_fonts
        }
        self._ft_weight_axis_index = {}
        for font in self._all_fonts:
            face = self._ft_face_map[font]
            info = face.get_variation_info()
            for i, axis in enumerate(info.axes):
                if axis.tag == "wght":
                    self._ft_weight_axis_index[font] = i
                    break

        # Create default white image:
        white_data = np.ones((1, 1, 4), dtype="<f4")
        self._default_white_image = GpuImage(
            device=self.gpu_device,
            usages=["texture-binding", "transfer-dst"],
            meta=GpuImageMeta(
                shape=(1, 1, 4),
                dtype="<f4",
                color_space="linear",
            ),
        )
        # Upload the white pixel:
        self._upload_image_data(self._default_white_image, white_data)
        self._default_white_batch = None

    def _on_dispose(self) -> None:
        # Dispose glyph atlas:
        self._glyph_atlas.dispose()

        # Dispose batches:
        for batch in self._text_quad_batches:
            batch.dispose()
        for batch in self._rgba_quad_batches.values():
            batch.dispose()
        if self._default_white_batch is not None:
            self._default_white_batch.dispose()

        # Dispose default white image:
        self._default_white_image.dispose()

        # Dispose pipelines:
        if self._cached_text_pipeline is not None:
            self._cached_text_pipeline.dispose()
        if self._cached_rgba_pipeline is not None:
            self._cached_rgba_pipeline.dispose()
        if self._cached_depth_image is not None:
            self._cached_depth_image.dispose()

        # Dispose shaders:
        self._text_vertex_shader.dispose()
        self._text_fragment_shader.dispose()
        self._rgba_vertex_shader.dispose()
        self._rgba_fragment_shader.dispose()

        # Dispose pipeline layout:
        self._pipeline_layout.dispose()

        # Dispose descriptor set layout:
        self._descriptor_set_layout.dispose()

        # Dispose samplers:
        self._nearest_sampler.dispose()
        self._linear_sampler.dispose()

    def _upload_image_data(self, gpu_image: GpuImage, data: np.ndarray) -> None:
        """
        Upload image data to GPU using a synchronous transfer.
        """

        staging_buf = GpuBuffer(
            device=self.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(
                element_count=data.size,
                element_dtype=data.dtype,
            ),
        )
        staging_buf.memory.write(data=data)

        encoder = GpuCommandEncoder(device=self.gpu_device, queue_type="transfer")
        encoder.transition_image_layout(image=gpu_image, layout="transfer-dst-optimal")
        encoder.copy_buffer_to_image(
            src=staging_buf,
            dst=gpu_image,
            regions=[
                GpuBufferImageCopyRegion(
                    buffer_offset=0,
                    image_offset=(0, 0, 0),
                    image_extent=(data.shape[1], data.shape[0], 1),
                )
            ],
        )
        encoder.submit().wait()
        staging_buf.dispose()

    ##----------------------------------------------------------------------------------
    ## clear()
    ##----------------------------------------------------------------------------------

    def clear(self) -> None:
        self._quad_count = 0

        for batch in self._text_quad_batches:
            batch.clear()

        for batch in self._rgba_quad_batches.values():
            batch.clear()

    ##----------------------------------------------------------------------------------
    ## add_quad()
    ##----------------------------------------------------------------------------------

    def add_quad(
        self,
        *,
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int] | None = None,
        src_xy: tuple[int, int] = (0, 0),
        src_wh: tuple[int, int] | None = None,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        border_color: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        border_thickness: tuple[int, int, int, int] = (0, 0, 0, 0),  # TRBL
        image: GpuImage | None = None,
        sampler: "SamplerType" = "nearest",
        _dip: bool = True,
    ) -> None:
        """
        Add a single quad using logical (device-independent) pixel coordinates.

        Coordinates are converted to physical pixels internally based on the
        renderer's scale factor if `_dip=True`.
        """

        # Determine the image to use:
        gpu_image = image if image is not None else self._default_white_image

        # Resolve sizes:
        if src_wh is None:
            src_wh = (gpu_image.width, gpu_image.height)
        if dst_wh is None:
            dst_wh = src_wh

        # Convert to physical pixels if needed:
        if _dip:
            dst_xy_phys = (
                int(dst_xy[0] * self.scale),
                int(dst_xy[1] * self.scale),
            )
            dst_wh_phys = (
                int(dst_wh[0] * self.scale),
                int(dst_wh[1] * self.scale),
            )
            border_thickness_phys = (
                int(border_thickness[0] * self.scale),
                int(border_thickness[1] * self.scale),
                int(border_thickness[2] * self.scale),
                int(border_thickness[3] * self.scale),
            )
        else:
            dst_xy_phys = dst_xy
            dst_wh_phys = dst_wh
            border_thickness_phys = border_thickness

        # Compute UV coordinates:
        src_uv = (
            src_xy[0] / gpu_image.width,
            src_xy[1] / gpu_image.height,
            src_wh[0] / gpu_image.width,
            src_wh[1] / gpu_image.height,
        )

        # Get or create batch:
        batch = self._get_rgba_quad_batch(gpu_image, sampler)

        # Add quad to batch:
        batch.add_quad(
            dst_xywh_px=(
                dst_xy_phys[0],
                dst_xy_phys[1],
                dst_wh_phys[0],
                dst_wh_phys[1],
            ),
            src_xywh_uv=src_uv,
            depth=float(self._quad_count),
            border_thickness=border_thickness_phys,
            color=color,
            border_color=border_color,
        )
        self._quad_count += 1

    def _get_rgba_quad_batch(
        self,
        image: GpuImage,
        sampler: "SamplerType",
    ) -> "QuadBatch":
        image_id = image.get_unique_image_id()

        if batch := self._rgba_quad_batches.get(image_id):
            return batch

        batch = QuadBatch(
            renderer=self,
            batch_type="rgba",
            gpu_image=image,
            sampler=self._nearest_sampler
            if sampler == "nearest"
            else self._linear_sampler,
        )
        self._rgba_quad_batches[image_id] = batch
        return batch

    ##----------------------------------------------------------------------------------
    ## add_text()
    ##----------------------------------------------------------------------------------

    def add_text(
        self,
        *,
        text: str,
        font: Font,
        dst_xy: tuple[int, int],
        dst_wh: tuple[int, int],
        font_size_px: int = 16,
        color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        wrap: bool = True,
        font_weight: int = 400,
        horizontal_alignment: HorizontalAlignment = "left",
        vertical_alignment: VerticalAlignment = "top",
    ) -> None:
        """
        Add quads for rendering the given text string with the given font.

        Supports horizontal and vertical alignment within the destination rectangle.
        """

        if not text:
            return

        # Strategy: Accumulate pen position in 26.6 fixed-point to preserve
        # subpixel precision. Only round to integer pixels when placing glyphs.

        renderer_scale = self.scale
        effective_size_px = int(font_size_px * renderer_scale)

        ft_face = self._ft_face_map[font]
        ft_face.set_pixel_sizes(0, effective_size_px)
        self._set_freetype_weight(font, font_weight)
        metrics = ft_face.size

        # FreeType metrics are in 26.6 fixed point:
        ascender_26_6 = metrics.ascender
        height_26_6 = metrics.height

        infos, positions = self._shape_text(font, text, font_size_px, font_weight)

        dst_x, dst_y = dst_xy
        dst_w, dst_h = dst_wh

        # Convert destination rect to physical 26.6 fixed-point:
        dst_x_26_6 = int(dst_x * renderer_scale * 64)
        dst_y_26_6 = int(dst_y * renderer_scale * 64)
        dst_w_26_6 = int(dst_w * renderer_scale * 64)
        dst_h_26_6 = int(dst_h * renderer_scale * 64)

        # Physical pixel versions for clipping:
        dst_x_phys = int(dst_x * renderer_scale)
        dst_y_phys = int(dst_y * renderer_scale)
        dst_w_phys = int(dst_w * renderer_scale)
        dst_h_phys = int(dst_h * renderer_scale)

        lines = self._layout_text_lines(infos, positions, text, wrap, dst_w_26_6)

        total_text_height_26_6 = len(lines) * height_26_6

        # Vertical alignment:
        start_y_26_6 = dst_y_26_6
        if vertical_alignment == "middle":
            start_y_26_6 += (dst_h_26_6 - total_text_height_26_6) // 2
        elif vertical_alignment == "bottom":
            start_y_26_6 += dst_h_26_6 - total_text_height_26_6

        pen_y_26_6 = start_y_26_6 + ascender_26_6

        # Get the text batch for this atlas state:
        text_batch = self._get_text_quad_batch()

        for start_idx, end_idx, line_width_26_6 in lines:
            # Horizontal alignment:
            pen_x_26_6 = dst_x_26_6

            min_ink, max_ink = self._get_line_optical_bounds(
                font, infos, positions, start_idx, end_idx
            )
            optical_width = max_ink - min_ink

            if horizontal_alignment == "center":
                pen_x_26_6 += (dst_w_26_6 - optical_width) // 2 - min_ink
            elif horizontal_alignment == "right":
                pen_x_26_6 += dst_w_26_6 - max_ink
            elif horizontal_alignment == "left":
                pen_x_26_6 -= min_ink
            else:
                raise NotImplementedError(f"{horizontal_alignment=}")

            for i in range(start_idx, end_idx):
                info = infos[i]
                pos = positions[i]

                codepoint = info.codepoint

                # HarfBuzz positions are in 26.6 fixed point:
                x_advance_26_6 = pos.x_advance
                y_advance_26_6 = pos.y_advance
                x_offset_26_6 = pos.x_offset
                y_offset_26_6 = pos.y_offset

                glyph_entry = self._glyph_atlas.get_glyph(
                    font, codepoint, font_size_px, font_weight
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

                    # Intersection with dst rect:
                    ix_phys = max(qx_phys, dst_x_phys)
                    iy_phys = max(qy_phys, dst_y_phys)
                    ir_phys = min(qx_phys + qw_phys, dst_x_phys + dst_w_phys)
                    ib_phys = min(qy_phys + qh_phys, dst_y_phys + dst_h_phys)

                    if ir_phys > ix_phys and ib_phys > iy_phys:
                        # Clipping offset:
                        src_off_x = ix_phys - qx_phys
                        src_off_y = iy_phys - qy_phys
                        src_w = ir_phys - ix_phys
                        src_h = ib_phys - iy_phys

                        # Ensure we don't exceed the glyph bounds:
                        src_w = min(src_w, qw_phys - src_off_x)
                        src_h = min(src_h, qh_phys - src_off_y)

                        # Compute UV coordinates within the atlas:
                        atlas_x, atlas_y, atlas_w, atlas_h = glyph_entry.uv_xywh
                        src_uv = (
                            atlas_x + (src_off_x / self._glyph_atlas.page_size),
                            atlas_y + (src_off_y / self._glyph_atlas.page_size),
                            src_w / self._glyph_atlas.page_size,
                            src_h / self._glyph_atlas.page_size,
                        )

                        # Add quad:
                        text_batch.add_quad(
                            dst_xywh_px=(ix_phys, iy_phys, src_w, src_h),
                            src_xywh_uv=src_uv,
                            depth=float(self._quad_count),
                            border_thickness=(0, 0, 0, 0),
                            color=color,
                            border_color=(0.0, 0.0, 0.0, 0.0),
                        )
                        self._quad_count += 1

                # Accumulate pen position:
                pen_x_26_6 += x_advance_26_6
                pen_y_26_6 += y_advance_26_6

            pen_y_26_6 += height_26_6

    def _get_text_quad_batch(self) -> "QuadBatch":
        """Get or create a text quad batch for the current glyph atlas state."""
        # For simplicity, we use a single batch for all text.
        # If the atlas is rebuilt, we'd need to invalidate batches.
        if not self._text_quad_batches:
            batch = QuadBatch(
                renderer=self,
                batch_type="text",
                gpu_image=self._glyph_atlas.gpu_image,
                sampler=self._nearest_sampler,
            )
            self._text_quad_batches.append(batch)
        return self._text_quad_batches[0]

    ##----------------------------------------------------------------------------------
    ## Font helpers
    ##----------------------------------------------------------------------------------

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
    def _load_harfbuzz_font(font: Font) -> hb.Font:
        file_path = Draw2dRenderer._get_font_file_path(font)
        with open(file_path, "rb") as f:
            hb_blob = f.read()
        hb_face = hb.Face(hb_blob)
        return hb.Font(hb_face)

    @staticmethod
    def _load_freetype_font(font: Font) -> ft.Face:
        file_path = Draw2dRenderer._get_font_file_path(font)
        return ft.Face(str(file_path))

    def _set_freetype_weight(self, font: Font, weight: int) -> None:
        if font not in self._ft_weight_axis_index:
            return

        face = self._ft_face_map[font]
        axis_idx = self._ft_weight_axis_index[font]
        coords = list(face.get_var_design_coords())
        coords[axis_idx] = float(weight)
        face.set_var_design_coords(coords)

    def _shape_text(
        self,
        font: Font,
        text: str,
        font_size_px: int,
        font_weight: int,
    ) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition]]:
        hb_font = self._hb_font_map[font]
        renderer_scale = self.scale
        effective_size_px = int(font_size_px * renderer_scale)
        scale = effective_size_px * 64  # HarfBuzz uses 26.6 fixed point
        hb_font.scale = (scale, scale)
        hb_font.set_variations({"wght": font_weight})

        hb_buffer = hb.Buffer()
        hb_buffer.add_str(text)
        hb_buffer.guess_segment_properties()

        hb.shape(hb_font, hb_buffer)

        return hb_buffer.glyph_infos, hb_buffer.glyph_positions

    def _layout_text_lines(
        self,
        infos: list[hb.GlyphInfo],
        positions: list[hb.GlyphPosition],
        text: str,
        wrap: bool,
        max_width_26_6: int,
    ) -> list[tuple[int, int, int]]:
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
        font: Font,
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
        hb_font = self._hb_font_map[font]
        min_x = 2147483647
        max_x = -2147483648

        pen_x = 0
        has_ink = False

        for i in range(start_idx, end_idx):
            info = infos[i]
            pos = positions[i]

            extents = hb_font.get_glyph_extents(info.codepoint)

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

    ##----------------------------------------------------------------------------------
    ## draw()
    ##----------------------------------------------------------------------------------

    def draw(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        target: GpuImage,
        clear_color: Literal["black", "transparent"] | None = "black",
    ) -> None:
        """Render all queued quads to the given target image.
        
        Args:
            command_encoder: The command encoder to use.
            target: The target image to render to.
            clear_color: The color to clear the target to before rendering.
                If None, the target is not cleared (renders on top of existing content).
        """
        if self._quad_count == 0:
            return

        # Flush glyph atlas:
        self._glyph_atlas.flush(command_encoder=command_encoder)

        # Get or create pipelines:
        text_pipeline = self._get_text_pipeline(target)
        rgba_pipeline = self._get_rgba_pipeline(target)

        # Flush all batches (copy staging -> device buffers) BEFORE render pass:
        for batch in self._rgba_quad_batches.values():
            if batch.quad_count > 0:
                batch.flush(
                    command_encoder=command_encoder,
                    target=target,
                    total_quad_count=self._quad_count,
                )
        for batch in self._text_quad_batches:
            if batch.quad_count > 0:
                batch.flush(
                    command_encoder=command_encoder,
                    target=target,
                    total_quad_count=self._quad_count,
                )

        # Ensure target is in correct layout:
        command_encoder.transition_image_layout(
            image=target,
            layout="color-attachment-optimal",
        )

        # Begin render pass:
        with command_encoder.render(
            color_attachment=target,
            depth_attachment=None,
            clear_color=clear_color,
        ) as render_pass:
            # Draw RGBA batches:
            render_pass.bind_pipeline(pipeline=rgba_pipeline)
            for batch in self._rgba_quad_batches.values():
                if batch.quad_count > 0:
                    batch.draw(render_pass=render_pass)

            # Draw text batches:
            render_pass.bind_pipeline(pipeline=text_pipeline)
            for batch in self._text_quad_batches:
                if batch.quad_count > 0:
                    batch.draw(render_pass=render_pass)

    def _get_text_pipeline(self, target: GpuImage) -> GpuPipeline:
        if self._cached_text_pipeline is not None:
            if (
                self._cached_text_pipeline.vk_color_format == target._vk_format
                and self._cached_text_pipeline.viewport_width == target.width
                and self._cached_text_pipeline.viewport_height == target.height
            ):
                return self._cached_text_pipeline
            self._cached_text_pipeline.dispose()

        self._cached_text_pipeline = GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._text_vertex_shader,
            fragment_shader=self._text_fragment_shader,
            vk_color_format=target._vk_format,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._pipeline_layout,
        )
        return self._cached_text_pipeline

    def _get_rgba_pipeline(self, target: GpuImage) -> GpuPipeline:
        if self._cached_rgba_pipeline is not None:
            if (
                self._cached_rgba_pipeline.vk_color_format == target._vk_format
                and self._cached_rgba_pipeline.viewport_width == target.width
                and self._cached_rgba_pipeline.viewport_height == target.height
            ):
                return self._cached_rgba_pipeline
            self._cached_rgba_pipeline.dispose()

        self._cached_rgba_pipeline = GpuPipeline(
            device=self.gpu_device,
            vertex_shader=self._rgba_vertex_shader,
            fragment_shader=self._rgba_fragment_shader,
            vk_color_format=target._vk_format,
            enable_depth_test=False,
            enable_alpha_blending=True,
            viewport_width=target.width,
            viewport_height=target.height,
            layout=self._pipeline_layout,
        )
        return self._cached_rgba_pipeline


##--------------------------------------------------------------------------------------
## Quad Dtype
##--------------------------------------------------------------------------------------

QUAD_DTYPE = np.dtype(
    [
        ("dst_xywh_px", "<i4", 4),
        ("src_xywh_uv", "<f4", 4),
        ("depth", "<f4"),
        ("border_thickness_t", "<i4"),
        ("border_thickness_r", "<i4"),
        ("border_thickness_b", "<i4"),
        ("border_thickness_l", "<i4"),
        ("_pad0", "<i4"),
        ("_pad1", "<i4"),
        ("_pad2", "<i4"),
        ("color", "<f4", 4),
        ("border_color", "<f4", 4),
    ]
)

UNIFORM_DTYPE = np.dtype(
    [
        ("framebuffer_size_px", "<i4", 2),
        ("quad_count", np.uint32),
        ("_rsv0", np.uint32),
    ]
)


##--------------------------------------------------------------------------------------
## Sampler Type
##--------------------------------------------------------------------------------------

type SamplerType = Literal["nearest", "linear"]

type QuadBatchType = Literal["rgba", "text"]


##--------------------------------------------------------------------------------------
## Quad Batch
##--------------------------------------------------------------------------------------


class QuadBatch(BaseResource):
    """
    Batch of quads for rendering with a single texture.

    Can be used for either RGBA textures or text (glyph atlas).
    The batch_type determines which pipeline to use during rendering.
    """

    renderer: Draw2dRenderer
    batch_type: QuadBatchType
    gpu_image: GpuImage
    sampler: GpuSampler

    _uniform_staging: GpuBuffer
    _uniform_device: GpuBuffer
    _quad_staging: GpuEzBuffer
    _quad_device: GpuEzBuffer
    _descriptor_set: GpuDescriptorSet | None
    _quad_count: int

    def __init__(
        self,
        *,
        renderer: Draw2dRenderer,
        batch_type: QuadBatchType,
        gpu_image: GpuImage,
        sampler: GpuSampler,
    ):
        super().__init__(parent_resource=renderer)

        self.renderer = renderer
        self.batch_type = batch_type
        self.gpu_image = gpu_image
        self.sampler = sampler
        self._quad_count = 0

        # Create uniform buffers:
        self._uniform_staging = GpuBuffer(
            device=renderer.gpu_device,
            usages=["staging", "copy-src"],
            meta=GpuBufferMeta(element_count=1, element_dtype=UNIFORM_DTYPE),
        )
        self._uniform_device = GpuBuffer(
            device=renderer.gpu_device,
            usages=["uniform", "copy-dst"],
            meta=GpuBufferMeta(element_count=1, element_dtype=UNIFORM_DTYPE),
        )

        # Create quad buffers:
        self._quad_staging = GpuEzBuffer(
            device=renderer.gpu_device,
            dtype=QUAD_DTYPE,
            capacity=64,
            usages=["staging", "copy-src"],
        )
        self._quad_device = GpuEzBuffer(
            device=renderer.gpu_device,
            dtype=QUAD_DTYPE,
            capacity=64,
            usages=["storage", "copy-dst"],
        )

        # Descriptor set created lazily (image may not be ready at init for text batches):
        self._descriptor_set = None

    def _on_dispose(self) -> None:
        if self._descriptor_set is not None:
            self._descriptor_set.dispose()
        self._quad_device.dispose()
        self._quad_staging.dispose()
        self._uniform_device.dispose()
        self._uniform_staging.dispose()

    @property
    def quad_count(self) -> int:
        return self._quad_count

    def clear(self) -> None:
        self._quad_count = 0
        self._quad_staging.clear()

    def add_quad(
        self,
        *,
        dst_xywh_px: tuple[int, int, int, int],
        src_xywh_uv: tuple[float, float, float, float],
        depth: float,
        border_thickness: tuple[int, int, int, int],
        color: tuple[float, float, float, float],
        border_color: tuple[float, float, float, float],
    ) -> None:
        self._quad_staging.extend(
            values=np.array(
                [
                    (
                        dst_xywh_px,
                        src_xywh_uv,
                        depth,
                        border_thickness[0],  # T
                        border_thickness[1],  # R
                        border_thickness[2],  # B
                        border_thickness[3],  # L
                        0,  # _pad0
                        0,  # _pad1
                        0,  # _pad2
                        color,
                        border_color,
                    )
                ],
                dtype=QUAD_DTYPE,
            )
        )
        self._quad_count += 1

    def flush(
        self,
        *,
        command_encoder: GpuCommandEncoder,
        target: GpuImage,
        total_quad_count: int,
    ) -> None:
        """Flush staging buffers to device buffers (before render pass)."""
        if self._quad_count == 0:
            return

        # Ensure device buffer capacity:
        need_recreate_descriptor = False
        if self._quad_device.capacity < self._quad_count:
            self._quad_device.dispose()
            self._quad_device = GpuEzBuffer(
                device=self.renderer.gpu_device,
                dtype=QUAD_DTYPE,
                capacity=round_up_to_po2(self._quad_count),
                usages=["storage", "copy-dst"],
            )
            need_recreate_descriptor = True

        # Recreate descriptor set if needed:
        if need_recreate_descriptor or self._descriptor_set is None:
            if self._descriptor_set is not None:
                self._descriptor_set.dispose()

            self._descriptor_set = GpuDescriptorSet(
                device=self.renderer.gpu_device,
                layout=self.renderer._descriptor_set_layout,
                bindings={
                    "uniform": self._uniform_device,
                    "quads": self._quad_device.device_buffer,
                    "atlasTexture": self.gpu_image,
                    "atlasSampler": self.sampler,
                },
            )

        # Update uniform:
        uniform_data = np.array(
            [((target.width, target.height), total_quad_count, 0)],
            dtype=UNIFORM_DTYPE,
        )
        self._uniform_staging.memory.write(data=uniform_data)
        command_encoder.copy_buffer_to_buffer(
            src=self._uniform_staging,
            dst=self._uniform_device,
            size=uniform_data.nbytes,
        )

        # Copy quad data from staging EzBuffer to device EzBuffer:
        self._quad_device._data[: self._quad_count] = self._quad_staging.array
        self._quad_device._length = self._quad_count
        self._quad_device.flush(command_encoder=command_encoder)

        # Transition texture:
        command_encoder.transition_image_layout(
            image=self.gpu_image,
            layout="texture-binding",
        )

    def draw(
        self,
        *,
        render_pass: GpuRenderPassCommandEncoder,
    ) -> None:
        """Draw the batch (inside render pass, after flush)."""

        assert self._descriptor_set is not None

        if self._quad_count == 0:
            return

        render_pass.bind_descriptor_set(set_index=0, set_=self._descriptor_set)
        render_pass.draw(vertex_count=6, instance_count=self._quad_count)


##--------------------------------------------------------------------------------------
## Glyph Atlas
##--------------------------------------------------------------------------------------


class GlyphEntry:
    """Entry for a single glyph in the atlas."""

    uv_xywh: tuple[float, float, float, float]
    width: int
    height: int
    bitmap_left: int
    bitmap_top: int

    def __init__(
        self,
        *,
        uv_xywh: tuple[float, float, float, float],
        width: int,
        height: int,
        bitmap_left: int,
        bitmap_top: int,
    ):
        self.uv_xywh = uv_xywh
        self.width = width
        self.height = height
        self.bitmap_left = bitmap_left
        self.bitmap_top = bitmap_top


class GlyphAtlas(BaseResource):
    """
    Dynamic texture atlas for glyph bitmaps.

    Uses simple row-based packing. The atlas is a single monochrome texture.
    """

    renderer: Draw2dRenderer
    page_size: int
    gpu_image: GpuImage

    _pixel_data: np.ndarray
    _glyph_cache: dict[tuple[Font, int, int, int], GlyphEntry | None]
    _cursor_x: int
    _cursor_y: int
    _row_height: int
    _dirty: bool
    _staging_buffer: GpuBuffer | None

    def __init__(
        self,
        *,
        renderer: Draw2dRenderer,
        page_size: int = 2048,
    ):
        super().__init__(parent_resource=renderer)

        self.renderer = renderer
        self.page_size = page_size

        # Create GPU image (monochrome):
        self.gpu_image = GpuImage(
            device=renderer.gpu_device,
            usages=["texture-binding", "transfer-dst"],
            meta=GpuImageMeta(
                shape=(page_size, page_size, 1),
                dtype="<f4",
                color_space="linear",
            ),
        )

        # CPU-side pixel data:
        self._pixel_data = np.zeros((page_size, page_size, 1), dtype="<f4")

        # Glyph cache: (font, glyph_index, size, weight) -> GlyphEntry
        self._glyph_cache = {}

        # Packing state:
        self._cursor_x = 0
        self._cursor_y = 0
        self._row_height = 0
        self._dirty = False

        # Staging buffer:
        self._staging_buffer = None

    def _on_dispose(self) -> None:
        if self._staging_buffer is not None:
            self._staging_buffer.dispose()
        self.gpu_image.dispose()

    def get_glyph(
        self,
        font: Font,
        glyph_index: int,
        font_size_px: int,
        font_weight: int,
    ) -> GlyphEntry | None:
        """
        Get a glyph entry, rasterizing and adding to atlas if needed.
        """

        renderer_scale = self.renderer.scale
        effective_size_px = int(font_size_px * renderer_scale)
        cache_key = (font, glyph_index, effective_size_px, font_weight)

        if cache_key in self._glyph_cache:
            return self._glyph_cache[cache_key]

        # Rasterize the glyph:
        ft_face = self.renderer._ft_face_map[font]
        ft_face.set_pixel_sizes(0, effective_size_px)
        self.renderer._set_freetype_weight(font, font_weight)
        ft_face.load_glyph(glyph_index, ft.FT_LOAD_RENDER | ft.FT_LOAD_TARGET_NORMAL)

        bitmap_left = ft_face.glyph.bitmap_left
        bitmap_top = ft_face.glyph.bitmap_top
        bitmap = ft_face.glyph.bitmap

        if not bitmap.buffer or bitmap.width == 0 or bitmap.rows == 0:
            self._glyph_cache[cache_key] = None
            return None

        h, w = bitmap.rows, bitmap.width
        pitch = bitmap.pitch

        # Load buffer:
        buffer_array = np.array(bitmap.buffer, dtype=np.uint8).reshape(h, pitch)
        if pitch != w:
            buffer_array = buffer_array[:, :w]

        # Normalize to float:
        glyph_data = buffer_array.astype("<f4") / 255.0

        # Allocate in atlas:
        entry = self._allocate_glyph(
            width=w,
            height=h,
            bitmap_left=bitmap_left,
            bitmap_top=bitmap_top,
            data=glyph_data,
        )

        self._glyph_cache[cache_key] = entry
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
        # Try to fit on current row:
        if self._cursor_x + width > self.page_size:
            # Move to next row:
            self._cursor_x = 0
            self._cursor_y += self._row_height
            self._row_height = 0

        # Check if we have space:
        if self._cursor_y + height > self.page_size:
            LOG.warning("Glyph atlas full, cannot allocate more glyphs")
            return None

        # Allocate:
        x = self._cursor_x
        y = self._cursor_y

        # Copy data:
        self._pixel_data[y : y + height, x : x + width, 0] = data

        # Update cursor:
        self._cursor_x += width
        self._row_height = max(self._row_height, height)
        self._dirty = True

        # Create entry:
        uv_xywh = (
            x / self.page_size,
            y / self.page_size,
            width / self.page_size,
            height / self.page_size,
        )

        return GlyphEntry(
            uv_xywh=uv_xywh,
            width=width,
            height=height,
            bitmap_left=bitmap_left,
            bitmap_top=bitmap_top,
        )

    def flush(self, *, command_encoder: GpuCommandEncoder) -> None:
        """Upload dirty atlas data to GPU."""
        if not self._dirty:
            return

        # Create staging buffer if needed:
        if self._staging_buffer is None:
            self._staging_buffer = GpuBuffer(
                device=self.renderer.gpu_device,
                usages=["staging", "copy-src"],
                meta=GpuBufferMeta(
                    element_count=self.page_size * self.page_size,
                    element_dtype="<f4",
                ),
            )

        # Write data to staging:
        self._staging_buffer.memory.write(data=self._pixel_data)

        # Copy to GPU:
        command_encoder.transition_image_layout(
            image=self.gpu_image,
            layout="transfer-dst-optimal",
        )
        command_encoder.copy_buffer_to_image(
            src=self._staging_buffer,
            dst=self.gpu_image,
            regions=[
                GpuBufferImageCopyRegion(
                    buffer_offset=0,
                    image_offset=(0, 0, 0),
                    image_extent=(self.page_size, self.page_size, 1),
                )
            ],
        )

        self._dirty = False


##--------------------------------------------------------------------------------------
## Helpers
##--------------------------------------------------------------------------------------
