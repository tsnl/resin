import wgpu


#
# Device request
#


def get_required_wgpu_features() -> list[str]:
    """Get the list of required wgpu features for zfw."""
    return [
        "shader-f16",
        "texture-compression-bc",
    ]


def get_required_wgpu_limits() -> dict[str, int | None]:
    """
    Get the dict of required wgpu limits for zfw.
    """
    return {
        "maxBufferSize": 1 << 30,  # 1 GiB
        "maxStorageBufferBindingSize": 1 << 30,  # 1 GiB
    }


def help_request_wgpu_device(adapter: wgpu.GPUAdapter) -> wgpu.GPUDevice:
    """
    Convenience function to request a wgpu device with zfw-required features and
    limits.
    """

    return adapter.request_device_sync(
        label="ZfwDevice",
        required_features=get_required_wgpu_features(),
        required_limits=get_required_wgpu_limits(),
    )


#
# BlitRenderer
#


class BlitRenderer:
    """
    A renderer that blits (copies) a source texture to a destination texture using
    a full-screen triangle pass. This is useful for inter-format copies that cannot
    be done with copy_texture_to_texture.
    """

    device: wgpu.GPUDevice
    bind_group_layout: wgpu.GPUBindGroupLayout
    pipeline_layout: wgpu.GPUPipelineLayout
    sampler: wgpu.GPUSampler
    shader_module: wgpu.GPUShaderModule
    pipelines: dict[wgpu.TextureFormat, wgpu.GPURenderPipeline]

    def __init__(self, device: wgpu.GPUDevice) -> None:
        self.device = device

        self.bind_group_layout = device.create_bind_group_layout(
            label="BlitRenderer.BindGroupLayout",
            entries=[
                wgpu.BindGroupLayoutEntry(
                    binding=0,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    sampler=wgpu.SamplerBindingLayout(
                        type=wgpu.SamplerBindingType.non_filtering,
                    ),
                ),
                wgpu.BindGroupLayoutEntry(
                    binding=1,
                    visibility=wgpu.ShaderStage.FRAGMENT,
                    texture=wgpu.TextureBindingLayout(
                        sample_type=wgpu.TextureSampleType.unfilterable_float,
                        view_dimension=wgpu.TextureViewDimension.d2,
                        multisampled=False,
                    ),
                ),
            ],
        )

        self.pipeline_layout = device.create_pipeline_layout(
            label="BlitRenderer.PipelineLayout",
            bind_group_layouts=[self.bind_group_layout],
        )

        self.sampler = device.create_sampler(
            label="BlitRenderer.Sampler",
            address_mode_u=wgpu.AddressMode.clamp_to_edge,
            address_mode_v=wgpu.AddressMode.clamp_to_edge,
            address_mode_w=wgpu.AddressMode.clamp_to_edge,
            mag_filter=wgpu.FilterMode.nearest,
            min_filter=wgpu.FilterMode.nearest,
            mipmap_filter=wgpu.MipmapFilterMode.nearest,
        )

        with open(__file__.replace(".py", ".wgsl"), "r") as f:
            shader_source = f.read()

        self.shader_module = device.create_shader_module(code=shader_source)

        # Cache pipelines per output format
        self.pipelines = {}

    def _get_pipeline(
        self, output_format: wgpu.TextureFormat
    ) -> wgpu.GPURenderPipeline:
        """Get or create a pipeline for the given output format."""
        if output_format not in self.pipelines:
            self.pipelines[output_format] = self.device.create_render_pipeline(
                label=f"BlitRenderer.Pipeline.{output_format}",
                layout=self.pipeline_layout,
                vertex=wgpu.VertexState(
                    module=self.shader_module,
                    entry_point="vs_blit",
                ),
                fragment=wgpu.FragmentState(
                    module=self.shader_module,
                    entry_point="fs_blit",
                    targets=[
                        wgpu.ColorTargetState(
                            format=str(output_format),
                            write_mask=wgpu.ColorWrite.ALL,
                        )
                    ],
                ),
                primitive=wgpu.PrimitiveState(
                    topology=wgpu.PrimitiveTopology.triangle_list,
                ),
            )
        return self.pipelines[output_format]

    def record(
        self,
        input_texture: wgpu.GPUTexture,
        output_texture: wgpu.GPUTexture,
        command_encoder: wgpu.GPUCommandEncoder,
    ) -> None:
        """
        Record commands to blit the input texture to the output texture.

        Args:
            input_texture: The source texture to read from.
            output_texture: The destination texture to write to.
            command_encoder: The command encoder to record commands into.
        """
        bind_group = self.device.create_bind_group(
            label="BlitRenderer.BindGroup",
            layout=self.bind_group_layout,
            entries=[
                wgpu.BindGroupEntry(
                    binding=0,
                    resource=self.sampler,
                ),
                wgpu.BindGroupEntry(
                    binding=1,
                    resource=input_texture.create_view(),
                ),
            ],
        )

        pipeline = self._get_pipeline(
            wgpu.TextureFormat[output_texture.format.replace("-", "_")]
        )

        render_pass = command_encoder.begin_render_pass(
            label="BlitRenderer.RenderPass",
            color_attachments=[
                wgpu.RenderPassColorAttachment(
                    view=output_texture.create_view(),
                    resolve_target=None,
                    load_op="clear",
                    store_op="store",
                    clear_value=(0.0, 0.0, 0.0, 0.0),
                )
            ],
        )

        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group, [], 0, 0)
        render_pass.draw(3, 1, 0, 0)  # 3 vertices for full-screen triangle

        render_pass.end()
