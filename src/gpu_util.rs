
//
// Buffer wrappers: StorageBuffer, UniformBuffer, VertexBuffer, IndexBuffer, StagingBuffer, ReadbackBuffer
//

pub trait BufferWrapper {
    fn wgpu_buffer(&self) -> &wgpu::Buffer;
    fn len(&self) -> usize;

    fn copy_to_buffer<T: BufferWrapper>(
        &self,
        dst: &T,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        command_encoder.copy_buffer_to_buffer(
            self.wgpu_buffer(),
            0,
            dst.wgpu_buffer(),
            0,
            (self.len() * std::mem::size_of::<u8>()) as wgpu::BufferAddress,
        );
    }
    fn copy_to_texture<T: TextureWrapper>(
        &self,
        dst: &T,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        command_encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: self.wgpu_buffer(),
                layout: dst.texel_copy_buffer_layout(),
            },
            dst.texel_copy_texture_info(),
            dst.extent_3d(),
        );
    }
    fn binding_size() -> Option<wgpu::BufferSize> {
        Some(wgpu::BufferSize::new(std::mem::size_of::<u8>() as u64).unwrap())
    }
}

macro_rules! buffer_wrapper {
    ($type_vis:vis struct $name:ident { $usage:expr }) => {
        $type_vis struct $name <T: bytemuck::Pod> {
            device: wgpu::Device,
            buffer: wgpu::Buffer,
            marker: std::marker::PhantomData<T>,
            count: usize,
        }
        impl<T: bytemuck::Pod> BufferWrapper for $name<T> {
            fn wgpu_buffer(&self) -> &wgpu::Buffer {
                &self.buffer
            }
            fn len(&self) -> usize {
                self.count
            }
        }
        impl<T: bytemuck::Pod> $name<T> {
            pub fn new(device: &wgpu::Device, count: usize, label: &str) -> Self {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: (count * std::mem::size_of::<T>()) as wgpu::BufferAddress,
                    usage: $usage,
                    mapped_at_creation: false,
                });
                Self {
                    device: device.clone(),
                    buffer,
                    marker: std::marker::PhantomData,
                    count,
                }
            }
        }
    }
}
buffer_wrapper!(pub struct StorageBuffer { wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
buffer_wrapper!(pub struct UniformBuffer { wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
buffer_wrapper!(pub struct VertexBuffer { wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
buffer_wrapper!(pub struct IndexBuffer { wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST });
buffer_wrapper!(pub struct StagingBuffer { wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC });
buffer_wrapper!(pub struct ReadbackBuffer { wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST });

impl<T: bytemuck::Pod + Send> StagingBuffer<T> {
    pub fn map_sync(&self, map_mode: wgpu::MapMode) {
        let buffer_slice = self.buffer.slice(..);
        let (tx, rx) = oneshot::channel();
        buffer_slice.map_async(map_mode, move |result| {
            if let Err(e) = result {
                panic!("Failed to map staging buffer for write: {:?}", e);
            }
            tx.send(()).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap();
    }
    pub fn write(&self, data: &[T]) {
        assert_eq!(data.len(), self.len());
        self.map_sync(wgpu::MapMode::Write);
        let mut mapped_range = self.buffer.get_mapped_range_mut(..);
        mapped_range.copy_from_slice(bytemuck::cast_slice(data));
        self.buffer.unmap();
    }
    pub fn read(&self) -> Box<[T]> {
        self.map_sync(wgpu::MapMode::Read);
        let mapped_range = self.buffer.get_mapped_range(..);
        let data = bytemuck::cast_slice(&mapped_range)
            .to_vec()
            .into_boxed_slice();
        self.buffer.unmap();
        data
    }
}

//
// Texture wrappers:
//

pub trait TextureWrapper {
    fn wgpu_texture(&self) -> &wgpu::Texture;

    fn extent_3d(&self) -> wgpu::Extent3d {
        self.wgpu_texture().size()
    }
    fn width(&self) -> u16 {
        self.wgpu_texture().width() as u16
    }
    fn height(&self) -> u16 {
        self.wgpu_texture().height() as u16
    }
    fn format(&self) -> wgpu::TextureFormat {
        self.wgpu_texture().format()
    }
    fn texel_copy_buffer_layout(&self) -> wgpu::TexelCopyBufferLayout {
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some({
                // TODO: When we support BC formats, this needs to be adjusted to account for block sizes.
                self.width() as u32 * self.format().target_pixel_byte_cost().unwrap()
            }),
            rows_per_image: None,
        }
    }
    fn texel_copy_texture_info<'a>(&'a self) -> wgpu::TexelCopyTextureInfo<'a> {
        wgpu::TexelCopyTextureInfo {
            texture: self.wgpu_texture(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }
}

macro_rules! texture2d_wrapper {
    ($type_vis:vis struct $name:ident { $format:expr }) => {
        $type_vis struct $name {
            device: wgpu::Device,
            texture: wgpu::Texture,
        }
        impl $name {
            pub fn new(device: &wgpu::Device, size_wh: [u16; 2], label: &str) -> Self {
                let device = device.clone();
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d {
                        width: size_wh[0] as u32,
                        height: size_wh[1] as u32,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: $format,
                    usage: wgpu::TextureUsages::COPY_SRC
                        | wgpu::TextureUsages::COPY_DST
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::STORAGE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                Self { device, texture }
            }
        }
        impl TextureWrapper for $name {
            fn wgpu_texture(&self) -> &wgpu::Texture {
                &self.texture
            }
        }
    };
}

texture2d_wrapper!(pub struct Rgba32FloatTexture { wgpu::TextureFormat::Rgba32Float });
texture2d_wrapper!(pub struct Rgba8UnormTexture { wgpu::TextureFormat::Rgba8Unorm });

//
// Fixed-point encoding:
//

pub fn fp32_to_fx_u16(value: f32) -> u16 {
    let normalized = value.clamp(0.0, 1.0);
    (normalized * 65535.0).round() as u16
}

pub fn fp32_to_fx_i16(value: f32) -> i16 {
    // Rescale from [-1.0, 1.0] to [0.0, 1.0] and encode as unsigned.
    let value = (value.clamp(-1.0, 1.0) + 1.0) / 2.0;
    fp32_to_fx_u16(value) as i16
}
