//! Texture resources contain image data.

use crate::{
    res::generic::{GenericBackend, GenericManager, GenericResource},
    util::RectAllocator,
};

//
// API:
//

pub type TextureManager = GenericManager<TextureBackend>;
pub type Texture = GenericResource<TextureBackend>;

pub struct TextureManagerArgs {
    page_width: usize,
    page_height: usize,
    page_count: usize,
    format: TextureFormat,
}

pub struct TextureArgs<'a> {
    data: Option<&'a [u8]>,
    bytes_per_row: usize,
    rows_per_image: usize,
    texture_format: TextureFormat,
}

pub struct TextureInfo {}

pub enum TextureFormat {
    Rgba8Unorm,
    Rgba32Float,
    Bc1RgbaUnorm,
    Bc4RUnorm,
    Bc5RgUnorm,
}

#[derive(thiserror::Error, Debug)]
pub enum TextureAllocationError {
    #[error("Failed to allocate required texture")]
    NoFreeSpace(),
}

//
// Implementation:
//

struct TextureBackend {
    rect_allocator: RectAllocator,
}

impl GenericBackend for TextureBackend {
    type ManagerCreateArgs = TextureManagerArgs;
    type ResourceCreateArgs<'a> = TextureArgs<'a>;
    type ResourceCreateError = TextureAllocationError;
    type ResourceInfo = TextureInfo;

    fn new<'a>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        create_info: Self::ManagerCreateArgs,
    ) -> Self {
        todo!()
    }

    fn add_impl<'a>(
        &mut self,
        create_info: Self::ResourceCreateArgs<'a>,
    ) -> Result<Self::ResourceInfo, Self::ResourceCreateError> {
        todo!()
    }

    fn del_impl(&mut self, resource: &Self::ResourceInfo) {
        todo!()
    }
}

impl Default for TextureManagerArgs {
    fn default() -> Self {
        Self {
            page_width: 0,
            page_height: 0,
            page_count: 0,
            format: TextureFormat::Rgba8Unorm,
        }
    }
}
