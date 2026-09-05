use std::{ffi::OsStr, path::Path};

use resin_runtime::{ResinGpu, ResinMemory, ResinStatus};

use crate::{
    backend::{
        Error,
        glsl::{self, Stage},
    },
    ir::Module,
    toolchain,
};

pub enum Pipeline {
    Compute,
    Graphics,
}

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub fn render(module: &Module, pipeline: Pipeline, compiler: &OsStr) -> Result<Image, Error> {
    let shader = |stage: Stage| -> Result<Vec<u8>, Error> {
        toolchain::compile_glsl(&glsl::emit(module, stage.entry(), stage)?, stage, compiler)
    };
    let shaders = match pipeline {
        Pipeline::Compute => vec![shader(Stage::Compute)?],
        Pipeline::Graphics => vec![shader(Stage::Vertex)?, shader(Stage::Fragment)?],
    };
    let width = 256;
    let height = 256;
    let bytes = (width * height * 4) as usize;
    let mut gpu = ResinGpu::create().map_err(status)?;
    // SAFETY: generated shaders can only write the bounded output. All resources
    // belong to this GPU, outlive submission, and are read after its synchronous wait.
    let rgba = unsafe {
        match pipeline {
            Pipeline::Compute => {
                #[repr(C)]
                struct Root {
                    count: u32,
                    padding: u32,
                    pixels: u64,
                }
                let pipeline = gpu.create_compute_pipeline(&shaders[0]).map_err(status)?;
                let pixels = gpu.malloc(bytes, 4, ResinMemory::Default).map_err(status)?;
                let root = gpu
                    .malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default)
                    .map_err(status)?;
                root.host_pointer().cast::<Root>().write(Root {
                    count: width * height,
                    padding: 0,
                    pixels: pixels.device_pointer(),
                });
                let mut commands = gpu.start_command_recording().map_err(status)?;
                commands.set_pipeline(&pipeline).map_err(status)?;
                commands
                    .dispatch(root.device_pointer(), (width * height).div_ceil(64), 1, 1)
                    .map_err(status)?;
                gpu.submit(commands).map_err(status)?;
                pixels
                    .host_bytes()
                    .ok_or_else(|| Error("output is not host visible".into()))?
                    .to_vec()
            }
            Pipeline::Graphics => {
                let pipeline = gpu
                    .create_graphics_pipeline(&shaders[0], &shaders[1])
                    .map_err(status)?;
                let mut image = gpu.create_image(width, height).map_err(status)?;
                let pixels = gpu
                    .malloc(bytes, 4, ResinMemory::Readback)
                    .map_err(status)?;
                let mut commands = gpu.start_command_recording().map_err(status)?;
                commands
                    .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
                    .map_err(status)?;
                commands.set_pipeline(&pipeline).map_err(status)?;
                commands.draw(0, 3).map_err(status)?;
                commands.end_rendering().map_err(status)?;
                commands
                    .copy_image_to_buffer(&mut image, &pixels)
                    .map_err(status)?;
                gpu.submit(commands).map_err(status)?;
                pixels
                    .host_bytes()
                    .ok_or_else(|| Error("output is not host visible".into()))?
                    .to_vec()
            }
        }
    };
    Ok(Image {
        width,
        height,
        rgba,
    })
}

pub fn write_png(image: &Image, output: &Path) -> Result<(), Error> {
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temp = toolchain::TempDir::new(parent).map_err(|e| Error(e.to_string()))?;
    let path = temp.path().join("image.png");
    resin_runtime::image_write_png(&path, image.width, image.height, 4, &image.rgba)
        .map_err(status)?;
    std::fs::rename(path, output).map_err(|e| Error(e.to_string()))
}

fn status(status: ResinStatus) -> Error {
    Error(format!("GPU runtime: {status:?}"))
}
