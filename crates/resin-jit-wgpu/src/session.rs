//! Backend sessions: device context + buffer handle types.
//!
//! [`Session`] supplies the device, queue, and buffer I/O primitives. Each
//! [`crate::pipeline::Pipeline`] instance owns a full buffer pool (params,
//! intermediates, sinks) allocated in [`crate::pipeline::PipelineFactory::create`].
//!
//! App code uses compile-time [`resin_core::Tree`] shapes end-to-end — e.g.
//! `TrainStepIn { minibatch, model }` in, `TrainStepOut` out via
//! [`crate::pipeline::Pipeline::call_trees`]. Dotted string names exist only inside
//! the compiler's IR tables, not in user-facing APIs.

use std::sync::mpsc;

use wgpu::util::DeviceExt;

use crate::pipeline::PipelineError;

/// Device-resident storage handle for one admitted param/sink leaf.
pub trait BufferHandle: Clone + Send + Sync + 'static {
    fn byte_len(&self) -> u64;
}

/// Sub-range of a parent [`WgpuBuffer`] (byte offset + length).
#[derive(Clone)]
pub struct BufferView {
    pub buffer: WgpuBuffer,
    pub offset: u64,
    pub byte_len: u64,
}

impl BufferView {
    pub fn new(buffer: WgpuBuffer, offset: u64, byte_len: u64) -> Self {
        Self {
            buffer,
            offset,
            byte_len,
        }
    }

    pub fn whole(buffer: WgpuBuffer) -> Self {
        let byte_len = buffer.size();
        Self {
            buffer,
            offset: 0,
            byte_len,
        }
    }
}

impl BufferHandle for BufferView {
    fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Leaf type for GPU [`resin_core::Tree`] I/O: whole buffer or a view into one.
#[derive(Clone)]
pub enum GpuLeaf {
    Buffer(WgpuBuffer),
    View(BufferView),
}

impl BufferHandle for GpuLeaf {
    fn byte_len(&self) -> u64 {
        match self {
            Self::Buffer(b) => b.byte_len(),
            Self::View(v) => v.byte_len(),
        }
    }
}

impl From<WgpuBuffer> for GpuLeaf {
    fn from(buffer: WgpuBuffer) -> Self {
        Self::Buffer(buffer)
    }
}

impl From<BufferView> for GpuLeaf {
    fn from(view: BufferView) -> Self {
        Self::View(view)
    }
}

/// GPU execution context and leaf buffer type.
///
/// [`WgpuSession`] is the wgpu implementation (`Buffer = wgpu::Buffer`). Future
/// backends (Vulkan-native, WebGPU in browser) implement the same trait with their
/// own buffer handle.
pub trait Session: Clone + Send + Sync + 'static {
    type Buffer: BufferHandle;

    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), PipelineError>;

    fn read_buffer(&self, buffer: &Self::Buffer) -> Result<Vec<u8>, PipelineError>;

    fn alloc_buffer(&self, byte_len: u64) -> Result<Self::Buffer, PipelineError>;

    fn alloc_buffer_init(&self, data: &[u8]) -> Result<Self::Buffer, PipelineError>;

    /// Read the first F32 from a buffer (e.g. a rank-0 loss leaf).
    fn read_buffer_f32(&self, buffer: &Self::Buffer) -> Result<f32, PipelineError> {
        let bytes = self.read_buffer(buffer)?;
        let word: [u8; 4] = bytes
            .get(..4)
            .ok_or_else(|| PipelineError::Program("buffer shorter than 4 bytes".into()))?
            .try_into()
            .map_err(|_| PipelineError::Program("buffer shorter than 4 bytes".into()))?;
        Ok(f32::from_le_bytes(word))
    }
}

impl BufferHandle for wgpu::Buffer {
    fn byte_len(&self) -> u64 {
        self.size()
    }
}

/// wgpu device + queue. Open once, then pass to [`crate::pipeline::compile`].
#[derive(Clone)]
pub struct WgpuSession {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Session leaf buffer type for the wgpu backend.
pub type WgpuBuffer = <WgpuSession as Session>::Buffer;

impl WgpuSession {
    pub fn open(config: &crate::pipeline::DeviceConfig) -> Result<Self, PipelineError> {
        let (device, queue) = match &config.device_name {
            None => request_default_device()?,
            Some(name) => request_named_device(name)?,
        };
        Ok(Self { device, queue })
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Back-compat alias for [`Self::open`].
    pub fn from_config(config: &crate::pipeline::DeviceConfig) -> Result<Self, PipelineError> {
        Self::open(config)
    }
}

impl Session for WgpuSession {
    type Buffer = wgpu::Buffer;

    fn write_buffer(
        &self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), PipelineError> {
        if offset + data.len() as u64 > buffer.size() {
            return Err(PipelineError::Program(format!(
                "write_buffer: {} bytes at offset {offset} exceeds buffer size {}",
                data.len(),
                buffer.size()
            )));
        }
        self.queue.write_buffer(buffer, offset, data);
        Ok(())
    }

    fn read_buffer(&self, buffer: &Self::Buffer) -> Result<Vec<u8>, PipelineError> {
        let size = buffer.size();
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resin-read-staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resin-read"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging.slice(..);
        let (sender, receiver) = mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| PipelineError::BufferMapFailed)?
            .map_err(|_| PipelineError::BufferMapFailed)?;
        let data = {
            let mapped = buffer_slice.get_mapped_range();
            mapped.to_vec()
        };
        staging.unmap();
        Ok(data)
    }

    fn alloc_buffer(&self, byte_len: u64) -> Result<Self::Buffer, PipelineError> {
        let size = byte_len.max(4);
        Ok(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resin-buffer"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    fn alloc_buffer_init(&self, data: &[u8]) -> Result<Self::Buffer, PipelineError> {
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resin-buffer"),
                contents: data,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            }))
    }
}

fn request_default_device() -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter_options = wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    };
    let adapter = pollster::block_on(instance.request_adapter(&adapter_options))
        .or_else(|| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                force_fallback_adapter: true,
                ..adapter_options
            }))
        })
        .ok_or_else(|| PipelineError::Backend("no suitable GPU adapter found".into()))?;
    request_device_from_adapter(adapter)
}

fn request_named_device(device_name: &str) -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .find(|adapter| adapter.get_info().name == device_name)
        .ok_or_else(|| {
            PipelineError::Config(format!(
                "no GPU adapter found with device_name '{device_name}'"
            ))
        })?;
    request_device_from_adapter(adapter)
}

fn request_device_from_adapter(
    adapter: wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), PipelineError> {
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("resin"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))?;
    Ok((device, queue))
}
