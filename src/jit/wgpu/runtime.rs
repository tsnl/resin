//! WebGPU runtime: arenas for the invoke, one bind group, dispatch, sinks.
//!
//! Pipelines and the bind-group layout live on the [`super::WgpuProgram`]
//! artifact. Arena GPU buffers are created each invoke. Sinks become compact
//! [`super::WgpuArray`] storage buffers (GPU copy from the arena — no map).
//!
//! **Invariant:** this module never `poll(Wait)`s except inside
//! [`host_download`] (used only by [`super::WgpuArray::host`]).

use std::sync::{Arc, OnceLock};

use wgpu::util::DeviceExt;

use super::{WgpuArray, WgpuProgram};
use crate::ir::{Accessor, BufferData, Kernel, element_count};
use crate::jit::{ArrayData, DeviceValue, Error, HostArray};
use crate::ops::ElementType;

pub struct Context {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn shared_context() -> Result<&'static Context, Error> {
    static CTX: OnceLock<Result<Context, String>> = OnceLock::new();
    match CTX.get_or_init(create_context) {
        Ok(ctx) => Ok(ctx),
        Err(msg) => Err(Error::Wgpu(msg.clone())),
    }
}

fn create_context() -> Result<Context, String> {
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
        .ok_or_else(|| "no suitable GPU adapter found".to_string())?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("resin"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .map_err(|e| format!("request_device: {e}"))?;

    Ok(Context { device, queue })
}

fn storage_copy_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
}

/// Upload a host array into a new storage buffer (no GPU wait).
pub fn upload_host(ctx: &Context, host: &HostArray) -> Result<WgpuArray, Error> {
    let bytes = array_data_to_bytes(host.as_data());
    let size = byte_len(host.as_data().len());
    let mut contents = bytes;
    contents.resize(size as usize, 0);
    let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("resin-value"),
        contents: &contents,
        usage: storage_copy_usage(),
    });
    Ok(WgpuArray {
        shape: host.shape().into(),
        element_type: host.element_type(),
        buffer: Arc::new(buffer),
    })
}

/// Empty storage buffer for an output leaf (filled by [`run`]).
pub fn alloc_empty(ctx: &Context, shape: &[usize], element_type: ElementType) -> WgpuArray {
    let n = element_count(shape);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-value"),
        size: byte_len(n),
        usage: storage_copy_usage(),
        mapped_at_creation: false,
    });
    WgpuArray {
        shape: shape.into(),
        element_type,
        buffer: Arc::new(buffer),
    }
}

/// Params → arenas → dispatch → compact sink buffers. Submits work; does not wait.
pub fn run(
    ctx: &Context,
    program: &WgpuProgram,
    params: &[&WgpuArray],
    outputs: &mut [&mut WgpuArray],
) -> Result<(), Error> {
    let ir = &program.ir;

    let host_shadow: Vec<Vec<u8>> = ir
        .buffers
        .iter()
        .map(|spec| {
            let size = byte_len(spec.len()) as usize;
            match &spec.init {
                Some(init) => {
                    let mut bytes = buffer_data_to_bytes(init);
                    bytes.resize(size, 0);
                    bytes
                }
                None => vec![0u8; size],
            }
        })
        .collect();

    let arenas: Vec<wgpu::Buffer> = host_shadow
        .iter()
        .map(|bytes| {
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("resin-arena"),
                contents: bytes,
                usage: storage_copy_usage(),
            })
        })
        .collect();

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resin-arenas"),
        layout: &program.gpu.bind_group_layout,
        entries: &arenas
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resin-run") });

    for (array, &param) in params.iter().zip(&ir.params) {
        let view = ir.view(param);
        let etype = ir.buffer(view.buffer).element_type;
        if array.element_type() != etype {
            return Err(Error::ElementType {
                expected: etype,
                got: array.element_type(),
            });
        }
        let count = element_count(&view.accessor.shape);
        if array.nelem() != count {
            return Err(Error::Size {
                expected: count,
                got: array.nelem(),
            });
        }
        copy_dense_into_arena(&mut encoder, array, &arenas[view.buffer.0], &view.accessor)?;
    }

    for (dispatch, &pipeline_index) in ir.queue.iter().zip(&program.pipeline_of) {
        let slot = &program.gpu.pipelines[pipeline_index];
        if let Kernel::Remap { info } = &dispatch.kernel
            && info.is_scatter()
        {
            clear_dense_region_gpu(&mut encoder, &arenas, ir, dispatch.output)?;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&slot.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let [x, y, z] = slot.workgroups;
        if x > 0 {
            pass.dispatch_workgroups(x, y, z);
        }
    }

    for (array, &sink) in outputs.iter_mut().zip(&ir.sinks) {
        let view = ir.view(sink);
        let etype = ir.buffer(view.buffer).element_type;
        let acc = &view.accessor;
        let count = element_count(&acc.shape);
        if array.element_type() != etype {
            return Err(Error::ElementType {
                expected: etype,
                got: array.element_type(),
            });
        }
        if array.nelem() != count {
            return Err(Error::Size {
                expected: count,
                got: array.nelem(),
            });
        }
        if !acc.is_dense() {
            return Err(Error::Wgpu(
                "sink view must be dense for device-local copy".into(),
            ));
        }
        let sink_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resin-sink"),
            size: byte_len(count),
            usage: storage_copy_usage(),
            mapped_at_creation: false,
        });
        let src_off = (acc.offset as u64) * 4;
        let size = byte_len(count);
        if count > 0 {
            encoder.copy_buffer_to_buffer(
                &arenas[view.buffer.0],
                src_off,
                &sink_buf,
                0,
                size,
            );
        }
        **array = WgpuArray {
            shape: (*acc.shape).into(),
            element_type: etype,
            buffer: Arc::new(sink_buf),
        };
    }

    ctx.queue.submit(Some(encoder.finish()));
    Ok(())
}

/// Map a dense storage buffer to a [`HostArray`]. **Waits on the GPU.**
pub fn host_download(ctx: &Context, array: &WgpuArray) -> Result<HostArray, Error> {
    let n = array.nelem();
    let raw = read_buffer(ctx, &array.buffer, byte_len(n))?;
    match array.element_type {
        ElementType::F32 => {
            let vals = bytes_to_f32s(&raw);
            debug_assert!(vals.len() >= n);
            Ok(HostArray::from_f32(array.shape(), &vals[..n]))
        }
        ElementType::U32 => {
            let vals = bytes_to_u32s(&raw);
            debug_assert!(vals.len() >= n);
            Ok(HostArray::from_u32(array.shape(), &vals[..n]))
        }
    }
}

fn copy_dense_into_arena(
    encoder: &mut wgpu::CommandEncoder,
    src: &WgpuArray,
    arena: &wgpu::Buffer,
    accessor: &Accessor,
) -> Result<(), Error> {
    if !accessor.is_dense() {
        return Err(Error::Wgpu(
            "param view must be dense for device-local copy".into(),
        ));
    }
    let count = element_count(&accessor.shape);
    let dst_off = (accessor.offset as u64) * 4;
    let size = byte_len(count);
    if count > 0 {
        encoder.copy_buffer_to_buffer(&src.buffer, 0, arena, dst_off, size);
    }
    Ok(())
}

fn clear_dense_region_gpu(
    encoder: &mut wgpu::CommandEncoder,
    arenas: &[wgpu::Buffer],
    ir: &crate::ir::Program,
    output: crate::ir::BufferViewRef,
) -> Result<(), Error> {
    let view = ir.view(output);
    let acc = &view.accessor;
    if !acc.is_dense() {
        return Err(Error::Wgpu("scatter clear requires dense output view".into()));
    }
    let count = element_count(&acc.shape) as u64;
    let offset_bytes = (acc.offset as u64) * 4;
    let size_bytes = count * 4;
    if size_bytes > 0 {
        encoder.clear_buffer(&arenas[view.buffer.0], offset_bytes, Some(size_bytes));
    }
    Ok(())
}

fn byte_len(elements: usize) -> u64 {
    (elements as u64 * 4).max(4)
}

fn buffer_data_to_bytes(data: &BufferData) -> Vec<u8> {
    match data {
        BufferData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        BufferData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn array_data_to_bytes(data: &ArrayData) -> Vec<u8> {
    match data {
        ArrayData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ArrayData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    }
}

fn bytes_to_f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn bytes_to_u32s(raw: &[u8]) -> Vec<u32> {
    raw.chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Map + wait. Only called from [`host_download`].
fn read_buffer(ctx: &Context, buffer: &wgpu::Buffer, size: u64) -> Result<Vec<u8>, Error> {
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resin-readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resin-read") });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    ctx.queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    ctx.device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|_| Error::Wgpu("readback channel closed".into()))?
        .map_err(|e| Error::Wgpu(format!("map_async: {e}")))?;
    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    Ok(data)
}
