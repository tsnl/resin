//! WebGPU runtime: upload params into existing arenas, dispatch, read back.
//!
//! Pipelines, arena buffers, bind-group layout, and bind group live on the
//! [`super::WgpuProgram`] artifact. Invoke does not allocate those objects.

use std::sync::OnceLock;

use super::WgpuProgram;
use crate::ir::{Accessor, Kernel, element_count};
use crate::jit::{Array, ArrayData, Error};
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

/// Write params → dispatch queue → densify sinks. Uses artifact arenas/pipelines.
pub fn run(
    ctx: &Context,
    program: &WgpuProgram,
    params: &[&Array],
    outputs: &mut [&mut Array],
) -> Result<(), Error> {
    let ir = &program.ir;
    let arenas = &program.gpu.arenas;

    // Paint param slices into existing arenas (dense views → contiguous write).
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
        if array.as_data().len() != count {
            return Err(Error::Size {
                expected: count,
                got: array.as_data().len(),
            });
        }
        write_dense_region_queue(ctx, &arenas[view.buffer.0], &view.accessor, array.as_data())?;
    }

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("resin-run") });

    for (dispatch, &pipeline_index) in ir.queue.iter().zip(&program.pipeline_of) {
        let slot = &program.gpu.pipelines[pipeline_index];
        if let Kernel::Remap { info } = &dispatch.kernel
            && info.is_scatter()
        {
            clear_dense_region_gpu(&mut encoder, arenas, ir, dispatch.output)?;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("resin-compute"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&slot.pipeline);
        pass.set_bind_group(0, &program.gpu.bind_group, &[]);
        let [x, y, z] = slot.workgroups;
        if x > 0 {
            pass.dispatch_workgroups(x, y, z);
        }
    }

    ctx.queue.submit(Some(encoder.finish()));

    for (array, &sink) in outputs.iter_mut().zip(&ir.sinks) {
        let view = ir.view(sink);
        let etype = ir.buffer(view.buffer).element_type;
        let arena_len = ir.buffer(view.buffer).len();
        // Full-arena readback (simple); densify the sink view on the host.
        let raw = read_buffer(ctx, &arenas[view.buffer.0], byte_len(arena_len))?;
        match (etype, array.as_data_mut()) {
            (ElementType::F32, ArrayData::F32(dst)) => {
                densify_f32(&bytes_to_f32s(&raw), &view.accessor, dst)?;
            }
            (ElementType::U32, ArrayData::U32(dst)) => {
                densify_u32(&bytes_to_u32s(&raw), &view.accessor, dst)?;
            }
            _ => {
                return Err(Error::ElementType {
                    expected: etype,
                    got: array.element_type(),
                });
            }
        }
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

/// Dense param views are contiguous in the arena → one `write_buffer`.
fn write_dense_region_queue(
    ctx: &Context,
    arena: &wgpu::Buffer,
    accessor: &Accessor,
    data: &ArrayData,
) -> Result<(), Error> {
    if !accessor.is_dense() {
        return Err(Error::Wgpu("param upload requires dense view".into()));
    }
    let count = element_count(&accessor.shape);
    if data.len() != count {
        return Err(Error::Size {
            expected: count,
            got: data.len(),
        });
    }
    let offset = (accessor.offset as u64) * 4;
    let bytes = match data {
        ArrayData::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
        ArrayData::U32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    };
    ctx.queue.write_buffer(arena, offset, &bytes);
    Ok(())
}

fn byte_len(elements: usize) -> u64 {
    (elements as u64 * 4).max(4)
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

fn densify_f32(buffer: &[f32], accessor: &Accessor, out: &mut [f32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &accessor.shape, &mut coords);
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn densify_u32(buffer: &[u32], accessor: &Accessor, out: &mut [u32]) -> Result<(), Error> {
    let count = element_count(&accessor.shape);
    if out.len() != count {
        return Err(Error::Size { expected: count, got: out.len() });
    }
    let mut coords = vec![0; accessor.rank()];
    for (linear, slot) in out.iter_mut().enumerate() {
        decode(linear, &accessor.shape, &mut coords);
        *slot = buffer[accessor.index(&coords)];
    }
    Ok(())
}

fn decode(linear: usize, shape: &[usize], coords: &mut [usize]) {
    debug_assert_eq!(coords.len(), shape.len());
    let mut rem = linear;
    for axis in (0..shape.len()).rev() {
        coords[axis] = rem % shape[axis];
        rem /= shape[axis];
    }
}
