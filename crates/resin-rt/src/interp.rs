use crate::program::{WgpuBufferSpec, WgpuPipelineSpec, WgpuProgram};
use std::borrow::Cow;
use thiserror::Error;
use wgpu::util::DeviceExt;

#[derive(Debug, Error)]
pub enum WgpuInterpError {
    #[error("no suitable GPU adapter found")]
    NoAdapter,
    #[error("request device failed: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("program error: {0}")]
    Program(String),
    #[error("buffer map failed")]
    BufferMapFailed,
}

struct PreparedDispatch {
    pipeline_index: usize,
    bind_group: wgpu::BindGroup,
    dispatch_size: [u32; 3],
    clear_output_before_dispatch: bool,
    output_buffer_index: usize,
}

pub struct WgpuInterp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    program: WgpuProgram,
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    prepared_dispatches: Vec<PreparedDispatch>,
}

/// Acquire a default GPU device and queue for standalone use (e.g. Python dev).
///
/// Game engines should pass their existing `Device` and `Queue` to [`WgpuInterp::new`].
pub fn request_default_device() -> Result<(wgpu::Device, wgpu::Queue), WgpuInterpError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or(WgpuInterpError::NoAdapter)?;

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

impl WgpuInterp {
    /// Build an interpreter against an existing wgpu context.
    ///
    /// `device` and `queue` are cloned (they are cheap handles) and used for the
    /// lifetime of this interpreter.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        program: WgpuProgram,
    ) -> Result<Self, WgpuInterpError> {
        let device = device.clone();
        let queue = queue.clone();

        let buffers = program
            .buffers
            .iter()
            .map(|spec| create_buffer(&device, spec))
            .collect::<Result<Vec<_>, _>>()?;

        let pipelines = program
            .pipelines
            .iter()
            .map(|spec| create_pipeline(&device, spec))
            .collect::<Result<Vec<_>, _>>()?;

        let prepared_dispatches =
            prepare_dispatches(&device, &program, &buffers, &pipelines)?;

        Ok(Self {
            device,
            queue,
            program,
            buffers,
            pipelines,
            prepared_dispatches,
        })
    }

    pub fn program(&self) -> &WgpuProgram {
        &self.program
    }

    pub fn run(&self) -> Result<(), WgpuInterpError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resin-run"),
            });

        for prepared in &self.prepared_dispatches {
            if prepared.clear_output_before_dispatch {
                let output_spec = self
                    .program
                    .buffers
                    .get(prepared.output_buffer_index)
                    .ok_or_else(|| {
                        WgpuInterpError::Program(format!(
                            "invalid output buffer index {}",
                            prepared.output_buffer_index
                        ))
                    })?;
                encoder.clear_buffer(
                    &self.buffers[prepared.output_buffer_index],
                    0,
                    Some(output_spec.byte_len()),
                );
            }
        }

        for prepared in &self.prepared_dispatches {
            let pipeline = &self.pipelines[prepared.pipeline_index];
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resin-compute-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &prepared.bind_group, &[]);
            pass.dispatch_workgroups(
                prepared.dispatch_size[0],
                prepared.dispatch_size[1],
                prepared.dispatch_size[2],
            );
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn clear_buffer(&self, buffer_index: usize) -> Result<(), WgpuInterpError> {
        let spec = self.program.buffers.get(buffer_index).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid buffer index {buffer_index}"))
        })?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resin-clear"),
            });
        encoder.clear_buffer(&self.buffers[buffer_index], 0, Some(spec.byte_len()));
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn write_buffer(&self, buffer_index: usize, data: &[u8]) -> Result<(), WgpuInterpError> {
        let spec = self.program.buffers.get(buffer_index).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid buffer index {buffer_index}"))
        })?;
        if data.len() as u64 != spec.byte_len() {
            return Err(WgpuInterpError::Program(format!(
                "write_buffer size mismatch: expected {} bytes, got {}",
                spec.byte_len(),
                data.len()
            )));
        }

        self.queue
            .write_buffer(&self.buffers[buffer_index], 0, data);
        Ok(())
    }

    pub fn param_buffer_index(&self, param_id: u64) -> Result<usize, WgpuInterpError> {
        self.program
            .param_buffer_ids
            .get(&param_id)
            .copied()
            .ok_or_else(|| WgpuInterpError::Program(format!("unknown param id {param_id:#x}")))
    }

    pub fn copy_buffer_to_buffer(
        &self,
        src_buffer_index: usize,
        dst_buffer_index: usize,
    ) -> Result<(), WgpuInterpError> {
        let src = self.buffers.get(src_buffer_index).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid source buffer index {src_buffer_index}"))
        })?;
        let dst = self.buffers.get(dst_buffer_index).ok_or_else(|| {
            WgpuInterpError::Program(format!(
                "invalid destination buffer index {dst_buffer_index}"
            ))
        })?;
        let size = src.size();
        if size != dst.size() {
            return Err(WgpuInterpError::Program(format!(
                "copy_buffer_to_buffer size mismatch: source {} bytes, destination {} bytes",
                size,
                dst.size()
            )));
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resin-copy"),
            });
        encoder.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    pub fn read_buffer(&self, buffer_index: usize) -> Result<Vec<u8>, WgpuInterpError> {
        let buffer = self.buffers.get(buffer_index).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid buffer index {buffer_index}"))
        })?;
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
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| WgpuInterpError::BufferMapFailed)?
            .map_err(|_| WgpuInterpError::BufferMapFailed)?;

        let mapped = buffer_slice.get_mapped_range();
        Ok(mapped.to_vec())
    }
}

fn prepare_dispatches(
    device: &wgpu::Device,
    program: &WgpuProgram,
    buffers: &[wgpu::Buffer],
    pipelines: &[wgpu::ComputePipeline],
) -> Result<Vec<PreparedDispatch>, WgpuInterpError> {
    let mut prepared = Vec::with_capacity(program.queue.len());

    for dispatch in &program.queue {
        let compute = match &program.pipelines[dispatch.pipeline_index] {
            WgpuPipelineSpec::Compute(spec) => spec,
        };
        let pipeline = &pipelines[dispatch.pipeline_index];

        if compute.dispatch_size == [0, 0, 0] || compute.dispatch_size[0] == 0 {
            continue;
        }

        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: buffers[dispatch.output_buffer_index].as_entire_binding(),
        }];
        for (binding, &view_index) in dispatch.arg_buffer_view_indices.iter().enumerate() {
            let view = &program.buffer_views[view_index];
            entries.push(wgpu::BindGroupEntry {
                binding: (binding + 1) as u32,
                resource: buffers[view.buffer_index].as_entire_binding(),
            });
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resin-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });

        prepared.push(PreparedDispatch {
            pipeline_index: dispatch.pipeline_index,
            bind_group,
            dispatch_size: compute.dispatch_size,
            clear_output_before_dispatch: compute.clear_output_before_dispatch,
            output_buffer_index: dispatch.output_buffer_index,
        });
    }

    Ok(prepared)
}

fn create_buffer(
    device: &wgpu::Device,
    spec: &WgpuBufferSpec,
) -> Result<wgpu::Buffer, WgpuInterpError> {
    let size = spec.byte_len().max(4);
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;

    if let Some(init) = &spec.init {
        if init.len() as u64 != spec.byte_len() {
            return Err(WgpuInterpError::Program(format!(
                "buffer init size mismatch: expected {} bytes, got {}",
                spec.byte_len(),
                init.len()
            )));
        }
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resin-buffer"),
            contents: init,
            usage,
        });
        Ok(buffer)
    } else {
        Ok(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resin-buffer"),
            size,
            usage,
            mapped_at_creation: false,
        }))
    }
}

fn create_pipeline(
    device: &wgpu::Device,
    spec: &WgpuPipelineSpec,
) -> Result<wgpu::ComputePipeline, WgpuInterpError> {
    let compute = match spec {
        WgpuPipelineSpec::Compute(compute) => compute,
    };

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(compute.wgsl.as_str())),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resin-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some(compute.entry_point.as_str()),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(pipeline)
}