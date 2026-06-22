use super::{BufferId, Interp, InterpConfig, InterpError, ProgramId};
use crate::program::{
    WgpuBufferSpec, WgpuComputePipelineSpec, WgpuCopy, WgpuDispatch, WgpuProgram, WgpuQueueOp,
};
use rmpv::Value as MsgpackValue;
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

impl From<WgpuInterpError> for InterpError {
    fn from(err: WgpuInterpError) -> Self {
        match err {
            WgpuInterpError::NoAdapter | WgpuInterpError::RequestDevice(_) => {
                InterpError::Backend(err.to_string())
            }
            WgpuInterpError::Program(message) => InterpError::Program(message),
            WgpuInterpError::BufferMapFailed => InterpError::BufferMapFailed,
        }
    }
}

struct PreparedDispatch {
    pipeline_index: usize,
    bind_group: wgpu::BindGroup,
    dispatch_size: [u32; 3],
    clear_output_before_dispatch: bool,
    output_buffer_index: usize,
    output_byte_len: u64,
}

struct PreparedCopy {
    source_buffer_index: usize,
    output_buffer_index: usize,
    src_offset: u64,
    copy_bytes: u64,
}

enum PreparedQueueOp {
    Dispatch(PreparedDispatch),
    Copy(PreparedCopy),
}

struct LoadedProgram {
    program: WgpuProgram,
    buffers: Vec<wgpu::Buffer>,
    pipelines: Vec<wgpu::ComputePipeline>,
    prepared_queue: Vec<PreparedQueueOp>,
}

pub struct WgpuInterp {
    device: wgpu::Device,
    queue: wgpu::Queue,
    programs: Vec<LoadedProgram>,
}

fn request_device_from_adapter(
    adapter: wgpu::Adapter,
) -> Result<(wgpu::Device, wgpu::Queue), WgpuInterpError> {
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

/// Acquire a default GPU device and queue for standalone use (e.g. Python dev).
///
/// Game engines should pass their existing `Device` and `Queue` to [`WgpuInterp::empty`].
pub fn request_default_device() -> Result<(wgpu::Device, wgpu::Queue), WgpuInterpError> {
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
        .ok_or(WgpuInterpError::NoAdapter)?;

    request_device_from_adapter(adapter)
}

fn request_named_device(device_name: &str) -> Result<(wgpu::Device, wgpu::Queue), WgpuInterpError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter = instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .find(|adapter| adapter.get_info().name == device_name)
        .ok_or(WgpuInterpError::NoAdapter)?;

    request_device_from_adapter(adapter)
}

impl WgpuInterp {
    /// Build an interpreter against an existing wgpu context with no programs loaded.
    pub fn empty(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        Self {
            device: device.clone(),
            queue: queue.clone(),
            programs: Vec::new(),
        }
    }

    pub fn from_config(config: &InterpConfig) -> Result<Self, InterpError> {
        let device_name = parse_wgpu_config(config)?;
        let (device, queue) = match device_name.as_deref() {
            None | Some("") | Some("default") => {
                request_default_device().map_err(InterpError::from)?
            }
            Some(name) => request_named_device(name).map_err(|err| match err {
                WgpuInterpError::NoAdapter => InterpError::Config(format!(
                    "no GPU adapter found with device_name '{name}'"
                )),
                other => other.into(),
            })?,
        };
        Ok(Self::empty(&device, &queue))
    }

    pub fn program_count(&self) -> usize {
        self.programs.len()
    }

    pub fn admit_wgpu_program(
        &mut self,
        program: WgpuProgram,
    ) -> Result<ProgramId, WgpuInterpError> {
        let buffers = program
            .buffers
            .iter()
            .map(|spec| create_buffer(&self.device, spec))
            .collect::<Result<Vec<_>, _>>()?;

        let pipelines = program
            .pipelines
            .iter()
            .map(|spec| create_pipeline(&self.device, spec))
            .collect::<Result<Vec<_>, _>>()?;

        let prepared_queue =
            prepare_queue(&self.device, &program, &buffers, &pipelines)?;

        let program_id = self.programs.len();
        self.programs.push(LoadedProgram {
            program,
            buffers,
            pipelines,
            prepared_queue,
        });
        Ok(ProgramId(program_id))
    }

    pub fn program(&self, program_id: ProgramId) -> Result<&WgpuProgram, WgpuInterpError> {
        Ok(&self.loaded_program(program_id.0)?.program)
    }

    pub fn run(&self, program_id: ProgramId) -> Result<(), WgpuInterpError> {
        let loaded = self.loaded_program(program_id.0)?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("resin-run"),
            });

        for prepared in &loaded.prepared_queue {
            match prepared {
                PreparedQueueOp::Dispatch(prepared) => {
                    self.run_compute_dispatch(&mut encoder, loaded, prepared)?;
                }
                PreparedQueueOp::Copy(prepared) => {
                    self.run_copy(&mut encoder, loaded, prepared)?;
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn run_compute_dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        loaded: &LoadedProgram,
        prepared: &PreparedDispatch,
    ) -> Result<(), WgpuInterpError> {
        debug_assert_ne!(prepared.dispatch_size, [0, 0, 0]);

        if prepared.clear_output_before_dispatch {
            encoder.clear_buffer(
                &loaded.buffers[prepared.output_buffer_index],
                0,
                Some(prepared.output_byte_len),
            );
        }

        let pipeline = &loaded.pipelines[prepared.pipeline_index];
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
        Ok(())
    }

    fn run_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        loaded: &LoadedProgram,
        prepared: &PreparedCopy,
    ) -> Result<(), WgpuInterpError> {
        encoder.copy_buffer_to_buffer(
            &loaded.buffers[prepared.source_buffer_index],
            prepared.src_offset,
            &loaded.buffers[prepared.output_buffer_index],
            0,
            prepared.copy_bytes,
        );
        Ok(())
    }

    pub fn write_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
        data: &[u8],
    ) -> Result<(), WgpuInterpError> {
        let loaded = self.loaded_program(program_id.0)?;
        let spec = loaded
            .program
            .buffers
            .get(buffer_id.0)
            .ok_or_else(|| {
                WgpuInterpError::Program(format!("invalid buffer id {}", buffer_id.0))
            })?;
        if data.len() as u64 != spec.byte_len() {
            return Err(WgpuInterpError::Program(format!(
                "write_buffer size mismatch: expected {} bytes, got {}",
                spec.byte_len(),
                data.len()
            )));
        }

        let buffer = loaded.buffers.get(buffer_id.0).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid buffer id {}", buffer_id.0))
        })?;
        self.queue.write_buffer(buffer, 0, data);
        Ok(())
    }

    pub fn copy_buffer_to_buffer(
        &self,
        src_program_id: ProgramId,
        src_buffer_id: BufferId,
        dst_program_id: ProgramId,
        dst_buffer_id: BufferId,
    ) -> Result<(), WgpuInterpError> {
        let src = self.program_buffer(src_program_id.0, src_buffer_id.0)?;
        let dst = self.program_buffer(dst_program_id.0, dst_buffer_id.0)?;
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

    pub fn read_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
    ) -> Result<Vec<u8>, WgpuInterpError> {
        let buffer = self.program_buffer(program_id.0, buffer_id.0)?;
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

    fn program_buffer(
        &self,
        program_id: usize,
        buffer_id: usize,
    ) -> Result<&wgpu::Buffer, WgpuInterpError> {
        let loaded = self.loaded_program(program_id)?;
        loaded.buffers.get(buffer_id).ok_or_else(|| {
            WgpuInterpError::Program(format!(
                "invalid buffer id {buffer_id} for program {program_id}"
            ))
        })
    }

    fn loaded_program(&self, program_id: usize) -> Result<&LoadedProgram, WgpuInterpError> {
        self.programs.get(program_id).ok_or_else(|| {
            WgpuInterpError::Program(format!("invalid program id {program_id}"))
        })
    }
}

impl Interp for WgpuInterp {
    fn program_count(&self) -> usize {
        WgpuInterp::program_count(self)
    }

    fn admit_program(&mut self, program_msgpack: &[u8]) -> Result<ProgramId, InterpError> {
        let program = WgpuProgram::from_msgpack(program_msgpack)
            .map_err(|err| InterpError::Program(err.to_string()))?;
        WgpuInterp::admit_wgpu_program(self, program).map_err(Into::into)
    }

    fn run(&self, program_id: ProgramId) -> Result<(), InterpError> {
        WgpuInterp::run(self, program_id).map_err(Into::into)
    }

    fn write_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
        data: &[u8],
    ) -> Result<(), InterpError> {
        WgpuInterp::write_buffer(self, program_id, buffer_id, data).map_err(Into::into)
    }

    fn read_buffer(
        &self,
        program_id: ProgramId,
        buffer_id: BufferId,
    ) -> Result<Vec<u8>, InterpError> {
        WgpuInterp::read_buffer(self, program_id, buffer_id).map_err(Into::into)
    }

    fn copy_buffer_to_buffer(
        &self,
        src_program_id: ProgramId,
        src_buffer_id: BufferId,
        dst_program_id: ProgramId,
        dst_buffer_id: BufferId,
    ) -> Result<(), InterpError> {
        WgpuInterp::copy_buffer_to_buffer(
            self,
            src_program_id,
            src_buffer_id,
            dst_program_id,
            dst_buffer_id,
        )
        .map_err(Into::into)
    }
}

fn parse_wgpu_config(config: &InterpConfig) -> Result<Option<String>, InterpError> {
    match config {
        MsgpackValue::Nil => Ok(None),
        MsgpackValue::Map(map) => {
            let mut device_name = None;
            for (key, value) in map {
                let key_name = match key {
                    MsgpackValue::String(key_str) => key_str.as_str().ok_or_else(|| {
                        InterpError::Config("config keys must be UTF-8 strings".into())
                    })?,
                    _ => return Err(InterpError::Config("config keys must be strings".into())),
                };
                match key_name {
                    "device_name" => {
                        let name = match value {
                            MsgpackValue::String(name) => name.as_str().ok_or_else(|| {
                                InterpError::Config(
                                    "device_name must be a UTF-8 string".into(),
                                )
                            })?,
                            _ => {
                                return Err(InterpError::Config(
                                    "device_name must be a string".into(),
                                ));
                            }
                        };
                        if device_name.is_some() {
                            return Err(InterpError::Config(
                                "duplicate device_name config key".into(),
                            ));
                        }
                        device_name = Some(name.to_string());
                    }
                    other => {
                        return Err(InterpError::Config(format!(
                            "unknown config key: {other}"
                        )));
                    }
                }
            }
            Ok(device_name)
        }
        _ => Err(InterpError::Config("config must be a map or null".into())),
    }
}

#[cfg(test)]
mod config_tests {
    use super::*;
    use rmpv::Value as MsgpackValue;

    #[test]
    fn rejects_unknown_config_key() {
        let config = MsgpackValue::Map(vec![(
            MsgpackValue::String("bogus".into()),
            MsgpackValue::from(1),
        )]);
        assert!(matches!(
            parse_wgpu_config(&config),
            Err(InterpError::Config(message)) if message.contains("unknown config key")
        ));
    }

    #[test]
    fn rejects_non_string_device_name() {
        let config = MsgpackValue::Map(vec![(
            MsgpackValue::String("device_name".into()),
            MsgpackValue::from(1),
        )]);
        assert!(matches!(
            parse_wgpu_config(&config),
            Err(InterpError::Config(message)) if message.contains("device_name must be a string")
        ));
    }

    #[test]
    fn accepts_nil_config() {
        assert!(matches!(
            parse_wgpu_config(&MsgpackValue::Nil),
            Ok(None)
        ));
    }
}

fn prepare_queue(
    device: &wgpu::Device,
    program: &WgpuProgram,
    buffers: &[wgpu::Buffer],
    pipelines: &[wgpu::ComputePipeline],
) -> Result<Vec<PreparedQueueOp>, WgpuInterpError> {
    program
        .queue
        .iter()
        .map(|op| match op {
            WgpuQueueOp::Dispatch(dispatch) => Ok(PreparedQueueOp::Dispatch(
                prepare_dispatch_op(device, program, buffers, pipelines, dispatch)?,
            )),
            WgpuQueueOp::Copy(copy) => {
                Ok(PreparedQueueOp::Copy(prepare_copy_op(program, copy)?))
            }
        })
        .collect()
}

fn prepare_dispatch_op(
    device: &wgpu::Device,
    program: &WgpuProgram,
    buffers: &[wgpu::Buffer],
    pipelines: &[wgpu::ComputePipeline],
    dispatch: &WgpuDispatch,
) -> Result<PreparedDispatch, WgpuInterpError> {
    let compute = &program.pipelines[dispatch.pipeline_index];
    let pipeline = &pipelines[dispatch.pipeline_index];

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

    Ok(PreparedDispatch {
        pipeline_index: dispatch.pipeline_index,
        bind_group,
        dispatch_size: compute.dispatch_size,
        clear_output_before_dispatch: compute.clear_output_before_dispatch,
        output_buffer_index: dispatch.output_buffer_index,
        output_byte_len: program.buffers[dispatch.output_buffer_index].byte_len(),
    })
}

fn prepare_copy_op(
    program: &WgpuProgram,
    copy: &WgpuCopy,
) -> Result<PreparedCopy, WgpuInterpError> {
    let source_view = &program.buffer_views[copy.source_buffer_view_index];
    let output_spec = program
        .buffers
        .get(copy.output_buffer_index)
        .ok_or_else(|| {
            WgpuInterpError::Program(format!(
                "invalid output buffer index {}",
                copy.output_buffer_index
            ))
        })?;

    let accessor = &source_view.accessor;
    let element_nbytes = output_spec.etype.nbytes() as u64;
    let copy_bytes: u64 = accessor
        .shape
        .iter()
        .map(|&d| d as u64)
        .product::<u64>()
        * element_nbytes;
    if output_spec.byte_len() != copy_bytes {
        return Err(WgpuInterpError::Program(format!(
            "copy size mismatch: output {} bytes, view {} bytes",
            output_spec.byte_len(),
            copy_bytes
        )));
    }

    Ok(PreparedCopy {
        source_buffer_index: source_view.buffer_index,
        output_buffer_index: copy.output_buffer_index,
        src_offset: accessor.offset as u64 * element_nbytes,
        copy_bytes,
    })
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
    spec: &WgpuComputePipelineSpec,
) -> Result<wgpu::ComputePipeline, WgpuInterpError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resin-shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(spec.wgsl.as_str())),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resin-pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some(spec.entry_point.as_str()),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    Ok(pipeline)
}