use super::{Config, Result, Workload, build, expected, input, workload_path};
use resin_runtime::{
    ResinAllocation, ResinGpu, ResinGpuDeviceInfo, ResinMemory, ResinPipeline, ResinStatus,
};
use serde_json::{Value, json};
use std::ffi::CStr;

pub(super) struct Device {
    gpu: ResinGpu,
    info: ResinGpuDeviceInfo,
}

pub(super) fn list_devices() -> Result<()> {
    for info in devices()? {
        println!(
            "{}: {} ({:?}, {})",
            info.index,
            device_name(&info),
            info.kind,
            if info.suitable != 0 {
                "suitable"
            } else {
                "unsupported"
            }
        );
    }
    Ok(())
}

fn devices() -> Result<Vec<ResinGpuDeviceInfo>> {
    let count = checked("enumerate Vulkan devices", ResinGpu::device_count())?;
    // Zero is a valid representation for every field, including device kind.
    let mut infos = vec![unsafe { std::mem::zeroed() }; count as usize];
    checked(
        "enumerate Vulkan devices",
        ResinGpu::enumerate_devices(&mut infos),
    )?;
    Ok(infos)
}

fn device_name(info: &ResinGpuDeviceInfo) -> String {
    // Runtime enumeration writes a NUL-terminated Vulkan device name.
    unsafe { CStr::from_ptr(info.name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

impl Device {
    pub(super) fn open(index: Option<u32>) -> Result<Self> {
        let info = devices()?
            .into_iter()
            .find(|info| index.map_or(info.suitable != 0, |index| info.index == index))
            .ok_or("no matching suitable GPU; use --list-devices to inspect Vulkan devices")?;
        let gpu = checked("create GPU", ResinGpu::create_at(info.index))?;
        eprintln!("GPU {}: {}", info.index, device_name(&info));
        Ok(Self { gpu, info })
    }

    pub(super) fn description(&self) -> Value {
        json!({ "index": self.info.index, "name": device_name(&self.info),
            "kind": format!("{:?}", self.info.kind), "vendor_id": self.info.vendor_id,
            "device_id": self.info.device_id, "driver_version": self.info.driver_version,
            "api_version": self.info.api_version })
    }

    pub(super) fn run(&mut self, workload: &Workload, config: &Config) -> Result<Vec<f64>> {
        let built = build(workload, &workload_path(workload), None, &[])?;
        let shaders = built.generated.shaders();
        if shaders.len() != 1 {
            return Err("a compute benchmark must declare exactly one shader".into());
        }
        let bytes = std::fs::read(
            built
                .artifacts
                .path(shaders[0].spirv().file_name().unwrap()),
        )?;
        let pipeline = checked(
            "create compute pipeline",
            // These bytes were produced by this build's verified Resin shader.
            unsafe { self.gpu.create_compute_pipeline(&bytes) },
        )?;
        let count = config.size.unwrap_or(workload.count);
        let expected: Vec<_> = (0..count).map(|lane| expected(workload, lane)).collect();
        // Allocations are live, aligned, and coherent; every submission completes
        // before a host read or free. Pipelines and recordings belong to this GPU.
        unsafe {
            let buffers = Buffers::new(&mut self.gpu, workload, count)?;
            let result = self.measure(&pipeline, &buffers, config, &expected);
            buffers.free(&mut self.gpu);
            result
        }
    }

    unsafe fn measure(
        &self,
        pipeline: &ResinPipeline,
        buffers: &Buffers,
        config: &Config,
        expected: &[u32],
    ) -> Result<Vec<f64>> {
        unsafe {
            self.dispatch(pipeline, buffers)?;
            buffers.validate(expected)?;
            for _ in 0..config.warmup {
                self.dispatch(pipeline, buffers)?;
            }
            let mut samples = Vec::with_capacity(config.samples);
            for _ in 0..config.samples {
                let mut commands = checked(
                    "start GPU timing (timestamps must be supported)",
                    self.gpu.start_timed_command_recording(),
                )?;
                checked("bind compute pipeline", commands.set_pipeline(pipeline))?;
                checked(
                    "dispatch",
                    commands.dispatch(
                        buffers.root.device_pointer(),
                        buffers.count.div_ceil(64),
                        1,
                        1,
                    ),
                )?;
                samples.push(
                    checked("submit timed dispatch", self.gpu.submit_timed(commands))?
                        .as_secs_f64(),
                );
            }
            buffers.validate(expected)?;
            Ok(samples)
        }
    }

    unsafe fn dispatch(&self, pipeline: &ResinPipeline, buffers: &Buffers) -> Result<()> {
        unsafe {
            let mut commands = checked("start recording", self.gpu.start_command_recording())?;
            checked("bind compute pipeline", commands.set_pipeline(pipeline))?;
            checked(
                "dispatch",
                commands.dispatch(
                    buffers.root.device_pointer(),
                    buffers.count.div_ceil(64),
                    1,
                    1,
                ),
            )?;
            checked("submit dispatch", self.gpu.submit(commands))
        }
    }
}

#[repr(C)]
struct Root {
    count: u32,
    iterations: u32,
    input: u64,
    output: u64,
}

struct Buffers {
    input: ResinAllocation,
    output: ResinAllocation,
    root: ResinAllocation,
    count: u32,
}

const CANARY: u32 = 0xcdcdcdcd;

impl Buffers {
    unsafe fn new(gpu: &mut ResinGpu, workload: &Workload, count: u32) -> Result<Self> {
        unsafe {
            let bytes = count as usize * size_of::<u32>();
            let input_buffer = checked(
                "allocate inputs",
                gpu.malloc(bytes, align_of::<u32>(), ResinMemory::Default),
            )?;
            let output = checked(
                "allocate outputs",
                gpu.malloc(
                    bytes + size_of::<u32>(),
                    align_of::<u32>(),
                    ResinMemory::Default,
                ),
            )?;
            let root = checked(
                "allocate parameters",
                gpu.malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default),
            )?;
            for (lane, value) in std::slice::from_raw_parts_mut(
                input_buffer.host_pointer().cast::<u32>(),
                count as usize,
            )
            .iter_mut()
            .enumerate()
            {
                *value = input(lane as u32);
            }
            std::slice::from_raw_parts_mut(output.host_pointer().cast::<u32>(), count as usize + 1)
                .fill(CANARY);
            root.host_pointer().cast::<Root>().write(Root {
                count,
                iterations: workload.iterations,
                input: input_buffer.device_pointer(),
                output: output.device_pointer(),
            });
            Ok(Self {
                input: input_buffer,
                output,
                root,
                count,
            })
        }
    }

    unsafe fn validate(&self, expected: &[u32]) -> Result<()> {
        let output = unsafe {
            std::slice::from_raw_parts(
                self.output.host_pointer().cast::<u32>(),
                self.count as usize + 1,
            )
        };
        if output[self.count as usize] != CANARY {
            return Err("GPU workload wrote beyond its output".into());
        }
        for (lane, (&actual, &expected)) in output.iter().zip(expected).enumerate() {
            if actual != expected {
                return Err(format!(
                    "GPU validation failed at lane {lane}: {actual} != {expected}"
                )
                .into());
            }
        }
        Ok(())
    }

    unsafe fn free(self, gpu: &mut ResinGpu) {
        unsafe {
            gpu.free(&self.root);
            gpu.free(&self.output);
            gpu.free(&self.input);
        }
    }
}

fn checked<T>(operation: &str, result: std::result::Result<T, ResinStatus>) -> Result<T> {
    result.map_err(|status| format!("{operation}: {status:?}").into())
}
