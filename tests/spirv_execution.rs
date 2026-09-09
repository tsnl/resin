#![cfg(feature = "gpu")]
use resin_runtime::{ResinGpu, ResinMemory, ResinStatus, testing::lock_gpu};
use std::fs;

use support::shaders;
mod support;
#[path = "support/toolchain.rs"]
mod toolchain;

#[repr(C)]
struct Root {
    count: u64,
    inputs: u64,
    outputs: u64,
}

fn gpu() -> Option<ResinGpu> {
    match ResinGpu::create() {
        Ok(gpu) => Some(gpu),
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert!(
                std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "a suitable Vulkan device is required"
            );
            eprintln!("skipping: no suitable Vulkan device");
            None
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    }
}

fn execute<I: Copy, O: Copy>(source: &str, inputs: &[I], sentinel: O) -> Option<Vec<O>> {
    let optimizer = shaders::optimizer()?;
    // Keep the same GPU-before-build lock order as the other GPU suites.
    let _lock = lock_gpu();
    let mut gpu = gpu()?;
    let module = support::module(source);
    let project = support::project::Project::new(&module, None).unwrap();
    let built = project.build(&toolchain::spirv(&optimizer)).unwrap();
    let shader = &project.generated.shaders()[0];
    let bytes = fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
    // Each fixture uses Root and the supplied repr(C) element types. Resources stay
    // live through synchronous submission; mapped output is read only afterwards.
    unsafe {
        let pipeline = gpu.create_compute_pipeline(&bytes).unwrap();
        let input = gpu
            .malloc(size_of_val(inputs), align_of::<I>(), ResinMemory::Default)
            .unwrap();
        let output = gpu
            .malloc(
                (inputs.len() + 1) * size_of::<O>(),
                align_of::<O>(),
                ResinMemory::Default,
            )
            .unwrap();
        let root = gpu
            .malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default)
            .unwrap();
        std::slice::from_raw_parts_mut(input.host_pointer().cast::<I>(), inputs.len())
            .copy_from_slice(inputs);
        std::slice::from_raw_parts_mut(output.host_pointer().cast::<O>(), inputs.len() + 1)
            .fill(sentinel);
        root.host_pointer().cast::<Root>().write(Root {
            count: inputs.len() as u64,
            inputs: input.device_pointer(),
            outputs: output.device_pointer(),
        });
        let mut commands = gpu.start_command_recording().unwrap();
        commands.set_pipeline(&pipeline).unwrap();
        commands
            .dispatch(
                root.device_pointer(),
                (inputs.len() as u32).div_ceil(64),
                1,
                1,
            )
            .unwrap();
        gpu.submit(commands).unwrap();
        Some(
            std::slice::from_raw_parts(output.host_pointer().cast::<O>(), inputs.len() + 1)
                .to_vec(),
        )
    }
}

#[test]
fn dynamic_numeric_limits_and_failures_in_loop_conditions() {
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    struct Input {
        kind: u32,
        value: f32,
        wide: u64,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Output {
        progress: u32,
        value: u64,
    }
    let cases = [
        (0, -0.75, 0, Some(0)),
        (0, 255.75, 0, Some(255)),
        (0, 256.0, 0, None),
        (0, -1.0, 0, None),
        (0, f32::NAN, 0, None),
        (0, f32::INFINITY, 0, None),
        (0, f32::NEG_INFINITY, 0, None),
        (1, 4294967040.0, 0, Some(4294967040)),
        (1, 4294967296.0, 0, None),
        (2, -2147483648.0, 0, Some(i32::MIN as u64)),
        (2, 2147483520.0, 0, Some(2147483520)),
        (2, 2147483648.0, 0, None),
        (2, -2147483904.0, 0, None),
        (2, -12.75, 0, Some((-12_i32) as u64)),
        (3, f32::from_bits(0x5f7fffff), 0, Some(18446742974197923840)),
        (3, f32::from_bits(0x5f800000), 0, None),
        (3, -0.5, 0, Some(0)),
        (4, 0.0, 255, Some(255)),
        (4, 0.0, 256, None),
        (5, 0.0, 4294967295, Some(4294967295)),
        (5, 0.0, 4294967296, None),
        (6, 0.0, 2147483647, Some(2147483647)),
        (6, 0.0, 2147483648, None),
    ];
    let inputs: Vec<_> = cases
        .iter()
        .map(|&(kind, value, wide, _)| Input { kind, value, wide })
        .collect();
    let sentinel = Output {
        progress: 0,
        value: u64::MAX,
    };
    let Some(actual) = execute(
        include_str!("fixtures/spirv_numeric.resin"),
        &inputs,
        sentinel,
    ) else {
        return;
    };
    for (index, &(_, _, _, value)) in cases.iter().enumerate() {
        let expected = value.map_or(
            Output {
                progress: 2,
                ..sentinel
            },
            |value| Output { progress: 4, value },
        );
        assert_eq!(actual[index], expected, "case {index}: {:?}", inputs[index]);
    }
    assert_eq!(
        actual[cases.len()],
        sentinel,
        "excess invocation must preserve canary"
    );
}

#[test]
fn physical_byte_record_strides_and_mixed_record_copies() {
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Bytes3 {
        a: u8,
        b: u8,
        c: u8,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Rows {
        first: Bytes3,
        second: Bytes3,
    }
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Sample {
        lead: u8,
        rows: Rows,
        marker: u32,
        wide: u64,
        tail: u8,
    }
    assert_eq!(size_of::<Bytes3>(), 3);
    assert_eq!((size_of::<Sample>(), align_of::<Sample>()), (32, 8));
    assert_eq!(std::mem::offset_of!(Sample, rows), 1);
    assert_eq!(std::mem::offset_of!(Sample, marker), 8);
    assert_eq!(std::mem::offset_of!(Sample, wide), 16);
    assert_eq!(std::mem::offset_of!(Sample, tail), 24);
    let inputs: Vec<_> = (0..67)
        .map(|index| Sample {
            lead: 201,
            rows: Rows {
                first: Bytes3 {
                    a: index as u8,
                    b: 110,
                    c: 212,
                },
                second: Bytes3 {
                    a: 213,
                    b: 250,
                    c: index as u8 + 1,
                },
            },
            marker: 0xa5000000 + index,
            wide: (1_u64 << 55) + u64::from(index),
            tail: 202,
        })
        .collect();
    let sentinel = inputs[0];
    let Some(actual) = execute(
        include_str!("fixtures/spirv_packed.resin"),
        &inputs,
        sentinel,
    ) else {
        return;
    };
    for (index, input) in inputs.iter().enumerate() {
        let mut expected = *input;
        if index & 1 == 0 {
            expected.rows.first.b += 1;
        } else {
            expected.rows.second.b += 1;
        }
        expected.marker += 1;
        expected.wide += 4294967297;
        assert_eq!(actual[index], expected, "sample {index}");
    }
    assert_eq!(
        actual[inputs.len()],
        sentinel,
        "excess invocation must preserve canary"
    );
}

#[test]
fn an_unconditionally_failing_loop_condition_stops_before_body_and_caller_stores() {
    let source = r#"export { kernel };
        struct Root { count: ulong, inputs: Ptr<uint>, outputs: Ptr<uint> };
        @compute_shader def kernel(index: ulong, root: Ptr<Root>) = {
            if (index < root.count) {
                var output = Span<uint> { data = root.outputs, length = root.count };
                while ({
                    var value: None;
                    value := None;
                    var test: bool;
                    test := value!;
                    test
                }) { output.at(index).* := 1_ui; };
                output.at(index).* := 2_ui;
            };
        };"#;
    let sentinel = 0xabcd1234_u32;
    let Some(actual) = execute(source, &[0_u32; 65], sentinel) else {
        return;
    };
    assert_eq!(actual, vec![sentinel; 66]);
}
