#![cfg(feature = "gpu")]
use resin_runtime::{ResinGpu, ResinMemory, ResinStatus, testing::lock_gpu};
use std::fs;

use support::shaders;
mod support;
use support::toolchain;

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

#[test]
fn integer_arithmetic_matches_cpu_results_and_propagates_invocation_failure() {
    macro_rules! check {
        ($ty:ty) => {{
            #[repr(C)]
            #[derive(Clone, Copy, Debug)]
            struct Input {
                left: $ty,
                right: $ty,
                operation: u32,
            }
            #[repr(C)]
            #[derive(Clone, Copy, Debug, PartialEq)]
            struct Output {
                value: $ty,
                progress: u32,
            }
            let cases = support::integer_arithmetic::cases(stringify!($ty));
            let inputs = cases
                .iter()
                .map(|&(left, right, operation, _)| Input {
                    left: left as $ty,
                    right: right as $ty,
                    operation,
                })
                .collect::<Vec<_>>();
            let sentinel = Output {
                value: <$ty>::MAX,
                progress: 0,
            };
            let source = support::integer_arithmetic::source(stringify!($ty));
            if let Some(actual) = execute(&source, &inputs, sentinel) {
                for (index, &(_, _, _, value)) in cases.iter().enumerate() {
                    let expected = value.map_or(
                        Output {
                            progress: 1,
                            ..sentinel
                        },
                        |value| Output {
                            value: value as $ty,
                            progress: 3,
                        },
                    );
                    assert_eq!(
                        actual[index],
                        expected,
                        "{} case {:?}",
                        stringify!($ty),
                        inputs[index]
                    );
                }
                assert_eq!(actual[inputs.len()], sentinel);
            }
        }};
    }
    check!(u8);
    check!(i32);
    check!(u32);
    check!(i64);
    check!(u64);
}

fn execute<I: Copy, O: Copy>(source: &str, inputs: &[I], sentinel: O) -> Option<Vec<O>> {
    let optimizer = shaders::optimizer()?;
    // Keep the same GPU-before-build lock order as the other GPU suites.
    let _lock = lock_gpu();
    let mut gpu = gpu()?;
    let module = support::module(&format!(
        r#"{source} intrinsic "pointer_index" fn device_index<T>(data: Ptr<T>, length: u64, index: u64) -> Ptr<T>;"#
    ));
    let project = support::project::Project::new(&module, None).unwrap();
    for shader in project.generated.shaders() {
        shaders::validate(shader.unoptimized_spirv());
    }
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
            .dispatch(root.device_pointer(), inputs.len() as u32, 1, 1)
            .unwrap();
        gpu.submit(commands).unwrap();
        Some(
            std::slice::from_raw_parts(output.host_pointer().cast::<O>(), inputs.len() + 1)
                .to_vec(),
        )
    }
}

#[test]
fn local_reference_helpers_preserve_aliases_fields_and_dynamic_elements() {
    let source = r#"
        export { kernel };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }
        struct Pair { first: u32, second: u32 }
        fn add_both(left: RefMut<u32>, right: RefMut<u32>, amount: u32) {
            left = left + amount;
            right = right + left;
        }
        fn increment(value: RefMut<u32>, amount: u32) {
            let alias: RefMut<u32> = value;
            add_both(value, alias, amount);
        }
        @compute_shader
        fn kernel(index: u64, root: Ptr<Root>) {
            if (index >= root.count) { return; };
            let input = device_index(root.inputs, root.count, index).*;
            let mut pair = Pair { first = input, second = u32(100) };
            increment(pair.first, u32(1));
            let mut values = [u32(10), u32(20)];
            increment(values:at_mut(index & u64(1)), u32(3));
            let output = device_index(root.outputs, root.count, index);
            output.* = pair.first + pair.second + values:at(u64(0)) + values:at(u64(1));
            increment(output.*, u32(4));
        }
    "#;
    let inputs = (0..65).collect::<Vec<u32>>();
    let Some(actual) = execute(source, &inputs, u32::MAX) else {
        return;
    };
    for (index, input) in inputs.iter().enumerate() {
        let array_sum = if index % 2 == 0 { 26 + 20 } else { 10 + 46 };
        assert_eq!(actual[index], 2 * (2 * (input + 1) + 100 + array_sum + 4));
    }
    assert_eq!(actual[inputs.len()], u32::MAX);
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
fn physical_byte_record_strides_preserve_neighboring_elements() {
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Bytes3 {
        a: u8,
        b: u8,
        c: u8,
    }
    assert_eq!((size_of::<Bytes3>(), align_of::<Bytes3>()), (3, 1));
    let inputs: Vec<_> = (0..67).map(|a| Bytes3 { a, b: 100, c: 212 }).collect();
    let sentinel = Bytes3 {
        a: 255,
        b: 255,
        c: 255,
    };
    let Some(actual) = execute(
        r#"export { kernel };
        import { "$/span.resin" };
        struct Bytes3 { a: u8, b: u8, c: u8, }
        struct Root { count: u64, inputs: Ptr<Bytes3>, outputs: Ptr<Bytes3>, }
        @compute_shader fn kernel(index: u64, root: Ptr<Root>)  {
            if (index < root.count) {
                let mut inputs = Span<Bytes3> { data = root.inputs, length = root.count };
                let mut outputs = Span<Bytes3> { data = root.outputs, length = root.count };
                let output: RefMut<_> = device_index(outputs.data, outputs.length, index).*;
                let input: Ref<Bytes3> = device_index(inputs.data, inputs.length, index).*;
                output = Bytes3 { a = input.a, b = input.b, c = input.c };
                output.b = output.b + u8(1);
            };
        }
    "#,
        &inputs,
        sentinel,
    ) else {
        return;
    };
    for (index, input) in inputs.iter().enumerate() {
        assert_eq!(
            actual[index],
            Bytes3 { b: 101, ..*input },
            "element {index}"
        );
    }
    assert_eq!(
        actual[inputs.len()],
        sentinel,
        "excess invocation must preserve canary"
    );
}

#[test]
fn mixed_record_copies_preserve_nested_byte_fields() {
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
fn an_unconditionally_failing_nested_loop_condition_stops_before_caller_stores() {
    let source = r#"export { kernel }; import { "$/span.resin" };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32>, }
        fn fail() -> bool  {
            let mut value: None;
            value = None;
            let mut test: bool;
            test = value!;
            test
        }
        fn condition() -> bool  { fail() }
        @compute_shader fn kernel(index: u64, root: Ptr<Root>)  {
            if (index < root.count) {
                let mut output = Span<u32> { data = root.outputs, length = root.count };
                let mut step: u32 = 0;
                while (step < u32(1)) {
                    while (condition()) { device_index(output.data, output.length, index).* = u32(1); };
                    device_index(output.data, output.length, index).* = u32(2);
                    step = step + u32(1);
                };
                device_index(output.data, output.length, index).* = u32(3);
            };
        }"#;
    let sentinel = 0xabcd1234_u32;
    let Some(actual) = execute(source, &[0_u32; 65], sentinel) else {
        return;
    };
    assert_eq!(actual, vec![sentinel; 66]);
}

#[test]
fn local_reference_calls_in_loops_propagate_failure_and_evaluate_indices_once() {
    let source = r#"export { kernel };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }
        fn position(calls: RefMut<u32>, index: u64) -> u64 { calls = calls + 1; index }
        fn increment(value: RefMut<u32>, fail: bool) {
            assert(!fail);
            value = value + 1;
        }
        @compute_shader fn kernel(index: u64, root: Ptr<Root>) {
            if (index >= root.count) { return; };
            let mut values = [u32(10), u32(20)];
            let mut calls: u32 = 0;
            let mut step: u32 = 0;
            while (step < u32(3)) {
                let selected: RefMut<u32> = values:at_mut(position(calls, index & u64(1)));
                increment(selected, index == u64(1) && step == u32(1));
                step = step + 1;
            };
            assert(calls == u32(3));
            device_index(root.outputs, root.count, index).* = values:at(u64(0)) + values:at(u64(1));
        }
    "#;
    let sentinel = u32::MAX;
    let Some(actual) = execute(source, &[0_u32; 65], sentinel) else {
        return;
    };
    for (index, value) in actual.iter().enumerate() {
        assert_eq!(
            *value,
            if index == 1 || index == 65 {
                sentinel
            } else {
                33
            }
        );
    }
}

#[test]
fn boolean_local_references_work_in_helpers_without_device_storage_layout() {
    let source = r#"
        export { kernel };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }
        struct Flags { first: bool, second: bool }
        fn toggle(value: RefMut<bool>) { value = !value; }
        fn update(value: RefMut<Flags>, index: u64) {
            toggle(value.first);
            let mut flags = [value.first, value.second];
            toggle(flags:at_mut(index & u64(1)));
            value.second = flags:at(u64(0)) && flags:at(u64(1));
        }
        @compute_shader
        fn kernel(index: u64, root: Ptr<Root>) {
            if (index >= root.count) { return; };
            let mut flags = Flags { first = false, second = true };
            update(flags, index);
            device_index(root.outputs, root.count, index).* = if (flags.first && !flags.second) { u32(42) } else { u32(0) };
        }
    "#;
    let Some(actual) = execute(source, &[0_u32; 65], u32::MAX) else {
        return;
    };
    assert_eq!(&actual[..65], &[42; 65]);
    assert_eq!(actual[65], u32::MAX);
}

#[test]
fn cooperative_map_reduce_runs_once_per_group_and_shares_results() {
    let source = r#"
        export { kernel };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }
        fn bump(value: RefMut<u32>) { value = value + u32(1); }
        @compute_shader
        fn kernel(group: u64, root: Ptr<Root>) {
            let output = device_index(root.outputs, root.count, group);
            bump(output.*);
            let base = device_index(root.inputs, root.count, group).*;
            let values = parallel_map([u32(0), u32(1), u32(2), u32(3), u32(4), u32(5)]) |x| { x + base };
            let total = parallel_reduce(values, u32(7)) |a, b| { a + b };
            output.* = output.* + total;
        }
    "#;
    if let Some(values) = execute(source, &[10u32, 20, 30, 40, 50], 0u32) {
        assert_eq!(values, [83, 143, 203, 263, 323, 0]);
    }
}

#[test]
fn cooperative_regions_handle_loops_large_arrays_and_nested_helpers() {
    let batch = (0..129)
        .map(|n| format!("u32({n})"))
        .collect::<Vec<_>>()
        .join(",");
    let source = format!(
        r#"
        export {{ kernel }};
        struct Root {{ count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }}
        fn read(value: Ref<u32>) -> u32 {{ value }}
        fn inner(value: u32) -> u32 {{
            let results = parallel_map([u32(1), u32(2)]) |x| {{ x * value }};
            parallel_reduce(results, u32(0)) |a, b| {{ a + b }}
        }}
        @compute_shader
        fn kernel(group: u64, root: Ptr<Root>) {{
            let base = device_index(root.inputs, root.count, group).*;
            let mut total = u32(0);
            let mut round = u32(0);
            while (round < u32(2)) {{
                let values = parallel_map([{batch}]) |x| {{
                    let nested = parallel_map([u32(0)]) |y| {{ x + y }};
                    inner(nested(0)) + read(base) + round
                }};
                if (round == u32(0)) {{ total = parallel_reduce(values, u32(0)) |a, b| {{ a + b }}; }}
                else {{ total = total + values(128); }};
                round = round + u32(1);
            }};
            let identity = u32(7);
            let one = parallel_reduce([u32(5)], u32(0)) |a, b| {{ a + b }};
            device_index(root.outputs, root.count, group).* = total + identity + one;
        }}
    "#
    );
    if let Some(values) = execute(&source, &[1u32, 2, 3], 0u32) {
        assert_eq!(values, [25295, 25425, 25555, 0]);
    }
}

#[test]
fn cooperative_failures_converge_before_barriers_and_leave_other_groups_running() {
    let source = r#"
        export { kernel };
        struct Root { count: u64, inputs: Ptr<u32>, outputs: Ptr<u32> }
        fn checked(value: u32, mode: u32) -> u32 {
            if (mode == u32(1) && value == u32(2)) { assert(false); };
            value + u32(1)
        }
        @compute_shader
        fn kernel(group: u64, root: Ptr<Root>) {
            let mode = device_index(root.inputs, root.count, group).*;
            let output = device_index(root.outputs, root.count, group);
            output.* = u32(1);
            if (mode == u32(2)) { assert(false); };
            let values = parallel_map([u32(0), u32(1), u32(2), u32(3), u32(4)]) |x| { checked(x, mode) };
            output.* = u32(2);
            let total = parallel_reduce(values, u32(0)) |a, b| {
                if (mode == u32(3) && a == u32(2)) { assert(false); };
                a + b
            };
            output.* = total;
        }
    "#;
    if let Some(values) = execute(source, &[0u32, 1, 2, 3, 0], 0u32) {
        assert_eq!(values, [15, 1, 1, 2, 15, 0]);
    }
}
