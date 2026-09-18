mod support;

const EXAMPLE: &str = include_str!("../examples/workgroup.resin");

#[test]
fn ordinary_host_calls_borrow_state_and_use_one_lane() {
    let module = support::module(EXAMPLE);
    let project = support::project::Project::new(&module, Some("host_main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn shared_state_helpers_emit_real_workgroup_storage_and_barriers() {
    let module = support::module(EXAMPLE);
    let project = support::project::Project::new(&module, None).unwrap();
    let shader = &project.generated.shaders()[0];
    support::shaders::validate(shader.unoptimized_spirv());
    let bytes = std::fs::read(shader.unoptimized_spirv()).unwrap();
    assert!(support::shaders::instructions(&bytes, 59).any(|args| args[2] == 4));
    assert!(support::shaders::instructions(&bytes, 224).count() >= 2);
}

#[test]
fn shader_operations_require_the_actual_shared_state_and_cannot_trap() {
    for (body, message) in [
        ("let mut local = u32(0); local:sync();", "shared state"),
        ("root.*:sync();", "shared state"),
        ("assert(index == 0); group:sync();", "checked operations"),
        ("root.* = u32(index); group:sync();", "checked operations"),
    ] {
        let source = format!(
            r#"import {{ "$/workgroup.resin" }};
            @compute_shader fn kernel(index: u64, root: PtrMut<u32>, group: Workgroup<u32>) {{ {body} }}"#
        );
        let module = support::module(&source);
        let error = match support::project::Project::new(&module, None) {
            Ok(_) => panic!("unsupported workgroup operation was accepted"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains(message), "{source}\n{error}");
    }
}

#[test]
fn invalid_workgroup_parameters_and_oversized_state_are_diagnosed() {
    for state in [
        "Ref<u32>".to_owned(),
        "PtrMut<u32>".to_owned(),
        format!("RefMut<({})>", vec!["u32"; 2049].join(",")),
    ] {
        let source = format!(
            "@compute_shader fn kernel(index: u64, root: PtrMut<u32>, group: {state}) {{}}"
        );
        let error = support::pipeline::source_module(&source)
            .unwrap_err()
            .to_string();
        let expected = if state.starts_with("RefMut") {
            "16 KiB"
        } else {
            "signature"
        };
        assert!(error.contains(expected), "{error}");
    }
}

#[cfg(feature = "gpu")]
#[test]
fn separate_workgroups_share_state_and_execute_without_changing_dispatch() {
    use resin_runtime::{ResinGpu, ResinMemory, ResinStatus, testing::lock_gpu};
    let _lock = lock_gpu();
    let mut gpu = match ResinGpu::create() {
        Ok(gpu) => gpu,
        Err(ResinStatus::Unsupported | ResinStatus::VulkanUnavailable) => {
            assert_ne!(std::env::var("RESIN_REQUIRE_GPU").as_deref(), Ok("1"));
            return;
        }
        Err(error) => panic!("GPU initialization failed: {error:?}"),
    };
    let optimizer = support::shaders::optimizer().unwrap();
    for (source, repeated) in [(EXAMPLE, false), (PHASES, true)] {
        let module = support::module(source);
        let project = support::project::Project::new(&module, None).unwrap();
        support::shaders::validate(project.generated.shaders()[0].unoptimized_spirv());
        let built = project
            .build(&support::toolchain::spirv(&optimizer))
            .unwrap();
        let shader = &project.generated.shaders()[0];
        let bytes = std::fs::read(built.path(shader.spirv().file_name().unwrap())).unwrap();
        #[repr(C)]
        struct Root {
            data: u64,
            length: u64,
        }
        let width = gpu.compute_workgroup_size() as usize;
        let count = width * 3;
        // All buffers remain live and mapped until synchronous submission completes.
        unsafe {
            let pipeline = gpu.create_compute_pipeline(&bytes).unwrap();
            let output = gpu.malloc(count * 4, 4, ResinMemory::Default).unwrap();
            std::slice::from_raw_parts_mut(output.host_pointer().cast::<f32>(), count).fill(-1.0);
            let root = gpu
                .malloc(size_of::<Root>(), align_of::<Root>(), ResinMemory::Default)
                .unwrap();
            root.host_pointer().cast::<Root>().write(Root {
                data: output.device_pointer(),
                length: count as u64,
            });
            let mut commands = gpu.start_command_recording().unwrap();
            commands.set_pipeline(&pipeline).unwrap();
            commands.dispatch(root.device_pointer(), 3, 1, 1).unwrap();
            gpu.submit(commands).unwrap();
            let values = std::slice::from_raw_parts(output.host_pointer().cast::<f32>(), count);
            for (index, &value) in values.iter().enumerate() {
                let expected = if repeated {
                    3.0
                } else if index % width == 0 {
                    (index * 2 + 3) as f32
                } else {
                    -1.0
                };
                assert_eq!(value, expected, "output {index}");
            }
        }
    }
    // Exercise Resin's typed pipeline and launch projection too: the host supplies
    // only the root record; shared state is allocated by the shader wrapper.
    let module = support::module(TYPED_PIPELINE);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let output = project.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

const TYPED_PIPELINE: &str = r#"
    export { main };
    import { "$/workgroup.resin", "$/gpu.resin" };
    struct Root<T> { output: T }
    @compute_shader
    fn kernel(index: u64, root: PtrMut<Root<PtrMut<u32>>>, group: Workgroup<u32>) {
        if (group:lane_index() == 0) { group = u32(41); };
        group:sync();
        if (group:lane_index() == 0) { root.output.* = group + u32(1); };
    }
    fn main() -> i32 | Err<_> {
        let gpu = gpu_new()?;
        let output = gpu:create(u32(0))?;
        let pipeline = gpu:create_compute_pipeline(kernel)?;
        let commands = gpu:start_command_recording()?;
        commands:dispatch(pipeline, Root<_> { output = output }, 1, 1, 1)?;
        commands:submit()?;
        assert(output:load() == u32(42));
        0
    }
"#;

const PHASES: &str = r#"
    export { kernel, main };
    import { "$/workgroup.resin", "$/span.resin" };
    struct State { value: u32, ready: bool }
    struct Root { output: SpanMut<f32> }
    fn phase(group: Workgroup<State>) {
        if (group:lane_index() == 0) {
            group.value = group.value + u32(1);
            group.ready = true;
        };
        group:sync();
    }
    @compute_shader
    fn kernel(index: u64, root: PtrMut<Root>, group: Workgroup<State>) {
        let mut round = u32(0);
        while (round < u32(3)) { phase(group); round = round + u32(1); };
        root.output:at_mut(index) = if (group.ready) { f32(group.value) } else { f32(0) };
    }
    fn main() -> i32 {
        let mut state = State { value = u32(7), ready = false };
        phase(state); phase(state); phase(state);
        assert(state.value == u32(10) && state.ready);
        0
    }
"#;

#[test]
fn host_calls_preserve_the_callers_initial_state() {
    let module = support::module(PHASES);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    assert!(project.run().status.success());
}
