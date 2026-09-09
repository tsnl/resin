#[allow(dead_code)]
mod common;

use resin_runtime::{ResinMemory, ResinStatus};

#[test]
fn compute_writes_are_visible_across_pipeline_switches() {
    let Some(mut gpu) = common::require_gpu() else {
        return;
    };
    let source = include_str!("synchronization.glsl");
    let Some(comp) = common::compile_shader(source, "comp", Some("COMPUTE")) else {
        return;
    };
    let Some(vert) = common::compile_shader(source, "vert", Some("VERTEX")) else {
        return;
    };
    let Some(frag) = common::compile_shader(source, "frag", Some("FRAGMENT")) else {
        return;
    };
    // SAFETY: all addresses/resources belong to gpu and remain live through its synchronous submit.
    unsafe {
        let compute = gpu.create_compute_pipeline(&comp).unwrap();
        let graphics = gpu.create_graphics_pipeline(&vert, &frag).unwrap();
        let root = gpu.malloc(16, 16, ResinMemory::Default).unwrap();
        root.host_pointer().cast::<[f32; 4]>().write([0.0; 4]);
        let mut commands = gpu.start_command_recording().unwrap();
        let mut outputs = Vec::new();
        for (width, height) in [(16, 16), (31, 7), (7, 31)] {
            let mut image = gpu.create_image(width, height).unwrap();
            let pixels = gpu
                .malloc((width * height * 4) as usize, 4, ResinMemory::Readback)
                .unwrap();
            commands.set_pipeline(&compute).unwrap();
            commands.dispatch(root.device_pointer(), 1, 1, 1).unwrap();
            commands.set_pipeline(&graphics).unwrap();
            commands.begin_rendering(&mut image, [0.0; 4]).unwrap();
            assert_eq!(
                commands.set_pipeline(&compute),
                Err(ResinStatus::InvalidArgument)
            );
            commands.draw(root.device_pointer(), 3).unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            outputs.push((image, pixels));
        }
        gpu.submit(commands).unwrap();
        for (_, pixels) in &outputs {
            for pixel in pixels.host_bytes().unwrap().chunks_exact(4) {
                assert_eq!(pixel, [255, 0, 0, 255]);
            }
        }
    }
}

#[test]
fn cancelled_transitions_do_not_affect_later_submissions() {
    let Some(mut gpu) = common::require_gpu() else {
        return;
    };
    // SAFETY: resources stay live, all recording/submission is sequential, reads follow the wait.
    unsafe {
        let mut image = gpu.create_image(4, 4).unwrap();
        let pixels = gpu.malloc(4 * 4 * 4, 4, ResinMemory::Readback).unwrap();
        let mut cancelled = gpu.start_command_recording().unwrap();
        cancelled
            .begin_rendering(&mut image, [1.0, 0.0, 0.0, 1.0])
            .unwrap();
        cancelled.end_rendering().unwrap();
        cancelled.copy_image_to_buffer(&mut image, &pixels).unwrap();
        drop(cancelled);

        let mut stale = gpu.start_command_recording().unwrap();
        stale
            .begin_rendering(&mut image, [1.0, 0.0, 0.0, 1.0])
            .unwrap();
        stale.end_rendering().unwrap();
        let mut commands = gpu.start_command_recording().unwrap();
        commands
            .begin_rendering(&mut image, [0.0, 1.0, 0.0, 1.0])
            .unwrap();
        commands.end_rendering().unwrap();
        commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
        gpu.submit(commands).unwrap();
        assert_eq!(gpu.submit(stale), Err(ResinStatus::InvalidArgument));
        for pixel in pixels.host_bytes().unwrap().chunks_exact(4) {
            assert_eq!(pixel, [0, 255, 0, 255]);
        }
    }
}

#[test]
fn timed_recordings_preserve_submission_and_cancellation() {
    let Some(mut gpu) = common::require_gpu() else {
        return;
    };
    // SAFETY: resources belong to this GPU, survive recordings, and are accessed
    // sequentially. Every host read follows synchronous submission.
    unsafe {
        let untimed = gpu.start_command_recording().unwrap();
        assert_eq!(gpu.submit_timed(untimed), Err(ResinStatus::InvalidArgument));
        let mut cancelled = match gpu.start_timed_command_recording() {
            Ok(commands) => commands,
            Err(ResinStatus::Unsupported) => {
                // Timestamps are optional even on an otherwise supported GPU.
                let commands = gpu.start_command_recording().unwrap();
                gpu.submit(commands).unwrap();
                return;
            }
            Err(status) => panic!("start_timed_command_recording failed: {status:?}"),
        };
        let mut image = gpu.create_image(4, 4).unwrap();
        let pixels = gpu.malloc(4 * 4 * 4, 4, ResinMemory::Readback).unwrap();
        cancelled.begin_rendering(&mut image, [1.0; 4]).unwrap();
        cancelled.end_rendering().unwrap();
        drop(cancelled);

        let mut invalid = gpu.start_timed_command_recording().unwrap();
        invalid.begin_rendering(&mut image, [1.0; 4]).unwrap();
        assert_eq!(gpu.submit_timed(invalid), Err(ResinStatus::InvalidArgument));

        for red in [0.0, 1.0] {
            let mut commands = gpu.start_timed_command_recording().unwrap();
            commands
                .begin_rendering(&mut image, [red, 0.0, 1.0, 1.0])
                .unwrap();
            commands.end_rendering().unwrap();
            commands.copy_image_to_buffer(&mut image, &pixels).unwrap();
            let _elapsed = gpu.submit_timed(commands).unwrap();
            for pixel in pixels.host_bytes().unwrap().chunks_exact(4) {
                assert_eq!(pixel, [(red * 255.0) as u8, 0, 255, 255]);
            }
        }
        // Callers may discard the result without leaving a pending query read.
        let commands = gpu.start_timed_command_recording().unwrap();
        gpu.submit(commands).unwrap();
    }
}
