mod common;

use resin_runtime::{ResinMemory, ResinStatus};

const GLSL: &str = include_str!("hello_triangle.glsl");
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

#[test]
fn hello_triangle() {
    // SAFETY: resources share one GPU, outlive submission, and are read only after its wait.
    unsafe {
        let Some(mut gpu) = common::require_gpu() else {
            return;
        };
        let Some(vert) = common::compile_shader(GLSL, "vert", Some("VERTEX")) else {
            return;
        };
        let Some(frag) = common::compile_shader(GLSL, "frag", Some("FRAGMENT")) else {
            return;
        };

        assert!(matches!(
            gpu.create_graphics_pipeline(&vert, &[]),
            Err(ResinStatus::InvalidArgument)
        ));
        let mut unaligned = vec![0];
        unaligned.extend_from_slice(&vert);
        let pipeline = gpu
            .create_graphics_pipeline(&unaligned[1..], &frag)
            .expect("graphics pipeline");
        let mut image = gpu.create_image(WIDTH, HEIGHT).expect("image");
        let pixels = gpu
            .malloc(
                WIDTH as usize * HEIGHT as usize * 4,
                4,
                ResinMemory::Readback,
            )
            .expect("malloc");

        let mut commands = gpu.start_command_recording().expect("record");
        commands
            .begin_rendering(&mut image, [0.0, 0.0, 0.0, 1.0])
            .expect("begin rendering");
        commands.set_pipeline(&pipeline).expect("set pipeline");
        commands.draw(0, 3).expect("draw");
        commands.end_rendering().expect("end rendering");
        commands
            .copy_image_to_buffer(&mut image, &pixels)
            .expect("copy");
        gpu.submit(commands).expect("submit");

        let host = pixels.host_bytes().expect("mapped readback");
        // Rasterization may round RGB by one UNORM step across drivers; alpha stays exact.
        common::assert_reftest(file!(), WIDTH, HEIGHT, host, 1);
    }
}
