mod common;

use resin_runtime::ResinMemory;

const GLSL: &str = include_str!("compute.glsl");
const WIDTH: u32 = 256;
const HEIGHT: u32 = 256;

#[repr(C)]
struct Root {
    width: u32,
    height: u32,
    pixels: u64,
}

#[test]
fn compute_gradient() {
    let Some(mut gpu) = common::require_gpu() else {
        return;
    };
    let Some(spv) = common::compile_shader(GLSL, "comp", None) else {
        return;
    };

    let pipeline = gpu.create_compute_pipeline(&spv).expect("compute pipeline");
    let pixels = gpu
        .malloc(
            WIDTH as usize * HEIGHT as usize * 4,
            4,
            ResinMemory::Default,
        )
        .expect("malloc pixels");
    let root = gpu
        .malloc(size_of::<Root>(), 8, ResinMemory::Default)
        .expect("malloc root");

    let pixels_device = gpu
        .host_to_device(pixels.host_pointer())
        .expect("pixels device address");
    unsafe {
        root.host_pointer().cast::<Root>().write(Root {
            width: WIDTH,
            height: HEIGHT,
            pixels: pixels_device,
        });
    }
    let root_device = gpu
        .host_to_device(root.host_pointer())
        .expect("root device address");

    let mut commands = gpu.start_command_recording().expect("record");
    commands.set_pipeline(&pipeline).expect("set pipeline");
    commands
        .dispatch(root_device, WIDTH.div_ceil(8), HEIGHT.div_ceil(8), 1)
        .expect("dispatch");
    gpu.submit(commands).expect("submit");

    let host = pixels.host_bytes().expect("mapped default allocation");
    common::assert_reftest(file!(), WIDTH, HEIGHT, host);
}
