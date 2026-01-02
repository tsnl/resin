use std::{fs, path::PathBuf};

use zfw::*;

#[test]
fn basic_draw_3d_test() {
    let wgpu_instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(wgpu_instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let (device, queue) =
        pollster::block_on(adapter.unwrap().request_device(&wgpu::DeviceDescriptor {
            label: Some("BasicDraw3dTest.Device"),
            ..Default::default()
        }))
        .unwrap();

    let renderer = Draw3dRenderer::new(&device, &queue, [1024, 1024]);
    let mut frame = Draw3dFrame::new(&renderer);
    let readback_buffer =
        ReadbackBuffer::<[f32; 4]>::new(&device, 1024 * 1024, "BasicDraw3dTest.ReadbackBuffer");

    let scene = Draw3dScene::default();

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("BasicDraw2dTest.CommandEncoder"),
    });
    {
        renderer.record(&scene, &mut frame, &mut command_encoder);
        readback_buffer.copy_from_texture(frame.output_image(), &mut command_encoder);
    }
    _ = queue.submit(std::iter::once(command_encoder.finish()));

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let readback_data: Box<[[f32; 4]]> = readback_buffer.read();
    let readback_data: Box<[f32]> = bytemuck::cast_slice_box(readback_data);

    let tonemapped_readback_data = readback_data
        .into_iter()
        .map(|c| (c * 255.0) as u8)
        .collect();

    let output_path = PathBuf::from("output/draw_3d/basic_render_test.png");
    fs::create_dir_all(output_path.parent().unwrap()).unwrap();
    image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(1024, 1024, tonemapped_readback_data).unwrap(),
    )
    .save(output_path)
    .unwrap();
}
