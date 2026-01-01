use std::iter;

use zfw::*;

fn main() {
    let wgpu_instance = wgpu::Instance::new(&Default::default());
    let adapter = pollster::block_on(wgpu_instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }));
    let (device, queue) =
        pollster::block_on(adapter.unwrap().request_device(&wgpu::DeviceDescriptor {
            label: Some("TestDevice"),
            ..Default::default()
        }))
        .unwrap();

    let renderer = Draw2dRenderer::create(&device, &queue, [256, 256]);
    let mut frame = Draw2dFrame::new(&device, [256, 256]);
    let readback_buffer = ReadbackBuffer::<[u8; 4]>::new(&device, 256 * 256, "TestReadbackBuffer");

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("TestCommandEncoder"),
    });
    {
        let quads = vec![Draw2dQuad {
            dst_xy_px: [0, 0],
            dst_wh_px: Some([128, 128]),
            fill_color_rgba: [1.0, 0.0, 0.0, 1.0],
            ..Default::default()
        }];
        renderer.record(&quads, &mut frame, &mut command_encoder);

        readback_buffer.copy_from_texture(frame.output_image(), &mut command_encoder);
    }
    queue.submit(iter::once(command_encoder.finish()));

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(
            256,
            256,
            bytemuck::cast_vec(readback_buffer.read().into_vec()),
        )
        .unwrap(),
    )
    .save("test_output.png")
    .unwrap();
}
