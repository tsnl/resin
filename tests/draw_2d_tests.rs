use std::{fs, iter, path::PathBuf};

use zfw::*;

#[test]
fn basic_draw_2d_test() {
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

    let rainbow_image = image::open("tests/data/rainbow-512x512.png").unwrap();
    let rainbow_texture = Rgba8UnormTexture::new(&device, [512, 512], "TestRainbowImage");
    queue.write_texture(
        rainbow_texture.texel_copy_texture_info(),
        rainbow_image.as_bytes(),
        rainbow_texture.texel_copy_buffer_layout(),
        rainbow_texture.size(),
    );
    // FIXME: need to convert loaded textures to linear colorspace for correct alpha blending

    // TODO: add a test-case to ensure our sorting logic works as expected

    let renderer = Draw2dRenderer::create(&device, &queue, [1024, 1024]);
    let mut frame = Draw2dFrame::new(&device, [1024, 1024]);
    let readback_buffer =
        ReadbackBuffer::<[u8; 4]>::new(&device, 1024 * 1024, "BasicDraw2dTest.ReadbackBuffer");

    let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("BasicDraw2dTest.CommandEncoder"),
    });
    {
        let quads = vec![
            Draw2dQuad {
                dst_xy_px: [10, 10],
                dst_wh_px: Some([497, 497]),
                fill_color_rgba: [1.0, 0.0, 0.0, 1.0],
                ..Default::default()
            },
            Draw2dQuad {
                dst_xy_px: [517, 10],
                dst_wh_px: Some([497, 497]),
                fill_texture: Some(rainbow_texture.wgpu_texture().clone()),
                fill_color_rgba: [1.0, 1.0, 1.0, 1.0],
                ..Default::default()
            },
            Draw2dQuad {
                dst_xy_px: [20, 527],
                dst_wh_px: Some([477, 477]),
                fill_color_rgba: [0.0, 1.0, 0.0, 1.0],
                border_thickness_px: [5, 10, 15, 20],
                border_color_rgba: [0.0, 0.5, 0.0, 1.0],
                ..Default::default()
            },
        ];
        renderer.record(&quads, &mut frame, &mut command_encoder);

        readback_buffer.copy_from_texture(frame.output_image(), &mut command_encoder);
    }
    queue.submit(iter::once(command_encoder.finish()));

    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

    let output_path = PathBuf::from("output/draw_2d/basic_render_test.png");
    fs::create_dir_all(output_path.parent().unwrap()).unwrap();
    image::DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(
            1024,
            1024,
            bytemuck::cast_vec(readback_buffer.read().into_vec()),
        )
        .unwrap(),
    )
    .save(output_path)
    .unwrap();
}
