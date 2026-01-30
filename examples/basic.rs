use resin::expr::Expr;

async fn main_impl() {
    // Create wgpu device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an adapter");
    let (device, _queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Failed to create device");

    let x = Expr::new_vector(&device, &[1.0]) * Expr::new_vector(&device, &[2.0]);
    eprintln!("{:#?}", x);

    let lt = Expr::new_matrix(&device, &[[1.0, 0.0], [0.0, 1.0]]);
    let rt = Expr::new_matrix(&device, &[[2.0, 1.0], [0.0, 2.0]]);
    let x = lt.matmul(rt);
    eprintln!("{:#?}", x);
}

fn main() {
    pollster::block_on(main_impl());
}
