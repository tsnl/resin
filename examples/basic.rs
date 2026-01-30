use resin::expr::Expr;
use resin::interp::Interp;

async fn main_impl() {
    // Create wgpu device
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find an adapter");
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .expect("Failed to create device");

    // Create the interpreter
    let interp = Interp::new(&device, &queue);

    // Build and evaluate a matmul expression: identity * [[2,1],[0,2]]
    let lt = Expr::new_matrix(&device, &[[1.0, 0.0], [0.0, 1.0]]);
    let rt = Expr::new_matrix(&device, &[[2.0, 1.0], [0.0, 2.0]]);
    let expr = lt.matmul(rt);

    // Evaluate and readback
    let result_buf = interp.eval(&expr);
    let result = interp.readback(&result_buf, expr.numel());

    println!("Shape: {:?}", expr.shape);
    println!("Result: {:?}", result);
}

fn main() {
    pollster::block_on(main_impl());
}
