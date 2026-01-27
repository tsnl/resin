use resin::interp::Interpreter;
use resin::tensor::Tensor;

fn main() {
    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        .expect("Failed to find GPU adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None))
        .expect("Failed to create device");

    let mut interp = Interpreter::new(device, queue);

    // Create constant tensors
    let a: Tensor = (&[1.0f32, 2.0, 3.0, 4.0][..]).into();
    let b: Tensor = (&[5.0f32, 6.0, 7.0, 8.0][..]).into();
    let c: Tensor = (&[2.0f32, 2.0, 2.0, 2.0][..]).into();

    // Build an expression graph: (a + b) * c
    let expr = (a + b) * c;

    println!("Expression (before eval):");
    println!("{:#?}", expr);

    // Evaluate to get a constant Tensor
    let result = interp.eval(&expr);

    println!("\nResult (after eval):");
    println!("{:#?}", result);

    // Extract and print the actual values
    if let resin::tensor::TensorDetail::Constant(data) = &result.detail {
        let values: &[f32] = bytemuck::cast_slice(data);
        println!("\nValues: {:?}", values);
        // Expected: (1+5)*2=12, (2+6)*2=16, (3+7)*2=20, (4+8)*2=24
    }

    // Demonstrate 2D array formatting
    println!("\n--- 2D Matrix Example ---");
    let matrix: Tensor = (&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0][..]).into();
    let matrix = matrix.reshape(&[2, 3]);
    let matrix = interp.eval(&matrix);

    println!("Compact ({{:?}}):");
    println!("{:?}", matrix);

    println!("\nPretty ({{:#?}}):");
    println!("{:#?}", matrix);
}
