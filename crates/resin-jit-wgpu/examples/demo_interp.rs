//! E2E: compose a simple const graph and interpret it on the GPU.

use resin_core::{named, Empty};
use resin_dsl::{constant, View};
use resin_jit_wgpu::{compile_open, DeviceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = constant([3], &[1.0f32, 2.0, 3.0]);
    let factory = compile_open(
        &Empty::<View>::new(),
        &named("out", t),
        DeviceConfig::default(),
        None,
    )?;
    let mut pipe = factory.create()?;
    let outs = pipe.call([])?;
    let values: Vec<f32> = outs["out"]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    eprintln!("sink values: {values:?}");
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
    Ok(())
}
