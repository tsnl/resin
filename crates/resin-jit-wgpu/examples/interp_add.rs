//! Smoke: const and param paths via PipelineFactory.

use resin_core::F4;
use resin_core::{named, Empty};
use resin_dsl::{constant, param, View};
use resin_jit_wgpu::{compile_open, DeviceConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sum =
        &constant([4], &[1.0f32, 2.0, 3.0, 4.0]) + &constant([4], &[10.0f32, 20.0, 30.0, 40.0]);
    let factory = compile_open(
        &Empty::<View>::new(),
        &named("out", sum),
        DeviceConfig::default(),
        None,
    )?;
    let mut pipe = factory.create()?;
    let outs = pipe.call([])?;
    let const_out: Vec<f32> = outs["out"]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(const_out, vec![11., 22., 33., 44.]);

    let a = param([4], F4);
    let b = param([4], F4);
    let out = &a + &b;
    let factory = compile_open(
        &(named("a", a.clone()), named("b", b.clone())),
        &named("out", out),
        DeviceConfig::default(),
        None,
    )?;
    let mut pipe = factory.create()?;
    let a_bytes: Vec<u8> = [1f32, 2., 3., 4.]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let b_bytes: Vec<u8> = [10f32, 20., 30., 40.]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let outs = pipe.call([("a", a_bytes.as_slice()), ("b", b_bytes.as_slice())])?;
    let values: Vec<f32> = outs["out"]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("out = {values:?}");
    assert_eq!(values, vec![11., 22., 33., 44.]);
    Ok(())
}
