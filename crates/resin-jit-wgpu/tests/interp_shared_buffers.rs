//! Two pipeline instances can hold independent buffer state.

use resin_core::F4;
use resin_dsl::param;
use resin_jit_wgpu::{compile, DeviceConfig};

#[test]
fn two_instances_independent_buffers() {
    let weights = param([2], F4, "weights");
    let factory = compile(
        &[("weights", &weights)],
        &[("out", &weights)],
        DeviceConfig::default(),
        None,
    )
    .expect("compile");
    let mut a = factory.create().unwrap();
    let mut b = factory.create().unwrap();

    let payload_a: Vec<u8> = [1.0f32, 2.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let payload_b: Vec<u8> = [3.0f32, 4.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();

    let out_a = a.call([("weights", payload_a.as_slice())]).unwrap();
    let out_b = b.call([("weights", payload_b.as_slice())]).unwrap();
    let va: Vec<f32> = out_a["out"]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let vb: Vec<f32> = out_b["out"]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(va, vec![1.0, 2.0]);
    assert_eq!(vb, vec![3.0, 4.0]);
}
