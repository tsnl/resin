//! Smoke example: compile `a + b`, admit, write params, run, read sink.

use resin_compile::compile_program;
use resin_core::F4;
use resin_dsl::param;
use resin_jit_wgpu::{create_interp, parse_backend, BufferId, InterpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a = param([4], F4, "a");
    let b = param([4], F4, "b");
    let out = &a + &b;

    let compiled = compile_program(&[("out", &out)], &[("a", &a), ("b", &b)], None)?;
    let mut interp = create_interp(parse_backend("wgpu")?, InterpConfig::Nil)?;
    let program_id = interp.admit_program(compiled.artifact.clone())?;

    let a_bytes: Vec<u8> = [1f32, 2., 3., 4.]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    let b_bytes: Vec<u8> = [10f32, 20., 30., 40.]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();

    let a_idx = compiled.artifact.param_buffers["a"];
    let b_idx = compiled.artifact.param_buffers["b"];
    interp.write_buffer(program_id, BufferId(a_idx), &a_bytes)?;
    interp.write_buffer(program_id, BufferId(b_idx), &b_bytes)?;
    interp.run(program_id)?;

    let sink_view = compiled.artifact.sinks["out"];
    let buf_idx = compiled.artifact.buffer_views[sink_view].buffer_index;
    let out_bytes = interp.read_buffer(program_id, BufferId(buf_idx))?;
    let values: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("out = {values:?}");
    assert_eq!(values, vec![11., 22., 33., 44.]);
    Ok(())
}
