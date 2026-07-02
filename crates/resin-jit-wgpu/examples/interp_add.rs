//! Smoke example: compile `a + b`, admit, write params, run, read sink.

use resin_core::F4;
use resin_dsl::{constant, param};
use resin_jit_wgpu::{compile_program, create_interp, AdmitProgram, Interp, InterpConfig};

fn f32_param_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Const path (no params): exercise `constant`.
    let sum_const =
        &constant([4], &[1.0f32, 2.0, 3.0, 4.0]) + &constant([4], &[10.0f32, 20.0, 30.0, 40.0]);
    let no_params: [(&str, &resin_dsl::View); 0] = [];
    let compiled_c = compile_program(&[("out", &sum_const)], &no_params, None)?;
    let mut interp = create_interp(InterpConfig::default())?;
    let pid_c = interp.admit_program(compiled_c.artifact)?;
    interp.run(pid_c)?;
    let const_out: Vec<f32> = interp
        .read_sink(pid_c, "out")?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(const_out, vec![11., 22., 33., 44.]);

    // Param path: host writes.
    let a = param([4], F4, "a");
    let b = param([4], F4, "b");
    let out = &a + &b;
    let compiled = compile_program(&[("out", &out)], &[("a", &a), ("b", &b)], None)?;
    let program_id = interp.admit_program(compiled.artifact)?;

    interp.write_param(program_id, "a", &f32_param_bytes(&[1., 2., 3., 4.]))?;
    interp.write_param(program_id, "b", &f32_param_bytes(&[10., 20., 30., 40.]))?;
    interp.run(program_id)?;

    let values: Vec<f32> = interp
        .read_sink(program_id, "out")?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    println!("out = {values:?}");
    assert_eq!(values, vec![11., 22., 33., 44.]);
    Ok(())
}
