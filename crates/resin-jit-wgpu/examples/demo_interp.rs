//! E2E: compose a simple const graph and interpret it on the GPU (Python `demo_interp`).

use resin_dsl::constant;
use resin_ir::IrProgram;
use resin_jit_wgpu::{build_wgpu_program, create_interp, AdmitProgram, Interp, InterpConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t = constant([3], &[1.0f32, 2.0, 3.0]);

    let mut ir = IrProgram::new();
    ir.build_sink("out", &t)?;
    ir.seal_params()?;
    let artifact = build_wgpu_program(&ir, None);

    let mut interp = create_interp(InterpConfig::default())?;
    let program_id = interp.admit_program(artifact)?;
    interp.run(program_id)?;

    let values: Vec<f32> = interp
        .read_sink(program_id, "out")?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    eprintln!("sink values: {values:?}");
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
    Ok(())
}
