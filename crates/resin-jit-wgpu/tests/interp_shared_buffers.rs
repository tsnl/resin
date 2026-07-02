//! Python `test_interp_shared_buffers` parity: explicit cross-program buffer copy.

use resin_core::F4;
use resin_dsl::param;
use resin_ir::IrProgram;
use resin_jit_wgpu::{build_wgpu_program, create_interp, AdmitProgram, Interp, InterpConfig};

#[test]
fn copy_buffer_to_buffer_across_programs() {
    let weights = param([2], F4, "weights");

    let mut train_ir = IrProgram::new();
    train_ir.register_param("weights", &weights).unwrap();
    train_ir.build_sink("out", &weights).unwrap();
    train_ir.seal_params().unwrap();
    let train_prog = build_wgpu_program(&train_ir, None);

    let mut eval_ir = IrProgram::new();
    eval_ir.register_param("weights", &weights).unwrap();
    eval_ir.build_sink("out", &weights).unwrap();
    eval_ir.seal_params().unwrap();
    let eval_prog = build_wgpu_program(&eval_ir, None);

    let mut interp = create_interp(InterpConfig::default()).expect("interp");
    let train_id = interp.admit_program(train_prog).unwrap();
    let eval_id = interp.admit_program(eval_prog).unwrap();

    let train_w = interp.param(train_id, "weights").unwrap();
    let eval_w = interp.param(eval_id, "weights").unwrap();

    let payload: Vec<u8> = [1.0f32, 2.0]
        .into_iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();
    interp.write_buffer(train_id, train_w, &payload).unwrap();
    interp
        .copy_buffer_to_buffer(train_id, train_w, eval_id, eval_w)
        .unwrap();
    let raw = interp.read_buffer(eval_id, eval_w).unwrap();
    let vals: Vec<f32> = raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(vals, vec![1.0, 2.0]);
}
