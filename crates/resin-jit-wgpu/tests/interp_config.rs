//! Python `test_interp_config` parity (WGPU `InterpConfig`).

use resin_jit_wgpu::{create_interp, InterpConfig};

#[test]
fn unknown_device_name_errors() {
    let err = create_interp(InterpConfig::with_device_name(
        "nonexistent-adapter-name-xyz",
    ));
    let msg = match err {
        Ok(_) => panic!("should fail for unknown device_name"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("no GPU adapter found with device_name")
            || msg.contains("nonexistent-adapter-name-xyz"),
        "msg={msg}"
    );
}

#[test]
fn default_config_constructs() {
    let _ = create_interp(InterpConfig::default()).expect("default wgpu adapter");
}
