//! Device config / adapter selection.

use resin_jit_wgpu::{DeviceConfig, DeviceContext};

#[test]
fn unknown_device_name_errors() {
    let err = DeviceContext::from_config(&DeviceConfig::with_device_name(
        "nonexistent-adapter-name-xyz",
    ));
    let msg = match err {
        Ok(_) => panic!("should fail for unknown device_name"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("no GPU adapter") || msg.contains("nonexistent-adapter-name-xyz"),
        "msg={msg}"
    );
}

#[test]
fn default_config_constructs() {
    let _ = DeviceContext::from_config(&DeviceConfig::default()).expect("default adapter");
}
