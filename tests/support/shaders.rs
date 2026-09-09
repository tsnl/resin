#![allow(dead_code)]
use std::{ffi::OsString, io, process::Command};

pub fn optimizer() -> Option<OsString> {
    let explicit = std::env::var_os("SPIRV_OPT");
    let compiler = explicit.clone().unwrap_or_else(|| "spirv-opt".into());
    match Command::new(&compiler).arg("--version").output() {
        Ok(output) => assert!(output.status.success(), "spirv-opt --version failed"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            assert!(
                explicit.is_none()
                    && std::env::var("RESIN_REQUIRE_SPIRV_TOOLS").as_deref() != Ok("1")
                    && std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "spirv-opt is required but unavailable: {error}"
            );
            eprintln!("skipping: spirv-opt not found");
            return None;
        }
        Err(error) => panic!("cannot run spirv-opt: {error}"),
    }
    Some(compiler)
}
fn validator() -> Option<OsString> {
    let explicit = std::env::var_os("SPIRV_VAL");
    let compiler = explicit.clone().unwrap_or_else(|| "spirv-val".into());
    match Command::new(&compiler).arg("--version").output() {
        Ok(output) => assert!(output.status.success(), "spirv-val --version failed"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            assert!(
                explicit.is_none()
                    && std::env::var("RESIN_REQUIRE_SPIRV_TOOLS").as_deref() != Ok("1")
                    && std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "spirv-val is required but unavailable: {error}"
            );
            eprintln!("skipping: spirv-val not found");
            return None;
        }
        Err(error) => panic!("cannot run spirv-val: {error}"),
    }
    Some(compiler)
}

pub fn validate(path: &std::path::Path) {
    let Some(validator) = validator() else {
        return;
    };
    let output = Command::new(validator)
        .args(["--target-env", "vulkan1.3"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{} ({}): {}{}",
        path.display(),
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Inspect instruction structure without coupling tests to generated IDs or names.
pub fn instructions(bytes: &[u8], opcode: u32) -> impl Iterator<Item = Vec<u32>> {
    assert_eq!(bytes.len() % 4, 0);
    let words = bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    let mut matches = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count > 0 && offset + count <= words.len());
        if words[offset] & 0xffff == opcode {
            matches.push(words[offset + 1..offset + count].to_vec());
        }
        offset += count;
    }
    matches.into_iter()
}
