use std::{fs, process::Command};

#[test]
fn service_binary_embeds_arbitrary_bytes_without_a_server() {
    let directory = tempfile::tempdir().unwrap();
    for bytes in [vec![], vec![0, 255, 13, 10, 128]] {
        let input = directory.path().join("source.bin");
        let output = directory.path().join("output.h");
        fs::write(&input, &bytes).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_resin-server"))
            .arg("--embed")
            .arg(&input)
            .arg("--symbol")
            .arg("sample")
            .arg("-o")
            .arg(&output)
            .env_remove("RESIN_SERVER")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let header = fs::read_to_string(output).unwrap();
        assert!(
            header.contains(&format!("#define sample_length UINT64_C({})", bytes.len())),
            "{header}"
        );
    }
}
