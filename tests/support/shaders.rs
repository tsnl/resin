use std::{ffi::OsString, io, process::Command};

pub fn compiler() -> Option<OsString> {
    let explicit = std::env::var_os("GLSLC");
    let compiler = explicit.clone().unwrap_or_else(|| "glslc".into());
    match Command::new(&compiler).arg("--version").output() {
        Ok(output) => assert!(output.status.success(), "glslc --version failed"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            assert!(
                explicit.is_none()
                    && std::env::var("RESIN_REQUIRE_GLSLC").as_deref() != Ok("1")
                    && std::env::var("RESIN_REQUIRE_GPU").as_deref() != Ok("1"),
                "glslc is required but unavailable: {error}"
            );
            eprintln!("skipping: glslc not found");
            return None;
        }
        Err(error) => panic!("cannot run glslc: {error}"),
    }
    Some(compiler)
}
