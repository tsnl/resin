//! Header adapters are reusable independently of Resin bodies and preserve scalar C ABI.
#[allow(dead_code)]
mod support;
use std::{fs, path::Path, process::Command};
use support::service::Service;

fn build(service: &Service, source: &Path, output: &Path) -> i32 {
    let result = service
        .command()
        .arg(source)
        .arg("-o")
        .arg(output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Command::new(output)
        .output()
        .unwrap()
        .status
        .code()
        .unwrap()
}

#[test]
fn adapter_cache_survives_body_edits_and_rebuilds_changed_transitive_headers() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    let source = client.path().join("main.resin");
    let headers = client.path().join("include");
    fs::create_dir(&headers).unwrap();
    let header = headers.join("api.h");
    let detail = headers.join("value.h");
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    fs::write(
        &header,
        "#include \"value.h\"\nstatic inline int answer(void) { return VALUE; }\n",
    )
    .unwrap();
    fs::write(&detail, "#define VALUE 40\n").unwrap();
    let write = |tail| {
        fs::write(&source, format!("export {{ main }}; extern {{ \"include/api.h\": {{ def answer() -> int; }} }}; def main() -> int = {{ answer() + {tail} }};")).unwrap()
    };
    write(1);
    assert_eq!(build(&service, &source, &output), 41);
    let initial = service.server.counters();
    assert_eq!(initial.foreign_builds, 1);
    write(2);
    assert_eq!(build(&service, &source, &output), 42);
    let edited = service.server.counters();
    assert_eq!(edited.foreign_builds, initial.foreign_builds);
    assert_eq!(
        edited.native_object_builds,
        initial.native_object_builds + 1
    );
    fs::write(&detail, "#define VALUE 41\n").unwrap();
    assert_eq!(build(&service, &source, &output), 43);
    assert_eq!(
        service.server.counters().foreign_builds,
        edited.foreign_builds + 1
    );
}

#[test]
fn scalar_adapters_handle_narrow_signed_unsigned_bool_pointer_and_void() {
    let service = Service::new();
    let client = tempfile::tempdir().unwrap();
    fs::write(
        client.path().join("api.h"),
        r#"
        static inline signed char negative(signed char value) { return value; }
        static inline unsigned short wide(unsigned short value) { return value; }
        static inline _Bool truth(_Bool value) { return value; }
        static inline void store(int *value) { *value = 42; }
        #define sum(a, b) ((a) + (b))
    "#,
    )
    .unwrap();
    let source = client.path().join("main.resin");
    fs::write(
        &source,
        r#"
        export { main };
        extern { "api.h": {
            def negative(value: sbyte) -> sbyte;
            def wide(value: ushort) -> ushort;
            def truth(value: bool) -> bool;
            def store(value: Ptr<int>);
            def sum(a: int, b: int) -> int;
        } };
        def main() -> int = {
            var value = 0_i;
            store(&value);
            if (negative(-7_b) == -7_b && wide(65000_uh) == 65000_uh && truth(1 == 1)) {
                sum(value, 0)
            } else { 1 }
        };
    "#,
    )
    .unwrap();
    let output = client
        .path()
        .join(format!("program{}", std::env::consts::EXE_SUFFIX));
    assert_eq!(build(&service, &source, &output), 42);
}
