#[allow(dead_code)]
mod support;

#[test]
fn window_library_probe() {
    let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/window.resin");
    support::pipeline::host_entry(&source, "main").unwrap_or_else(|error| panic!("{error}"));
}
