#[allow(dead_code)]
mod support;

#[test]
fn source_float_images_keep_ownership_and_validate_bounded_writes() {
    let source = r#"export { main };
    import { "$/image.resin", "$/shared.resin", "$/span.resin", "$/status.resin" };
    fn main() -> () | Err<_> {
        let samples = arc_span_alloc::<f32>(8, 0)?;
        let view = samples:get();
        view:at_mut(0) = 16;
        view:at_mut(1) = -0.5;
        view:at_mut(3) = 0.25;
        view:at_mut(7) = 1;
        let name = bytes("samples.exr.not-part-of-the-path");
        let path = name:slice(0, 11);
        image_data_write_exr_pixels(path, 2, 1, view:read_only())?;
        let retained = { let loaded = image_data_read_exr(path)?; loaded:clone() };
        assert(retained:width() == 2 && retained:height() == 1 && retained:channels() == 4);
        let values = retained:pixels();
        assert(values.length == 8 && values:at(0) == 16 && values:at(1) == -0.5 && values:at(3) == 0.25);
        let rejected = match (image_data_write_exr_pixels(bytes("samples.exr"), 3, 1, view:read_only())) {
            ()(value) => { false }, Err(error) => { true },
        };
        assert(rejected);
        let unchanged = image_data_read_exr(bytes("samples.exr"))?;
        assert(unchanged:width() == 2);
        retained:write_exr(bytes("copy.exr"))?;
        match (image_data_read_exr(bytes("samples.exr\0ignored"))) {
            Err(error) => { assert(runtime_status_code(error) == 1); },
            FloatImageData(_) => { assert(false); },
        };
        match (retained:write_exr(bytes("samples.exr\0ignored"))) {
            Err(error) => { assert(runtime_status_code(error) == 1); },
            ()(_) => { assert(false); },
        };
    }"#;
    let module = support::module(source);
    let project = support::project::Project::new(&module, Some("main")).unwrap();
    let executable = project.build_executable();
    let directory = tempfile::tempdir().unwrap();
    let output = std::process::Command::new(executable.path())
        .current_dir(directory.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let image = resin_runtime::image_read_exr(directory.path().join("copy.exr")).unwrap();
    assert_eq!(image.pixels[0..4], [16.0, -0.5, 0.0, 0.25]);
}
