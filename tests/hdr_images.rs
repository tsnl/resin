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
        image_data_write_exr_pixels("samples.exr".data, 2, 1, view)?;
        let retained = { let loaded = image_data_read_exr("samples.exr".data)?; loaded:clone() };
        assert(retained:width() == 2 && retained:height() == 1 && retained:channels() == 4);
        let values = retained:pixels();
        assert(values.length == 8 && values:at(0) == 16 && values:at(1) == -0.5 && values:at(3) == 0.25);
        let rejected = match (image_data_write_exr_pixels("samples.exr".data, 3, 1, view)) {
            ()(value) => { false }, Err(error) => { true },
        };
        assert(rejected);
        let unchanged = image_data_read_exr("samples.exr".data)?;
        assert(unchanged:width() == 2);
        retained:write_exr("copy.exr".data)?;
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
