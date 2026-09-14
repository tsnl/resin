mod support;

const INDEX: &str = r#"
    intrinsic "pointer_index" def index<T>(data: Ptr<T>, length: ulong, position: ulong) -> Ptr<T>;
"#;

#[test]
fn generic_pointer_index_preserves_stride_mutation_and_host_bounds_diagnostics() {
    for (position, succeeds) in [(1, true), (3, false)] {
        let source = format!(
            r#"
            export {{ main }};
            {INDEX}
            def main() -> int = {{
                var values = [3_ul, 7_ul, 11_ul];
                index(values.at(0), 3, {position}).* := 42_ul;
                if (values.at(1).* == 42_ul) {{ 0 }} else {{ 1 }}
            }};
        "#
        );
        let output = support::project::Project::new(&support::module(&source), Some("main"))
            .unwrap()
            .run();
        assert_eq!(
            output.status.success(),
            succeeds,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if !succeeds {
            assert!(String::from_utf8_lossy(&output.stderr).contains("index"));
        }
    }
}
