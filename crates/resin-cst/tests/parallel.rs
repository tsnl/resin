mod common;

#[tokio::test]
async fn parallel_blocks_are_not_lambda_values_or_ordinary_calls() {
    for expression in [
        "parallel_map([1, 2]) |x| { x + 1 }",
        "parallel_reduce([1, 2], 0) |mut a, b| { a = a + b; a }",
        "parallel_map([1]) |x| { parallel_map([2]) |y| { x + y } }",
    ] {
        let source = format!("fn f() {{ let result = {expression}; }}");
        let document = common::parse(&source, None).await;
        assert!(!document.tree().root_node().has_error(), "{source}");
        let formatted = resin_cst::format_source(&source).unwrap();
        assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
        assert!(
            formatted.contains("|x|") || formatted.contains("|mut a, b|"),
            "{formatted}"
        );
    }
    for expression in [
        "parallel_map([1])",
        "parallel_map([1]) |x, y| { x }",
        "parallel_reduce([1], 0) |x| { x }",
        "|x| { x }",
    ] {
        let source = format!("fn f() {{ let result = {expression}; }}");
        assert!(
            common::parse(&source, None)
                .await
                .tree()
                .root_node()
                .has_error(),
            "{source}"
        );
    }
}
