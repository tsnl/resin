use resin_cst::Node;
mod common;
use common::parse;

fn nodes(node: Node<'_>) -> Vec<(String, usize, usize)> {
    let mut result = vec![(node.kind().into(), node.start_byte(), node.end_byte())];
    for child in node.children(&mut node.walk()) {
        result.extend(nodes(child));
    }
    result
}

#[tokio::test]
async fn incremental_edits_match_fresh_parsing_including_utf8_boundaries() {
    let texts = [
        "",
        "// é",
        "// ê\n",
        "// 🌲\n",
        "// 🌳\r\n",
        "fn main()  { print(\"é🌲\") }",
        "fn main()  { print(\"ê🌳\") }",
        "fn main() { print(\"ê🌳\")",
        "fn main() { print(\"",
        "fn main()  { let mut value: i32 = 4; value }",
    ];
    for before in texts {
        let original = parse(before, None).await;
        for after in texts {
            let incremental = parse(after, Some(&original)).await;
            let fresh = parse(after, None).await;
            assert_eq!(
                nodes(incremental.tree().root_node()),
                nodes(fresh.tree().root_node())
            );
            assert_eq!(original.source(), before);
        }
    }
}

#[tokio::test]
async fn queries_reject_invalid_utf8_and_out_of_range_offsets() {
    let source = "// é🌲\nfn main()  {}";
    let document = parse(source, None).await;
    for offset in 0..=source.len() + 1 {
        let token = document.token(offset);
        let types = document.type_context(offset);
        if !source.is_char_boundary(offset) {
            assert!(token.is_none());
            assert!(!types);
        }
    }
    assert!(document.token(usize::MAX).is_none());
    assert!(!document.type_context(usize::MAX));
}

#[tokio::test]
async fn delimiter_recovery_preserves_source_and_ignores_string_contents() {
    let source = "fn main() { print(\"[({é\")";
    let document = parse(source, None).await;
    let recovered = document
        .recovery()
        .expect("the block needs a closing brace");
    assert!(recovered.source().starts_with(source));
    assert!(!recovered.tree().root_node().has_error());
    assert!(document.tree().root_node().has_error());
    assert_eq!(document.source(), source);
    assert!(recovered.recovery().is_none());
}

#[test]
fn formatting_needs_no_semantic_context() {
    let source = "fn unknown()  { missing_name(42) }";
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(resin_cst::format_source("fn broken( = {").is_none());
}

#[test]
fn struct_fields_follow_delimiter_formatting() {
    for (source, expected) in [
        ("struct Empty{}", "struct Empty {}\n"),
        (
            "struct Point<T>{x:T,y:T}",
            "struct Point<T> { x: T, y: T }\n",
        ),
        (
            "struct Point<T>{x:T,y:T,}",
            "struct Point<T> {\n\tx: T,\n\ty: T,\n}\n",
        ),
    ] {
        assert_eq!(resin_cst::format_source(source).as_deref(), Some(expected));
        assert_eq!(
            resin_cst::format_source(expected).as_deref(),
            Some(expected)
        );
    }
}

#[tokio::test]
async fn source_wrapper_names_preserve_token_queries_and_formatting() {
    let source = "fn first(values: GpuSpan<u32>) -> GpuPtr<u32>  { values:at(u64(0)) }";
    let document = parse(source, None).await;
    assert!(!document.tree().root_node().has_error());
    for former in ["GpuSpan", "GpuPtr"] {
        let offset = source.find(former).unwrap() + 2;
        assert_eq!(document.token(offset).unwrap().kind(), "uid");
        assert!(document.type_context(offset));
    }
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(formatted.contains("GpuSpan<u32>"));
    assert!(formatted.contains("GpuPtr<u32>"));
}

#[tokio::test]
async fn gpu_pipeline_annotations_preserve_type_queries_and_formatting() {
    let source = "fn pipeline(value:GpuComputePipeline<Root,ArcPtr<Owner>>)->GpuGraphicsPipeline<None,_>  {value}";
    let document = parse(source, None).await;
    assert!(!document.tree().root_node().has_error());
    for former in ["GpuComputePipeline", "GpuGraphicsPipeline"] {
        let offset = source.find(former).unwrap() + 2;
        assert_eq!(document.token(offset).unwrap().kind(), "uid");
        assert!(document.type_context(offset));
    }
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(formatted.contains("GpuComputePipeline<Root, ArcPtr<Owner>>"));
    assert!(formatted.contains("GpuGraphicsPipeline<None, _>"));
}

#[tokio::test]
async fn associated_method_references_format_and_preserve_type_queries() {
    let source = "fn main(){let mut f=select::<i32, u64>;let mut g=create::<Ptr<i32>>;select::<i32, u64>(7,42);}";
    let document = parse(source, None).await;
    assert!(!document.tree().root_node().has_error());
    assert!(document.type_context(source.find("u64").unwrap()));
    assert_eq!(
        document
            .token(source.find("select").unwrap())
            .unwrap()
            .kind(),
        "lid"
    );
    let formatted = resin_cst::format_source(source).unwrap();
    assert!(formatted.contains("select::<i32, u64>"), "{formatted}");
    assert!(formatted.contains("create::<Ptr<i32>>"), "{formatted}");
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    assert!(!parse(formatted, None).await.tree().root_node().has_error());
}

#[tokio::test]
async fn editing_method_reference_arguments_matches_fresh_parsing() {
    let sources = [
        "fn main() { let mut f = select",
        "fn main() { let mut f = select::",
        "fn main() { let mut f = select::<",
        "fn main() { let mut f = select::<i32, u64",
        "fn main()  { let mut f = select::<i32, u64>; }",
        "fn main()  { let mut f = select::<i32, u64>(7, 42); }",
    ];
    for before in sources {
        let original = parse(before, None).await;
        for after in sources {
            let incremental = parse(after, Some(&original)).await;
            let fresh = parse(after, None).await;
            assert_eq!(
                nodes(incremental.tree().root_node()),
                nodes(fresh.tree().root_node())
            );
            assert_eq!(original.source(), before);
        }
    }
}

#[tokio::test]
async fn parallel_successors_preserve_the_shared_predecessor() {
    fn shareable<T: Send + Sync>() {}
    fn sendable<T: Send>(value: T) -> T {
        value
    }
    shareable::<resin_cst::Document>();

    let execution = resin_executor::Execution::new(std::num::NonZeroUsize::new(2).unwrap());
    let cancellation = resin_executor::Cancellation::new();
    let original = resin_cst::build_cst("fn value()  { \"🌲\" }", None, &execution, &cancellation)
        .await
        .unwrap();
    let clone = original.clone();
    let original_nodes = nodes(original.tree().root_node());
    assert_eq!(original.source().as_ptr(), clone.source().as_ptr());
    let left_text = "fn value()  { \"é🌳\" }";
    let right_text = "fn value()  { \"🍂\" }";
    let (left, right) = tokio::join!(
        sendable(resin_cst::build_cst(
            left_text,
            Some(&original),
            &execution,
            &cancellation
        )),
        sendable(resin_cst::build_cst(
            right_text,
            Some(&clone),
            &execution,
            &cancellation
        )),
    );
    let (left, right) = (left.unwrap(), right.unwrap());
    assert_eq!(left.source(), left_text);
    assert_eq!(right.source(), right_text);
    assert_eq!(
        nodes(left.tree().root_node()),
        nodes(parse(left_text, None).await.tree().root_node())
    );
    assert_eq!(
        nodes(right.tree().root_node()),
        nodes(parse(right_text, None).await.tree().root_node())
    );
    assert_eq!(original.source(), "fn value()  { \"🌲\" }");
    assert_eq!(nodes(original.tree().root_node()), original_nodes);
    assert_eq!(nodes(clone.tree().root_node()), original_nodes);
}

#[tokio::test]
async fn queued_parsing_can_be_cancelled_without_an_execution_slot() {
    let execution = resin_executor::Execution::new(std::num::NonZeroUsize::MIN);
    let cancellation = resin_executor::Cancellation::new();
    let permit = execution.acquire(&cancellation).await.unwrap();
    let pending = resin_cst::build_cst("fn value()  {}", None, &execution, &cancellation);
    tokio::pin!(pending);
    tokio::select! {
        biased;
        _ = &mut pending => panic!("parsing must wait for its execution slot"),
        _ = tokio::task::yield_now() => {},
    }
    cancellation.cancel();
    assert!(matches!(
        pending.await,
        Err(resin_executor::Error::Cancelled)
    ));
    drop(permit);
    execution.wait_idle().await;
}
