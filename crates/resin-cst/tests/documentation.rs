mod common;

#[tokio::test]
async fn attaches_line_and_block_markdown_to_module_items_and_fields() {
    let source = r#"//! Module **overview**.
/*!
 * A second paragraph.
 */
export { Item, read, Alias, limit };
/// A generic record.
struct Item<T> {
    /// Stored value.
    value: T,
}
/** Reads the value.
 *
 * ```resin
 * read(item)
 * ```
 */
fn read<T>(item: Ref<Item<T>>) -> T { item.value }
/// Transparent alias.
type Alias = Item<int>;
/// Maximum count.
const limit = 16;
//// Ordinary comment.
/*** Ordinary block. */
/**/ /***/
fn hidden() {}
"#;
    let document = common::parse(source, None).await;
    assert!(
        !document.tree().root_node().has_error(),
        "{}",
        document.tree().root_node().to_sexp()
    );
    let docs = document.documentation();
    assert!(docs.diagnostics.is_empty(), "{:?}", docs.diagnostics);
    assert_eq!(docs.module, "Module **overview**.\nA second paragraph.");
    let read = docs
        .declarations
        .iter()
        .find(|item| item.name.val == "read")
        .unwrap();
    assert_eq!(
        read.markdown,
        "Reads the value.\n\n```resin\nread(item)\n```"
    );
    assert_eq!(read.signature, "fn read<T>(item: Ref<Item<T>>) -> T");
    let item = docs
        .declarations
        .iter()
        .find(|item| item.name.val == "Item")
        .unwrap();
    let field = docs
        .declarations
        .iter()
        .find(|item| item.name.val == "value")
        .unwrap();
    assert_eq!(field.markdown, "Stored value.");
    assert_eq!(field.field_owner, Some(item.span));
    assert!(!field.exported);
    assert_eq!(
        docs.declarations
            .iter()
            .filter(|item| item.exported)
            .count(),
        4
    );
    assert!(
        docs.declarations
            .iter()
            .find(|item| item.name.val == "hidden")
            .unwrap()
            .markdown
            .is_empty()
    );
    for item in docs.declarations {
        assert_eq!(
            &source[item.name.span.start..item.name.span.end],
            item.name.val
        );
    }
}

#[tokio::test]
async fn decorated_foreign_intrinsic_and_grouped_constant_declarations_keep_docs() {
    let source = r#"extern { "native.h": {
    /// Native function.
    fn native(value: int) -> int;
}, };
/// Opaque type.
extern type Handle;
/// Intrinsic function.
intrinsic "pointer_index" fn index<T>(pointer: Ptr<T>, index: ulong) -> Ptr<T>;
/// Shader entry.
@compute_shader
fn kernel(index: ulong, root: Ptr<int>) {}
/// Group defaults.
const (
    /// First member.
    first, other = 1, 2;
    second = iota;
);
"#;
    let document = common::parse(source, None).await;
    assert!(!document.tree().root_node().has_error());
    let docs = document.documentation();
    assert!(docs.diagnostics.is_empty(), "{:?}", docs.diagnostics);
    for (name, expected) in [
        ("native", "Native function."),
        ("Handle", "Opaque type."),
        ("index", "Intrinsic function."),
        ("kernel", "Shader entry."),
        ("first", "Group defaults.\nFirst member."),
        ("other", "Group defaults.\nFirst member."),
        ("second", "Group defaults."),
    ] {
        assert_eq!(
            docs.declarations
                .iter()
                .find(|item| item.name.val == name)
                .unwrap()
                .markdown,
            expected
        );
    }
}

#[tokio::test]
async fn misplaced_documentation_is_diagnosed_at_its_own_span() {
    for source in [
        "/// Orphan\n",
        "/// Cannot document imports\nimport { \"library.resin\" };",
        "fn f() {}\n//! Too late",
        "struct Item { //! Not module scope\n value: int }",
        "fn f() { /// Not a declaration\n 1; }",
        "fn f(/// Not a field\nvalue: int) {}",
        "struct Item { value: int, /// Dangling\n}",
    ] {
        let document = common::parse(source, None).await;
        let docs = document.documentation();
        assert_eq!(docs.diagnostics.len(), 1, "{source}: {docs:?}");
        let error = &docs.diagnostics[0];
        assert!(source[error.span.start..error.span.end].starts_with("//"));
    }
}

#[tokio::test]
async fn formatting_and_incremental_edits_preserve_markdown_and_attachment() {
    let source = "//! Résumé 😀\r\nexport { read };\r\n/// First paragraph.\r\n///\r\n///     indented code\r\nfn read( value : int)->int{value}\r\n";
    let before = common::parse(source, None).await;
    let formatted = resin_cst::format_source(source).unwrap();
    assert_eq!(resin_cst::format_source(&formatted).unwrap(), formatted);
    let after = common::parse(&formatted, None).await;
    assert_eq!(before.documentation().module, after.documentation().module);
    assert_eq!(
        before.documentation().declarations[0].markdown,
        after.documentation().declarations[0].markdown
    );
    let changed = source.replace("First paragraph.", "Updated **text**.");
    let incremental = common::parse(changed, Some(&before)).await;
    assert!(
        incremental.documentation().declarations[0]
            .markdown
            .starts_with("Updated")
    );
    assert!(
        before.documentation().declarations[0]
            .markdown
            .starts_with("First")
    );
}

#[tokio::test]
async fn block_margins_preserve_unicode_and_relative_code_indentation() {
    let source = "/** Résumé\n  paragraph\n\u{2003}\n      code\n  😀\n */\nfn read() {}";
    let document = common::parse(source, None).await;
    assert_eq!(
        document.documentation().declarations[0].markdown,
        "Résumé\nparagraph\n\u{2003}\n    code\n😀"
    );
}
