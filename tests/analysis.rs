use resin::{
    analysis::{Analysis, Sources, normalize_path},
    ast::{self, SourceProvider},
};
use std::path::PathBuf;

#[test]
fn pointer_hover_and_completion_use_angle_bracket_types() {
    let source = "def main (value: Ptr<Span<int>>) -> Ptr<Span<int>> = { value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    assert_eq!(
        analysis
            .hover(&project.path("main.resin"), source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: Ptr<Span<int>>"
    );

    let source = "type Number = int; def main () -> () = { var value: Ptr<Num>; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.path("main.resin"),
        source.rfind("Num").unwrap() + 3,
    );
    assert_eq!(
        items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        ["Number"]
    );
}

struct Project {
    root: PathBuf,
    sources: Sources,
}

#[test]
fn deferred_bindings_keep_navigation_types_and_local_scopes() {
    let source = "def main() = { var value = 1; { defer { var inner: _; inner := value; print(\"{0}\", (inner,)); }; var value = 2; () }; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    let reference = source.find("inner := value").unwrap() + 9;
    assert_eq!(
        analysis.definition(&path, reference).unwrap().span.start,
        source.find("value").unwrap()
    );
    assert_eq!(analysis.hover(&path, reference).unwrap().text, "value: int");
    assert!(
        analysis
            .completions(&path, source.rfind("var value").unwrap())
            .iter()
            .all(|item| item.name != "inner")
    );
    assert!(
        analysis
            .completions(&path, source.find("defer").unwrap())
            .iter()
            .any(|item| item.name == "defer")
    );
}

#[test]
fn deferred_expressions_keep_lexical_navigation_and_recover_fields() {
    let source = "def main() = { var value = 1; { defer value + 1; var value = 2; }; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    let reference = source.find("defer value").unwrap() + 6;
    assert_eq!(
        analysis.definition(&path, reference).unwrap().span.start,
        source.find("value").unwrap()
    );
    assert_eq!(analysis.hover(&path, reference).unwrap().text, "value: int");

    let source = "def main() = { var point = { count = 42 }; defer point.; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let items = analysis.completions(
        &project.path("main.resin"),
        source.find("point.").unwrap() + 6,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "count: int");
}

#[test]
fn inferred_errors_and_match_payloads_have_editor_types() {
    let source = "struct Broken { code: int }; def fail() -> Result<int, _> = { err(Broken { code = 7 }) }; def main() = { var result = fail(); match (result) { ok(value) => { value; }, err(error) => { error.code; } }; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    assert_eq!(
        analysis
            .hover(&path, source.find("match (result").unwrap() + 7)
            .unwrap()
            .text,
        "result: Result<int, Broken>"
    );
    assert_eq!(
        analysis
            .hover(&path, source.find("error.code").unwrap())
            .unwrap()
            .text,
        "error: Broken"
    );
    let fields = analysis.completions(&path, source.find("error.code").unwrap() + 6);
    assert!(
        fields
            .iter()
            .any(|field| field.name == "code" && field.detail == "code: int")
    );
    let origin = analysis
        .definition(&path, source.find("error.code").unwrap())
        .unwrap();
    assert_eq!(origin.span.start, source.find("err(error)").unwrap() + 4);
    assert!(
        analysis
            .completions(&path, source.find("var result").unwrap())
            .iter()
            .all(|item| item.name != "error")
    );
}

#[test]
fn inferred_imported_results_and_local_annotations_support_editor_queries() {
    let source =
        "import { \"lib.resin\" }; def main() = { var value: _; value := make(); value.count; };";
    let library = "export { make }; struct Counter { count: int }; def make() -> _ = { Counter { count = 42 } };";
    let project = Project::new(&[("main.resin", source), ("lib.resin", library)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    assert_eq!(
        analysis
            .hover(&path, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: Counter"
    );
    let fields = analysis.completions(&path, source.rfind("count").unwrap());
    assert!(
        fields
            .iter()
            .any(|field| field.name == "count" && field.detail == "count: int")
    );
    let definition = analysis
        .definition(&path, source.find("make()").unwrap())
        .unwrap();
    assert_eq!(definition.path, project.path("lib.resin"));
}

#[test]
fn inference_does_not_publish_speculative_type_references() {
    let source = "type Value = int; def f() -> _ = { type Value = bool; type Wrapper = Value; Wrapper(Value(1 == 1)) };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let reference = source.find("Wrapper = Value").unwrap() + "Wrapper = ".len();
    let definition = analysis
        .definition(&project.path("main.resin"), reference)
        .unwrap();
    assert_eq!(
        definition.span.start,
        source.rfind("type Value").unwrap() + 5
    );
}

#[test]
fn field_completion_uses_receiver_types_and_replaces_only_the_field() {
    for (setup, receiver) in [
        ("var value = { count = 1, label = 2 };", "value"),
        (
            "var record = { count = 1, label = 2 }; var value = &record;",
            "value",
        ),
        (
            "var value = { inner = { count = 1, label = 2 } };",
            "value.inner",
        ),
        ("var value = { count = 1, label = 2 };", "(value)"),
    ] {
        for field in ["", "co", "count"] {
            let source = format!("def main () = {{ {setup} {receiver}.{field}; }};");
            let project = Project::new(&[("main.resin", &source)]);
            let start = source.rfind('.').unwrap() + 1;
            let offset = start + field.len().min(2);
            let items = project
                .analyze()
                .completions(&project.path("main.resin"), offset);
            let names = items
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>();
            assert_eq!(
                names,
                if field.is_empty() {
                    vec!["count", "label"]
                } else {
                    vec!["count"]
                },
                "{source}"
            );
            assert_eq!(items[0].detail, "count: int");
            assert_eq!(items[0].kind, resin::analysis::DefinitionKind::Field);
            assert_eq!(items[0].replace.start, start);
            assert_eq!(items[0].replace.end, start + field.len());
        }
    }
}

#[test]
fn field_completion_resolves_imported_nominal_function_results() {
    let source = "import { \"lib.resin\" }; def main () = { make().; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Counter { count: int }; def make () -> Counter = { Counter { count = 0 } };",
        ),
    ]);
    let items = project
        .analyze()
        .completions(&project.path("main.resin"), source.rfind('.').unwrap() + 1);
    assert_eq!(
        items
            .iter()
            .map(|item| item.name.as_str())
            .collect::<Vec<_>>(),
        ["count"]
    );
}

#[test]
fn field_completion_does_not_offer_unrelated_names() {
    for expression in ["1.", "missing.", "\"text.\"", "// value."] {
        let source = format!("def main () = {{ var value = {{ count = 1 }}; {expression}\n }};");
        let project = Project::new(&[("main.resin", &source)]);
        assert!(
            project
                .analyze()
                .completions(&project.path("main.resin"), source.rfind('.').unwrap() + 1)
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn field_completion_recovers_unfinished_functions_and_uninitialized_locals() {
    for source in [
        "struct Point { x: float32, y: float32 }; def main () = { var point: Point; point.; };",
        "def main (point: { x: float32, y: float32 }) = { point.",
        "def main () = { var point = { x = 1, y = 2 }; point.",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let items = project
            .analyze()
            .completions(&project.path("main.resin"), source.rfind('.').unwrap() + 1);
        assert_eq!(
            items
                .iter()
                .map(|item| item.name.as_str())
                .collect::<Vec<_>>(),
            ["x", "y"],
            "{source}"
        );
    }
}
impl Project {
    fn new(files: &[(&str, &str)]) -> Self {
        let root = normalize_path(
            &std::env::temp_dir().join(format!("resin-analysis-unsaved-{}", std::process::id())),
        )
        .unwrap();
        let overlays = files
            .iter()
            .map(|(path, text)| (root.join(path), text.to_string()))
            .collect();
        Self {
            root,
            sources: Sources { overlays },
        }
    }
    fn analyze(&self) -> Analysis {
        Analysis::new(
            &self.root.join("main.resin"),
            &self.sources,
            &self.root.join("std"),
        )
    }
    fn path(&self, name: &str) -> PathBuf {
        self.root.join(name)
    }
}

#[test]
fn inferred_types_and_parameter_definitions_come_from_compilation() {
    let source = "def main (argument: int) -> int = { var value = argument + 1; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    assert_eq!(
        analysis
            .hover(&path, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: int"
    );
    let location = analysis
        .definition(&path, source.rfind("argument").unwrap())
        .unwrap();
    assert_eq!(location.path, path);
    assert_eq!(&source[location.span.start..location.span.end], "argument");
    assert_eq!(location.span.start, source.find("argument").unwrap());
}

#[test]
fn reexports_keep_original_definitions_through_diamond_imports() {
    let source = "import { \"left.resin\", \"right.resin\" }; def main () -> Number = { make() };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "base.resin",
            "export { Number, make }; type Number = int; def make () -> Number = { Number(42) }; def secret () -> int = { 1 };",
        ),
        (
            "left.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Number, make }; import { \"base.resin\" };",
        ),
    ]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    for word in ["Number", "make"] {
        assert_eq!(
            analysis
                .definition(&path, source.rfind(word).unwrap())
                .unwrap()
                .path,
            project.path("base.resin")
        );
    }
    let completions = analysis.completions(&path, source.rfind("make").unwrap());
    assert!(completions.iter().any(|c| c.name == "make"));
    assert!(!completions.iter().any(|c| c.name == "secret"));
    assert_eq!(
        analysis
            .definition(&path, source.find("left.resin").unwrap())
            .unwrap()
            .path,
        project.path("left.resin")
    );
}

#[test]
fn nested_bindings_shadow_without_leaking_out_of_their_scope() {
    let source =
        "def main () -> int = { var value = 1; var inner = { var value = 2; value }; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    let inside = source.find("value };").unwrap();
    assert_eq!(
        analysis.definition(&path, inside).unwrap().span.start,
        source.find("value = 2").unwrap()
    );
    assert_eq!(
        analysis
            .definition(&path, source.rfind("value").unwrap())
            .unwrap()
            .span
            .start,
        source.find("value = 1").unwrap()
    );
}

#[test]
fn incomplete_code_keeps_parameters_and_prior_locals_available() {
    for source in [
        "def main (argument: int) -> int = { var local = 1; arg };",
        "def main (argument: int) -> int = { var local = 1; arg",
        "def main (argument: int) -> int = { var local = 1; print(arg };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let at = source.rfind("arg").unwrap() + 3;
        let completions = analysis.completions(&project.path("main.resin"), at);
        assert!(
            completions.iter().any(|c| c.name == "argument"),
            "{source}: {completions:?}"
        );
        let local = analysis.completions(&project.path("main.resin"), at - 3);
        assert!(
            local.iter().any(|c| c.name == "local"),
            "{source}: {local:?}"
        );
    }
}

#[test]
fn conflicting_imports_are_ambiguous_and_report_related_locations() {
    let source = "import { \"a.resin\", \"b.resin\" }; def main () -> int = { answer() };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "a.resin",
            "export { answer }; def answer () -> int = { 1 };",
        ),
        (
            "b.resin",
            "export { answer }; def answer () -> int = { 2 };",
        ),
    ]);
    let analysis = project.analyze();
    assert!(analysis.diagnostics.iter().any(|d| d.related.len() == 2));
    assert!(
        analysis
            .definition(&project.path("main.resin"), source.rfind("answer").unwrap())
            .is_none()
    );
    assert!(
        !analysis
            .completions(&project.path("main.resin"), source.rfind("answer").unwrap())
            .iter()
            .any(|c| c.name == "answer")
    );
}

#[test]
fn missing_imports_and_cycles_have_source_ranges() {
    let source = "import { \"missing.resin\" };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert_eq!(analysis.diagnostics.len(), 1);
    let error = &analysis.diagnostics[0];
    assert_eq!(error.location.path, project.path("main.resin"));
    assert_eq!(
        &source[error.location.span.start..error.location.span.end],
        "\"missing.resin\""
    );
    let project = Project::new(&[
        ("main.resin", "import { \"a.resin\" };"),
        ("a.resin", "import { \"main.resin\" };"),
    ]);
    assert!(
        project
            .analyze()
            .diagnostics
            .iter()
            .any(|d| d.message.contains("cyclic"))
    );
}

#[test]
fn imported_syntax_errors_stay_at_the_dependency() {
    let project = Project::new(&[
        ("main.resin", "import { \"a.resin\" };"),
        ("a.resin", "export { value }; def value () -> int = { + };"),
    ]);
    let error =
        ast::load_with(&project.path("main.resin"), &project.root, &project.sources).unwrap_err();
    assert_eq!(error.path, project.path("a.resin"));
    assert!(error.span.is_some());
    assert_eq!(error.related[0].location.path, project.path("main.resin"));
    assert!(
        project
            .analyze()
            .diagnostics
            .iter()
            .any(|d| d.location.path == project.path("a.resin"))
    );
}

#[test]
#[cfg(unix)]
fn new_paths_resolve_through_symlinks() {
    let temp = resin::toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
    assert_eq!(
        normalize_path(&root.join("alias/new.resin")).unwrap(),
        root.join("real/new.resin")
    );
}

#[test]
fn buffers_override_disk() {
    let temp = resin::toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    let path = temp.path().join("lib.resin");
    std::fs::write(&path, "disk").unwrap();
    let path = normalize_path(&path).unwrap();
    let sources = Sources {
        overlays: [(path.clone(), "buffer".into())].into(),
    };
    assert_eq!(sources.read(&path).unwrap(), "buffer");
}

#[test]
fn normalized_paths_are_stable_for_existing_and_unsaved_files() {
    let temp = resin::toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    std::fs::write(temp.path().join("saved.resin"), "").unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    for name in ["saved.resin", "unsaved.resin"] {
        let expected = root.join(name);
        assert_eq!(normalize_path(&temp.path().join(name)).unwrap(), expected);
        assert_eq!(normalize_path(&expected).unwrap(), expected);
    }
}

#[test]
fn completion_respects_type_context_and_ignores_strings_and_comments() {
    let source = "type Number = int; def main () -> () = { var value = 1; var other: Num; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let suggestions = analysis.completions(
        &project.path("main.resin"),
        source.rfind("Num").unwrap() + 3,
    );
    assert_eq!(
        suggestions
            .iter()
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>(),
        ["Number"]
    );
    for source in ["// val", "def main () -> () = { var value = \"val\"; };"] {
        let project = Project::new(&[("main.resin", source)]);
        let at = if source.contains('"') {
            source.rfind("val").unwrap() + 2
        } else {
            source.len() - 1
        };
        assert!(
            project
                .analyze()
                .completions(&project.path("main.resin"), at)
                .is_empty(),
            "{source}"
        );
    }
}

#[test]
fn completion_obeys_parameter_shadowing_and_module_type_order() {
    let source = "def main (value: int) -> int = { var value = 2; val };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let items = analysis.completions(
        &project.path("main.resin"),
        source.rfind("val").unwrap() + 3,
    );
    let value = items.iter().find(|item| item.name == "value").unwrap();
    assert_eq!(value.kind, resin::analysis::DefinitionKind::Variable);

    // Module type definitions run before function signatures, but run in source
    // order relative to each other. Completion must match those compiler phases.
    for (source, expected) in [
        ("def main (value: Num) -> () = {}; type Number = int;", true),
        ("type Earlier = Num; type Number = int;", false),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let items =
            analysis.completions(&project.path("main.resin"), source.find("Num").unwrap() + 2);
        assert_eq!(
            items.iter().any(|item| item.name == "Number"),
            expected,
            "{source}: {items:?}"
        );
    }
    let project = Project::new(&[("main.resin", "// comment")]);
    assert!(
        project
            .analyze()
            .completions(&project.path("main.resin"), 10)
            .is_empty()
    );
}

#[test]
fn module_analysis_needs_no_entry_and_rejects_runtime_globals() {
    let source = "export { run }; def run () -> int = { helper() }; def helper () -> int = { 1 };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    assert_eq!(
        analysis
            .module()
            .unwrap()
            .entries
            .keys()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>(),
        ["run"]
    );
    assert_eq!(
        analysis
            .definition(&project.path("main.resin"), source.find("run").unwrap())
            .unwrap()
            .span
            .start,
        source.find("run ()").unwrap()
    );
    let project = Project::new(&[("main.resin", "export {}; def helper () -> int = { 1 };")]);
    assert!(project.analyze().module().unwrap().entries.is_empty());

    for source in [
        "var global = 1; def main () -> int = { glo };",
        "var global: int; def main () -> int = { glo };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        assert!(!analysis.diagnostics.is_empty(), "{source}");
        let at = source.rfind("glo").unwrap();
        assert!(
            analysis
                .definition(&project.path("main.resin"), at)
                .is_none()
        );
        assert!(
            !analysis
                .completions(&project.path("main.resin"), at + 3)
                .iter()
                .any(|item| item.name == "global")
        );
    }
}

#[test]
fn implicit_unit_signatures_and_declaration_keywords_support_editor_features() {
    let source = "extern \"native.h\" def release(value: int); def run() = { release(1); };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics.is_empty(),
        "{:?}",
        analysis.diagnostics
    );
    let path = project.path("main.resin");
    for (name, signature) in [
        (
            "release",
            "extern \"native.h\" def release(value: int) -> ()",
        ),
        ("run", "def run() -> ()"),
    ] {
        let offset = source.rfind(name).unwrap();
        assert_eq!(analysis.hover(&path, offset).unwrap().text, signature);
        let items = analysis.completions(&path, offset + name.len());
        assert_eq!(
            items.iter().find(|item| item.name == name).unwrap().detail,
            signature
        );
        assert_eq!(
            analysis.definition(&path, offset).unwrap().span.start,
            source.find(name).unwrap()
        );
    }
    let project = Project::new(&[("main.resin", "def run() = { var value = 1; ")]);
    let items = project
        .analyze()
        .completions(&project.path("main.resin"), 28);
    for keyword in ["def", "var", "type"] {
        assert!(items.iter().any(|item| item.name == keyword), "{items:?}");
    }
    let project = Project::new(&[("main.resin", "def run() = { 1 };")]);
    assert!(
        !project.analyze().diagnostics.is_empty(),
        "unit returns are not inferred from bodies"
    );
}

#[test]
fn holes_preserve_later_locals_and_functions_without_producing_ir() {
    for broken in [
        "var broken = ;",
        "var broken: ;",
        "var broken = 1 + ;",
        "unknown_name;",
        "var broken = missing(1);",
    ] {
        let source = format!(
            "def first() = {{ {broken} var point = {{ x = 1, y = 2 }}; point.; }}; def later(arg: int) -> int = {{ var result = arg; result }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let path = project.path("main.resin");
        assert!(analysis.module().is_err(), "{source}");
        assert!(!analysis.diagnostics.is_empty(), "{source}");
        let items = analysis.completions(&path, source.find("point.").unwrap() + 6);
        assert_eq!(
            items.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
            ["x", "y"],
            "{source}"
        );
        let hover = analysis
            .hover(&path, source.rfind("result").unwrap())
            .unwrap();
        assert_eq!(hover.text, "result: int", "{source}");
    }
}

#[test]
fn unknown_bindings_shadow_outer_values_without_fabricating_types() {
    let source = "def main(point: { x: int }) = { var point = ; var alias = point; alias.; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let path = project.path("main.resin");
    assert_eq!(
        analysis
            .hover(&path, source.rfind("alias").unwrap())
            .unwrap()
            .text,
        "alias: ?"
    );
    assert!(
        analysis
            .completions(&path, source.find("alias.").unwrap() + 6)
            .is_empty()
    );
    assert_eq!(
        analysis
            .definition(&path, source.rfind("point").unwrap())
            .unwrap()
            .span
            .start,
        source.find("var point").unwrap() + 4
    );
}

#[test]
fn recovery_uses_unsaved_imports_and_keeps_nominal_field_types() {
    let source =
        "import { \"lib.resin\" }; def main() = { var broken = ; var point = make(); point.; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "lib.resin",
            "export { make }; struct Point { x: float32 }; def broken() = { var hole = ; }; def make() -> Point = { Point { x = 1 } };",
        ),
    ]);
    let analysis = project.analyze();
    let items = analysis.completions(
        &project.path("main.resin"),
        source.find("point.").unwrap() + 6,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "x: float32");
    assert!(analysis.module().is_err());
}

#[test]
fn recovered_ast_contains_expression_type_and_field_holes() {
    let source = "def main() = { var value = ; var typed: ; var point = { x = 1 }; point.; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let ast = resin::ast::print::format_source(
        analysis
            .recovered_file(&project.path("main.resin"))
            .unwrap(),
    );
    assert!(ast.contains("(hole "), "{ast}");
    assert!(ast.contains("(type-hole "), "{ast}");
    assert!(ast.contains("(field-hole "), "{ast}");
    assert!(analysis.program().is_err());
}

#[test]
fn editor_analysis_tolerates_truncation_and_deleted_tokens() {
    for source in [
        "export { main }; struct Point { x: int }; def main(arg: Ptr<Point>) = { var value = arg.x + 1; print(\"{}\", value); };",
        "def main(arg: int) -> int = { var pair = { left = arg, right = 1 }; if (arg == 0) (pair.left) else (pair.right) };",
        "def main() = { var values = [1, 2]; while (1 == 1) { var missing: Ptr<int>; }; };",
        "def main() = { var n = 42; defer { defer {}; print(\"{0}\", (n,)); }; };",
        "def main() = { var n = 42; defer if (n == 42) { print(\"{0}\", (n,)); } else {}; defer n := n + 1; };",
    ] {
        for end in 0..=source.len() {
            let project = Project::new(&[("main.resin", &source[..end])]);
            let analysis = project.analyze();
            analysis.completions(&project.path("main.resin"), end);
        }
        for index in 0..source.len() {
            let mut edited = source.to_owned();
            edited.remove(index);
            let project = Project::new(&[("main.resin", &edited)]);
            project.analyze();
        }
    }
}

#[test]
fn strict_lowering_rejects_holes_even_when_given_a_recovered_ast() {
    for source in [
        "def main() = { var value = ; };",
        "def main() = { var value: ; };",
        "def main(point: { x: int }) = { point.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let file = analysis
            .recovered_file(&project.path("main.resin"))
            .unwrap();
        let error = resin::ir::generate(file).unwrap_err();
        assert_eq!(
            error.kind,
            resin::ir::GenerateErrorKind::IncompleteSyntax,
            "{source}: {error}"
        );
        assert!(error.span.start <= error.span.end && error.span.end <= source.len());
    }
}
