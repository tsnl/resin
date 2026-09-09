use resin_compiler::{Compilation, Compiler};
use resin_hir::GenerateErrorKind;
use resin_source::normalize_path;
use resin_source::prelude::*;
use std::{collections::BTreeMap, path::Path, sync::Arc};
use tempfile::TempDir;

#[test]
fn option_payload_fields_remain_available_in_incomplete_code() {
    let source = "struct Item { count: int }; def f(value: Item | None) = { value!.; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
        source.find("value!.").unwrap() + 7,
    );
    assert_eq!(items.len(), 1);
    assert_eq!(items[0].detail, "count: int");
}

#[test]
fn weak_upgrade_recovery_exposes_the_shared_payload_and_handle_operations() {
    let source = "struct Item { count: int }; def f(weak: Weak<Item>) = { weak.upgrade()!.; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
        source.find("!.").unwrap() + 2,
    );
    assert!(
        items.iter().any(|item| item.detail == "count: int"),
        "{items:?}"
    );
    assert!(items.iter().any(|item| item.name == "get"), "{items:?}");
    assert!(
        items.iter().any(|item| item.name == "downgrade"),
        "{items:?}"
    );
}

#[test]
fn standard_library_resource_methods_support_editor_navigation_and_recovery() {
    for tail in ["ok(()) };", "buffer."] {
        let source = format!(
            r#"import {{ "$/std/gpu.resin" }};
            def f() -> Result<(), _> = {{
                var gpu = Gpu.new()?;
                var buffer = gpu.malloc(4_ul, 4_ul, Memory.default())?;
                buffer.host_pointer();
                {tail}"#
        );
        let mut loader = resin_source::Loader::new(resin_source::library_root());
        let input = loader
            .source_from_text(Path::new("main.resin"), source.clone())
            .unwrap();
        let analysis = Compiler::new().compile(input.clone(), &mut loader);
        let call = source.find("host_pointer").unwrap();
        let definition = analysis
            .definition(&input, call)
            .unwrap_or_else(|| panic!("tail: {tail}\n{:?}", analysis.diagnostics()));
        assert!(
            loader
                .path(&definition.source)
                .unwrap()
                .ends_with("resin/std/gpu.resin")
        );
        assert!(
            analysis
                .hover(&input, call)
                .unwrap()
                .text
                .starts_with("def host_pointer(")
        );
        let offset = if tail == "buffer." {
            source.len()
        } else {
            call
        };
        let items = analysis.completions(&input, offset);
        for name in ["host_pointer", "device_pointer", "size"] {
            assert!(
                items.iter().any(|item| item.name == name),
                "missing {name}: {items:?}"
            );
        }
        assert!(items.iter().any(|item| item.name == "drop"));
    }
}

#[test]
fn pointer_hover_and_completion_use_angle_bracket_types() {
    let source = "def main (value: Ptr<Span<int>>) -> Ptr<Span<int>> = { value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    assert_eq!(
        analysis
            .hover(
                &project.source("main.resin"),
                source.rfind("value").unwrap()
            )
            .unwrap()
            .text,
        "value: Ptr<Span<int>>"
    );

    let source = "type Number = int; def main () -> () = { var value: Ptr<Num>; };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
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
    sources: BTreeMap<String, Source>,
}

#[test]
fn inherent_methods_have_navigation_hover_and_member_completion() {
    let library = "export { Counter }; struct Counter { count: int }; impl Counter { def new() -> Counter = { Counter { count = 7 } }; def read(self: Ptr<Counter>) -> int = { self.count }; }";
    for incomplete in [None, Some("c"), Some("Counter")] {
        let tail = incomplete
            .map(|base| format!("{base}.;"))
            .unwrap_or_default();
        let source = format!(
            "import {{ \"lib.resin\" }}; def main() = {{ var c = Counter.new(); c.read(); {tail} }};"
        );
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert_eq!(
            analysis.diagnostics().is_empty(),
            incomplete.is_none(),
            "{:?}",
            analysis.diagnostics()
        );
        for method in ["new", "read"] {
            let call = source.find(&format!(".{method}()")).unwrap() + 1;
            let definition = analysis.definition(&input, call).unwrap();
            assert_eq!(definition.source, project.source("lib.resin"));
            assert_eq!(
                definition.span.start,
                library.find(&format!("{method}(")).unwrap()
            );
            assert!(
                analysis
                    .hover(&input, call)
                    .unwrap()
                    .text
                    .starts_with(&format!("def {method}("))
            );
            let items = analysis.completions(&input, call);
            assert!(items.iter().any(
                |item| item.name == method && item.kind == resin_hir::DefinitionKind::Function
            ));
        }
        let global = analysis.completions(&input, source.find("var c").unwrap());
        assert!(
            global
                .iter()
                .all(|item| item.name != "read" && item.name != "new")
        );
        if let Some(base) = incomplete {
            let items = analysis.completions(
                &input,
                source.rfind(&format!("{base}.;")).unwrap() + base.len() + 1,
            );
            assert_eq!(
                items
                    .iter()
                    .map(|item| item.name.as_str())
                    .collect::<Vec<_>>(),
                if base == "c" {
                    vec!["count", "read"]
                } else {
                    vec!["new", "read"]
                }
            );
        }
    }
    let project = Project::new(&[("main.resin", library)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
        library.find("self.count").unwrap(),
    );
    assert!(
        items
            .iter()
            .all(|item| item.name != "read" && item.name != "new")
    );
}

#[test]
fn at_indexing_has_hover_and_completion_in_valid_and_incomplete_code() {
    for receiver in ["values", "holder.values"] {
        for tail in ["", " values.;", " holder.values.;", " holder.values.at(; "] {
            let source = format!(
                "def main() = {{ var values = [1_i, 2_i]; var holder = {{ values = Span<int> {{ data = Ptr<int>(&values), length = 2_ul }} }}; {receiver}.at(0).* := 3;{tail} }};"
            );
            let project = Project::new(&[("main.resin", &source)]);
            let analysis = project.analyze();
            let input = project.source("main.resin");
            assert_eq!(
                analysis.diagnostics().is_empty(),
                tail.is_empty(),
                "{:?}",
                analysis.diagnostics()
            );
            let offset = source.find(".at(0)").unwrap() + 1;
            assert_eq!(
                analysis.hover(&input, offset).unwrap().text,
                "at: (ulong) -> Ptr<int>"
            );
            let items = analysis.completions(&input, offset);
            assert!(
                items
                    .iter()
                    .any(|item| item.name == "at"
                        && item.kind == resin_hir::DefinitionKind::Function)
            );
            if !tail.is_empty() {
                let offset = source.rfind(".;").or_else(|| source.rfind(".at(")).unwrap() + 1;
                let items = analysis.completions(&input, offset);
                assert!(
                    items.iter().any(|item| item.name == "at"),
                    "{source}\n{items:?}"
                );
            }
        }
    }
}

#[test]
fn shared_receiver_completion_and_navigation_include_ordinary_drop_methods() {
    let library = "export { Counter }; struct Counter { count: int }; impl Counter { def drop(self: Ptr<Counter>) = {}; def read(self: Ptr<Counter>) -> int = { self.count }; }";
    for tail in ["", "c.;"] {
        let source =
            format!("import {{ \"lib.resin\" }}; def f(c: Arc<Counter>) = {{ c.read(); {tail} }};");
        let project = Project::new(&[("main.resin", &source), ("lib.resin", library)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        let call = source.find("c.read").unwrap() + 2;
        assert_eq!(
            analysis
                .definition(&input, call)
                .unwrap_or_else(|| panic!(
                    "{:?}\n{:?}",
                    analysis.diagnostics(),
                    analysis.recovered_file(&input)
                ))
                .source,
            project.source("lib.resin")
        );
        assert!(
            analysis
                .hover(&input, call)
                .unwrap()
                .text
                .starts_with("def read(")
        );
        let offset = if tail.is_empty() {
            call
        } else {
            source.rfind("c.;").unwrap() + 2
        };
        let items = analysis.completions(&input, offset);
        assert!(items.iter().any(|item| item.name == "read"));
        assert!(items.iter().any(|item| item.name == "drop"));
    }
}

#[test]
fn inferred_errors_and_match_payloads_have_editor_types() {
    let source = "struct Broken { code: int }; def fail() -> Result<int, _> = { err(Broken { code = 7 }) }; def main() = { var result = fail(); match (result) { ok(value) => { value; }, err(error) => { error.code; } }; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("match (result").unwrap() + 7)
            .unwrap()
            .text,
        "result: Result<int, Broken>"
    );
    assert_eq!(
        analysis
            .hover(&input, source.find("error.code").unwrap())
            .unwrap()
            .text,
        "error: Broken"
    );
    let fields = analysis.completions(&input, source.find("error.code").unwrap() + 6);
    assert!(
        fields
            .iter()
            .any(|field| field.name == "code" && field.detail == "code: int")
    );
    let origin = analysis
        .definition(&input, source.find("error.code").unwrap())
        .unwrap();
    assert_eq!(origin.span.start, source.find("err(error)").unwrap() + 4);
    assert!(
        analysis
            .completions(&input, source.find("var result").unwrap())
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
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: Counter"
    );
    let fields = analysis.completions(&input, source.rfind("count").unwrap());
    assert!(
        fields
            .iter()
            .any(|field| field.name == "count" && field.detail == "count: int")
    );
    let definition = analysis
        .definition(&input, source.find("make()").unwrap())
        .unwrap();
    assert_eq!(definition.source, project.source("lib.resin"));
}

#[test]
fn inference_does_not_publish_speculative_type_references() {
    let source = "type Value = int; def f() -> _ = { type Value = bool; type Wrapper = Value; Wrapper(Value(1 == 1)) };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let reference = source.find("Wrapper = Value").unwrap() + "Wrapper = ".len();
    let definition = analysis
        .definition(&project.source("main.resin"), reference)
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
                .completions(&project.source("main.resin"), offset);
            let names = items
                .iter()
                .filter(|item| item.kind == resin_hir::DefinitionKind::Field)
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
            assert_eq!(items[0].detail, "count: long");
            assert_eq!(items[0].kind, resin_hir::DefinitionKind::Field);
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
    let items = project.analyze().completions(
        &project.source("main.resin"),
        source.rfind('.').unwrap() + 1,
    );
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
                .completions(
                    &project.source("main.resin"),
                    source.rfind('.').unwrap() + 1
                )
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
        let items = project.analyze().completions(
            &project.source("main.resin"),
            source.rfind('.').unwrap() + 1,
        );
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
        let sources = files
            .iter()
            .map(|(name, text)| (name.to_string(), Source::new(*name, *text)))
            .collect();
        Self { sources }
    }
    fn analyze(&self) -> Arc<Compilation> {
        let mut loader = resin_source::Loader::new(Default::default());
        for importer in self.sources.values() {
            for (reference, target) in &self.sources {
                loader
                    .set_import(importer, reference, target.clone())
                    .unwrap();
            }
        }
        Compiler::new().compile(self.source("main.resin"), &mut loader)
    }
    fn source(&self, name: &str) -> Source {
        self.sources[name].clone()
    }
}

#[test]
fn inferred_types_and_parameter_definitions_come_from_compilation() {
    let source = "def main (argument: int) -> int = { var value = argument + 1; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: int"
    );
    let location = analysis
        .definition(&input, source.rfind("argument").unwrap())
        .unwrap();
    assert_eq!(location.source, input);
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
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    for word in ["Number", "make"] {
        assert_eq!(
            analysis
                .definition(&input, source.rfind(word).unwrap())
                .unwrap()
                .source,
            project.source("base.resin")
        );
    }
    let completions = analysis.completions(&input, source.rfind("make").unwrap());
    assert!(completions.iter().any(|c| c.name == "make"));
    assert!(!completions.iter().any(|c| c.name == "secret"));
    assert_eq!(
        analysis
            .definition(&input, source.find("left.resin").unwrap())
            .unwrap()
            .source,
        project.source("left.resin")
    );
}

#[test]
fn nested_bindings_shadow_without_leaking_out_of_their_scope() {
    let source =
        "def main () -> int = { var value = 1; var inner = { var value = 2; value }; value };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    let inside = source.find("value };").unwrap();
    assert_eq!(
        analysis.definition(&input, inside).unwrap().span.start,
        source.find("value = 2").unwrap()
    );
    assert_eq!(
        analysis
            .definition(&input, source.rfind("value").unwrap())
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
        let completions = analysis.completions(&project.source("main.resin"), at);
        assert!(
            completions.iter().any(|c| c.name == "argument"),
            "{source}: {completions:?}"
        );
        let local = analysis.completions(&project.source("main.resin"), at - 3);
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
    assert!(analysis.diagnostics().iter().any(|d| d.related.len() == 2));
    assert!(
        analysis
            .definition(
                &project.source("main.resin"),
                source.rfind("answer").unwrap()
            )
            .is_none()
    );
    assert!(
        !analysis
            .completions(
                &project.source("main.resin"),
                source.rfind("answer").unwrap()
            )
            .iter()
            .any(|c| c.name == "answer")
    );
}

#[test]
fn missing_imports_and_cycles_have_source_ranges() {
    let source = "import { \"missing.resin\", \"available.resin\" }; def main() = { var point = make(); point.x; };";
    let project = Project::new(&[
        ("main.resin", source),
        (
            "available.resin",
            "export { make }; struct Point { x: int }; def make() -> Point = { Point { x = 1 } };",
        ),
    ]);
    let analysis = project.analyze();
    assert_eq!(analysis.diagnostics().len(), 1);
    let error = &analysis.diagnostics()[0];
    assert_eq!(error.location.source, project.source("main.resin"));
    assert_eq!(
        &source[error.location.span.start..error.location.span.end],
        "\"missing.resin\""
    );
    let fields = analysis.completions(
        &project.source("main.resin"),
        source.find("point.x").unwrap() + 6,
    );
    assert_eq!(
        fields[0].detail, "x: int",
        "a failed import must preserve later dependencies"
    );
    let project = Project::new(&[
        ("main.resin", "import { \"a.resin\" };"),
        ("a.resin", "import { \"main.resin\" };"),
    ]);
    assert!(
        project
            .analyze()
            .diagnostics()
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
    let analysis = project.analyze();
    let error = analysis.program().unwrap_err();
    assert_eq!(error.source, project.source("a.resin"));
    assert!(error.span.is_some());
    assert_eq!(
        error.related[0].location.source,
        project.source("main.resin")
    );
    assert!(
        project
            .analyze()
            .diagnostics()
            .iter()
            .any(|d| d.location.source == project.source("a.resin"))
    );
}

#[test]
fn malformed_foreign_headers_are_diagnostics_not_panics() {
    for source in [
        r#"extern "bad\q" def release();"#,
        "extern \"unfinished def release();",
        "extern def release();",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let input = project.source("main.resin");
        let analysis = project.analyze();
        let error = analysis.program().unwrap_err();
        assert_eq!(error.source, input);
        assert!(error.span.is_some(), "{source}: {error}");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        assert!(analysis.module().is_err(), "{source}");
    }
}

#[test]
fn malformed_function_names_do_not_create_editor_definitions() {
    let source = "extern \"native.h\" def releasex: int); def main() = {};";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let input = project.source("main.resin");
    assert!(analysis.module().is_err());
    assert!(
        analysis
            .definition(&input, source.find("releasex").unwrap())
            .is_none()
    );
    assert!(
        analysis
            .completions(&input, source.len())
            .iter()
            .all(|item| item.name != "releasex")
    );
}

#[test]
#[cfg(unix)]
fn new_paths_resolve_through_symlinks() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
    assert_eq!(
        normalize_path(&root.join("alias/new.resin")).unwrap(),
        root.join("real/new.resin")
    );
}

#[test]
fn source_buffers_need_no_disk_write() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
    let path = temp.path().join("lib.resin");
    std::fs::write(&path, "disk").unwrap();
    let mut loader = resin_source::Loader::new(temp.path().to_path_buf());
    let source = loader.source_from_text(&path, "buffer").unwrap();
    assert_eq!(source.text(), "buffer");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "disk");
}

#[test]
fn normalized_paths_are_stable_for_existing_and_unsaved_files() {
    let temp = TempDir::new_in(std::env::temp_dir()).unwrap();
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
        &project.source("main.resin"),
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
                .completions(&project.source("main.resin"), at)
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
        &project.source("main.resin"),
        source.rfind("val").unwrap() + 3,
    );
    let value = items.iter().find(|item| item.name == "value").unwrap();
    assert_eq!(value.kind, resin_hir::DefinitionKind::Variable);

    // Module type definitions run before function signatures, but run in source
    // order relative to each other. Completion must match those compiler phases.
    for (source, expected) in [
        ("def main (value: Num) -> () = {}; type Number = int;", true),
        ("type Earlier = Num; type Number = int;", false),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let items = analysis.completions(
            &project.source("main.resin"),
            source.find("Num").unwrap() + 2,
        );
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
            .completions(&project.source("main.resin"), 10)
            .is_empty()
    );
}

#[test]
fn module_analysis_needs_no_entry_and_rejects_runtime_globals() {
    let source = "export { run }; def run () -> int = { helper() }; def helper () -> int = { 1 };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
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
            .definition(&project.source("main.resin"), source.find("run").unwrap())
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
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        let at = source.rfind("glo").unwrap();
        assert!(
            analysis
                .definition(&project.source("main.resin"), at)
                .is_none()
        );
        assert!(
            !analysis
                .completions(&project.source("main.resin"), at + 3)
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
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    for (name, signature) in [
        (
            "release",
            "extern \"native.h\" def release(value: int) -> ()",
        ),
        ("run", "def run() -> ()"),
    ] {
        let offset = source.rfind(name).unwrap();
        assert_eq!(analysis.hover(&input, offset).unwrap().text, signature);
        let items = analysis.completions(&input, offset + name.len());
        assert_eq!(
            items.iter().find(|item| item.name == name).unwrap().detail,
            signature
        );
        assert_eq!(
            analysis.definition(&input, offset).unwrap().span.start,
            source.find(name).unwrap()
        );
    }
    let project = Project::new(&[("main.resin", "def run() = { var value = 1; ")]);
    let items = project
        .analyze()
        .completions(&project.source("main.resin"), 28);
    for keyword in ["def", "var", "type"] {
        assert!(items.iter().any(|item| item.name == keyword), "{items:?}");
    }
    let project = Project::new(&[("main.resin", "def run() = { 1 };")]);
    assert!(
        !project.analyze().diagnostics().is_empty(),
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
        let input = project.source("main.resin");
        assert!(analysis.module().is_err(), "{source}");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        let items = analysis.completions(&input, source.find("point.").unwrap() + 6);
        assert_eq!(
            items.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
            ["x", "y"],
            "{source}"
        );
        let hover = analysis
            .hover(&input, source.rfind("result").unwrap())
            .unwrap();
        assert_eq!(hover.text, "result: int", "{source}");
    }
}

#[test]
fn unknown_bindings_shadow_outer_values_without_fabricating_types() {
    for initializer in ["", "missing(1)", "1 + (1 == 1)"] {
        let source = format!(
            "def main(point: {{ x: int }}) = {{ var point = {initializer}; var alias = point; alias.; var healthy = {{ count = 42 }}; healthy.count; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert_eq!(
            analysis
                .hover(&input, source.rfind("alias").unwrap())
                .unwrap()
                .text,
            "alias: ?"
        );
        assert!(
            analysis
                .completions(&input, source.find("alias.").unwrap() + 6)
                .is_empty()
        );
        assert_eq!(
            analysis
                .definition(&input, source.rfind("point").unwrap())
                .unwrap()
                .span
                .start,
            source.find("var point").unwrap() + 4
        );
        assert!(analysis.module().is_err());
        let fields = analysis.completions(&input, source.rfind("count").unwrap());
        assert_eq!(
            fields
                .iter()
                .map(|item| item.detail.as_str())
                .collect::<Vec<_>>(),
            ["count: long"],
            "{source}"
        );
    }
}

#[test]
fn unrelated_errors_preserve_expression_types_and_field_completion() {
    let mut failures = Vec::new();
    for (setup, expected) in [
        ("var value = -128b;", "sbyte"),
        ("var value = float32(42);", "float32"),
        ("var unused: int; var value = size_of(unused);", "ulong"),
        (
            "var value: Result<int, Never>; value := ok(42);",
            "Result<int, Never>",
        ),
    ] {
        for broken in ["", "def broken() = { missing; };"] {
            let source = format!(
                "{broken} def main() = {{ {setup} value; var record = {{ payload = value }}; record.payload; }};"
            );
            let project = Project::new(&[("main.resin", &source)]);
            let analysis = project.analyze();
            let input = project.source("main.resin");
            assert_eq!(
                analysis.diagnostics().is_empty(),
                broken.is_empty(),
                "{source}: {:?}",
                analysis.diagnostics()
            );
            assert_eq!(analysis.module().is_ok(), broken.is_empty(), "{source}");
            let hover = analysis
                .hover(&input, source.find("value;").unwrap())
                .unwrap();
            let fields = analysis.completions(&input, source.rfind("payload").unwrap());
            let details: Vec<_> = fields.iter().map(|item| item.detail.as_str()).collect();
            if hover.text != format!("value: {expected}")
                || details != [format!("payload: {expected}")]
            {
                failures.push(format!(
                    "{source}\nexpected {expected}, got {} and {details:?}",
                    hover.text
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn failed_compound_constraints_do_not_poison_independent_inference() {
    let source = "def main() = { var value: _; var broken: { first: int, second: bool }; broken := { first = value, second = 0 }; value := 1.5f; value; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(!analysis.diagnostics().is_empty());
    assert!(analysis.module().is_err());
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("value").unwrap())
            .unwrap()
            .text,
        "value: float32"
    );
    assert_eq!(
        analysis
            .definition(&input, source.rfind("value").unwrap())
            .unwrap()
            .span
            .start,
        source.find("value").unwrap()
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
        &project.source("main.resin"),
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
    let ast = resin_ast::format_source(
        analysis
            .recovered_file(&project.source("main.resin"))
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
        "export { main }; struct Point { x: int }; def main(arg: Ptr<Point>) = { var value = arg.x + 1; print(fmt(\"{}\", value)); };",
        "def main(arg: int) -> int = { var pair = { left = arg, right = 1 }; if (arg == 0) (pair.left) else (pair.right) };",
        "def main() = { var values = [1, 2]; while (1 == 1) { var missing: Ptr<int>; }; };",
        "struct Cleanup { value: Ptr<int> }; impl Cleanup { def drop(self: Ptr<Cleanup>) = { self.value.* := 42; }; } def main() = { var n = 0; var cleanup = Cleanup { value = &n }; };",
        "struct Item { value: int }; def main() = { var owner = Arc<Item> { value = 42 }; var weak = owner.downgrade(); match (weak.upgrade()) { Arc<Item>(item) => { item.value; }, None => {} }; };",
    ] {
        for end in 0..=source.len() {
            let project = Project::new(&[("main.resin", &source[..end])]);
            let analysis = project.analyze();
            analysis.completions(&project.source("main.resin"), end);
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
fn checking_rejects_holes_even_when_given_a_recovered_ast() {
    for source in [
        "def main() = { var value = ; };",
        "def main() = { var value: ; };",
        "def main(point: { x: int }) = { point.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let file = analysis
            .recovered_file(&project.source("main.resin"))
            .unwrap();
        let error = resin_hir::generate(file).unwrap_err();
        assert_eq!(
            error.kind,
            GenerateErrorKind::IncompleteSyntax,
            "{source}: {error}"
        );
        assert!(error.span.start <= error.span.end && error.span.end <= source.len());
    }
}

#[test]
fn indexing_and_shader_artifacts_keep_editor_types_and_completions() {
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { var xs = [1, 2]; var p = xs(0); var code = kernel.spirv; code.length; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.find("p =").unwrap())
            .unwrap()
            .text,
        "p: Ptr<long>"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("code.length").unwrap())
            .unwrap()
            .text,
        "code: Span<ubyte>"
    );
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { kernel. };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
        source.find("kernel. }").unwrap() + 7,
    );
    assert!(
        items
            .iter()
            .any(|i| i.name == "spirv" && i.detail.contains("Span<ubyte>")),
        "{items:?}"
    );
    let source = "@compute_shader def kernel(invocation: ulong, output: Ptr<uint>) = { var i = uint(invocation); output.* := { i }; }; def main() = { var alias = kernel; alias. };";
    let project = Project::new(&[("main.resin", source)]);
    let items = project.analyze().completions(
        &project.source("main.resin"),
        source.find("alias. }").unwrap() + 6,
    );
    assert!(items.iter().all(|i| i.name != "spirv"));
}

#[test]
fn suffixes_and_one_armed_if_have_editor_types() {
    let source = "def main() = { var count = 42_ul; if (count > 0_ul) { var speed = 1.5_f; speed; }; count; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(
        analysis.diagnostics().is_empty(),
        "{:?}",
        analysis.diagnostics()
    );
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("count").unwrap())
            .unwrap()
            .text,
        "count: ulong"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("speed").unwrap())
            .unwrap()
            .text,
        "speed: float32"
    );
}

#[test]
fn failed_children_invalidate_composites_without_hiding_later_bindings() {
    for expression in [
        "if (1 == 1) { missing } else { 42 }",
        "[missing, 42]",
        "ok(missing)",
        "place := missing",
    ] {
        let source = format!(
            "def f() = {{ var place = 1; var bad = {expression}; var alias = bad; var healthy = 1.5f; alias; healthy; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert!(!analysis.diagnostics().is_empty(), "{source}");
        assert!(analysis.module().is_err());
        for (name, expected) in [
            ("bad", "bad: ?"),
            ("alias", "alias: ?"),
            ("healthy", "healthy: float32"),
        ] {
            assert_eq!(
                analysis
                    .hover(&input, source.rfind(name).unwrap())
                    .unwrap()
                    .text,
                expected,
                "{source}"
            );
        }
    }
}

#[test]
fn broken_annotations_and_duplicate_declarations_retain_recognizable_children() {
    for (source, name, expected) in [
        (
            "def f() -> { a: _, b: _, c: _, d: Missing } = {}; def later() = { var healthy = 1.5f; healthy; };",
            "healthy",
            "healthy: float32",
        ),
        (
            "def f() = { var duplicate = 1; var duplicate = { var healthy = 1.5f; healthy }; };",
            "healthy",
            "healthy: float32",
        ),
        (
            "type T = int; def f() = { type T = Missing; var value: T; value; };",
            "value",
            "value: ?",
        ),
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        assert!(!analysis.diagnostics().is_empty());
        assert!(analysis.module().is_err());
        assert_eq!(
            analysis
                .hover(&project.source("main.resin"), source.rfind(name).unwrap())
                .unwrap()
                .text,
            expected,
            "{source}"
        );
    }
}

#[test]
fn unknown_exports_retain_identity_and_poison_consumers_through_reexports() {
    let library =
        "export { Broken, broken }; type Broken = Missing; def broken() -> _ = { missing };";
    let source = "import { \"left.resin\", \"right.resin\" }; def main() = { var value: Broken; var result = broken(); value; result; };";
    let project = Project::new(&[
        ("main.resin", source),
        ("base.resin", library),
        (
            "left.resin",
            "export { Broken, broken }; import { \"base.resin\" };",
        ),
        (
            "right.resin",
            "export { Broken, broken }; import { \"base.resin\" };",
        ),
    ]);
    let analysis = project.analyze();
    assert!(analysis.module().is_err());
    let input = project.source("main.resin");
    for name in ["Broken", "broken"] {
        let offset = source.rfind(name).unwrap();
        let definition = analysis
            .definition(&input, offset)
            .unwrap_or_else(|| panic!("missing {name}: {:?}", analysis.diagnostics()));
        assert_eq!(definition.source, project.source("base.resin"));
        assert_eq!(&library[definition.span.start..definition.span.end], name);
        assert!(
            analysis
                .completions(&input, offset)
                .iter()
                .any(|item| item.name == name)
        );
    }
    assert_eq!(
        analysis
            .hover(&input, source.rfind("result").unwrap())
            .unwrap()
            .text,
        "result: ?"
    );
    assert!(
        analysis
            .diagnostics()
            .iter()
            .all(|d| d.location.source != input || !d.message.contains("Unbound")),
        "{:?}",
        analysis.diagnostics()
    );
}

#[test]
fn callers_cannot_resurrect_failed_result_inference() {
    for (result, body, caller_result, expected) in [
        ("_", "missing", "int", "alias: ?"),
        (
            "_",
            "if (1 == 1) { caller() } else { missing }",
            "int",
            "alias: ?",
        ),
        (
            "_",
            "if (1 == 1) { caller() } else { 1 + () }",
            "int",
            "alias: ?",
        ),
        (
            "Result<int, _>",
            "missing",
            "Result<int, Never>",
            "alias: ?",
        ),
        ("int", "missing", "int", "alias: (()) -> int"),
    ] {
        let source = format!(
            "def broken() -> {result} = {{ var healthy = 1.5f; healthy; {body} }}; def caller() -> {caller_result} = {{ broken() }}; def observer() = {{ var alias = broken; alias; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert!(analysis.program().is_ok(), "{source}");
        assert!(!analysis.diagnostics().is_empty());
        assert!(analysis.module().is_err());
        assert_eq!(
            analysis
                .hover(&input, source.rfind("alias").unwrap())
                .unwrap()
                .text,
            expected,
            "{source}"
        );
        assert_eq!(
            analysis
                .hover(&input, source.rfind("healthy").unwrap())
                .unwrap()
                .text,
            "healthy: float32"
        );
    }
}

#[test]
fn recursive_failure_discards_copied_caller_result_facts() {
    let source = "def broken() -> _ = { if (1 == 0) { caller() } else { 1 + () } }; def caller() -> _ = { broken() }; def observer() = { var forced = int(caller()); var alias = caller; var failed = broken; alias; failed; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    assert!(analysis.program().is_ok());
    assert!(!analysis.diagnostics().is_empty());
    assert!(analysis.module().is_err());
    let input = project.source("main.resin");
    assert_eq!(
        analysis
            .hover(&input, source.rfind("alias").unwrap())
            .unwrap()
            .text,
        "alias: ?"
    );
    assert_eq!(
        analysis
            .hover(&input, source.rfind("failed").unwrap())
            .unwrap()
            .text,
        "failed: ?"
    );
}

#[test]
fn incomplete_impls_and_method_arguments_keep_editor_recovery() {
    let source = "struct Counter { count: int }; impl Counter { def add(counter: Counter, amount: int) -> int = { counter.count + amount }; } def f(c: Counter) -> int = { c.add(1) };";
    for end in source
        .char_indices()
        .map(|(index, _)| index)
        .chain([source.len()])
    {
        let project = Project::new(&[("main.resin", &source[..end])]);
        project.analyze();
    }
    let source = "struct Counter { count: int }; impl Counter { def read(counter: Counter) -> int = { counter.count }; } def f(c: Counter) = { c.read(; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let offset = source.rfind("read").unwrap();
    assert!(
        analysis
            .definition(&project.source("main.resin"), offset)
            .is_some()
    );
}

#[test]
fn pointer_replace_has_ordinary_method_hover_and_recovery() {
    for tail in ["", "p.;"] {
        let source = format!("def f(p: Ptr<int>) = {{ p.replace(3); {tail} }};");
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert_eq!(
            analysis.diagnostics().is_empty(),
            tail.is_empty(),
            "{:?}",
            analysis.diagnostics()
        );
        let offset = source.find("replace").unwrap();
        assert_eq!(
            analysis.hover(&input, offset).unwrap().text,
            "replace: (int) -> int"
        );
        let items = analysis.completions(
            &input,
            if tail.is_empty() {
                offset
            } else {
                source.rfind("p.;").unwrap() + 2
            },
        );
        assert!(
            items
                .iter()
                .any(|item| item.name == "replace"
                    && item.kind == resin_hir::DefinitionKind::Function),
            "{items:?}"
        );
    }
}

#[test]
fn formatted_string_and_literal_string_types_survive_editor_recovery() {
    for source in [
        "def main() = { var text = fmt(\"{0}\", (42,)); text.bytes.; };",
        "def main() = { var text = \"literal\"; text.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let offset = source.rfind(".;").unwrap() + 1;
        let items = project
            .analyze()
            .completions(&project.source("main.resin"), offset);
        assert!(items.iter().any(|item| item.name == "data"), "{items:?}");
        assert!(items.iter().any(|item| item.name == "length"), "{items:?}");
    }
}

#[test]
fn invalid_method_arguments_preserve_receiver_facts_and_later_bindings() {
    for duplicate in ["", "def read(self: Missing) -> int = { 0 };"] {
        let source = format!(
            "struct Item {{ count: int }}; impl Item {{ def read(self: Ptr<Item>) -> int = {{ self.count }}; {duplicate} }} def f(c: Item) = {{ var bad = c.read(missing); var alias = bad; var healthy = 1.5f; alias; healthy; c.; }};"
        );
        let project = Project::new(&[("main.resin", &source)]);
        let analysis = project.analyze();
        let input = project.source("main.resin");
        assert!(analysis.module().is_err());
        for (name, expected) in [("alias", "alias: ?"), ("healthy", "healthy: float32")] {
            assert_eq!(
                analysis
                    .hover(&input, source.rfind(name).unwrap())
                    .unwrap()
                    .text,
                expected
            );
        }
        assert_eq!(
            analysis
                .definition(&input, source.find("c.read").unwrap() + 2)
                .unwrap()
                .span
                .start,
            source.find("read(self").unwrap()
        );
        let members = analysis.completions(&input, source.rfind("c.;").unwrap() + 2);
        assert!(members.iter().any(|member| member.name == "read"));
        assert!(members.iter().any(|member| member.detail == "count: int"));
    }
}

#[test]
fn string_constructor_is_an_ordinary_discoverable_static_method() {
    for source in [
        "def main() = { var text = String.from_str(\"title\"); text.bytes.; };",
        "def main() = { String.; };",
    ] {
        let project = Project::new(&[("main.resin", source)]);
        let analysis = project.analyze();
        let offset = source.rfind(".;").unwrap() + 1;
        let items = analysis.completions(&project.source("main.resin"), offset);
        let member = if source.contains("from_str") {
            "length"
        } else {
            "from_str"
        };
        assert!(items.iter().any(|item| item.name == member), "{items:?}");
        if member == "from_str" {
            assert!(
                items.iter().any(|item| item.name == "from_bytes"),
                "{items:?}"
            );
        }
    }
}

#[test]
fn literal_string_hover_preserves_its_distinct_primitive_type() {
    let source = "def main() = { var text = \"literal\"; text; };";
    let project = Project::new(&[("main.resin", source)]);
    let analysis = project.analyze();
    let hover = analysis
        .hover(&project.source("main.resin"), source.rfind("text").unwrap())
        .unwrap();
    assert_eq!(hover.text, "text: str");
}
