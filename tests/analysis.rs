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
fn buffers_override_disk_and_new_paths_resolve_through_symlinks() {
    let temp = resin::toolchain::TempDir::new(&std::env::temp_dir()).unwrap();
    let root = std::fs::canonicalize(temp.path()).unwrap();
    std::fs::create_dir(root.join("real")).unwrap();
    std::os::unix::fs::symlink(root.join("real"), root.join("alias")).unwrap();
    assert_eq!(
        normalize_path(&root.join("alias/new.resin")).unwrap(),
        root.join("real/new.resin")
    );
    std::fs::write(root.join("real/lib.resin"), "disk").unwrap();
    let path = normalize_path(&root.join("alias/lib.resin")).unwrap();
    let sources = Sources {
        overlays: [(path.clone(), "buffer".into())].into(),
    };
    assert_eq!(sources.read(&path).unwrap(), "buffer");
}

#[test]
fn completion_respects_type_context_and_ignores_fields_strings_and_comments() {
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
    for source in [
        "def main () -> () = { var value = {field = 1}; value.fi",
        "// val",
        "def main () -> () = { var value = \"val\"; };",
    ] {
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
