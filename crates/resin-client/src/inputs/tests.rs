use super::*;
use resin_protocol::{EntryProfile, HeaderInputs, ManagedHeaderRoot};
use std::fs;
use tempfile::TempDir;

fn options() -> CaptureOptions {
    CaptureOptions {
        include_roots: Vec::new(),
        capabilities: Arc::new(Capabilities {
            protocol: resin_protocol::PROTOCOL_VERSION,
            instance: "instance".into(),
            managed_snapshot: "managed".into(),
            targets: Vec::new(),
            entry_profiles: vec![EntryProfile::Host],
            header_roots: vec![ManagedHeaderRoot {
                id: "system".into(),
                headers: Vec::new(),
            }],
        }),
    }
}

async fn read(path: &Path) -> CapturedInputs {
    let mut loader = Loader::new(path.parent().unwrap().to_owned());
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let entry = loader
        .load_file_async(path, &execution, &cancellation)
        .await
        .unwrap();
    capture(entry, &mut loader, &options(), &execution, &cancellation)
        .await
        .unwrap()
}

#[tokio::test]
async fn recovering_sources_discover_nested_parent_and_cyclic_user_imports_only() {
    let directory = TempDir::new().unwrap();
    let project = directory.path().join("project");
    fs::create_dir_all(project.join("sub")).unwrap();
    let main = "import { \"sub/child.resin\", \"../shared.resin\", \"$/missing.resin\", \"$/deps/package/api.resin\" }; fn main()  { let mut hole = ; }";
    fs::write(project.join("main.resin"), main).unwrap();
    fs::write(
        project.join("sub/child.resin"),
        "import { \"../main.resin\", \"../../shared.resin\" }; fn child()  { 1 }",
    )
    .unwrap();
    fs::write(directory.path().join("shared.resin"), "fn shared()  { 2 }").unwrap();
    let captured = read(&project.join("main.resin")).await;
    assert_eq!(captured.inputs.entry, "main.resin");
    let names: Vec<_> = captured
        .inputs
        .sources
        .iter()
        .map(|source| source.name.as_str())
        .collect();
    assert_eq!(names, ["../shared.resin", "main.resin", "sub/child.resin"]);
    assert_eq!(captured.inputs.imports.len(), 4);
    assert!(
        captured
            .inputs
            .imports
            .iter()
            .all(|binding| !binding.reference.starts_with("$/"))
    );
    assert!(captured.inputs.acquisition_diagnostics.is_empty());
    assert_eq!(captured.origins["main.resin"].source.text(), main);
    let wire = serde_json::to_string(&captured.inputs).unwrap();
    assert!(!wire.contains(directory.path().to_str().unwrap()));
}

#[cfg(unix)]
#[tokio::test]
async fn aliases_select_one_physical_source_with_two_explicit_edges() {
    let directory = TempDir::new().unwrap();
    fs::write(
        directory.path().join("main.resin"),
        "import { \"value.resin\", \"alias.resin\" }; fn main()  { 0 }",
    )
    .unwrap();
    fs::write(directory.path().join("value.resin"), "fn value()  { 1 }").unwrap();
    std::os::unix::fs::symlink("value.resin", directory.path().join("alias.resin")).unwrap();
    let captured = read(&directory.path().join("main.resin")).await;
    assert_eq!(captured.inputs.sources.len(), 2);
    assert_eq!(captured.inputs.imports.len(), 2);
    assert!(
        captured
            .inputs
            .imports
            .iter()
            .all(|binding| binding.target == "value.resin")
    );
}

#[tokio::test]
async fn relocated_trees_upload_equal_inputs_with_separate_local_origins() {
    let directory = TempDir::new().unwrap();
    for name in ["left", "right"] {
        let root = directory.path().join(name);
        fs::create_dir(&root).unwrap();
        fs::write(
            root.join("main.resin"),
            "import { \"child.resin\" }; fn main()  { value() }",
        )
        .unwrap();
        fs::write(
            root.join("child.resin"),
            "export { value }; fn value()  { 1 }",
        )
        .unwrap();
    }
    let left = read(&directory.path().join("left/main.resin")).await;
    let right = read(&directory.path().join("right/main.resin")).await;
    assert_eq!(left.inputs, right.inputs);
    assert_ne!(
        left.origins["main.resin"].path,
        right.origins["main.resin"].path
    );
}

#[tokio::test]
async fn missing_imports_produce_portable_diagnostics_and_repair_on_next_capture() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, "import { \"missing.resin\" }; fn main()  { 1 }").unwrap();
    let missing = read(&path).await;
    assert_eq!(missing.inputs.acquisition_diagnostics.len(), 1);
    let diagnostic = &missing.inputs.acquisition_diagnostics[0];
    assert_eq!(diagnostic.span.as_ref().unwrap().source, "main.resin");
    assert!(diagnostic.message.contains("file not found"));
    assert!(
        !diagnostic
            .message
            .contains(directory.path().to_str().unwrap())
    );
    fs::write(directory.path().join("missing.resin"), "fn helper()  { 1 }").unwrap();
    let repaired = read(&path).await;
    assert!(repaired.inputs.acquisition_diagnostics.is_empty());
    assert_eq!(repaired.inputs.sources.len(), 2);
    assert_eq!(missing.inputs.sources.len(), 1);
}

#[tokio::test]
async fn captured_editor_snapshot_keeps_old_dependency_text_while_later_edits_and_disk_differ() {
    let directory = TempDir::new().unwrap();
    let main_path = directory.path().join("main.resin");
    let helper_path = directory.path().join("helper.resin");
    let main_text = "import { \"helper.resin\" }; fn main()  { 1 }";
    fs::write(&main_path, main_text).unwrap();
    fs::write(&helper_path, "fn value()  { 0 }").unwrap();
    let mut accepted = Loader::new(directory.path().into());
    let entry = accepted.source_from_text(&main_path, main_text).unwrap();
    accepted
        .source_from_text(&helper_path, "fn value()  { 1 }")
        .unwrap();
    let mut request_loader = accepted.supplied_snapshot();
    accepted
        .source_from_text(&helper_path, "fn value()  { 2 }")
        .unwrap();
    let captured = capture(
        entry,
        &mut request_loader,
        &options(),
        &Execution::default(),
        &Cancellation::new(),
    )
    .await
    .unwrap();
    assert_eq!(
        captured.origins["helper.resin"].source.text(),
        "fn value()  { 1 }"
    );
    let disk = read(&main_path).await;
    assert_eq!(
        disk.origins["helper.resin"].source.text(),
        "fn value()  { 0 }"
    );
    assert_ne!(captured.inputs, disk.inputs);
    assert_eq!(
        fs::read_to_string(helper_path).unwrap(),
        "fn value()  { 0 }"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn non_utf8_source_names_are_lossless_on_the_wire() {
    use std::{ffi::OsStr, os::unix::ffi::OsStrExt};
    let directory = TempDir::new().unwrap();
    let path = directory.path().join(OsStr::from_bytes(b"main\xff.resin"));
    fs::write(&path, "fn main()  { 1 }").unwrap();
    let captured = read(&path).await;
    assert_eq!(captured.inputs.entry, "main%FF.resin");
    assert_eq!(captured.origins["main%FF.resin"].path, path);
}

#[tokio::test]
async fn cancellation_returns_no_partial_captured_inputs() {
    let directory = TempDir::new().unwrap();
    let mut loader = Loader::new(directory.path().into());
    let entry = loader
        .source_from_text(&directory.path().join("main.resin"), "text")
        .unwrap();
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let result = capture(
        entry,
        &mut loader,
        &options(),
        &Execution::default(),
        &cancellation,
    )
    .await;
    assert!(result.is_err());
}

#[test]
fn delta_replaces_complete_metadata_and_only_changed_source_membership() {
    let old = Inputs {
        entry: "old.resin".into(),
        sources: vec![
            resin_protocol::SourceFile {
                name: "old.resin".into(),
                text: "old".into(),
            },
            resin_protocol::SourceFile {
                name: "same.resin".into(),
                text: "same".into(),
            },
            resin_protocol::SourceFile {
                name: "changed.resin".into(),
                text: "before".into(),
            },
        ],
        imports: Vec::new(),
        headers: HeaderInputs::default(),
        acquisition_diagnostics: Vec::new(),
        managed_snapshot: "old-managed".into(),
    };
    let mut current = old.clone();
    current.entry = "new.resin".into();
    current.sources[0] = resin_protocol::SourceFile {
        name: "new.resin".into(),
        text: "new".into(),
    };
    current.sources[2].text = "after".into();
    current.managed_snapshot = "new-managed".into();
    current.imports.push(resin_protocol::ImportBinding {
        importer: "new.resin".into(),
        reference: "same.resin".into(),
        target: "same.resin".into(),
    });
    let handle = InputHandle {
        instance: "server".into(),
        id: "handle".into(),
    };
    let InputSelection::Delta {
        base,
        entry,
        replacements,
        deleted,
        imports,
        headers,
        acquisition_diagnostics,
        managed_snapshot,
    } = selection(&current, Some((&handle, &old)))
    else {
        panic!("expected delta")
    };
    assert_eq!(base, handle);
    assert_eq!(entry, "new.resin");
    assert_eq!(
        replacements
            .iter()
            .map(|source| source.name.as_str())
            .collect::<Vec<_>>(),
        ["changed.resin", "new.resin"]
    );
    assert_eq!(deleted, ["old.resin"]);
    assert_eq!(imports, current.imports);
    assert_eq!(headers, current.headers);
    assert_eq!(acquisition_diagnostics, current.acquisition_diagnostics);
    assert_eq!(managed_snapshot, "new-managed");
    assert_eq!(
        selection(&current, None),
        InputSelection::Full { inputs: current }
    );
}
