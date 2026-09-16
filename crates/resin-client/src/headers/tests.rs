use super::*;
use base64::{Engine, engine::general_purpose::STANDARD};
use resin_protocol::{Capabilities, EntryProfile, ManagedHeaderRoot};
use std::sync::Arc;
use tempfile::TempDir;

fn options(include_roots: Vec<PathBuf>) -> CaptureOptions {
    CaptureOptions {
        include_roots,
        capabilities: Arc::new(Capabilities {
            protocol: resin_protocol::PROTOCOL_VERSION,
            instance: "server".into(),
            managed_snapshot: "managed".into(),
            targets: Vec::new(),
            entry_profiles: vec![EntryProfile::Host],
            header_roots: vec![
                ManagedHeaderRoot {
                    id: "runtime".into(),
                    headers: vec!["resin_runtime.h".into()],
                },
                ManagedHeaderRoot {
                    id: "package/demo/include".into(),
                    headers: vec!["package.h".into()],
                },
                ManagedHeaderRoot {
                    id: "system".into(),
                    headers: Vec::new(),
                },
            ],
        }),
    }
}

fn declaration(source: &str, directory: &Path, spelling: &str) -> Declaration {
    Declaration {
        source: source.into(),
        path: directory.join("main.resin"),
        spelling: spelling.into(),
        span: resin_protocol::Span {
            source: source.into(),
            start: 0,
            end: 0,
        },
    }
}

async fn snapshot(
    declarations: Vec<Declaration>,
    include_roots: Vec<PathBuf>,
) -> (HeaderInputs, Vec<Diagnostic>) {
    capture(
        declarations,
        &options(include_roots),
        &Execution::default(),
        &Cancellation::new(),
    )
    .await
    .unwrap()
}

fn bytes(headers: &HeaderInputs, target: &HeaderTarget) -> Vec<u8> {
    let HeaderTarget::Uploaded { bundle, path } = target else {
        panic!("expected uploaded header")
    };
    let bundle = headers
        .bundles
        .iter()
        .find(|candidate| candidate.id == *bundle)
        .unwrap();
    STANDARD
        .decode(
            &bundle
                .files
                .iter()
                .find(|file| file.path == *path)
                .unwrap()
                .contents_base64,
        )
        .unwrap()
}

#[tokio::test]
async fn directories_capture_nested_binary_inputs_and_keep_same_basename_bindings_distinct() {
    let directory = TempDir::new().unwrap();
    for (name, contents) in [("left", b"left".as_slice()), ("right", b"right".as_slice())] {
        let root = directory.path().join(name);
        fs::create_dir_all(root.join("nested")).unwrap();
        fs::write(root.join("api.h"), contents).unwrap();
        fs::write(root.join("nested/asset.bin"), [0, 255, 3, 0]).unwrap();
        fs::write(root.join("nested/config.inc"), "configuration").unwrap();
    }
    let (headers, diagnostics) = snapshot(
        vec![
            declaration("left/main.resin", &directory.path().join("left"), "api.h"),
            declaration("right/main.resin", &directory.path().join("right"), "api.h"),
        ],
        Vec::new(),
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(headers.bundles.len(), 2);
    assert_eq!(bytes(&headers, &headers.bindings[0].target), b"left");
    assert_eq!(bytes(&headers, &headers.bindings[1].target), b"right");
    for bundle in &headers.bundles {
        assert_eq!(bundle.files.len(), 3);
        let binary = bundle
            .files
            .iter()
            .find(|file| file.path == "nested/asset.bin")
            .unwrap();
        assert_eq!(
            STANDARD.decode(&binary.contents_base64).unwrap(),
            [0, 255, 3, 0]
        );
    }
}

#[tokio::test]
async fn explicit_roots_preserve_search_order_and_coalesce_overlapping_uploads() {
    let directory = TempDir::new().unwrap();
    let include = directory.path().join("include");
    fs::create_dir_all(include.join("sub")).unwrap();
    fs::create_dir(include.join("other")).unwrap();
    fs::write(include.join("api.h"), "outer").unwrap();
    fs::write(include.join("sub/api.h"), "first search root").unwrap();
    fs::write(directory.path().join("api.h"), "importer relative").unwrap();
    let (headers, diagnostics) = snapshot(
        vec![declaration("main.resin", directory.path(), "api.h")],
        vec![include.join("sub"), include.clone(), include.join("other")],
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(headers.bundles.len(), 1);
    assert_eq!(
        bytes(&headers, &headers.bindings[0].target),
        b"first search root"
    );
    let directories: Vec<_> = headers
        .include_roots
        .iter()
        .filter_map(|root| match root {
            IncludeRoot::Uploaded { directory, .. } => Some(directory.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(directories, ["sub", "", "other"]);
    assert!(
        matches!(&headers.bindings[0].target, HeaderTarget::Uploaded { path, .. } if path == "sub/api.h")
    );
}

#[tokio::test]
async fn local_headers_override_managed_catalogues_and_system_is_only_an_advertised_fallback() {
    let directory = TempDir::new().unwrap();
    fs::write(
        directory.path().join("resin_runtime.h"),
        "user runtime spelling",
    )
    .unwrap();
    let (headers, diagnostics) = snapshot(
        vec![
            declaration("main.resin", directory.path(), "resin_runtime.h"),
            declaration("main.resin", directory.path(), "package.h"),
            declaration("main.resin", directory.path(), "stdint.h"),
        ],
        Vec::new(),
    )
    .await;
    assert!(diagnostics.is_empty());
    assert!(
        matches!(&headers.bindings[0].target, HeaderTarget::Managed { root, .. } if root == "package/demo/include")
    );
    assert_eq!(
        bytes(&headers, &headers.bindings[1].target),
        b"user runtime spelling"
    );
    assert!(
        matches!(&headers.bindings[2].target, HeaderTarget::Managed { root, .. } if root == "system")
    );
    fs::remove_file(directory.path().join("resin_runtime.h")).unwrap();
    let (managed, diagnostics) = snapshot(
        vec![declaration(
            "main.resin",
            directory.path(),
            "resin_runtime.h",
        )],
        Vec::new(),
    )
    .await;
    assert!(diagnostics.is_empty());
    assert!(
        matches!(&managed.bindings[0].target, HeaderTarget::Managed { root, .. } if root == "runtime")
    );
    let mut no_system = options(Vec::new());
    Arc::make_mut(&mut no_system.capabilities)
        .header_roots
        .retain(|root| root.id != "system");
    let (missing, diagnostics) = capture(
        vec![declaration("main.resin", directory.path(), "missing.h")],
        &no_system,
        &Execution::default(),
        &Cancellation::new(),
    )
    .await
    .unwrap();
    assert!(missing.bindings.is_empty());
    assert_eq!(diagnostics.len(), 1);
}

#[tokio::test]
async fn absolute_headers_bind_uploaded_paths_and_never_fall_back_to_server_files() {
    let directory = TempDir::new().unwrap();
    let includes = directory.path().join("includes");
    fs::create_dir(&includes).unwrap();
    let path = includes.join("api.h");
    fs::write(&path, "absolute local input").unwrap();
    let (headers, diagnostics) = snapshot(
        vec![declaration(
            "main.resin",
            directory.path(),
            path.to_str().unwrap(),
        )],
        Vec::new(),
    )
    .await;
    assert!(diagnostics.is_empty());
    assert!(
        matches!(&headers.bindings[0].target, HeaderTarget::Uploaded { path, .. } if path == "api.h")
    );
    assert_eq!(
        bytes(&headers, &headers.bindings[0].target),
        b"absolute local input"
    );
    fs::remove_file(&path).unwrap();
    let (missing, diagnostics) = snapshot(
        vec![declaration(
            "main.resin",
            directory.path(),
            path.to_str().unwrap(),
        )],
        Vec::new(),
    )
    .await;
    assert!(missing.bindings.is_empty());
    assert_eq!(diagnostics.len(), 1);
    assert!(
        !diagnostics[0]
            .message
            .contains(directory.path().to_str().unwrap())
    );
}

#[tokio::test]
async fn relocation_preserves_bundle_identity_and_all_content_changes_invalidate_it() {
    let directory = TempDir::new().unwrap();
    for name in ["left", "right"] {
        let root = directory.path().join(name);
        fs::create_dir(&root).unwrap();
        fs::write(root.join("api.h"), "header").unwrap();
        fs::write(root.join("unused.inc"), "unused bytes").unwrap();
    }
    let left_root = directory.path().join("left");
    let right_root = directory.path().join("right");
    let (left, _) = snapshot(
        vec![declaration("main.resin", &left_root, "api.h")],
        Vec::new(),
    )
    .await;
    let (right, _) = snapshot(
        vec![declaration("main.resin", &right_root, "api.h")],
        Vec::new(),
    )
    .await;
    assert_eq!(left, right);
    let original = &left.bundles[0].id;
    fs::write(left_root.join("unused.inc"), "changed bytes").unwrap();
    let (changed, _) = snapshot(
        vec![declaration("main.resin", &left_root, "api.h")],
        Vec::new(),
    )
    .await;
    assert_ne!(&changed.bundles[0].id, original);
    fs::write(left_root.join("added.dat"), [1, 2, 3]).unwrap();
    let (added, _) = snapshot(
        vec![declaration("main.resin", &left_root, "api.h")],
        Vec::new(),
    )
    .await;
    assert_ne!(added.bundles[0].id, changed.bundles[0].id);
    fs::remove_file(left_root.join("unused.inc")).unwrap();
    let (deleted, _) = snapshot(
        vec![declaration("main.resin", &left_root, "api.h")],
        Vec::new(),
    )
    .await;
    assert_ne!(deleted.bundles[0].id, added.bundles[0].id);
}

#[cfg(unix)]
#[tokio::test]
async fn contained_symlinks_flatten_and_external_links_need_an_explicit_root() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new().unwrap();
    let local = directory.path().join("local");
    let external = directory.path().join("external");
    fs::create_dir(&local).unwrap();
    fs::create_dir(&external).unwrap();
    fs::write(local.join("real.h"), "header bytes").unwrap();
    symlink("real.h", local.join("api.h")).unwrap();
    let (contained, diagnostics) =
        snapshot(vec![declaration("main.resin", &local, "api.h")], Vec::new()).await;
    assert!(diagnostics.is_empty());
    assert_eq!(contained.bundles[0].files.len(), 2);
    assert_eq!(
        bytes(&contained, &contained.bindings[0].target),
        b"header bytes"
    );
    fs::write(external.join("blob.dat"), [9, 0, 255]).unwrap();
    symlink(external.join("blob.dat"), local.join("external.dat")).unwrap();
    let (rejected, diagnostics) =
        snapshot(vec![declaration("main.resin", &local, "api.h")], Vec::new()).await;
    assert!(rejected.bindings.is_empty());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("outside the selected roots"))
    );
    let (supported, diagnostics) = snapshot(
        vec![declaration("main.resin", &local, "api.h")],
        vec![external],
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(supported.bundles.len(), 2);
    assert!(
        supported
            .bundles
            .iter()
            .any(|bundle| bundle.files.iter().any(|file| file.path == "external.dat"))
    );
}

#[cfg(unix)]
#[tokio::test]
async fn direct_directory_symlinks_do_not_implicitly_select_external_roots() {
    use std::os::unix::fs::symlink;
    let directory = TempDir::new().unwrap();
    let local = directory.path().join("local");
    let external = directory.path().join("external");
    fs::create_dir(&local).unwrap();
    fs::create_dir(&external).unwrap();
    fs::write(external.join("api.h"), "external header").unwrap();
    symlink(&external, local.join("linked")).unwrap();
    let (rejected, diagnostics) = snapshot(
        vec![declaration("main.resin", &local, "linked/api.h")],
        Vec::new(),
    )
    .await;
    assert!(rejected.bindings.is_empty());
    assert!(rejected.bundles.is_empty());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("outside the selected roots"))
    );
    let (allowed, diagnostics) = snapshot(
        vec![declaration("main.resin", &local, "linked/api.h")],
        vec![external],
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(
        bytes(&allowed, &allowed.bindings[0].target),
        b"external header"
    );

    fs::create_dir(local.join("nested")).unwrap();
    fs::write(local.join("nested/local.h"), "contained header").unwrap();
    symlink("nested", local.join("alias")).unwrap();
    let (contained, diagnostics) = snapshot(
        vec![declaration("main.resin", &local, "alias/local.h")],
        Vec::new(),
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(
        bytes(&contained, &contained.bindings[0].target),
        b"contained header"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn symlink_cycles_and_nonportable_names_are_diagnostics_without_partial_bundles() {
    use std::{
        ffi::OsStr,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let directory = TempDir::new().unwrap();
    fs::write(directory.path().join("api.h"), "header").unwrap();
    symlink(".", directory.path().join("cycle")).unwrap();
    let (cyclic, diagnostics) = snapshot(
        vec![declaration("main.resin", directory.path(), "api.h")],
        Vec::new(),
    )
    .await;
    assert!(cyclic.bundles.is_empty());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("symlink cycle"))
    );
    fs::remove_file(directory.path().join("cycle")).unwrap();
    fs::write(
        directory.path().join(OsStr::from_bytes(b"invalid\xff.inc")),
        "bytes",
    )
    .unwrap();
    let (invalid, diagnostics) = snapshot(
        vec![declaration("main.resin", directory.path(), "api.h")],
        Vec::new(),
    )
    .await;
    assert!(invalid.bundles.is_empty());
    assert!(
        diagnostics
            .iter()
            .any(|diagnostic| diagnostic.message.contains("UTF-8"))
    );
}

#[tokio::test]
async fn duplicate_declarations_keep_one_binding_and_empty_explicit_roots_have_stable_bundles() {
    let directory = TempDir::new().unwrap();
    let (headers, diagnostics) = snapshot(
        vec![
            declaration("main.resin", directory.path(), "stdint.h"),
            declaration("main.resin", directory.path(), "stdint.h"),
        ],
        vec![directory.path().into()],
    )
    .await;
    assert!(diagnostics.is_empty());
    assert_eq!(headers.bindings.len(), 1);
    assert_eq!(headers.bundles.len(), 1);
    assert!(headers.bundles[0].files.is_empty());
    assert_eq!(headers.bundles[0].id.len(), 64);
}

#[tokio::test]
async fn cancellation_does_not_turn_into_an_acquisition_diagnostic() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let result = capture(
        Vec::new(),
        &options(Vec::new()),
        &Execution::default(),
        &cancellation,
    )
    .await;
    let error = result.expect_err("cancelled capture");
    assert_eq!(
        error.downcast_ref::<resin_executor::Error>(),
        Some(&resin_executor::Error::Cancelled)
    );
}

#[test]
fn portable_paths_reject_windows_device_names_in_every_component() {
    for name in [
        "CON", "prn.h", "Aux.fn", "nul.bin", "com1", "COM9.h", "lpt1.inc", "LPT9",
    ] {
        assert!(!tree::portable(name), "{name}");
        assert!(!tree::portable(&format!("nested/{name}/api.h")), "{name}");
    }
    for name in ["com0.h", "COM10.h", "auxiliary.h", "é/日.h"] {
        assert!(tree::portable(name), "{name}");
    }
}

#[cfg(target_os = "linux")]
#[tokio::test]
async fn ascii_case_aliases_are_rejected_for_files_and_directory_prefixes() {
    for paths in [["A.h", "a.h"], ["A/x.h", "a/y.h"], ["A", "a/y.h"]] {
        let directory = TempDir::new().unwrap();
        for path in paths {
            let path = directory.path().join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, "bytes").unwrap();
        }
        let (headers, diagnostics) = snapshot(Vec::new(), vec![directory.path().into()]).await;
        assert!(headers.bundles.is_empty());
        assert!(
            diagnostics
                .iter()
                .any(|diagnostic| diagnostic.message.contains("collide ignoring ASCII case")),
            "{paths:?}"
        );
    }
}
