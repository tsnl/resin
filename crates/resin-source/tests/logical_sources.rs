use resin_executor::{Cancellation, Execution};
use resin_source::{ImportBinding, LoadError, Loader, Source, SourceGraph, SourceId};
use std::{
    collections::{BTreeMap, hash_map::DefaultHasher},
    fs,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    path::{Path, PathBuf},
};
use tempfile::TempDir;

struct Checkout {
    _directory: TempDir,
    root: PathBuf,
    loader: Loader,
    entry: Source,
    helper: Source,
    library: Source,
}

impl Checkout {
    fn new() -> Self {
        let directory = TempDir::new().unwrap();
        let root = directory.path().join("checkout/src");
        let library_root = directory.path().join("toolchain/lib");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(root.parent().unwrap().join("shared")).unwrap();
        fs::create_dir_all(&library_root).unwrap();
        fs::write(
            root.join("main.resin"),
            "import { \"../shared/helper.resin\", \"$/core.resin\" };",
        )
        .unwrap();
        fs::write(
            root.parent().unwrap().join("shared/helper.resin"),
            "fn helper()  {}",
        )
        .unwrap();
        fs::write(library_root.join("core.resin"), "fn core()  {}").unwrap();
        let mut loader = Loader::new(library_root);
        let entry = loader.load_file(&root.join("main.resin")).unwrap();
        let helper = loader
            .load_import(&entry, "../shared/helper.resin")
            .unwrap();
        let library = loader.load_import(&entry, "$/core.resin").unwrap();
        Self {
            _directory: directory,
            root,
            loader,
            entry,
            helper,
            library,
        }
    }

    async fn mapping(&self) -> BTreeMap<SourceId, Source> {
        self.loader
            .logical_sources(
                [
                    self.entry.clone(),
                    self.helper.clone(),
                    self.library.clone(),
                ],
                &self.root,
                &Execution::default(),
                &Cancellation::new(),
            )
            .await
            .unwrap()
    }

    fn graph(&self, mapped: &BTreeMap<SourceId, Source>) -> SourceGraph {
        let entry = mapped[&self.entry.id()].clone();
        SourceGraph::new(
            entry.clone(),
            mapped.values().cloned(),
            [
                ImportBinding {
                    source: entry.id(),
                    reference: "../shared/helper.resin".into(),
                    target: mapped[&self.helper.id()].id(),
                },
                ImportBinding {
                    source: entry.id(),
                    reference: "$/core.resin".into(),
                    target: mapped[&self.library.id()].id(),
                },
            ],
        )
        .unwrap()
    }
}

#[tokio::test]
async fn independent_checkouts_share_complete_user_and_library_graphs() {
    let first = Checkout::new();
    let second = Checkout::new();
    assert_ne!(first.entry, second.entry);
    let left = first.mapping().await;
    let right = second.mapping().await;
    assert_eq!(
        left[&first.entry.id()],
        Source::new("main.resin", first.entry.text())
    );
    assert_eq!(left[&first.helper.id()].name(), "../shared/helper.resin");
    assert_eq!(left[&first.library.id()].name(), "$/core.resin");
    assert_eq!(left[&first.entry.id()], right[&second.entry.id()]);
    let left = first.graph(&left);
    let right = second.graph(&right);
    assert_eq!(left, right);
    let hash = |graph: &SourceGraph| {
        let mut state = DefaultHasher::new();
        graph.hash(&mut state);
        state.finish()
    };
    assert_eq!(hash(&left), hash(&right));
    for source in left.sources() {
        assert!(
            !source
                .name()
                .contains(first._directory.path().to_str().unwrap())
        );
    }
}

#[tokio::test]
async fn logical_mapping_preserves_originals_and_reuses_exact_text_identity() {
    let checkout = Checkout::new();
    let before = checkout.entry.clone();
    let mapped = checkout.mapping().await;
    assert_eq!(checkout.entry, before);
    assert_eq!(
        checkout.loader.path(&before),
        Some(checkout.root.join("main.resin").as_path())
    );
    let logical = &mapped[&before.id()];
    assert_ne!(logical.id(), before.id());
    assert_eq!(logical.content_hash(), before.content_hash());
    assert_eq!(
        logical.text().as_ptr(),
        before.text().as_ptr(),
        "mapping shares text storage"
    );
    assert_eq!(checkout.mapping().await, mapped);
}

#[tokio::test]
async fn reserved_namespace_and_literal_percent_names_cannot_collide() {
    let directory = TempDir::new().unwrap();
    let root = directory.path().join("user");
    let mut loader = Loader::new(directory.path().join("library"));
    let user = loader
        .source_from_text(&root.join("$/core.resin"), "same")
        .unwrap();
    let escaped = loader
        .source_from_text(&root.join("%24/core.resin"), "same")
        .unwrap();
    let library = loader
        .source_from_text(&directory.path().join("library/core.resin"), "same")
        .unwrap();
    let mapped = loader
        .logical_sources(
            [user.clone(), escaped.clone(), library.clone()],
            &root,
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(mapped[&user.id()].name(), "%24/core.resin");
    assert_eq!(mapped[&escaped.id()].name(), "%2524/core.resin");
    assert_eq!(mapped[&library.id()].name(), "$/core.resin");
    let identities: std::collections::BTreeSet<_> = mapped.values().map(Source::id).collect();
    assert_eq!(identities.len(), 3);
}

#[tokio::test]
async fn sources_without_loader_origins_preserve_their_explicit_identity() {
    let source =
        Source::with_identity(SourceId::new("generated-id"), "generated label", "captured");
    let loader = Loader::new("not-needed".into());
    let mapped = loader
        .logical_sources(
            [source.clone()],
            Path::new("not-needed"),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(mapped[&source.id()], source);
    assert_eq!(mapped[&source.id()].text().as_ptr(), source.text().as_ptr());
}

#[cfg(unix)]
#[tokio::test]
async fn non_utf8_and_literal_escape_names_remain_lossless_and_distinct() {
    use std::{ffi::OsString, os::unix::ffi::OsStringExt};
    let directory = TempDir::new().unwrap();
    let mut loader = Loader::new(directory.path().join("library"));
    let names = [
        OsString::from_vec(b"bad\xFF.resin".to_vec()),
        OsString::from_vec(b"bad\xFE.resin".to_vec()),
        "bad%FF.resin".into(),
        "bad�.resin".into(),
    ];
    let sources: Vec<_> = names
        .into_iter()
        .map(|name| {
            loader
                .source_from_text(&directory.path().join(name), "same")
                .unwrap()
        })
        .collect();
    let mapped = loader
        .logical_sources(
            sources.clone(),
            directory.path(),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    let actual: Vec<_> = sources
        .iter()
        .map(|source| mapped[&source.id()].name())
        .collect();
    assert_eq!(
        actual,
        [
            "bad%FF.resin",
            "bad%FE.resin",
            "bad%25FF.resin",
            "bad�.resin"
        ]
    );
    let identities: std::collections::BTreeSet<_> = mapped.values().map(Source::id).collect();
    assert_eq!(identities.len(), 4);
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_library_root_keeps_the_library_role() {
    let directory = TempDir::new().unwrap();
    let actual = directory.path().join("actual-library");
    fs::create_dir(&actual).unwrap();
    fs::write(actual.join("core.resin"), "library").unwrap();
    let alias = directory.path().join("library-alias");
    std::os::unix::fs::symlink(&actual, &alias).unwrap();
    let mut loader = Loader::new(alias);
    let source = loader.load_file(&actual.join("core.resin")).unwrap();
    let mapped = loader
        .logical_sources(
            [source.clone()],
            directory.path(),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await
        .unwrap();
    assert_eq!(mapped[&source.id()].name(), "$/core.resin");
}

#[tokio::test]
async fn cancellation_interrupts_queued_mapping_without_changing_inputs() {
    let checkout = Checkout::new();
    let before = checkout.entry.clone();
    let execution = Execution::new(NonZeroUsize::MIN);
    let cancellation = Cancellation::new();
    let permit = execution.acquire(&cancellation).await.unwrap();
    let mapping = checkout.loader.logical_sources(
        [before.clone()],
        &checkout.root,
        &execution,
        &cancellation,
    );
    tokio::pin!(mapping);
    tokio::select! {
        biased;
        _ = &mut mapping => panic!("mapping must wait for a worker slot"),
        _ = tokio::task::yield_now() => {},
    }
    cancellation.cancel();
    assert!(matches!(
        mapping.await,
        Err(LoadError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    drop(permit);
    execution.wait_idle().await;
    assert_eq!(checkout.entry, before);
    assert_eq!(checkout.mapping().await[&before.id()].text(), before.text());
}

#[tokio::test]
async fn contradictory_versions_cannot_silently_overwrite_a_mapping() {
    let source = Source::new("module", "old");
    let loader = Loader::new("unused".into());
    let result = loader
        .logical_sources(
            [source.clone(), source.with_text("new")],
            Path::new("unused"),
            &Execution::default(),
            &Cancellation::new(),
        )
        .await;
    assert!(
        matches!(result, Err(LoadError::Io { error }) if error.kind() == std::io::ErrorKind::InvalidInput)
    );
    assert_eq!(source.text(), "old");
}
