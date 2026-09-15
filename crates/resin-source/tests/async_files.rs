use resin_executor::{Cancellation, Execution};
use resin_source::{LoadError, Loader, Source};
use std::{fs, num::NonZeroUsize};
use tempfile::TempDir;

#[tokio::test]
async fn async_reads_match_sync_identity_and_preserve_supplied_buffers() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("helper.resin");
    fs::write(&path, "disk before").unwrap();
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let mut loader = Loader::new(directory.path().into());
    let original = loader.load_file(&path).unwrap();
    assert_eq!(
        loader
            .load_file_async(&path, &execution, &cancellation)
            .await
            .unwrap(),
        original
    );
    let supplied = loader.source_from_text(&path, "editor").unwrap();
    let root = loader
        .source_from_text(&directory.path().join("main.resin"), "entry")
        .unwrap();
    fs::write(&path, "disk after").unwrap();
    assert_eq!(
        loader
            .load_import_async(&root, "helper.resin", &execution, &cancellation)
            .await
            .unwrap(),
        supplied
    );
    let after = loader
        .load_file_async(&path, &execution, &cancellation)
        .await
        .unwrap();
    assert_eq!(after.text(), "disk after");
    assert_eq!(after.id(), supplied.id());
    assert_eq!(
        loader
            .load_import_async(&root, "helper.resin", &execution, &cancellation)
            .await
            .unwrap(),
        supplied
    );
    loader.remove_source(&path).unwrap();
    assert_eq!(
        loader
            .load_import_async(&root, "helper.resin", &execution, &cancellation)
            .await
            .unwrap(),
        after
    );
    assert_eq!(original.text(), "disk before");
    assert_eq!(supplied.text(), "editor");
}

#[tokio::test]
async fn explicit_bindings_need_neither_a_worker_slot_nor_a_file() {
    let execution = Execution::new(NonZeroUsize::MIN);
    let cancellation = Cancellation::new();
    let permit = execution.acquire(&cancellation).await.unwrap();
    let mut loader = Loader::new("missing-library".into());
    let entry = Source::new("entry", "entry");
    let library = Source::new("library", "captured");
    loader
        .set_import(&entry, "$/library.resin", library.clone())
        .unwrap();
    let future = loader.load_import_async(&entry, "$/library.resin", &execution, &cancellation);
    tokio::select! {
        biased;
        result = future => assert_eq!(result.unwrap(), library),
        _ = tokio::task::yield_now() => panic!("an explicit binding should resolve immediately"),
    }
    drop(permit);
}

#[tokio::test]
async fn cancellation_interrupts_waiting_for_normalization_and_preserves_sources() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    fs::write(&path, "original").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let original = loader.load_file(&path).unwrap();
    fs::write(&path, "changed").unwrap();
    let execution = Execution::new(NonZeroUsize::MIN);
    let cancellation = Cancellation::new();
    let permit = execution.acquire(&cancellation).await.unwrap();
    let pending = loader.load_file_async(&path, &execution, &cancellation);
    tokio::pin!(pending);
    tokio::select! {
        biased;
        _ = &mut pending => panic!("normalization must wait for its worker slot"),
        _ = tokio::task::yield_now() => {},
    }
    cancellation.cancel();
    assert!(matches!(
        pending.await,
        Err(LoadError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    drop(permit);
    assert_eq!(original.text(), "original");
    execution.wait_idle().await;
}

#[tokio::test]
async fn io_failures_and_cancellation_are_distinct() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("missing.resin");
    let mut loader = Loader::new(directory.path().into());
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let error = loader
        .load_file_async(&path, &execution, &cancellation)
        .await
        .unwrap_err();
    assert!(
        matches!(error, LoadError::Io { ref error } if error.kind() == std::io::ErrorKind::NotFound)
    );
    assert!(error.to_string().contains("missing.resin"));
    cancellation.cancel();
    assert!(matches!(
        loader
            .load_file_async(&path, &execution, &cancellation)
            .await,
        Err(LoadError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn disk_reading_keeps_the_async_runtime_responsive() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("large.resin");
    let text = "// immutable source\n".repeat(100_000);
    fs::write(&path, &text).unwrap();
    let mut loader = Loader::new(directory.path().into());
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let stop = Arc::new(AtomicBool::new(false));
    let progress = Arc::new(AtomicUsize::new(0));
    let observer = tokio::spawn({
        let stop = stop.clone();
        let progress = progress.clone();
        async move {
            while !stop.load(Ordering::Relaxed) {
                progress.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
        }
    });
    let source = loader
        .load_file_async(&path, &execution, &cancellation)
        .await
        .unwrap();
    stop.store(true, Ordering::Relaxed);
    observer.await.unwrap();
    assert_eq!(source.text(), text);
    assert!(
        progress.load(Ordering::Relaxed) > 0,
        "the runtime must poll unrelated work while loading"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn async_files_preserve_lossless_paths_and_symlink_aliases() {
    use std::{
        ffi::OsStr,
        os::unix::{ffi::OsStrExt, fs::symlink},
    };
    let directory = TempDir::new().unwrap();
    let first_path = directory.path().join(OsStr::from_bytes(b"file\xff.resin"));
    let second_path = directory.path().join(OsStr::from_bytes(b"file\xfe.resin"));
    let alias = directory.path().join("alias.resin");
    fs::write(&first_path, "identical").unwrap();
    fs::write(&second_path, "identical").unwrap();
    symlink(&first_path, &alias).unwrap();
    let execution = Execution::default();
    let cancellation = Cancellation::new();
    let mut loader = Loader::new(directory.path().into());
    let first = loader
        .load_file_async(&first_path, &execution, &cancellation)
        .await
        .unwrap();
    let second = loader
        .load_file_async(&second_path, &execution, &cancellation)
        .await
        .unwrap();
    assert_eq!(first.name(), second.name());
    assert_ne!(first.id(), second.id());
    let canonical = resin_source::normalize_path(&first_path).unwrap();
    assert_eq!(loader.path(&first), Some(canonical.as_path()));
    assert_eq!(
        loader
            .load_file_async(&alias, &execution, &cancellation)
            .await
            .unwrap(),
        first
    );
    assert_eq!(loader.load_file(&first_path).unwrap(), first);
}
