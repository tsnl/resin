use resin_source::{Loader, Source, normalize_path};
use std::{fs, sync::Arc};
use tempfile::TempDir;

#[test]
fn snapshots_discard_disk_and_closed_history_but_keep_current_supplied_text() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    let closed_path = directory.path().join("closed.resin");
    let disk_path = directory.path().join("disk.resin");
    fs::write(&disk_path, "disk-only").unwrap();
    let old: Arc<str> = "old editor text".into();
    let old_weak = Arc::downgrade(&old);
    let closed: Arc<str> = "closed editor text".into();
    let closed_weak = Arc::downgrade(&closed);
    let current_text: Arc<str> = "current editor text".into();
    let current_weak = Arc::downgrade(&current_text);
    let mut loader = Loader::new(directory.path().into());
    loader.source_from_text(&path, old).unwrap();
    let current = loader.source_from_text(&path, current_text).unwrap();
    let closed = loader.source_from_text(&closed_path, closed).unwrap();
    let disk = loader.load_file(&disk_path).unwrap();
    loader.remove_source(&closed_path).unwrap();

    let mut snapshot = loader.supplied_snapshot();
    assert_eq!(snapshot.path(&current), loader.path(&current));
    assert!(snapshot.path(&closed).is_none());
    assert!(snapshot.path(&disk).is_none());
    assert!(old_weak.upgrade().is_some());
    drop(closed);
    drop(loader);
    assert!(old_weak.upgrade().is_none());
    assert!(closed_weak.upgrade().is_none());
    assert_eq!(
        snapshot.source_from_text(&path, current.text()).unwrap(),
        current
    );
    assert_eq!(snapshot.load_import(&current, "disk.resin").unwrap(), disk);
    assert!(snapshot.supplied_snapshot().path(&disk).is_none());
    snapshot.remove_source(&path).unwrap();
    snapshot = snapshot.supplied_snapshot();
    assert!(snapshot.path(&current).is_none());
    assert!(current_weak.upgrade().is_some());
    drop(current);
    assert!(current_weak.upgrade().is_none());
}

#[test]
fn snapshots_preserve_explicit_bindings_and_their_file_origins() {
    let directory = TempDir::new().unwrap();
    let target_path = directory.path().join("target.resin");
    fs::write(&target_path, "bound target").unwrap();
    fs::write(directory.path().join("helper.resin"), "helper").unwrap();
    let root = Source::new("generated root", "root");
    let mut loader = Loader::new(directory.path().into());
    let target = loader.load_file(&target_path).unwrap();
    loader.set_import(&root, "bound", target.clone()).unwrap();
    let mut snapshot = loader.supplied_snapshot();
    drop(loader);
    assert_eq!(snapshot.load_import(&root, "bound").unwrap(), target);
    assert_eq!(
        snapshot.path(&target),
        Some(normalize_path(&target_path).unwrap().as_path())
    );
    assert_eq!(
        snapshot
            .load_import(&target, "helper.resin")
            .unwrap()
            .text(),
        "helper"
    );
}

#[test]
fn independent_snapshots_cannot_replace_each_others_buffers() {
    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    let mut loader = Loader::new(directory.path().into());
    let first = loader.source_from_text(&path, "first").unwrap();
    let mut left = loader.supplied_snapshot();
    let mut right = loader.supplied_snapshot();
    let changed = left.source_from_text(&path, "left").unwrap();
    right.remove_source(&path).unwrap();
    let right = right.supplied_snapshot();
    assert!(right.path(&first).is_none());
    assert_eq!(loader.source_from_text(&path, "first").unwrap(), first);
    assert_eq!(left.source_from_text(&path, "left").unwrap(), changed);
    assert_ne!(first, changed);
}

#[cfg(unix)]
#[test]
fn snapshot_preserves_open_identity_and_reopen_registers_the_new_symlink_target() {
    use std::os::unix::fs::symlink;

    let directory = TempDir::new().unwrap();
    let path = directory.path().join("main.resin");
    let target = directory.path().join("target.resin");
    fs::write(&path, "original disk").unwrap();
    fs::write(&target, "target disk").unwrap();
    let mut loader = Loader::new(directory.path().into());
    let open = loader.source_from_text(&path, "open buffer").unwrap();
    let registered = loader.path(&open).unwrap().to_path_buf();
    fs::remove_file(&path).unwrap();
    symlink(&target, &path).unwrap();

    let mut snapshot = loader.supplied_snapshot();
    let edited = snapshot
        .source_from_text(&registered, "edited buffer")
        .unwrap();
    assert_eq!(edited.id(), open.id());
    assert_eq!(snapshot.path(&edited), Some(registered.as_path()));
    assert_eq!(open.text(), "open buffer");
    snapshot.remove_source(&registered).unwrap();
    let mut reopened = snapshot.supplied_snapshot();
    let new = reopened.source_from_text(&path, "reopened buffer").unwrap();
    assert_ne!(new.id(), open.id());
    assert_eq!(
        reopened.path(&new),
        Some(normalize_path(&target).unwrap().as_path())
    );
    assert_eq!(loader.path(&open), Some(registered.as_path()));
}
