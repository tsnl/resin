//! Compiler behavior exercised through the public facade by an external client.
use resin_common::prelude::*;
use resin_compiler::Session;
use std::{fs, sync::Arc};

fn temporary_directory() -> TempDir {
    TempDir::new(&fs::canonicalize(std::env::temp_dir()).unwrap()).unwrap()
}

#[test]
fn disk_notifications_preserve_an_open_overlay_until_it_closes() {
    let temp = temporary_directory();
    let path = temp.path().join("main.resin");
    fs::write(&path, "def main() -> int = { 1 }; ").unwrap();
    let mut session = Session::default();
    session
        .set_overlay(&path, "def main() -> int = { 2 }; ".into())
        .unwrap();
    let old = session.analyze(&path).unwrap();
    assert!(old.module().is_ok());
    fs::write(&path, "def main() -> int = { false }; ").unwrap();
    let revision = session.revision();
    session.file_changed(&path).unwrap();
    assert_eq!(revision, session.revision());
    assert!(Arc::ptr_eq(&old, &session.analyze(&path).unwrap()));
    session.remove_overlay(&path).unwrap();
    assert!(session.analyze(&path).unwrap().module().is_err());
    assert!(old.module().is_ok());
}

#[test]
fn changing_stdlib_rebuilds_the_root_import_graph() {
    let temp = temporary_directory();
    let path = temp.path().join("main.resin");
    let first = temp.path().join("first");
    let second = temp.path().join("second");
    let mut session = Session::new(first.clone());
    session
        .set_overlay(
            &path,
            "import { \"std/value.resin\" }; def main() -> int = { value() };".into(),
        )
        .unwrap();
    session
        .set_overlay(
            &first.join("value.resin"),
            "export { value }; def value() -> int = { 1 };".into(),
        )
        .unwrap();
    session
        .set_overlay(
            &second.join("value.resin"),
            "export { renamed }; def renamed() -> int = { 2 };".into(),
        )
        .unwrap();
    let old = session.analyze(&path).unwrap();
    assert!(old.module().is_ok(), "{:?}", old.diagnostics());
    session.set_stdlib(second.clone());
    let new = session.analyze(&path).unwrap();
    assert!(new.module().is_err());
    assert!(new.dependencies().contains(&second.join("value.resin")));
    assert!(!new.dependencies().contains(&first.join("value.resin")));
    assert!(old.module().is_ok());
}

#[test]
fn later_errors_keep_completed_earlier_passes_available() {
    let temp = temporary_directory();
    let path = temp.path().join("main.resin");
    let mut session = Session::default();
    session
        .set_overlay(
            &path,
            "def first() -> int = { var n: int; n }; def second() -> bool = { var b: bool; b };"
                .into(),
        )
        .unwrap();
    let lowered = session.analyze(&path).unwrap();
    assert!(lowered.program().is_ok());
    assert!(lowered.hir().is_ok());
    assert!(lowered.module().is_err());
    assert_eq!(
        lowered.diagnostics().len(),
        2,
        "independent lowering errors are collected"
    );
    assert!(
        lowered
            .diagnostics()
            .iter()
            .all(|diagnostic| diagnostic.location.path == path)
    );
    session
        .set_overlay(&path, "def main() -> int = { false };".into())
        .unwrap();
    let typed = session.analyze(&path).unwrap();
    assert!(typed.program().is_ok());
    assert!(typed.hir().is_err());
    session
        .set_overlay(&path, "def main( = { false };".into())
        .unwrap();
    let parsed = session.analyze(&path).unwrap();
    assert!(parsed.program().is_err());
    assert!(parsed.recovered_file(&path).is_some());
}

#[test]
fn creating_a_transitively_missing_disk_dependency_recovers() {
    let temp = temporary_directory();
    let path = temp.path().join("main.resin");
    let dependency = temp.path().join("leaf.resin");
    fs::write(&path, "import { \"middle.resin\" }; ").unwrap();
    fs::write(
        temp.path().join("middle.resin"),
        "import { \"leaf.resin\" }; ",
    )
    .unwrap();
    let mut session = Session::default();
    let before = session.analyze(&path).unwrap();
    assert!(before.module().is_err());
    fs::write(&dependency, "def leaf() = {}; ").unwrap();
    session.file_changed(&dependency).unwrap();
    let after = session.analyze(&path).unwrap();
    assert!(after.module().is_ok(), "{:?}", after.diagnostics());
    assert!(!Arc::ptr_eq(&before, &after));
}

#[cfg(unix)]
mod symlinks {
    use super::*;
    use resin_compiler::Compilation;
    use std::path::PathBuf;

    const INTEGER: &str = "export { value }; def value() -> int = { 1 };";
    const BOOLEAN: &str = "export { value }; def value() -> bool = { true };";

    struct Project {
        _temp: TempDir,
        root: PathBuf,
        alias: PathBuf,
        second: PathBuf,
        session: Session,
    }

    impl Project {
        fn new() -> Self {
            let temp = temporary_directory();
            let root = temp.path().join("main.resin");
            let alias = temp.path().join("alias.resin");
            let first = temp.path().join("first.resin");
            let second = temp.path().join("second.resin");
            fs::write(
                &root,
                "import { \"./alias.resin\" }; def main() -> int = { value() };",
            )
            .unwrap();
            fs::write(&first, INTEGER).unwrap();
            fs::write(&second, BOOLEAN).unwrap();
            std::os::unix::fs::symlink(first, &alias).unwrap();
            Self {
                _temp: temp,
                root,
                alias,
                second,
                session: Session::default(),
            }
        }

        fn original(&mut self) -> Arc<Compilation> {
            let old = self.session.analyze(&self.root).unwrap();
            assert!(old.module().is_ok(), "{:?}", old.diagnostics());
            assert!(Arc::ptr_eq(
                &old,
                &self.session.analyze(&self.root).unwrap()
            ));
            old
        }

        fn retarget(&self) {
            fs::remove_file(&self.alias).unwrap();
            std::os::unix::fs::symlink(&self.second, &self.alias).unwrap();
        }

        fn changed(&mut self, old: &Arc<Compilation>) -> Arc<Compilation> {
            let updated = self.session.analyze(&self.root).unwrap();
            assert!(
                !Arc::ptr_eq(old, &updated),
                "retargeting invalidates the cache"
            );
            assert!(
                updated.module().is_err(),
                "new target returns bool where int is required"
            );
            assert!(old.module().is_ok(), "retained compilation remains stable");
            updated
        }
    }

    #[test]
    fn disk_notification_invalidates_a_retargeted_import() {
        let mut project = Project::new();
        let old = project.original();
        project.retarget();
        project.session.file_changed(&project.alias).unwrap();
        project.changed(&old);
    }

    #[test]
    fn disk_notification_preserves_the_new_targets_overlay_after_retargeting() {
        let mut project = Project::new();
        fs::write(&project.second, INTEGER).unwrap();
        project
            .session
            .set_overlay(&project.second, BOOLEAN.into())
            .unwrap();
        let old = project.original();
        project.retarget();
        project.session.file_changed(&project.alias).unwrap();
        project.changed(&old);
    }

    #[test]
    fn setting_an_overlay_invalidates_a_retargeted_import() {
        let mut project = Project::new();
        fs::write(&project.second, INTEGER).unwrap();
        let old = project.original();
        project.retarget();
        project
            .session
            .set_overlay(&project.alias, BOOLEAN.into())
            .unwrap();
        project.changed(&old);
    }

    #[test]
    fn unchanged_overlay_text_still_invalidates_a_retargeted_import() {
        let mut project = Project::new();
        fs::write(&project.second, INTEGER).unwrap();
        project
            .session
            .set_overlay(&project.second, BOOLEAN.into())
            .unwrap();
        let old = project.original();
        project.retarget();
        project
            .session
            .set_overlay(&project.alias, BOOLEAN.into())
            .unwrap();
        let updated = project.changed(&old);
        let revision = project.session.revision();
        project
            .session
            .set_overlay(&project.alias, BOOLEAN.into())
            .unwrap();
        assert_eq!(project.session.revision(), revision);
        assert!(Arc::ptr_eq(
            &updated,
            &project.session.analyze(&project.root).unwrap()
        ));
    }

    #[test]
    fn removing_an_overlay_invalidates_a_retargeted_import() {
        let mut project = Project::new();
        project
            .session
            .set_overlay(&project.second, INTEGER.into())
            .unwrap();
        let old = project.original();
        project.retarget();
        project.session.remove_overlay(&project.alias).unwrap();
        project.changed(&old);
    }

    #[test]
    fn closing_an_overlay_invalidates_a_retargeted_import_without_a_new_overlay() {
        let mut project = Project::new();
        project
            .session
            .set_overlay(&project.alias, INTEGER.into())
            .unwrap();
        let old = project.original();
        project.retarget();
        project.session.remove_overlay(&project.alias).unwrap();
        project.changed(&old);
    }
}
