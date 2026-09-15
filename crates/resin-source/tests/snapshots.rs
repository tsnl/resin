use resin_source::{
    EditError, GraphError, ImportBinding, Source, SourceEdit, SourceGraph, SourceId, Span,
};
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

fn hash(value: &impl Hash) -> u64 {
    let mut hash = DefaultHasher::new();
    value.hash(&mut hash);
    hash.finish()
}

fn binding(source: &Source, reference: &str, target: &Source) -> ImportBinding {
    ImportBinding {
        source: source.id(),
        reference: reference.into(),
        target: target.id(),
    }
}

#[test]
fn independent_sources_reconstruct_identity_without_an_interner() {
    let original = Source::new("src/main.resin", "first");
    let reconstructed = Source::new("src/main.resin", "first");
    assert_eq!(original, reconstructed);
    assert_eq!(original.cmp(&reconstructed), std::cmp::Ordering::Equal);
    assert_eq!(hash(&original), hash(&reconstructed));
    assert_eq!(original.id(), reconstructed.id());

    let edited = original.with_text("second");
    assert_eq!(original.id(), edited.id());
    assert_ne!(original, edited);
    assert_ne!(original.content_hash(), edited.content_hash());
    assert_eq!(edited.with_text("first"), original);
    assert_eq!(original.text(), "first");
    assert_eq!(edited.text(), "second");
}

#[test]
fn logical_modules_and_retained_names_are_part_of_complete_identity() {
    let left = Source::with_identity(SourceId::new("left"), "same label", "same text");
    let right = Source::with_identity(SourceId::new("right"), "same label", "same text");
    assert_ne!(left.id(), right.id());
    assert_ne!(left, right);
    assert_eq!(left.content_hash(), right.content_hash());
    let renamed = Source::with_identity(left.id(), "different label", left.text());
    assert_eq!(left.id(), renamed.id());
    assert_ne!(left, renamed, "cached diagnostics retain the display name");
}

#[test]
fn full_text_and_different_edit_histories_converge() {
    let id = SourceId::new("main.resin");
    let original = Source::with_identity(id.clone(), "main.resin", "a🌲c");
    let edited = Source::from_edits(
        id.clone(),
        "main.resin",
        Some(&original),
        &[
            SourceEdit {
                range: Span { start: 1, end: 5 },
                text: "é".into(),
            },
            SourceEdit {
                range: Span { start: 3, end: 4 },
                text: "d".into(),
            },
        ],
    )
    .unwrap();
    let uploaded = Source::from_edits(
        id,
        "main.resin",
        None,
        &[SourceEdit {
            range: Span { start: 0, end: 0 },
            text: "aéd".into(),
        }],
    )
    .unwrap();
    assert_eq!(edited, uploaded);
    assert_eq!(edited, Source::new("main.resin", "aéd"));
    assert_eq!(hash(&edited), hash(&uploaded));
    assert_eq!(original.text(), "a🌲c");
}

#[test]
fn invalid_edits_never_change_the_predecessor() {
    let original = Source::new("main", "a🌲c");
    let edit = |range| SourceEdit {
        range,
        text: "replacement".into(),
    };
    let apply = |change| Source::from_edits(original.id(), "main", Some(&original), &[change]);
    assert!(matches!(
        Source::from_edits(SourceId::new("other"), "main", Some(&original), &[]),
        Err(EditError::WrongPredecessor { .. })
    ));
    assert!(matches!(
        apply(edit(Span { start: 2, end: 5 })),
        Err(EditError::InvalidUtf8Boundary { offset: 2, .. })
    ));
    for range in [
        Span { start: 5, end: 1 },
        Span { start: 0, end: 7 },
        Span {
            start: usize::MAX,
            end: usize::MAX,
        },
    ] {
        assert!(matches!(
            apply(edit(range)),
            Err(EditError::InvalidRange { .. })
        ));
    }
    assert_eq!(original.text(), "a🌲c");
}

#[test]
fn graphs_canonicalize_discovery_order_and_drop_unreachable_sources() {
    let entry = Source::new("main.resin", "import left and right");
    let left = Source::new("left.resin", "left");
    let right = Source::new("right.resin", "right");
    let unused = Source::new("unused.resin", "old retained text");
    let left_edge = binding(&entry, "left.resin", &left);
    let right_edge = binding(&entry, "right.resin", &right);
    let forward = SourceGraph::new(
        entry.clone(),
        [left.clone(), right.clone(), unused.clone()],
        [left_edge.clone(), right_edge.clone()],
    )
    .unwrap();
    let reverse = SourceGraph::new(
        entry.clone(),
        [right.clone(), left.clone()],
        [right_edge, left_edge],
    )
    .unwrap();
    assert_eq!(forward, reverse);
    assert_eq!(hash(&forward), hash(&reverse));
    assert_eq!(forward.sources().count(), 3);
    assert!(forward.source(&unused.id()).is_none());
    assert_eq!(forward.resolve(&entry.id(), "left.resin"), Some(&left));
    assert!(forward.resolve(&entry.id(), "missing.resin").is_none());
    assert!(forward.resolve(&entry.id(), "$/span.resin").is_none());
}

#[test]
fn graph_keys_include_bindings_versions_and_selected_entry() {
    let entry = Source::new("main", "same entry bytes");
    let left = Source::new("left", "same helper bytes");
    let right = Source::new("right", "same helper bytes");
    let make = |target: &Source| {
        SourceGraph::new(
            entry.clone(),
            [target.clone()],
            [binding(&entry, "helper", target)],
        )
        .unwrap()
    };
    let first = make(&left);
    assert_ne!(first, make(&right));
    assert_ne!(first, make(&left.with_text("edited helper")));
    let alias = SourceGraph::new(
        entry.clone(),
        [left.clone()],
        [binding(&entry, "alias", &left)],
    )
    .unwrap();
    assert_ne!(first, alias);
    let other_entry = SourceGraph::new(
        left.clone(),
        [entry.clone()],
        [binding(&left, "helper", &entry)],
    )
    .unwrap();
    assert_ne!(first, other_entry);
    assert_eq!(
        first.resolve(&entry.id(), "helper").unwrap().text(),
        "same helper bytes"
    );
}

#[test]
fn graph_rejects_ambiguous_versions_and_bindings_but_retains_cycles() {
    let entry = Source::new("main", "entry");
    let helper = Source::new("helper", "helper");
    assert!(matches!(
        SourceGraph::new(entry.clone(), [entry.with_text("different")], []),
        Err(GraphError::ConflictingSource { .. })
    ));
    assert!(matches!(
        SourceGraph::new(entry.clone(), [], [binding(&entry, "helper", &helper)]),
        Err(GraphError::MissingSource { .. })
    ));
    assert!(matches!(
        SourceGraph::new(
            entry.clone(),
            [helper.clone()],
            [
                binding(&entry, "same", &helper),
                binding(&entry, "same", &entry)
            ]
        ),
        Err(GraphError::ConflictingImport { .. })
    ));
    let cyclic = SourceGraph::new(
        entry.clone(),
        [helper.clone()],
        [
            binding(&entry, "helper", &helper),
            binding(&helper, "main", &entry),
        ],
    )
    .unwrap();
    assert_eq!(cyclic.sources().count(), 2);
    assert_eq!(cyclic.resolve(&helper.id(), "main"), Some(&entry));
}

#[test]
fn source_graphs_and_edits_are_safe_to_send_between_workers() {
    fn shareable<T: Send + Sync>() {}
    shareable::<Source>();
    shareable::<SourceId>();
    shareable::<SourceGraph>();
    shareable::<SourceEdit>();
}
