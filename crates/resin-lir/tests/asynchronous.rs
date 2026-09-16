use resin_executor::{Cancellation, Execution};
use resin_hir::{Annotation, Constant, Function, Module, Signature, Term, TermKind, Type};
use resin_lir::{
    BuildError, Entry, LirKey, LoweringOptions, Profile, VerificationError, VerifiedModule,
};
use resin_source::{Source, SourceGraph, Span};
use resin_types::FunctionId;
use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    num::NonZeroUsize,
    sync::Arc,
    time::Duration,
};

fn execution(jobs: usize) -> Execution {
    Execution::new(NonZeroUsize::new(jobs).unwrap())
}

fn program(count: usize) -> Arc<Module> {
    let span = Span { start: 0, end: 0 };
    let function = Function {
        location: None,
        name: "unit".into(),
        signature: Signature {
            type_params: vec![],
            params: vec![],
            result: Annotation {
                ty: Type::Unit,
                span,
            },
        },
        foreign_header: None,
        body: Some(Term {
            span,
            ty: Type::Unit,
            kind: TermKind::Constant {
                value: Constant::Unit,
            },
        }),
    };
    Arc::new(Module {
        functions: vec![function; count],
        ..Default::default()
    })
}

#[tokio::test]
async fn parallel_lowering_and_verification_share_immutable_input() {
    let input = program(64);
    let pool = execution(2);
    let cancellation = Cancellation::new();
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..2 {
        let input = input.clone();
        let pool = pool.clone();
        let cancellation = cancellation.clone();
        tasks.spawn(async move {
            let lir = resin_lir::build_lir(
                input,
                vec![],
                LoweringOptions::default(),
                &pool,
                &cancellation,
            )
            .await
            .unwrap();
            VerifiedModule::build(lir, &pool, &cancellation)
                .await
                .unwrap()
        });
    }
    let first = tasks.join_next().await.unwrap().unwrap();
    let second = tasks.join_next().await.unwrap().unwrap();
    assert_eq!(first.view().module(), second.view().module());
    assert_eq!(input.functions.len(), 64);
    assert!(
        input
            .functions
            .iter()
            .all(|function| function.body.is_some())
    );
}

#[tokio::test]
async fn cancellation_is_not_a_lowering_or_verification_diagnostic() {
    let cancellation = Cancellation::new();
    cancellation.cancel();
    let pool = execution(1);
    let error = resin_lir::build_lir(
        program(1),
        vec![],
        LoweringOptions::default(),
        &pool,
        &cancellation,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        BuildError::Execution {
            error: resin_executor::Error::Cancelled
        }
    ));
    let error = VerifiedModule::build(Default::default(), &pool, &cancellation)
        .await
        .err()
        .unwrap();
    assert!(matches!(
        error,
        VerificationError::Execution {
            error: resin_executor::Error::Cancelled
        }
    ));
}

#[tokio::test]
async fn large_specialization_requests_can_cancel_while_a_worker_owns_them() {
    let input = program(40_000);
    let pool = execution(1);
    let cancellation = Cancellation::new();
    let worker_pool = pool.clone();
    let worker_token = cancellation.clone();
    let task = tokio::spawn(async move {
        resin_lir::build_lir(
            input,
            vec![],
            LoweringOptions::default(),
            &worker_pool,
            &worker_token,
        )
        .await
    });
    tokio::task::yield_now().await;
    let probe = Cancellation::new();
    assert!(
        tokio::time::timeout(Duration::from_millis(1), pool.acquire(&probe))
            .await
            .is_err()
    );
    cancellation.cancel();
    let result = tokio::time::timeout(Duration::from_secs(2), task)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        result,
        Err(BuildError::Execution {
            error: resin_executor::Error::Cancelled
        })
    ));
    pool.wait_idle().await;
}

fn graph(text: &str) -> Arc<SourceGraph> {
    Arc::new(SourceGraph::new(Source::new("main.resin", text), [], []).unwrap())
}

fn entry(name: &str, function: usize, profile: Profile) -> Entry {
    Entry {
        name: name.into(),
        function: FunctionId::from_index(function),
        arguments: vec![],
        profile,
    }
}

fn hash(value: &impl Hash) -> u64 {
    let mut state = DefaultHasher::new();
    value.hash(&mut state);
    state.finish()
}

#[test]
fn lowering_keys_canonicalize_targets_and_include_sources_profiles_and_limits() {
    let first = entry("first", 0, Profile::Host);
    let second = entry("second", 1, Profile::Host);
    let options = LoweringOptions::default();
    let left = LirKey::new(
        graph("source"),
        [second.clone(), first.clone(), second.clone()],
        options.clone(),
    );
    let right = LirKey::new(
        graph("source"),
        [first.clone(), second.clone()],
        options.clone(),
    );
    assert_eq!(left, right);
    assert_eq!(hash(&left), hash(&right));
    assert_eq!(left.entries().len(), 2);
    assert_ne!(
        left,
        LirKey::new(
            graph("edited"),
            [first.clone(), second.clone()],
            options.clone()
        )
    );
    assert_ne!(
        left,
        LirKey::new(
            graph("source"),
            [first.clone(), entry("second", 1, Profile::Shader)],
            options
        )
    );
    assert_ne!(
        left,
        LirKey::new(
            graph("source"),
            [first, second],
            LoweringOptions {
                max_monomorphs_per_function: NonZeroUsize::MIN
            }
        )
    );
}

#[test]
fn completed_languages_keys_and_certificates_are_shareable() {
    fn shared<T: Send + Sync>() {}
    shared::<resin_hir::Module>();
    shared::<resin_lir::Module>();
    shared::<LirKey>();
    shared::<VerifiedModule>();
}
