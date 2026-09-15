use resin_executor::{Cancellation, Error, Execution};
use std::{
    num::NonZeroUsize,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Default)]
struct Gate {
    open: Mutex<bool>,
    changed: Condvar,
}

impl Gate {
    fn wait(&self) {
        let mut open = self.open.lock().unwrap();
        while !*open {
            open = self.changed.wait(open).unwrap();
        }
    }

    fn release(&self) {
        *self.open.lock().unwrap() = true;
        self.changed.notify_all();
    }
}

struct ReleaseOnDrop(Arc<Gate>);

impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

#[tokio::test]
async fn cancellation_is_shared_and_never_loses_a_notification() {
    for _ in 0..64 {
        let token = Cancellation::new();
        let waiting = token.clone();
        let task = tokio::spawn(async move { waiting.cancelled().await });
        token.cancel();
        tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(token.check(), Err(Error::Cancelled));
        token.cancelled().await;
        token.cancel();
        assert!(token.is_cancelled());
    }
}

#[tokio::test]
async fn cpu_jobs_overlap_with_bounded_capacity_and_leave_io_responsive() {
    let execution = Execution::new(NonZeroUsize::new(2).unwrap());
    let gate = Arc::new(Gate::default());
    let release = ReleaseOnDrop(gate.clone());
    let active = Arc::new(AtomicUsize::new(0));
    let maximum = Arc::new(AtomicUsize::new(0));
    let (started, mut arrivals) = tokio::sync::mpsc::unbounded_channel();
    let mut tasks = tokio::task::JoinSet::new();
    for value in 0..8 {
        let execution = execution.clone();
        let gate = gate.clone();
        let active = active.clone();
        let maximum = maximum.clone();
        let started = started.clone();
        tasks.spawn(async move {
            execution
                .run(&Cancellation::new(), move |_| {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(count, Ordering::SeqCst);
                    started.send(()).unwrap();
                    gate.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                    value
                })
                .await
                .unwrap()
        });
    }
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(1), arrivals.recv())
            .await
            .unwrap()
            .unwrap();
    }
    assert_eq!(active.load(Ordering::SeqCst), 2);
    let (mut writer, mut reader) = tokio::io::duplex(1);
    let io = async {
        writer.write_u8(42).await.unwrap();
        assert_eq!(reader.read_u8().await.unwrap(), 42);
        tokio::time::sleep(Duration::from_millis(1)).await;
    };
    tokio::time::timeout(Duration::from_secs(1), io)
        .await
        .unwrap();
    drop(release);
    let mut sum = 0;
    while let Some(result) = tasks.join_next().await {
        sum += result.unwrap();
    }
    assert_eq!(sum, 28);
    assert_eq!(maximum.load(Ordering::SeqCst), 2);
    execution.wait_idle().await;
}

#[tokio::test]
async fn cancellation_removes_queued_work_without_running_its_callback() {
    let execution = Execution::new(NonZeroUsize::MIN);
    let occupied = execution.acquire(&Cancellation::new()).await.unwrap();
    let cancellation = Cancellation::new();
    let token = cancellation.clone();
    let pool = execution.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let task = tokio::spawn(async move {
        pool.run(&token, move |_| count.fetch_add(1, Ordering::SeqCst))
            .await
    });
    tokio::task::yield_now().await;
    cancellation.cancel();
    assert_eq!(task.await.unwrap(), Err(Error::Cancelled));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    drop(occupied);
    execution.wait_idle().await;
}

#[tokio::test]
async fn started_work_observes_cancellation_before_its_result_returns() {
    let execution = Execution::new(NonZeroUsize::MIN);
    let cancellation = Cancellation::new();
    let token = cancellation.clone();
    let pool = execution.clone();
    let (started, ready) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        pool.run(&token, move |cancellation| {
            started.send(()).unwrap();
            while !cancellation.is_cancelled() {
                std::thread::sleep(Duration::from_millis(1));
            }
            42
        })
        .await
    });
    ready.await.unwrap();
    cancellation.cancel();
    assert_eq!(task.await.unwrap(), Err(Error::Cancelled));
    assert_eq!(execution.run(&Cancellation::new(), |_| 7).await.unwrap(), 7);
}

#[tokio::test]
async fn dropped_future_keeps_its_slot_until_the_worker_exits() {
    let execution = Execution::new(NonZeroUsize::MIN);
    let gate = Arc::new(Gate::default());
    let release = ReleaseOnDrop(gate.clone());
    let (started, ready) = tokio::sync::oneshot::channel();
    let pool = execution.clone();
    let task = tokio::spawn(async move {
        pool.run(&Cancellation::new(), move |_| {
            started.send(()).unwrap();
            gate.wait();
        })
        .await
    });
    ready.await.unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());
    assert!(
        tokio::time::timeout(Duration::from_millis(10), execution.wait_idle())
            .await
            .is_err()
    );
    drop(release);
    tokio::time::timeout(Duration::from_secs(1), execution.wait_idle())
        .await
        .unwrap();
}

#[tokio::test]
async fn worker_panics_release_capacity_and_remain_execution_failures() {
    let execution = Execution::new(NonZeroUsize::MIN);
    let error = execution
        .run(&Cancellation::new(), |_| panic!("broken worker"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Worker { .. }));
    assert_eq!(
        execution.run(&Cancellation::new(), |_| 42).await.unwrap(),
        42
    );
}

#[test]
fn execution_handles_can_be_shared() {
    fn shared<T: Send + Sync>() {}
    shared::<Execution>();
    shared::<Cancellation>();
}
