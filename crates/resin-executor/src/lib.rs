//! Bounded execution and cooperative cancellation, without compiler or cache state.

use std::{fmt, num::NonZeroUsize, sync::Arc};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};

//
// Request cancellation
//

/// A shared, monotonic cancellation signal. Clones cancel the same work.
#[derive(Clone, Debug)]
pub struct Cancellation {
    state: watch::Sender<bool>,
}

impl Cancellation {
    pub fn new() -> Self {
        let (state, _) = watch::channel(false);
        Self { state }
    }

    pub fn cancel(&self) {
        self.state.send_replace(true);
    }

    pub fn is_cancelled(&self) -> bool {
        *self.state.borrow()
    }

    pub fn check(&self) -> Result<(), Error> {
        if self.is_cancelled() {
            Err(Error::Cancelled)
        } else {
            Ok(())
        }
    }

    /// Wait without polling; a previously cancelled token returns immediately.
    pub async fn cancelled(&self) {
        let mut changes = self.state.subscribe();
        while !*changes.borrow_and_update() {
            changes.changed().await.expect("this token owns the sender");
        }
    }
}

impl Default for Cancellation {
    fn default() -> Self {
        Self::new()
    }
}

//
// Shared execution capacity
//

/// A shared bound on running CPU jobs or explicitly reserved native work.
///
/// Async orchestration does not reserve a slot. Reserve only while doing work;
/// a caller holding a permit must not wait for another job in this same pool.
#[derive(Clone, Debug)]
pub struct Execution {
    jobs: NonZeroUsize,
    available: Arc<Semaphore>,
}

/// One execution slot, released when its owner finishes or drops it.
#[derive(Debug)]
pub struct Permit {
    _permit: OwnedSemaphorePermit,
}

impl Execution {
    /// Construct a pool. The job count must fit Tokio's semaphore and `u32`.
    pub fn new(jobs: NonZeroUsize) -> Self {
        assert!(jobs.get() <= Semaphore::MAX_PERMITS && jobs.get() <= u32::MAX as usize);
        Self {
            jobs,
            available: Arc::new(Semaphore::new(jobs.get())),
        }
    }

    pub fn jobs(&self) -> usize {
        self.jobs.get()
    }

    /// Reserve a slot for async native work. Cancellation also interrupts waiting.
    pub async fn acquire(&self, cancellation: &Cancellation) -> Result<Permit, Error> {
        let permit = tokio::select! {
            biased;
            _ = cancellation.cancelled() => return Err(Error::Cancelled),
            permit = self.available.clone().acquire_owned() => permit.expect("private pool remains open"),
        };
        cancellation.check()?;
        Ok(Permit { _permit: permit })
    }

    /// Run CPU work off the async runtime, subject to this pool's capacity.
    ///
    /// The callback checks its token at meaningful work boundaries. Cancellation
    /// discards its result, but a started callback is awaited until it returns.
    /// Dropping this future cannot stop a blocking callback: the callback retains
    /// its slot, and [`Self::wait_idle`] can wait for it during shutdown.
    pub async fn run<T, F>(&self, cancellation: &Cancellation, work: F) -> Result<T, Error>
    where
        T: Send + 'static,
        F: FnOnce(&Cancellation) -> T + Send + 'static,
    {
        let permit = self.acquire(cancellation).await?;
        let cancellation = cancellation.clone();
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            cancellation.check()?;
            let result = work(&cancellation);
            cancellation.check()?;
            Ok(result)
        })
        .await
        .map_err(|error| Error::Worker {
            message: error.to_string(),
        })?
    }

    /// Wait until all running work releases its slots.
    /// Stop submitting work before calling this during shutdown.
    pub async fn wait_idle(&self) {
        let _all = self
            .available
            .acquire_many(self.jobs.get() as u32)
            .await
            .expect("private pool remains open");
    }
}

impl Default for Execution {
    fn default() -> Self {
        Self::new(std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    }
}

//
// Execution failures
//

/// Cancellation and worker failure are distinct from a compiler's diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Cancelled,
    Worker { message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => formatter.write_str("operation cancelled"),
            Self::Worker { message } => write!(formatter, "worker failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}
