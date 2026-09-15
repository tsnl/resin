//! Supervise owned native processes through cancellation and future abandonment.
use crate::Error;
use resin_executor::Cancellation;
use std::{
    future::Future,
    process::{ExitStatus, Stdio},
};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::{Child, Command},
};

#[cfg(unix)]
#[path = "process/unix.rs"]
mod tree;
#[cfg(windows)]
#[path = "process/windows.rs"]
mod tree;

/// The task owns staging locks and child processes until cancellation cleanup finishes.
/// Dropping the caller's future signals this task instead of abandoning its resources.
pub(super) async fn supervise<T, F, W>(cancellation: &Cancellation, work: W) -> Result<T, Error>
where
    T: Send + 'static,
    F: Future<Output = Result<T, Error>> + Send + 'static,
    W: FnOnce(Cancellation) -> F,
{
    cancellation.check()?;
    let active = Cancellation::new();
    let cancel_on_drop = CancelOnDrop(active.clone());
    let mut task = tokio::spawn(work(active.clone()));
    let result = tokio::select! {
        biased;
        _ = cancellation.cancelled() => {
            active.cancel();
            let _ = task.await;
            return Err(Error::cancelled());
        }
        result = &mut task => result.map_err(|error| Error::new(format!("native worker failed: {error}")))?,
    };
    drop(cancel_on_drop);
    result
}

struct CancelOnDrop(Cancellation);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub(super) async fn run(
    command: Command,
    kind: &str,
    cancellation: &Cancellation,
) -> Result<(), Error> {
    capture(command, kind, cancellation).await.map(|_| ())
}

pub(super) async fn capture(
    mut command: Command,
    kind: &str,
    cancellation: &Cancellation,
) -> Result<(Vec<u8>, Vec<u8>), Error> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut process = spawn(command, cancellation).await?;
    let stdout = process.child.stdout.take().expect("piped stdout");
    let stderr = process.child.stderr.take().expect("piped stderr");
    let (status, stdout, stderr) =
        tokio::join!(process.wait(cancellation), output(stdout), output(stderr));
    let status = status?;
    let (stdout, stderr) = (stdout?, stderr?);
    if status.success() {
        return Ok((stdout, stderr));
    }
    Err(Error::new(format!(
        "{kind} failed ({status}):\n{}{}",
        String::from_utf8_lossy(&stdout),
        String::from_utf8_lossy(&stderr)
    )))
}

/// Stream exact preprocessing bytes to the caller's unpublished state file.
pub(super) async fn preprocess(
    mut command: Command,
    output_file: &mut tokio::fs::File,
    cancellation: &Cancellation,
) -> Result<u64, Error> {
    command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut process = spawn(command, cancellation).await?;
    let stdout = process.child.stdout.take().expect("piped stdout");
    let stderr = process.child.stderr.take().expect("piped stderr");
    let (status, digest, stderr) = tokio::join!(
        process.wait(cancellation),
        write_preprocessed(stdout, output_file),
        output(stderr)
    );
    let status = status?;
    let (digest, stderr) = (digest?, stderr?);
    if !status.success() {
        return Err(Error::new(format!(
            "native preprocessing failed ({status}):\n{}",
            String::from_utf8_lossy(&stderr)
        )));
    }
    Ok(digest)
}

async fn write_preprocessed(
    mut stream: impl AsyncRead + Unpin,
    output: &mut tokio::fs::File,
) -> Result<u64, Error> {
    let mut length = 0u64;
    let mut bytes = vec![0; 64 * 1024];
    loop {
        let count = stream.read(&mut bytes).await?;
        if count == 0 {
            return Ok(length);
        }
        output.write_all(&bytes[..count]).await?;
        length += count as u64;
    }
}

pub(super) async fn status(
    command: Command,
    cancellation: &Cancellation,
) -> Result<ExitStatus, Error> {
    spawn(command, cancellation).await?.wait(cancellation).await
}

async fn spawn(mut command: Command, cancellation: &Cancellation) -> Result<Process, Error> {
    cancellation.check()?;
    tree::prepare(&mut command);
    command.kill_on_drop(true);
    let mut child = command.spawn().map_err(|error| {
        Error::new(format!(
            "cannot run {}: {error}",
            command.as_std().get_program().to_string_lossy()
        ))
    })?;
    let tree = match tree::Tree::attach(&child) {
        Ok(tree) => tree,
        Err(error) => {
            let _ = child.kill().await;
            return Err(error.into());
        }
    };
    Ok(Process { child, tree })
}

struct Process {
    child: Child,
    tree: tree::Tree,
}

impl Process {
    async fn wait(&mut self, cancellation: &Cancellation) -> Result<ExitStatus, Error> {
        let cancelled = tokio::select! {
            biased;
            _ = cancellation.cancelled() => true,
            result = self.tree.wait_exit(&mut self.child) => { result?; false },
        };
        if cancelled {
            self.tree.interrupt();
            let _ = tokio::time::timeout(
                std::time::Duration::from_millis(200),
                self.tree.wait_exit(&mut self.child),
            )
            .await;
        }
        // Unix keeps the session leader unreaped until cleanup completes, preventing
        // its numeric session identity from being reused while descendants are killed.
        self.tree.terminate().await;
        self.tree.finished().await;
        let status = self.child.wait().await?;
        if cancelled {
            Err(Error::cancelled())
        } else {
            Ok(status)
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        // Runtime shutdown can drop the supervisor itself. Kill its complete tree and
        // reap the direct child synchronously in this exceptional path.
        self.tree.terminate_sync();
        let _ = self.child.start_kill();
        while matches!(self.child.try_wait(), Ok(None)) {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }
}

async fn output(mut stream: impl AsyncRead + Unpin) -> Result<Vec<u8>, Error> {
    // Drain every byte so children never block on pipes, while bounding diagnostics.
    const LIMIT: usize = 4 * 1024 * 1024;
    let mut result = Vec::new();
    let mut buffer = [0; 8192];
    loop {
        let count = stream.read(&mut buffer).await?;
        if count == 0 {
            return Ok(result);
        }
        let retained = count.min(LIMIT.saturating_sub(result.len()));
        result.extend_from_slice(&buffer[..retained]);
    }
}
