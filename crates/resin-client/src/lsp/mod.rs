//! Editor revisions and local file acquisition around the shared HTTP client.
mod analysis;
mod build;
mod mirrors;
mod query;
mod server;
mod text;
mod worker;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) fn serve(
    project: std::path::PathBuf,
    include_roots: Vec<std::path::PathBuf>,
) -> Result<i32> {
    server::run(project, include_roots)
}
