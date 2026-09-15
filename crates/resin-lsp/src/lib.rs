//! Resin's Language Server Protocol adapter. The `resin` executable owns command-line parsing.
mod analysis;
mod build;
mod caches;
mod inputs;
mod publication;
mod query;
mod server;
mod text;
mod worker;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Serve editor requests over stdin/stdout until the client shuts down.
pub fn serve(
    project: std::path::PathBuf,
    default_library_root: std::path::PathBuf,
) -> std::result::Result<i32, Box<dyn std::error::Error + Send + Sync>> {
    server::run(project, default_library_root)
}
