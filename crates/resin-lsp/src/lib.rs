//! Resin's Language Server Protocol adapter. The `resin` executable owns command-line parsing.
mod server;
mod text;
mod worker;

/// Serve editor requests over stdin/stdout until the client shuts down.
pub fn serve(
    default_stdlib: std::path::PathBuf,
) -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
    server::run(default_stdlib)
}
