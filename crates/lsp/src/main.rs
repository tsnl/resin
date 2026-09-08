mod server;
mod text;
mod worker;

fn main() {
    match start() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("resin-lsp: {error}");
            std::process::exit(1);
        }
    }
}

fn start() -> Result<i32, Box<dyn std::error::Error + Send + Sync>> {
    let mut stdlib = None;
    let mut args = std::env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--stdio") => {}
            Some("--stdlib") => {
                stdlib = Some(std::path::PathBuf::from(
                    args.next().ok_or("--stdlib requires a directory")?,
                ))
            }
            Some("--version") => {
                println!("resin-lsp {}", env!("CARGO_PKG_VERSION"));
                return Ok(0);
            }
            Some("--help" | "-h") => {
                println!(
                    "Usage: resin-lsp [--stdio] [--stdlib PATH]\n\nServe Resin editor requests over stdio. RESIN_STDLIB also configures the standard library."
                );
                return Ok(0);
            }
            _ => return Err(format!("unknown argument: {}", arg.to_string_lossy()).into()),
        }
    }
    let default_stdlib = std::env::var_os("RESIN_STDLIB")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(resin::ast::stdlib_path);
    server::run(stdlib, default_stdlib)
}
