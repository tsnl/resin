fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/parser.rs");
    println!("cargo:rerun-if-changed=src/parser.lalrpop");
    lalrpop::process_root()?;
    Ok(())
}
