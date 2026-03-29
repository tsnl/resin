fn main() {
    println!("cargo::rerun-if-changed=src/parser.lalrpop");
    lalrpop::process_root().unwrap();
}
