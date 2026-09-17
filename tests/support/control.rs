//! Sequential control-flow fixtures shared by regressions and compiler benchmarks.
#![allow(dead_code)]

pub fn host_source(branches: usize) -> String {
    let mut source = String::from(
        "export { main }; struct Failed {} fn step() -> (() | Err<Failed>) { (()) } fn main() -> (i32 | Err<Failed>) { let mut value = 0; ",
    );
    for _ in 0..branches {
        source.push_str("if (value == 0) { value = 1; } else { value = 0; }; step()?; ");
    }
    source.push_str("(value) }");
    source
}

pub fn shader_source(branches: usize) -> String {
    let mut source = String::from(
        "export { kernel }; struct Failed {} fn step() -> (() | Err<Failed>) { (()) } fn helper(value: u32) -> (u32 | Err<Failed>) { let mut result = value; ",
    );
    for _ in 0..branches {
        source.push_str(
            "if (result == u32(0)) { result = u32(1); } else { result = u32(0); }; step()?; ",
        );
    }
    source.push_str("(result) } @compute_shader fn kernel(i: u64, output: Ptr<u32>) { output.* = match (helper(u32(i))) { u32(value) => { value }, Err(error) => { u32(99) } }; }");
    source
}
