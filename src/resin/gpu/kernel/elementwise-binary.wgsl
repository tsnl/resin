//! Performs elementwise binary operations on the GPU using WebGPU.
//! TODO: Need to handle sparse arg views.

const BOP_POW: u32 = 0;
const BOP_MUL: u32 = 1;
const BOP_DIV: u32 = 2;
const BOP_ADD: u32 = 3;
const BOP_SUB: u32 = 4;
const BOP_MAX: u32 = 5;
const BOP_MIN: u32 = 6;
const BOP_EQ: u32 = 7;
const BOP_NE: u32 = 8;
const BOP_GT: u32 = 9;
const BOP_LT: u32 = 10;
const BOP_GE: u32 = 11;
const BOP_LE: u32 = 12;

// Template aliases:
/* template */ alias T = f32;

// Template consts:
/* template */ const SELECTED_BOP = BOP_POW;
/* template */ const N = 16;

@group(0) @binding(0) var<storage, write> out: array<T>;
@group(1) @binding(0) var<storage, read> arg0: array<T>;
@group(1) @binding(1) var<storage, read> arg1: array<T>;

fn dispatch_bop(x: T, y: T) -> T {
    switch (SELECTED_BOP) {
        case BOP_POW: return pow(x, y),
        case BOP_MUL: return x * y,
        case BOP_DIV: return x / y,
        case BOP_ADD: return x + y,
        case BOP_SUB: return x - y,
        case BOP_MAX: return max(x, y),
        case BOP_MIN: return min(x, y),
    };
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= N) {
        return;
    }

    out[i] = dispatch_bop(arg0[i], arg1[i]);
}
