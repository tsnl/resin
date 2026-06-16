const UOP_NEG: u32 = 0;
const UOP_EXP: u32 = 1;
const UOP_LOG: u32 = 2;
const UOP_NOT: u32 = 3;
const UOP_SIN: u32 = 4;
const UOP_COS: u32 = 5;

// Template aliases:
/* template */ alias T = f32;

// Template consts:
/* template */ const SELECTED_UOP = UOP_NEG;
/* template */ const N = 16;

@group(0) @binding(0) var<storage, write> out: array<T>;
@group(1) @binding(0) var<storage, read> arg0: array<T>;

fn dispatch_uop(x: T) -> T {
    switch (SELECTED_UOP) {
        case UOP_NEG: return -x,
        case UOP_EXP: return exp(x),
        case UOP_LOG: return log(x),
        case UOP_NOT: return abs(1.0 - x), // Assuming x is 0.0 or 1.0
        case UOP_SIN: return sin(x),
        case UOP_COS: return cos(x),
    };
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= N) {
        return;
    }

    out[i] = dispatch_uop(arg0[i]);
}
