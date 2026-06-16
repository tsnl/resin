//! Performs matrix multiplication on the GPU using WebGPU.
//! TODO: Need to handle sparse arg views.
//! TODO: Naive implementation: need to use tiling and cooperative matrix extensions.

// Template types:
/* template */ alias T = f32;

// Template consts:
/* template */ const B = 8;
/* template */ const M = 16;
/* template */ const K = 16;
/* template */ const N = 16;

@group(0) @binding(0) var<storage, write> out: array<ScalarType>;
@group(1) @binding(0) var<storage, read> arg0: array<ScalarType>;
@group(1) @binding(1) var<storage, read> arg1: array<ScalarType>;

@compute @workgroup_size(B, M, N)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;
    let m = global_id.y;
    let n = global_id.z;
    if (b >= B || m >= M || n >= N) {
        return;
    }
    var sum: ScalarType = 0.0;
    for k in 0..K {
        let a_index = b * M * K + m * K + k;
        let b_index = b * K * N + k * N + n;
        sum += arg0[a_index] * arg1[b_index];
    }
    let out_index = b * M * N + m * N + n;
    out[out_index] = sum;
}
