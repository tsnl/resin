//! Template WGSL shader for General Matrix-Matrix Multiplication (GEMM).
//! TODO: Add support for special cooperative matrix multiplication instructions.

enable f16;

alias Scalar = /* <template name="Scalar"> */ f32 /* </template> */;

struct GemmParams {
    a_dim: vec3<u32>,   // b, r, c
    b_dim: vec3<u32>,   // b, r, c
    c_dim: vec3<u32>,   // b, r, c
    a_coeff: Scalar,
    b_coeff: Scalar,
    c_coeff: Scalar,
};

@group(0) @binding(0) var<uniform> u: GemmParams;
@group(0) @binding(1) var<storage, read> a: array<Scalar>;
@group(0) @binding(2) var<storage, read> b: array<Scalar>;
@group(0) @binding(3) var<storage, read_write> c: array<Scalar>;

fn mat_index(dim: vec3<u32>, bi: u32, ri: u32, ci: u32) -> u32 {
    let nb = dim.x;
    let nr = dim.y;
    let nc = dim.z;
    return (
        ci +
        ri * nc +
        bi * nc * nr
    );
}

@compute
@workgroup_size(8, 8, 1)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if any(global_id >= u.c_dim) {
        return;
    }
    
    let bi = global_id.z;
    
    var sum: Scalar = 0.0;
    for (var k = 0u; k < u.a_dim.z; k++) {
        let a_index: u32 = mat_index(u.a_dim, bi, global_id.y, k);
        let b_index: u32 = mat_index(u.b_dim, bi, k, global_id.x);
        sum += (a[a_index] * u.a_coeff) * (b[b_index] * u.b_coeff);
    }

    let c_index: u32 = mat_index(u.c_dim, bi, global_id.y, global_id.x);
    c[c_index] = c[c_index] * u.c_coeff + sum;
}
