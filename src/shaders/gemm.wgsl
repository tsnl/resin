//! Template WGSL shader for General Matrix-Matrix Multiplication (GEMM).
//! TODO: Add support for special cooperative matrix multiplication instructions.

struct GemmParams {
    a_dim: vec3<u32>,   // b, r, c
    b_dim: vec3<u32>,   // b, r, c
    c_dim: vec3<u32>,   // b, r, c
    a_coeff: f32,
    b_coeff: f32,
    c_coeff: f32,
};

@group(0) @binding(0) var<uniform> u: GemmParams;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

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
fn unpack_mat_index(dim: vec3<u32>, i: u32) -> vec3<u32> {
    let nr = dim.y;
    let nc = dim.z;
    let bi = i / (nc * nr);
    let remainder = i % (nc * nr);
    let ri = remainder / nc;
    let ci = remainder % nc;
    return vec3<u32>(bi, ri, ci);
}

@compute
@workgroup_size(32, 1, 1)
fn gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.c_dim.x * u.c_dim.y * u.c_dim.z {
        return;
    }
    
    let c_index = global_id.x;
    let c_coords = unpack_mat_index(u.c_dim, c_index);
    let bi = c_coords.x;
    let ri = c_coords.y;
    let ci = c_coords.z;

    var sum: f32 = 0.0;
    for (var k = 0u; k < u.a_dim.z; k++) {
        let a_index: u32 = mat_index(u.a_dim, bi, global_id.y, k);
        let b_index: u32 = mat_index(u.b_dim, bi, k, global_id.x);
        sum += (a[a_index] * u.a_coeff) * (b[b_index] * u.b_coeff);
    }

    c[c_index] = c[c_index] * u.c_coeff + sum;
}
