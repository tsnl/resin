//! Template WGSL shader for binary operations between two tensors.

enable f16;

alias Scalar = /* <template name="Scalar"> */ f32 /* </template> */;

struct BinaryOpParams {
    a_dim: vec3<u32>,
    b_dim: vec3<u32>,
};

@group(0) @binding(0) var<uniform> u: BinaryOpParams;
@group(0) @binding(1) var<storage, read_write> a: array<Scalar>;
@group(0) @binding(2) var<storage, read> b: array<Scalar>;

@compute
@workgroup_size(32, 1, 1)
fn binary_op_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x {
        return;
    }
    a[global_id.x] *= b[global_id.x];
}

@compute
@workgroup_size(32, 1, 1)
fn binary_op_div(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x {
        return;
    }
    a[global_id.x] /= b[global_id.x];
}


@compute
@workgroup_size(32, 1, 1)
fn binary_op_mod(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x {
        return;
    }
    a[global_id.x] %= b[global_id.x];
}

@compute
@workgroup_size(32, 1, 1)
fn binary_op_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x {
        return;
    }
    a[global_id.x] += b[global_id.x];
}

@compute
@workgroup_size(32, 1, 1)
fn binary_op_sub(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x {
        return;
    }
    a[global_id.x] -= b[global_id.x];
}