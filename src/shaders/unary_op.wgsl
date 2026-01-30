//! Template WGSL shader for binary operations between two tensors.

struct UnaryOpParams {
    a_dim: vec3<u32>,
};

@group(0) @binding(0) var<uniform> u: UnaryOpParams;
@group(0) @binding(1) var<storage, read_write> a: array<f32>;

@compute
@workgroup_size(32, 1, 1)
fn unary_op_neg(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x * u.a_dim.y * u.a_dim.z {
        return;
    }
    a[global_id.x] = -a[global_id.x];
}

@compute
@workgroup_size(32, 1, 1)
fn unary_op_abs(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x * u.a_dim.y * u.a_dim.z {
        return;
    }
    a[global_id.x] = abs(a[global_id.x]);
}

@compute
@workgroup_size(32, 1, 1)
fn unary_op_exp(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x * u.a_dim.y * u.a_dim.z {
        return;
    }
    a[global_id.x] = exp(a[global_id.x]);
}

@compute
@workgroup_size(32, 1, 1)
fn unary_op_log(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= u.a_dim.x * u.a_dim.y * u.a_dim.z {
        return;
    }
    a[global_id.x] = log(a[global_id.x]);
}
