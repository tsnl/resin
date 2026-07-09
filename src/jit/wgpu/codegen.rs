//! WGSL emission: one IR dispatch → one compute shader.
//!
//! Naive by design — no fusion, tiling, or shared-memory tricks; those belong
//! in the IR layer. Addressing is emitted as flat scalar arithmetic (no
//! arrays or helper functions), which keeps shaders trivial for drivers to
//! compile.

use crate::ir::{
    Accessor, Element, Expr, Kernel, TILE, TILE_LANES, dense_pitch, element_count,
};
use crate::jit::Error;
use crate::ops::{AssocOp, BinaryOp, Op, UnaryOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    /// log2 of elements processed per thread (naive strip-mining).
    pub lg2_items_per_thread: u32,
    pub workgroup_size: u32,
    /// Emit tiled kernels with the `chromium_experimental_subgroup_matrix`
    /// WGSL extension, decomposing each 16x16 tile into MMAs of this size
    /// (8 or 16, per the adapter's subgroup-matrix configs). `None` emits a
    /// portable per-lane fallback. wgpu/naga cannot parse the extension yet,
    /// so the extension path is for shaders executed by Dawn/Chrome.
    pub subgroup_matrix_size: Option<usize>,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self { lg2_items_per_thread: 3, workgroup_size: 64, subgroup_matrix_size: None }
    }
}

/// Workgroup x-size for subgroup-matrix kernels: a multiple of every common
/// `maxSubgroupSize` (extra subgroups redundantly compute the same tile).
const SUBGROUP_WORKGROUP: usize = 64;

pub fn workgroups(kernel: &Kernel, out: &Accessor, config: &KernelConfig) -> [u32; 3] {
    let mut count = element_count(&out.shape()) as u64;
    if count == 0 {
        return [0, 1, 1];
    }
    if let Kernel::Elementwise { element: Element::F32Tile16, .. } = kernel {
        if config.subgroup_matrix_size.is_some() {
            // One workgroup per output tile.
            return [count as u32, 1, 1];
        }
        count *= TILE_LANES as u64; // fallback: one thread strip per lane
    }
    let threads = count.div_ceil(1 << config.lg2_items_per_thread);
    [threads.div_ceil(u64::from(config.workgroup_size)).max(1) as u32, 1, 1]
}

pub fn emit(
    kernel: &Kernel,
    args: &[&Accessor],
    out: &Accessor,
    config: &KernelConfig,
) -> Result<String, Error> {
    let mut w = Writer::default();
    if let (Kernel::Elementwise { element: Element::F32Tile16, .. }, Some(_)) =
        (kernel, config.subgroup_matrix_size)
    {
        w.print("enable chromium_experimental_subgroup_matrix;");
    }
    w.print("@group(0) @binding(0)\nvar<storage, read_write> output: array<f32>;");
    for i in 0..args.len() {
        w.print(&format!(
            "@group(0) @binding({})\nvar<storage, read> arg{i}: array<f32>;",
            i + 1
        ));
    }
    match kernel {
        Kernel::Elementwise { expr, element: Element::F32 } => {
            emit_elementwise(&mut w, expr, args, out, config)
        }
        Kernel::Elementwise { expr, element: Element::F32Tile16 } => {
            emit_tiled(&mut w, expr, args, out, config)?
        }
        Kernel::Matmul => emit_matmul(&mut w, args, out, config),
        Kernel::Reduction { op, axes } => emit_reduction(&mut w, *op, axes, args, out, config),
    }
    Ok(w.finish())
}

#[derive(Default)]
struct Writer {
    lines: Vec<String>,
}

impl Writer {
    fn print(&mut self, text: &str) {
        self.lines
            .extend(text.lines().map(|line| line.trim_start().to_string()));
    }

    fn block(&mut self, prefix: &str, body: impl FnOnce(&mut Self)) {
        self.print(prefix);
        self.print("{");
        body(self);
        self.print("}");
    }

    fn finish(self) -> String {
        self.lines.join("\n")
    }
}

/// Emit `let i{k} = …` statements decoding `lin` (a row-major linear index
/// over `shape`) into coordinates; returns the coordinate names.
fn emit_decode(w: &mut Writer, shape: &[usize], lin: &str) -> Vec<String> {
    let pitch = dense_pitch(shape);
    let coords: Vec<String> = (0..shape.len()).map(|k| format!("i{k}")).collect();
    for (k, coord) in coords.iter().enumerate() {
        w.print(&format!("let {coord} = ({lin} / {}u) % {}u;", pitch[k], shape[k]));
    }
    coords
}

/// Inline address expression: `offset + Σ coordᵢ · pitchᵢ` (zero-pitch terms
/// dropped, so broadcasts read one address).
fn address(accessor: &Accessor, coords: &[String]) -> String {
    let s = accessor.strided();
    let mut expr = format!("{}u", s.offset);
    for (coord, &pitch) in coords.iter().zip(&s.pitch) {
        if pitch != 0 {
            expr.push_str(&format!(" + {coord} * {pitch}u"));
        }
    }
    expr
}

/// Emit the entry point. `body` writes code for one output element; it gets
/// the names of the output coordinates (`i0…`, decoded from `lin`).
fn per_output_element(
    w: &mut Writer,
    out: &Accessor,
    config: &KernelConfig,
    body: impl FnOnce(&mut Writer, &[String]),
) {
    let count = element_count(&out.shape());
    let items = 1u32 << config.lg2_items_per_thread;
    w.block(
        &format!(
            "@compute @workgroup_size({})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>)",
            config.workgroup_size
        ),
        |w| {
            if count == 0 {
                w.print("return;");
                return;
            }
            // Clamping by arrayLength never changes `count` for a valid
            // program (views are bounds-checked against their buffers); it
            // keeps the loop bound out of const-folding, which llvmpipe's
            // JIT has been seen to miscompile for fully-constant guards.
            w.print(&format!("let count = min({count}u, arrayLength(&output));"));
            w.print(&format!("let lin_beg = gid.x << {}u;", config.lg2_items_per_thread));
            w.block(
                &format!("for (var lin = lin_beg; lin < lin_beg + {items}u; lin += 1u)"),
                |w| {
                    w.block("if (lin >= count)", |w| w.print("return;"));
                    let coords = emit_decode(w, &out.shape(), "lin");
                    body(w, &coords);
                },
            );
        },
    );
}

fn emit_elementwise(
    w: &mut Writer,
    expr: &Expr,
    args: &[&Accessor],
    out: &Accessor,
    config: &KernelConfig,
) {
    let free = expr.loads();
    debug_assert_eq!(free.len(), args.len());
    emit_expr_fn(w, expr, args.len());
    per_output_element(w, out, config, |w, coords| {
        let loads: Vec<String> = args
            .iter()
            .enumerate()
            .map(|(i, a)| format!("arg{i}[{}]", address(a, coords)))
            .collect();
        w.print(&format!(
            "output[{}] = elem({});",
            address(out, coords),
            loads.join(", ")
        ));
    });
}

/// `fn elem(a0: f32, …) -> f32` evaluating the expression as nested WGSL.
fn emit_expr_fn(w: &mut Writer, expr: &Expr, num_args: usize) {
    let params: Vec<String> = (0..num_args).map(|i| format!("a{i}: f32")).collect();
    let free = expr.loads();
    w.block(&format!("fn elem({}) -> f32", params.join(", ")), |w| {
        w.print(&format!("return {};", spell_expr(expr, &free)));
    });
}

fn spell_expr(expr: &Expr, free: &[crate::ir::BufferViewRef]) -> String {
    match expr {
        Expr::Load(v) => {
            let i = free.iter().position(|x| x == v).expect("load in free list");
            format!("a{i}")
        }
        Expr::Op { op, args } => {
            let parts: Vec<String> = args.iter().map(|a| spell_expr(a, free)).collect();
            spell_op(*op, &parts)
        }
    }
}

fn spell_op(op: Op, args: &[String]) -> String {
    match op {
        Op::Unary(op) => {
            let x = &args[0];
            match op {
                UnaryOp::Neg => format!("(-({x}))"),
                UnaryOp::Exp => format!("exp({x})"),
                UnaryOp::Log => format!("log({x})"),
                UnaryOp::Relu => format!("max({x}, 0.0)"),
                UnaryOp::Abs => format!("abs({x})"),
                UnaryOp::Sqrt => format!("sqrt({x})"),
                UnaryOp::Sin => format!("sin({x})"),
                UnaryOp::Cos => format!("cos({x})"),
            }
        }
        Op::Binary(op) => {
            let (a, b) = (&args[0], &args[1]);
            match op {
                BinaryOp::Assoc(AssocOp::Add) => format!("(({a}) + ({b}))"),
                BinaryOp::Assoc(AssocOp::Mul) => format!("(({a}) * ({b}))"),
                BinaryOp::Assoc(AssocOp::Max) => format!("max({a}, {b})"),
                BinaryOp::Assoc(AssocOp::Min) => format!("min({a}, {b})"),
                BinaryOp::Sub => format!("(({a}) - ({b}))"),
                BinaryOp::Div => format!("(({a}) / ({b}))"),
                BinaryOp::Pow => format!("pow({a}, {b})"),
            }
        }
    }
}

/// Tiled elementwise kernels: v1 supports exactly `tileC = tileA × tileB`,
/// which is all the tiling pass generates.
fn tiled_mul_args(expr: &Expr) -> Result<(usize, usize), Error> {
    match expr {
        Expr::Op { op: crate::ops::Op::MUL, args } => match args.as_ref() {
            [Expr::Load(a), Expr::Load(b)] => {
                let free = expr.loads();
                let ia = free.iter().position(|v| v == a).unwrap();
                let ib = free.iter().position(|v| v == b).unwrap();
                Ok((ia, ib))
            }
            _ => Err(Error::Wgpu(
                "tiled elementwise kernels support exactly `Load * Load` for now".into(),
            )),
        },
        _ => Err(Error::Wgpu(
            "tiled elementwise kernels support exactly `Load * Load` for now".into(),
        )),
    }
}

fn emit_tiled(
    w: &mut Writer,
    expr: &Expr,
    args: &[&Accessor],
    out: &Accessor,
    config: &KernelConfig,
) -> Result<(), Error> {
    let (a, b) = tiled_mul_args(expr)?;
    match config.subgroup_matrix_size {
        Some(mma) => emit_tiled_subgroup(w, args[a], args[b], out, mma),
        None => emit_tiled_fallback(w, args[a], args[b], out, config),
    }
    Ok(())
}

/// Portable path: one thread per output lane, plain dot product within the
/// 16x16 tile. Accessors address tiles; lane address = tile address x 256.
fn emit_tiled_fallback(
    w: &mut Writer,
    a: &Accessor,
    b: &Accessor,
    out: &Accessor,
    config: &KernelConfig,
) {
    let tiles = element_count(&out.shape());
    let lanes = tiles * TILE_LANES;
    let items = 1u32 << config.lg2_items_per_thread;
    w.block(
        &format!(
            "@compute @workgroup_size({})\nfn main(@builtin(global_invocation_id) gid: vec3<u32>)",
            config.workgroup_size
        ),
        |w| {
            if lanes == 0 {
                w.print("return;");
                return;
            }
            w.print(&format!("let count = min({lanes}u, arrayLength(&output));"));
            w.print(&format!("let lin_beg = gid.x << {}u;", config.lg2_items_per_thread));
            w.block(
                &format!("for (var lin = lin_beg; lin < lin_beg + {items}u; lin += 1u)"),
                |w| {
                    w.block("if (lin >= count)", |w| w.print("return;"));
                    w.print(&format!("let tile = lin / {TILE_LANES}u;"));
                    w.print(&format!("let lane = lin % {TILE_LANES}u;"));
                    w.print(&format!("let row = lane / {TILE}u;"));
                    w.print(&format!("let col = lane % {TILE}u;"));
                    let coords = emit_decode(w, &out.shape(), "tile");
                    w.print(&format!("let a_tile = {};", address(a, &coords)));
                    w.print(&format!("let b_tile = {};", address(b, &coords)));
                    w.print(&format!("let out_tile = {};", address(out, &coords)));
                    w.print("var sum: f32 = 0.0;");
                    w.block(&format!("for (var s: u32 = 0u; s < {TILE}u; s += 1u)"), |w| {
                        w.print(&format!(
                            "sum += arg0[a_tile * {TILE_LANES}u + row * {TILE}u + s] * arg1[b_tile * {TILE_LANES}u + s * {TILE}u + col];"
                        ));
                    });
                    w.print(&format!(
                        "output[out_tile * {TILE_LANES}u + row * {TILE}u + col] = sum;"
                    ));
                },
            );
        },
    );
}

/// Cooperative-matrix path (Dawn/Chrome only until wgpu's WGSL frontend
/// learns the extension): one workgroup per output tile, decomposed into
/// `mma`-sized subgroup matrix multiply-accumulates.
fn emit_tiled_subgroup(w: &mut Writer, a: &Accessor, b: &Accessor, out: &Accessor, mma: usize) {
    assert!(TILE.is_multiple_of(mma), "subgroup matrix size must divide {TILE}");
    let blocks = TILE / mma;
    let tiles = element_count(&out.shape());
    w.block(
        &format!(
            "@compute @workgroup_size({SUBGROUP_WORKGROUP})\nfn main(@builtin(workgroup_id) wg: vec3<u32>)"
        ),
        |w| {
            w.block(&format!("if (wg.x >= {tiles}u)"), |w| w.print("return;"));
            let coords = emit_decode(w, &out.shape(), "wg.x");
            w.print(&format!("let a_base = ({}) * {TILE_LANES}u;", address(a, &coords)));
            w.print(&format!("let b_base = ({}) * {TILE_LANES}u;", address(b, &coords)));
            w.print(&format!("let out_base = ({}) * {TILE_LANES}u;", address(out, &coords)));
            w.block(&format!("for (var bi = 0u; bi < {blocks}u; bi += 1u)"), |w| {
                w.block(&format!("for (var bj = 0u; bj < {blocks}u; bj += 1u)"), |w| {
                    w.print(&format!("var acc: subgroup_matrix_result<f32, {mma}, {mma}>;"));
                    w.block(&format!("for (var s = 0u; s < {blocks}u; s += 1u)"), |w| {
                        w.print(&format!(
                            "let a = subgroupMatrixLoad<subgroup_matrix_left<f32, {mma}, {mma}>>(&arg0, a_base + (bi * {TILE}u + s) * {mma}u, false, {TILE}u);"
                        ));
                        w.print(&format!(
                            "let b = subgroupMatrixLoad<subgroup_matrix_right<f32, {mma}, {mma}>>(&arg1, b_base + (s * {TILE}u + bj) * {mma}u, false, {TILE}u);"
                        ));
                        w.print("acc = subgroupMatrixMultiplyAccumulate(a, b, acc);");
                    });
                    w.print(&format!(
                        "subgroupMatrixStore(&output, out_base + (bi * {TILE}u + bj) * {mma}u, acc, false, {TILE}u);"
                    ));
                });
            });
        },
    );
}

fn emit_matmul(w: &mut Writer, args: &[&Accessor], out: &Accessor, config: &KernelConfig) {
    let rank = out.rank();
    let k = args[0].shape()[rank - 1];
    per_output_element(w, out, config, |w, coords| {
        // A reads [batch…, i, t]; B reads [batch…, t, j].
        let mut a_coords = coords.to_vec();
        a_coords[rank - 1] = "t".into();
        let mut b_coords = coords.to_vec();
        b_coords[rank - 2] = "t".into();

        w.print("var sum: f32 = 0.0;");
        w.block(&format!("for (var t: u32 = 0u; t < {k}u; t += 1u)"), |w| {
            w.print(&format!(
                "sum += arg0[{}] * arg1[{}];",
                address(args[0], &a_coords),
                address(args[1], &b_coords),
            ));
        });
        w.print(&format!("output[{}] = sum;", address(out, coords)));
    });
}

fn emit_reduction(
    w: &mut Writer,
    op: AssocOp,
    axes: &[usize],
    args: &[&Accessor],
    out: &Accessor,
    config: &KernelConfig,
) {
    let input = args[0];
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    let count: usize = sorted_axes.iter().map(|&axis| input.shape()[axis]).product();

    per_output_element(w, out, config, |w, coords| {
        w.print(&format!("var acc: f32 = {};", identity_literal(op)));
        w.block(&format!("for (var ri: u32 = 0u; ri < {count}u; ri += 1u)"), |w| {
            // Decode `ri` into the reduced axes (last sorted axis fastest);
            // other coordinates come from the output position.
            let mut in_coords = coords.to_vec();
            let mut stride = 1;
            for &axis in sorted_axes.iter().rev() {
                let dim = input.shape()[axis];
                in_coords[axis] = format!("r{axis}");
                w.print(&format!("let r{axis} = (ri / {stride}u) % {dim}u;"));
                stride *= dim;
            }
            let value = format!("arg0[{}]", address(input, &in_coords));
            w.print(&match op {
                AssocOp::Add => format!("acc += {value};"),
                AssocOp::Mul => format!("acc *= {value};"),
                AssocOp::Max => format!("acc = max(acc, {value});"),
                AssocOp::Min => format!("acc = min(acc, {value});"),
            });
        });
        w.print(&format!("output[{}] = acc;", address(out, coords)));
    });
}

/// WGSL literal for the fold identity (bit patterns for ±inf).
fn identity_literal(op: AssocOp) -> &'static str {
    match op {
        AssocOp::Add => "0.0",
        AssocOp::Mul => "1.0",
        AssocOp::Max => "bitcast<f32>(0xff800000u)",
        AssocOp::Min => "bitcast<f32>(0x7f800000u)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::BufferViewRef;
    use crate::ops::Op;

    fn load_op(op: Op, n: usize) -> Kernel {
        let views = (0..n).map(BufferViewRef);
        Kernel::elementwise(Expr::new_op(op, views.map(Expr::Load)))
    }

    #[test]
    fn elementwise_add_shader_shape() {
        let a = Accessor::dense([4], 0);
        let wgsl = emit(
            &load_op(Op::ADD, 2),
            &[&a, &a],
            &a,
            &KernelConfig::default(),
        ).unwrap();
        assert!(wgsl.contains("@compute"), "{wgsl}");
        assert!(wgsl.contains("fn elem(a0: f32, a1: f32) -> f32"), "{wgsl}");
        assert!(wgsl.contains("((a0) + (a1))"), "{wgsl}");
    }

    #[test]
    fn relu_emits_max() {
        let a = Accessor::dense([8], 0);
        let wgsl = emit(
            &load_op(Op::RELU, 1),
            &[&a],
            &a,
            &KernelConfig::default(),
        ).unwrap();
        assert!(wgsl.contains("max(a0, 0.0)"), "{wgsl}");
    }

    #[test]
    fn broadcast_arg_reads_single_address() {
        let out = Accessor::dense([2, 3], 0);
        let scalar = Accessor::dense([], 0).broadcast_to(&[2, 3]).unwrap();
        let wgsl = emit(
            &load_op(Op::MUL, 2),
            &[&out, &scalar],
            &out,
            &KernelConfig::default(),
        ).unwrap();
        // Zero-pitch dims contribute no address terms: the scalar loads `arg1[0u]`.
        assert!(wgsl.contains("arg1[0u]"), "{wgsl}");
    }

    #[test]
    fn matmul_shader_loops_over_k() {
        let a = Accessor::dense([2, 4], 0);
        let b = Accessor::dense([4, 3], 0);
        let out = Accessor::dense([2, 3], 0);
        let wgsl = emit(&Kernel::Matmul, &[&a, &b], &out, &KernelConfig::default()).unwrap();
        assert!(wgsl.contains("t < 4u"), "{wgsl}");
        assert!(wgsl.contains("i0 * 4u + t * 1u"), "{wgsl}");
    }

    #[test]
    fn reduction_starts_from_identity() {
        let input = Accessor::dense([2, 3], 0);
        let out = Accessor::dense([2, 1], 0);
        let wgsl = emit(
            &Kernel::Reduction { op: AssocOp::Max, axes: Box::from([1]) },
            &[&input],
            &out,
            &KernelConfig::default(),
        ).unwrap();
        assert!(wgsl.contains("bitcast<f32>(0xff800000u)"), "{wgsl}");
    }
}
