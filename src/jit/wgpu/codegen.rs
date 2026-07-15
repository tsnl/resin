//! WGSL emission: one IR dispatch → one compute shader.
//!
//! Naive by design — no fusion or shared-memory tricks; those belong in the
//! IR layer. Addressing is emitted as flat scalar arithmetic (no arrays or
//! helper functions), which keeps shaders trivial for drivers to compile.

use crate::ir::{
    Accessor, BufferViewRef, Dispatch, Expr, Kernel, Program, RemapInfo, dense_stride, element_count,
};
use crate::ops::{AssocOp, BinaryOp, ElementType, Op, UnaryOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    /// log2 of elements processed per thread (naive strip-mining).
    pub lg2_items_per_thread: u32,
    pub workgroup_size: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            lg2_items_per_thread: 3,
            workgroup_size: 64,
        }
    }
}

/// Thread count for a dispatch (scatter remaps iterate the source).
pub fn workgroups_for(program: &Program, dispatch: &Dispatch, config: &KernelConfig) -> [u32; 3] {
    let shape = thread_shape(program, dispatch);
    workgroups_for_shape(shape, config)
}

fn thread_shape<'a>(program: &'a Program, dispatch: &'a Dispatch) -> &'a [usize] {
    match &dispatch.kernel {
        Kernel::Remap { info } if info.is_scatter() => {
            &program.view(dispatch.args[0]).accessor.shape
        }
        _ => &program.view(dispatch.output).accessor.shape,
    }
}

fn workgroups_for_shape(shape: &[usize], config: &KernelConfig) -> [u32; 3] {
    let count = element_count(shape) as u64;
    if count == 0 {
        return [0, 1, 1];
    }
    let threads = count.div_ceil(1 << config.lg2_items_per_thread);
    [
        threads.div_ceil(u64::from(config.workgroup_size)).max(1) as u32,
        1,
        1,
    ]
}

/// Program-wide heap layout: `@binding(i)` ↔ `program.buffers[i]`.
///
/// A shader declares only the heaps it uses, each at its program-wide binding
/// index; the pipeline's bind-group layout still covers every arena, so one
/// bind group serves all dispatches.
pub fn program_heap_keys(program: &Program) -> Vec<(ElementType, bool)> {
    program
        .buffers
        .iter()
        .map(|b| (b.element_type, b.atomic))
        .collect()
}

/// Emit WGSL for one dispatch.
///
/// Bindings are the full program heap layout (not per-kernel). Atomic RMW
/// targets live on separate arenas from plain storage of the same etype.
pub fn emit_dispatch(program: &Program, dispatch: &Dispatch, config: &KernelConfig) -> String {
    let out_view = program.view(dispatch.output);
    let out = &out_view.accessor;
    let out_buf = program.buffer(out_view.buffer);
    let out_etype = out_buf.element_type;
    let out_atomic = out_buf.atomic;

    let args: Vec<&Accessor> = dispatch
        .args
        .iter()
        .map(|&r| &program.view(r).accessor)
        .collect();
    let arg_meta: Vec<(ElementType, bool)> = dispatch
        .args
        .iter()
        .map(|&r| {
            let b = program.buffer(program.view(r).buffer);
            (b.element_type, b.atomic)
        })
        .collect();

    let program_heaps = program_heap_keys(program);
    emit(
        &dispatch.kernel,
        &args,
        &arg_meta,
        out,
        out_etype,
        out_atomic,
        &program_heaps,
        config,
    )
}

pub fn emit(
    kernel: &Kernel,
    args: &[&Accessor],
    // Per-arg (element_type, atomic_heap).
    arg_meta: &[(ElementType, bool)],
    out: &Accessor,
    out_etype: ElementType,
    out_atomic: bool,
    // Full program heap list in binding order (`@binding(i)` = entry i).
    program_heaps: &[(ElementType, bool)],
    config: &KernelConfig,
) -> String {
    let mut w = Writer::default();
    let used = used_heaps(out_etype, out_atomic, arg_meta);
    emit_heap_bindings(&mut w, program_heaps, &used);
    match kernel {
        Kernel::Elementwise { expr } => {
            emit_elementwise(
                &mut w, expr, args, arg_meta, out, out_etype, out_atomic, config,
            );
        }
        Kernel::Matmul => {
            emit_matmul(&mut w, args, arg_meta, out, out_etype, out_atomic, config);
        }
        Kernel::Reduction { op, axes } => {
            emit_reduction(
                &mut w, *op, axes, args, arg_meta, out, out_etype, out_atomic, config,
            );
        }
        Kernel::Remap { info } => {
            emit_remap(
                &mut w, info, args, arg_meta, out, out_etype, out_atomic, config,
            );
        }
    }
    w.finish()
}

/// Heaps one kernel touches: its output plus every argument, deduped.
fn used_heaps(
    out_etype: ElementType,
    out_atomic: bool,
    arg_meta: &[(ElementType, bool)],
) -> Vec<(ElementType, bool)> {
    let mut used = vec![(out_etype, out_atomic)];
    for &m in arg_meta {
        if !used.contains(&m) {
            used.push(m);
        }
    }
    used
}

fn spell_etype(etype: ElementType) -> &'static str {
    match etype {
        ElementType::F32 => "f32",
        ElementType::U32 => "u32",
    }
}

fn heap_name(etype: ElementType, atomic: bool) -> &'static str {
    match (etype, atomic) {
        (ElementType::F32, false) => "heap_f32",
        (ElementType::F32, true) => "heap_atomic_f32",
        (ElementType::U32, false) => "heap_u32",
        (ElementType::U32, true) => "heap_atomic_u32",
    }
}

/// Declare the heaps in `used`, each at its program-wide binding index. Unused
/// arenas are skipped — the shader binds a subset of the pipeline layout, which
/// wgpu permits, so no filler is needed to keep bindings alive.
fn emit_heap_bindings(
    w: &mut Writer,
    program_heaps: &[(ElementType, bool)],
    used: &[(ElementType, bool)],
) {
    for (i, &(etype, atomic)) in program_heaps.iter().enumerate() {
        if !used.contains(&(etype, atomic)) {
            continue;
        }
        let name = heap_name(etype, atomic);
        if atomic {
            // RMW targets only; f32 values stored as bit patterns.
            w.print(&format!(
                "@group(0) @binding({i})\nvar<storage, read_write> {name}: array<atomic<u32>>;"
            ));
        } else {
            let t = spell_etype(etype);
            w.print(&format!(
                "@group(0) @binding({i})\nvar<storage, read_write> {name}: array<{t}>;"
            ));
        }
    }
}

fn read_at(etype: ElementType, atomic: bool, addr: &str) -> String {
    let name = heap_name(etype, atomic);
    if atomic {
        match etype {
            ElementType::U32 => format!("atomicLoad(&{name}[{addr}])"),
            ElementType::F32 => format!("bitcast<f32>(atomicLoad(&{name}[{addr}]))"),
        }
    } else {
        format!("{name}[{addr}]")
    }
}

fn write_at(etype: ElementType, atomic: bool, addr: &str, value: &str) -> String {
    let name = heap_name(etype, atomic);
    if atomic {
        match etype {
            ElementType::U32 => format!("atomicStore(&{name}[{addr}], {value});"),
            ElementType::F32 => format!("atomicStore(&{name}[{addr}], bitcast<u32>({value}));"),
        }
    } else {
        format!("{name}[{addr}] = {value};")
    }
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

/// Emit `let i{k} = …` statements decoding `lin` into coordinates.
fn emit_decode(w: &mut Writer, shape: &[usize], lin: &str) -> Vec<String> {
    let stride = dense_stride(shape);
    let coords: Vec<String> = (0..shape.len()).map(|k| format!("i{k}")).collect();
    for (k, coord) in coords.iter().enumerate() {
        w.print(&format!(
            "let {coord} = ({lin} / {}u) % {}u;",
            stride[k], shape[k]
        ));
    }
    coords
}

fn address(accessor: &Accessor, coords: &[String]) -> String {
    let mut expr = format!("{}u", accessor.offset);
    for (coord, &stride) in coords.iter().zip(&accessor.stride) {
        if stride != 0 {
            expr.push_str(&format!(" + {coord} * {stride}u"));
        }
    }
    expr
}

fn per_output_element(
    w: &mut Writer,
    out: &Accessor,
    config: &KernelConfig,
    body: impl FnOnce(&mut Writer, &[String]),
) {
    let count = element_count(&out.shape);
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
            // Literal count: after arena packing, arrayLength is the whole heap,
            // not this view's region.
            w.print(&format!("let count = {count}u;"));
            w.print(&format!(
                "let lin_beg = gid.x << {}u;",
                config.lg2_items_per_thread
            ));
            w.block(
                &format!("for (var lin = lin_beg; lin < lin_beg + {items}u; lin += 1u)"),
                |w| {
                    w.block("if (lin >= count)", |w| w.print("return;"));
                    let coords = emit_decode(w, &out.shape, "lin");
                    body(w, &coords);
                },
            );
        },
    );
}

/// Loop over a source shape with a literal element-count guard (scatter).
fn per_thread_element(
    w: &mut Writer,
    shape: &[usize],
    config: &KernelConfig,
    body: impl FnOnce(&mut Writer, &str, &[String]),
) {
    let count = element_count(shape);
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
            w.print(&format!(
                "let lin_beg = gid.x << {}u;",
                config.lg2_items_per_thread
            ));
            w.block(
                &format!("for (var lin = lin_beg; lin < lin_beg + {items}u; lin += 1u)"),
                |w| {
                    w.block(&format!("if (lin >= {count}u)"), |w| w.print("return;"));
                    let coords = emit_decode(w, shape, "lin");
                    body(w, "lin", &coords);
                },
            );
        },
    );
}

fn emit_elementwise(
    w: &mut Writer,
    expr: &Expr,
    args: &[&Accessor],
    arg_meta: &[(ElementType, bool)],
    out: &Accessor,
    out_etype: ElementType,
    out_atomic: bool,
    config: &KernelConfig,
) {
    let free = expr.loads();
    debug_assert_eq!(free.len(), args.len());
    emit_expr_fn(w, expr, args.len(), out_etype);
    per_output_element(w, out, config, |w, coords| {
        let loads: Vec<String> = args
            .iter()
            .zip(arg_meta.iter())
            .map(|(a, &(et, atomic))| read_at(et, atomic, &address(a, coords)))
            .collect();
        let dest = address(out, coords);
        let call = format!("elem({})", loads.join(", "));
        w.print(&write_at(out_etype, out_atomic, &dest, &call));
    });
}

fn emit_expr_fn(w: &mut Writer, expr: &Expr, num_args: usize, out_etype: ElementType) {
    // Argument types: for type-changing unaries the free load may differ.
    // We emit load params as the same type as their buffers — callers pass
    // typed arg bindings; spell_expr inserts casts as needed. For simplicity
    // we type every `a{i}` from a walk that uses out_etype for non-cast trees
    // and f32/u32 for each free load via the op tree.
    let free = expr.loads();
    let arg_types = infer_load_types(expr, &free, out_etype);
    let params: Vec<String> = (0..num_args)
        .map(|i| format!("a{i}: {}", spell_etype(arg_types[i])))
        .collect();
    let ret = spell_etype(out_etype);
    w.block(&format!("fn elem({}) -> {ret}", params.join(", ")), |w| {
        w.print(&format!("return {};", spell_expr(expr, &free, out_etype)));
    });
}

/// Load types for `elem` params. Cast/bitcast kernels convert from the other
/// width type; everything else inherits the kernel output type.
fn infer_load_types(
    expr: &Expr,
    free: &[BufferViewRef],
    out_etype: ElementType,
) -> Vec<ElementType> {
    let mut types = vec![out_etype; free.len()];
    fn walk(expr: &Expr, free: &[BufferViewRef], expected: ElementType, types: &mut [ElementType]) {
        match expr {
            Expr::Load(v) => {
                if let Some(i) = free.iter().position(|x| x == v) {
                    types[i] = expected;
                }
            }
            Expr::Op { op, args } => match op {
                Op::Unary(UnaryOp::Cast { to } | UnaryOp::Bitcast { to }) => {
                    // Result is `to`; operand is the other type.
                    let src = match to {
                        ElementType::F32 => ElementType::U32,
                        ElementType::U32 => ElementType::F32,
                    };
                    walk(&args[0], free, src, types);
                }
                _ => {
                    for arg in args.iter() {
                        walk(arg, free, expected, types);
                    }
                }
            },
        }
    }
    walk(expr, free, out_etype, &mut types);
    types
}

fn spell_expr(expr: &Expr, free: &[BufferViewRef], out_etype: ElementType) -> String {
    match expr {
        Expr::Load(v) => {
            let i = free.iter().position(|x| x == v).expect("load in free list");
            format!("a{i}")
        }
        Expr::Op { op, args } => {
            let parts: Vec<String> = args
                .iter()
                .map(|a| spell_expr(a, free, out_etype))
                .collect();
            spell_op(*op, &parts, out_etype)
        }
    }
}

fn spell_op(op: Op, args: &[String], out_etype: ElementType) -> String {
    let t = spell_etype(out_etype);
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
                UnaryOp::Floor => format!("floor({x})"),
                UnaryOp::Ceil => format!("ceil({x})"),
                UnaryOp::Bitcast { to } => format!("bitcast<{}>({x})", spell_etype(to)),
                UnaryOp::Cast { to } => format!("{}({x})", spell_etype(to)),
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
                BinaryOp::CmpEq => format!("select({t}(0), {t}(1), ({a}) == ({b}))"),
                BinaryOp::CmpNe => format!("select({t}(0), {t}(1), ({a}) != ({b}))"),
                BinaryOp::CmpLt => format!("select({t}(0), {t}(1), ({a}) < ({b}))"),
                BinaryOp::CmpLe => format!("select({t}(0), {t}(1), ({a}) <= ({b}))"),
                BinaryOp::CmpGt => format!("select({t}(0), {t}(1), ({a}) > ({b}))"),
                BinaryOp::CmpGe => format!("select({t}(0), {t}(1), ({a}) >= ({b}))"),
                BinaryOp::Band => format!("(({a}) & ({b}))"),
                BinaryOp::Bor => format!("(({a}) | ({b}))"),
                BinaryOp::Bxor => format!("(({a}) ^ ({b}))"),
                BinaryOp::Shl => format!("(({a}) << ({b}))"),
                BinaryOp::Shr => format!("(({a}) >> ({b}))"),
            }
        }
    }
}

fn emit_matmul(
    w: &mut Writer,
    args: &[&Accessor],
    arg_meta: &[(ElementType, bool)],
    out: &Accessor,
    out_etype: ElementType,
    out_atomic: bool,
    config: &KernelConfig,
) {
    let rank = out.rank();
    let k = args[0].shape[rank - 1];
    per_output_element(w, out, config, |w, coords| {
        let mut a_coords = coords.to_vec();
        a_coords[rank - 1] = "t".into();
        let mut b_coords = coords.to_vec();
        b_coords[rank - 2] = "t".into();

        w.print("var sum: f32 = 0.0;");
        w.block(&format!("for (var t: u32 = 0u; t < {k}u; t += 1u)"), |w| {
            w.print(&format!(
                "sum += {} * {};",
                read_at(arg_meta[0].0, arg_meta[0].1, &address(args[0], &a_coords)),
                read_at(arg_meta[1].0, arg_meta[1].1, &address(args[1], &b_coords)),
            ));
        });
        w.print(&write_at(
            out_etype,
            out_atomic,
            &address(out, coords),
            "sum",
        ));
    });
}

fn emit_reduction(
    w: &mut Writer,
    op: AssocOp,
    axes: &[usize],
    args: &[&Accessor],
    arg_meta: &[(ElementType, bool)],
    out: &Accessor,
    out_etype: ElementType,
    out_atomic: bool,
    config: &KernelConfig,
) {
    let input = args[0];
    let mut sorted_axes: Vec<usize> = axes.to_vec();
    sorted_axes.sort_unstable();
    let count: usize = sorted_axes.iter().map(|&axis| input.shape[axis]).product();
    let t = spell_etype(out_etype);

    per_output_element(w, out, config, |w, coords| {
        w.print(&format!(
            "var acc: {t} = {};",
            identity_literal(op, out_etype)
        ));
        w.block(
            &format!("for (var ri: u32 = 0u; ri < {count}u; ri += 1u)"),
            |w| {
                let mut in_coords = coords.to_vec();
                let mut stride = 1;
                for &axis in sorted_axes.iter().rev() {
                    let dim = input.shape[axis];
                    in_coords[axis] = format!("r{axis}");
                    w.print(&format!("let r{axis} = (ri / {stride}u) % {dim}u;"));
                    stride *= dim;
                }
                let value = read_at(arg_meta[0].0, arg_meta[0].1, &address(input, &in_coords));
                w.print(&match op {
                    AssocOp::Add => format!("acc += {value};"),
                    AssocOp::Mul => format!("acc *= {value};"),
                    AssocOp::Max => format!("acc = max(acc, {value});"),
                    AssocOp::Min => format!("acc = min(acc, {value});"),
                });
            },
        );
        w.print(&write_at(
            out_etype,
            out_atomic,
            &address(out, coords),
            "acc",
        ));
    });
}

fn identity_literal(op: AssocOp, etype: ElementType) -> String {
    match (op, etype) {
        (AssocOp::Add, ElementType::F32) => "0.0".into(),
        (AssocOp::Mul, ElementType::F32) => "1.0".into(),
        (AssocOp::Max, ElementType::F32) => "bitcast<f32>(0xff800000u)".into(),
        (AssocOp::Min, ElementType::F32) => "bitcast<f32>(0x7f800000u)".into(),
        (AssocOp::Add, ElementType::U32) => "0u".into(),
        (AssocOp::Mul, ElementType::U32) => "1u".into(),
        (AssocOp::Max, ElementType::U32) => "0u".into(),
        (AssocOp::Min, ElementType::U32) => "0xffffffffu".into(),
    }
}

fn emit_remap(
    w: &mut Writer,
    info: &RemapInfo,
    args: &[&Accessor],
    arg_meta: &[(ElementType, bool)],
    out: &Accessor,
    out_etype: ElementType,
    out_atomic: bool,
    config: &KernelConfig,
) {
    match info {
        RemapInfo::GatherRows => {
            let src_rows = args[0].shape[0];
            per_output_element(w, out, config, |w, coords| {
                let idx_coords = [coords[0].clone()];
                w.print(&format!(
                    "let row = min({}, {}u);",
                    read_at(arg_meta[1].0, arg_meta[1].1, &address(args[1], &idx_coords)),
                    src_rows.saturating_sub(1),
                ));
                let mut src_coords = coords.to_vec();
                src_coords[0] = "row".into();
                let val = read_at(arg_meta[0].0, arg_meta[0].1, &address(args[0], &src_coords));
                w.print(&write_at(
                    out_etype,
                    out_atomic,
                    &address(out, coords),
                    &val,
                ));
            });
        }
        RemapInfo::ScatterRows { operator } => {
            let src = args[0];
            let out_rows = out.shape[0];
            let out_heap = heap_name(out_etype, out_atomic);
            // Atomic scatter-add targets live on the atomic heap; plain scatter-write
            // uses the plain heap and normal stores.
            if operator.is_some() && out_etype == ElementType::F32 {
                debug_assert!(out_atomic, "f32 scatter-add output must be on atomic arena");
                w.block("fn atomic_add_f32(addr: u32, value: f32)", |w| {
                    w.print(&format!("var old = atomicLoad(&{out_heap}[addr]);"));
                    w.block("loop", |w| {
                        w.print(&format!(
                            "let new_bits = bitcast<u32>(bitcast<f32>(old) + value);\nlet result = atomicCompareExchangeWeak(&{out_heap}[addr], old, new_bits);"
                        ));
                        w.block("if (result.exchanged)", |w| {
                            w.print("break;");
                        });
                        w.print("old = result.old_value;");
                    });
                });
            }
            per_thread_element(w, &src.shape, config, |w, _lin, coords| {
                let idx_coords = [coords[0].clone()];
                w.print(&format!(
                    "let row = {};",
                    read_at(arg_meta[1].0, arg_meta[1].1, &address(args[1], &idx_coords))
                ));
                w.block(&format!("if (row < {out_rows}u)"), |w| {
                    let mut o_coords = coords.to_vec();
                    o_coords[0] = "row".into();
                    let out_addr = address(out, &o_coords);
                    let src_val = read_at(arg_meta[0].0, arg_meta[0].1, &address(src, coords));
                    match (operator, out_etype) {
                        (None, _) => {
                            w.print(&write_at(out_etype, out_atomic, &out_addr, &src_val));
                        }
                        (Some(_), ElementType::U32) => {
                            w.print(&format!("atomicAdd(&{out_heap}[{out_addr}], {src_val});"));
                        }
                        (Some(_), ElementType::F32) => {
                            w.print(&format!("atomic_add_f32({out_addr}, {src_val});"));
                        }
                    }
                });
            });
        }
        RemapInfo::ScatterView { map, operator } => {
            // Dense-logical linear index into the output; OOR drops.
            let src = args[0];
            let out_n = element_count(&out.shape);
            let out_heap = heap_name(out_etype, out_atomic);
            if operator.is_some() && out_etype == ElementType::F32 {
                debug_assert!(out_atomic, "f32 scatter_view-add output must be on atomic arena");
                w.block("fn atomic_add_f32(addr: u32, value: f32)", |w| {
                    w.print(&format!("var old = atomicLoad(&{out_heap}[addr]);"));
                    w.block("loop", |w| {
                        w.print(&format!(
                            "let new_bits = bitcast<u32>(bitcast<f32>(old) + value);\nlet result = atomicCompareExchangeWeak(&{out_heap}[addr], old, new_bits);"
                        ));
                        w.block("if (result.exchanged)", |w| {
                            w.print("break;");
                        });
                        w.print("old = result.old_value;");
                    });
                });
            }
            per_thread_element(w, &src.shape, config, |w, _lin, coords| {
                let logical = address(map, coords);
                w.print(&format!("let logical = {logical};"));
                w.block(&format!("if (logical < {out_n}u)"), |w| {
                    // Decode logical → dense out coords.
                    let mut stride = 1usize;
                    let mut decode_parts = Vec::new();
                    for axis in (0..out.rank()).rev() {
                        let dim = out.shape[axis];
                        decode_parts.push(format!(
                            "let o{axis} = (logical / {stride}u) % {dim}u;"
                        ));
                        stride *= dim;
                    }
                    for line in decode_parts.into_iter().rev() {
                        w.print(&line);
                    }
                    let out_coords: Vec<String> =
                        (0..out.rank()).map(|a| format!("o{a}")).collect();
                    let out_addr = address(out, &out_coords);
                    let src_val = read_at(arg_meta[0].0, arg_meta[0].1, &address(src, coords));
                    match (operator, out_etype) {
                        (None, _) => {
                            w.print(&write_at(out_etype, out_atomic, &out_addr, &src_val));
                        }
                        (Some(_), ElementType::U32) => {
                            w.print(&format!("atomicAdd(&{out_heap}[{out_addr}], {src_val});"));
                        }
                        (Some(_), ElementType::F32) => {
                            w.print(&format!("atomic_add_f32({out_addr}, {src_val});"));
                        }
                    }
                });
            });
        }
        RemapInfo::GatherView { map } => {
            // Dense-logical linear index into the source view shape, then
            // address through the source accessor (handles broadcast).
            let src = args[0];
            let n = element_count(&src.shape);
            let (src_et, src_atomic) = arg_meta[0];
            let zero = match out_etype {
                ElementType::F32 => "0.0",
                ElementType::U32 => "0u",
            };
            per_output_element(w, out, config, |w, coords| {
                let logical = address(map, coords);
                w.print(&format!("let logical = {logical};"));
                // Clamp before decode so the load is always in-bounds; select
                // restores OOR → zero (WGSL `select` evaluates both arms).
                if n == 0 {
                    w.print(&write_at(
                        out_etype,
                        out_atomic,
                        &address(out, coords),
                        zero,
                    ));
                    return;
                }
                w.print(&format!("let safe = min(logical, {}u);", n - 1));
                let mut stride = 1usize;
                let mut decode_parts = Vec::new();
                for axis in (0..src.rank()).rev() {
                    let dim = src.shape[axis];
                    decode_parts.push(format!("let s{axis} = (safe / {stride}u) % {dim}u;"));
                    stride *= dim;
                }
                for line in decode_parts.into_iter().rev() {
                    w.print(&line);
                }
                let src_coords: Vec<String> =
                    (0..src.rank()).map(|a| format!("s{a}")).collect();
                let src_addr = address(src, &src_coords);
                let loaded = read_at(src_et, src_atomic, &src_addr);
                w.print(&format!(
                    "let val = select({zero}, {loaded}, logical < {n}u);"
                ));
                w.print(&write_at(
                    out_etype,
                    out_atomic,
                    &address(out, coords),
                    "val",
                ));
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
            &[(ElementType::F32, false), (ElementType::F32, false)],
            &a,
            ElementType::F32,
            false,
            &[(ElementType::F32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("@compute"), "{wgsl}");
        assert!(wgsl.contains("heap_f32"), "{wgsl}");
        assert!(!wgsl.contains("heap_atomic"), "{wgsl}");
        assert!(wgsl.contains("fn elem(a0: f32, a1: f32) -> f32"), "{wgsl}");
        assert!(wgsl.contains("((a0) + (a1))"), "{wgsl}");
    }

    #[test]
    fn relu_emits_max() {
        let a = Accessor::dense([8], 0);
        let wgsl = emit(
            &load_op(Op::RELU, 1),
            &[&a],
            &[(ElementType::F32, false)],
            &a,
            ElementType::F32,
            false,
            &[(ElementType::F32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("max(a0, 0.0)"), "{wgsl}");
    }

    #[test]
    fn broadcast_arg_reads_single_address() {
        let out = Accessor::dense([2, 3], 0);
        let scalar = Accessor::dense([], 0).broadcast_to(&[2, 3]).unwrap();
        let wgsl = emit(
            &load_op(Op::MUL, 2),
            &[&out, &scalar],
            &[(ElementType::F32, false), (ElementType::F32, false)],
            &out,
            ElementType::F32,
            false,
            &[(ElementType::F32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("heap_f32[0u]"), "{wgsl}");
    }

    #[test]
    fn matmul_shader_loops_over_k() {
        let a = Accessor::dense([2, 4], 0);
        let b = Accessor::dense([4, 3], 0);
        let out = Accessor::dense([2, 3], 0);
        let wgsl = emit(
            &Kernel::Matmul,
            &[&a, &b],
            &[(ElementType::F32, false), (ElementType::F32, false)],
            &out,
            ElementType::F32,
            false,
            &[(ElementType::F32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("t < 4u"), "{wgsl}");
        assert!(wgsl.contains("i0 * 4u + t * 1u"), "{wgsl}");
    }

    #[test]
    fn reduction_starts_from_identity() {
        let input = Accessor::dense([2, 3], 0);
        let out = Accessor::dense([2, 1], 0);
        let wgsl = emit(
            &Kernel::Reduction {
                op: AssocOp::Max,
                axes: Box::from([1]),
            },
            &[&input],
            &[(ElementType::F32, false)],
            &out,
            ElementType::F32,
            false,
            &[(ElementType::F32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("bitcast<f32>(0xff800000u)"), "{wgsl}");
    }

    #[test]
    fn gather_rows_shader_emits() {
        let src = Accessor::dense([4, 2], 0);
        let idx = Accessor::dense([3], 0);
        let out = Accessor::dense([3, 2], 0);
        let wgsl = emit(
            &Kernel::Remap {
                info: RemapInfo::GatherRows,
            },
            &[&src, &idx],
            &[(ElementType::F32, false), (ElementType::U32, false)],
            &out,
            ElementType::F32,
            false,
            &[(ElementType::F32, false), (ElementType::U32, false)],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("heap_u32["), "{wgsl}");
        assert!(wgsl.contains("heap_f32["), "{wgsl}");
        assert!(wgsl.contains("min("), "{wgsl}");
    }

    #[test]
    fn scatter_add_binds_atomic_heap_not_plain() {
        let src = Accessor::dense([4], 0);
        let idx = Accessor::dense([4], 0);
        let out = Accessor::dense([3], 0);
        let wgsl = emit(
            &Kernel::Remap {
                info: RemapInfo::ScatterRows {
                    operator: Some(AssocOp::Add),
                },
            },
            &[&src, &idx],
            &[(ElementType::F32, false), (ElementType::U32, false)],
            &out,
            ElementType::F32,
            true, // output on atomic arena
            &[
                (ElementType::F32, false),
                (ElementType::U32, false),
                (ElementType::F32, true),
            ],
            &KernelConfig::default(),
        );
        assert!(wgsl.contains("heap_atomic_f32"), "{wgsl}");
        assert!(wgsl.contains("heap_f32"), "{wgsl}"); // plain source
        assert!(wgsl.contains("atomic_add_f32"), "{wgsl}");
        // Source reads must not go through atomics.
        assert!(wgsl.contains("heap_f32["), "{wgsl}");
    }
}
