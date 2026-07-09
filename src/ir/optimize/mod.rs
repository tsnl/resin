//! IR→IR optimization passes.
//!
//! Implemented:
//!
//! - **Tiled matmul** ([`tile_matmuls`]) — matmul as an *elementwise* multiply
//!   of 16×16 tiles plus a reduction. The tile product is an ordinary
//!   elementwise dispatch over [`Element::F32Tile16`].
//! - **Elementwise fusion** ([`fuse_elementwise`]) — producer→consumer chains
//!   of same-[`Element`] elementwise dispatches collapse into single kernels,
//!   to a fixed point. Fusion is tree substitution ([`Expr::substitute`]).
//!
//! Planned:
//!
//! - **Matmul epilogues** — fuse a trailing expression into a matmul
//!   (subsumed by the tiled representation once that lands).
//! - **Constant folding and dead-dispatch elimination.**

mod kernel_fusion;

pub use kernel_fusion::fuse_elementwise;

use super::{
    Accessor, Buffer, BufferRef, BufferView, BufferViewRef, Dispatch, Element, Expr, Kernel,
    Program, TILE,
};
use crate::ops::{AssocOp, Op};

/// Which IR optimization passes to run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OptPasses {
    /// No IR passes — identity (for A/B benchmarks).
    None,
    /// Elementwise fusion only.
    Fuse,
    /// Tiled matmul rewrite only.
    Tile,
    /// Full pipeline: tile, then fuse.
    #[default]
    All,
}

/// Run the default middle-end pipeline ([`OptPasses::All`]).
pub fn optimize(program: Program) -> Program {
    optimize_with(program, OptPasses::All)
}

/// Run a selected set of middle-end passes.
pub fn optimize_with(program: Program, passes: OptPasses) -> Program {
    match passes {
        OptPasses::None => program,
        OptPasses::Fuse => fuse_elementwise(program),
        OptPasses::Tile => tile_matmuls(program),
        OptPasses::All => {
            let program = tile_matmuls(program);
            fuse_elementwise(program)
        }
    }
}

/// Rewrite every eligible matmul as `pack A → pack B → tile-multiply →
/// reduce`, leaving ineligible matmuls untouched.
///
/// Eligible: rank-2, all dims divisible by 16, args and output read/written
/// through dense identity views (no transposed operands yet).
///
/// For `[M,K] @ [K,N]` with `Mt = M/16` etc., the replacement is:
///
/// 1. **pack A** — plain f32 copy into a tile-blocked buffer `[Mt,Kt,16,16]`
///    (source view walks the row-major matrix in tile order).
/// 2. **pack B** — likewise into `[Kt,Nt,16,16]`.
/// 3. **tile multiply** — elementwise `Mul` over [`Element::F32Tile16`] with
///    iteration space `[Mt,Nt,Kt]`.
/// 4. **reduce** — f32 `Add` reduction over the `Kt` axis into the original
///    row-major `[M,N]` output.
pub fn tile_matmuls(mut program: Program) -> Program {
    let mut index = 0;
    while index < program.queue.len() {
        match tile_one(&mut program, index) {
            Some(replacement) => {
                let len = replacement.len();
                program.queue.splice(index..=index, replacement);
                index += len;
            }
            None => index += 1,
        }
    }
    program
}

fn tile_one(program: &mut Program, index: usize) -> Option<Vec<Dispatch>> {
    let dispatch = &program.queue[index];
    if !matches!(dispatch.kernel, Kernel::Matmul) {
        return None;
    }
    let is_identity = |r: BufferViewRef| {
        let view = program.view(r);
        view.accessor == Accessor::dense(program.buffer(view.buffer).shape.clone(), 0)
    };
    if !dispatch.args.iter().chain([&dispatch.output]).all(|&r| is_identity(r)) {
        return None;
    }
    let a = program.view(dispatch.args[0]);
    let b = program.view(dispatch.args[1]);
    let out = program.view(dispatch.output);
    if a.accessor.rank() != 2 {
        return None; // batched matmul: keep the naive kernel
    }
    let a_shape = a.accessor.shape();
    let b_shape = b.accessor.shape();
    let (m, k) = (a_shape[0], a_shape[1]);
    let n = b_shape[1];
    if !(m.is_multiple_of(TILE) && k.is_multiple_of(TILE) && n.is_multiple_of(TILE)) {
        return None;
    }
    let (a_buffer, b_buffer, out_buffer) = (a.buffer, b.buffer, out.buffer);
    let (mt, kt, nt) = (m / TILE, k / TILE, n / TILE);

    let mut buffer = |shape: &[usize]| {
        program.buffers.push(Buffer { shape: shape.into(), init: None });
        BufferRef(program.buffers.len() - 1)
    };
    let a_tiles = buffer(&[mt, kt, TILE, TILE]);
    let b_tiles = buffer(&[kt, nt, TILE, TILE]);
    let product = buffer(&[mt, nt, kt, TILE, TILE]);

    let mut view = |buffer: BufferRef, accessor: Accessor| {
        program.views.push(BufferView { buffer, accessor });
        BufferViewRef(program.views.len() - 1)
    };
    let affine = |shape: &[usize], pitch: &[usize]| Accessor::Affine {
        offset: 0,
        shape: shape.into(),
        pitch: pitch.into(),
    };
    // Pack sources: walk row-major matrices tile-by-tile (custom pitches).
    let a_src = view(a_buffer, affine(&[mt, kt, TILE, TILE], &[TILE * k, TILE, k, 1]));
    let b_src = view(b_buffer, affine(&[kt, nt, TILE, TILE], &[TILE * n, TILE, n, 1]));
    let a_packed = view(a_tiles, Accessor::dense([mt, kt, TILE, TILE], 0));
    let b_packed = view(b_tiles, Accessor::dense([kt, nt, TILE, TILE], 0));
    // Tile grids over packed buffers (tile units), broadcast to [Mt,Nt,Kt].
    let a_grid = view(a_tiles, affine(&[mt, nt, kt], &[kt, 0, 1]));
    let b_grid = view(b_tiles, affine(&[mt, nt, kt], &[0, 1, nt]));
    let product_grid = view(product, affine(&[mt, nt, kt], &[nt * kt, kt, 1]));
    let product_lanes = view(product, Accessor::dense([mt, nt, kt, TILE, TILE], 0));
    let out_lanes = view(out_buffer, affine(&[mt, nt, 1, TILE, TILE], &[TILE * n, TILE, 0, n, 1]));

    Some(vec![
        Dispatch {
            kernel: Kernel::elementwise(Expr::Load(a_src)),
            args: vec![a_src],
            output: a_packed,
        },
        Dispatch {
            kernel: Kernel::elementwise(Expr::Load(b_src)),
            args: vec![b_src],
            output: b_packed,
        },
        Dispatch {
            kernel: Kernel::Elementwise {
                expr: Expr::new_op(Op::MUL, [Expr::Load(a_grid), Expr::Load(b_grid)]),
                element: Element::F32Tile16,
            },
            args: vec![a_grid, b_grid],
            output: product_grid,
        },
        Dispatch {
            kernel: Kernel::Reduction { op: AssocOp::Add, axes: Box::from([2]) },
            args: vec![product_lanes],
            output: out_lanes,
        },
    ])
}
