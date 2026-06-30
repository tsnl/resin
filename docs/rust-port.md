# Port resin to Rust (Arc graph, no Python seam)

**Status:** draft sketch (this PR)  
**Goal:** move the entire stack into Rust, delete the Python↔Rust msgpack/PyO3 boundary, keep serde only for debugging and expect tests.

## Motivation

Today the product is three layers with a hard seam:

```text
Python DSL / grad / IR / WGSL codegen
        │  msgpack(WgpuProgram)
        ▼
resin-rt-pybind (PyO3)
        ▼
resin-rt (WgpuInterp)
```

That forces a dual `WgpuProgram` schema (Python TypedDict + Rust serde), maturin rebuilds, and buffer traffic through `bytes`. The runtime already lives in Rust; the compiler front-end should too.

## Non-goals (this port)

- Tensor shape typing (rung 4): named dims / const-generic contracts — later epic.
- Etype-as-type-param (rung 2), rank types (rung 3), layout types (rung 5).
- Typed param trees / PyTree derive (rung 6) — runtime paths or ad hoc structs in examples for now.
- Grad effect typing (rung 7).
- Type-system “improvements” beyond graph identity and a typed IR→JIT handoff.
- 3DGS / multi-output / custom-kernel roadmap features (implement once, on Rust, after the port).

## Goals

| Goal | Approach |
| --- | --- |
| No ser/de **execution** boundary | `admit(WgpuProgram)` (or IR→backend artifact) by value |
| DSL graph composition in Rust | `resin-dsl` builds an identity-keyed DAG |
| Identity like Python `eq=False` nodes | Encapsulated `Arc` / `Rc` inside `View`; pointer equality |
| Low verbosity / `std::ops` ergonomics | No arena lifetimes on the public API |
| Examples as normal Rust examples | `examples/` under the workspace |
| Debug / tests | `Serialize` on IR and backend programs; optional graph dump via ID assignment pass |

## Identity model: encapsulated Arc (not arena)

Arenas + `'g` give co-graph proofs and dense IDs but force `View<'g>`, graph-threaded allocation, and noisier `Add` impls.

**Choice:** private `Arc<NodeInner>` (or `Rc` if we never need `Send`) inside public `View`. End users never see `Arc`.

```rust
// resin-dsl (sketch)

#[derive(Clone)]
pub struct View {
    node: Arc<NodeInner>,
    accessor: Accessor,
}

struct NodeInner {
    shape: Box<[u32]>,
    etype: ElementType,
    kind: NodeKind,
}

enum NodeKind {
    Const { init: Box<[u8]> },
    Param { name: Arc<str> },
    Elementwise {
        op: ElementOperator,
        args: Box<[View]>,
    },
    Matmul { args: [View; 2] },
    Reduction {
        op: BinaryAssocElementOperator,
        axes: Box<[u32]>,
        arg: View,
    },
    Remap {
        info: RemapInfo,
        args: Box<[View]>,
    },
}

impl View {
    /// Identity for IR memo / grad-by-node (Python object identity).
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.node, &b.node)
    }

    fn node_key(&self) -> *const NodeInner {
        Arc::as_ptr(&self.node)
    }
}
```

**Invariants**

- Graph is a **DAG** (args only reference existing nodes). No cycles → no `Weak` required yet.
- Nodes are **immutable** after creation (matches frozen Python dataclasses).
- Structural `PartialEq` on `View` is **not** used for memoization; use `ptr_eq` / pointer-keyed maps.
- `Clone` on `View` is cheap (clone the `Arc` + copy `Accessor`).

**Operators** (no `&mut Graph`):

```rust
impl Add for View {
    type Output = View;
    fn add(self, rhs: Self) -> Self {
        View::elementwise(ElementOperator::Add, [self, rhs])
    }
}
// Prefer also impl for &View so `a + b` need not consume.
```

Matmul: method or dedicated trait — **not** `Mul` (reserve `*` for elementwise, matching Python `*` vs `@`).

**`Rc` vs `Arc`:** start with `Arc` so `View: Send` stays easy; switch to `Rc` only if profiling demands it.

**Serialization of live graphs:** not required for execution. For debug dumps, walk reachable nodes from sinks, assign dense IDs, emit an adjacency list. Prefer serializing **`IrProgram` / backend programs** for expect tests.

## Crate layout

```text
resin (workspace)
├── crates/resin-core      # etype, accessor, shared small types
├── crates/resin-dsl       # View, NodeInner, ops, prelude (frontend)
├── crates/resin-grad      # df_do, grad, accessor_adjoint
├── crates/resin-ir        # IrProgram, kernels, builder (Arc DAG → IR)
├── crates/resin-jit-wgpu  # WGSL codegen + WgpuProgram + WgpuInterp (today's resin-rt)
├── crates/resin-jit-vk    # stub / later (consumes resin-ir)
├── crates/resin-dataset   # MNIST etc. (host I/O only)
└── examples/              # interp smoke, mnist, …
```

**Dependency direction**

```text
resin-dsl ──► resin-core
resin-grad ──► resin-dsl
resin-ir ──► resin-dsl (or resin-core + view handles)
resin-jit-wgpu ──► resin-ir (+ resin-core)
resin-jit-vk   ──► resin-ir
resin-dataset ──► (minimal; not on the hot path)
examples ──► dsl + grad + ir + jit-wgpu + dataset
```

JIT backends consume **`resin-ir` values**, not msgpack. Optional: `resin-jit-wgpu` also exposes admitting a prebuilt `WgpuProgram` for tests.

**Delete after port**

- `src/resin/` (Python package)
- `crates/resin-rt-pybind/`
- Python half of `Makefile` / `pyproject.toml` as the primary workflow (or shrink to docs-only if notebooks remain)
- Dual Python `WgpuProgram` TypedDict mirror

**Rename:** today’s `crates/resin-rt` becomes (or merges into) `resin-jit-wgpu`. Keep history via `git mv` where practical.

## Layer responsibilities

### 1. Frontend — `resin-dsl`

Port of `dsl/view.py`, `dsl/node.py`, `core/accessor.py`, `core/etype.py` (runtime checks).

- Public: `View`, `param`, `const`, `zeros` / `ones`, slicing / broadcast / permute, elementwise, reduce, matmul, remap.
- Private: `NodeInner`, `NodeKind`.
- Prelude: `F4`, `F2`, `U4` as values/constants (not type params in this port).

### 2. Transforms — `resin-grad`

Port of `grad/__init__.py`: `accessor_adjoint`, `df_do`, `grad` / `grad_by_node`.

- Accumulate grads **by node identity** (`ptr` keys), same as Python `grad_by_node`.
- Param trees for `wrt=`: **deferred** (rung 6). v1: pass explicit `&[View]` or a simple struct in examples.

### 3. IR — `resin-ir`

Port of `ir/ir.py` (+ RPN as needed).

- `IrProgramBuilder`: memo `*const NodeInner → IrBuffer`, `View` identity → `IrBufferView`.
- Kernels: elementwise RPN, matmul, reduction, remap (current Python surface).
- Output: owned `IrProgram` (index-based buffers/views/queue) — **serde-friendly**.

### 4. Backend JIT — `resin-jit-wgpu`

Port of `wgpu/codegen.py`, `wgpu/lowering.py`, plus existing `resin-rt` interp.

```rust
// sketch — execution API (no msgpack required)
pub struct WgpuProgram { /* existing fields */ }

pub trait Interp: Send + Sync {
    fn admit_program(&mut self, program: WgpuProgram) -> Result<ProgramId, InterpError>;
    // write_buffer / read_buffer / run / copy_buffer_to_buffer — unchanged spirit
}

impl WgpuProgram {
    pub fn to_msgpack(&self) -> Result<Vec<u8>, …> { … }      // debug / tests
    pub fn from_msgpack(bytes: &[u8]) -> Result<Self, …> { … }
}
```

Lowering: `IrProgram` → `WgpuProgram` (WGSL strings + dispatch metadata).

### 5. Host — `resin-dataset` + examples

MNIST loader and training loop as a Rust example. Datasets are **not** part of the DSL type story.

## Typing ladder (for orientation)

| Rung | This port |
| --- | --- |
| 0 Runtime shape/etype | **yes** |
| 1 Identity (Arc `ptr_eq`) | **yes** (no arena lifetime) |
| 2 Etype type params | no |
| 3 Rank types | no |
| 4 Shape / dim variables | no (later epic) |
| 5 Layout types | no |
| 6 ParamTree derive | no |
| 7 Grad effects | no |
| 8 IR/JIT as Rust values, no msgpack handoff | **yes** |

## Migration plan (big-bang)

Project is early; **no long dual-stack**. One branch replaces Python as the product surface.

| Phase | Work | Exit criteria |
| --- | --- | --- |
| **P0** | Crate skeleton; move/rename `resin-rt` → `resin-jit-wgpu`; `admit_program(WgpuProgram)`; msgpack optional | Rust tests admit a hand-built `WgpuProgram` without Python |
| **P1** | `resin-core` + `resin-dsl` (`View` + Arc, accessor, ops) | Unit tests for accessor + small graph construction |
| **P2** | `resin-ir` builder + kernel mapping | IR structural / snapshot tests |
| **P3** | WGSL codegen + lowering in `resin-jit-wgpu` | Snapshot WGSL + end-to-end GPU smoke (existing interp tests as oracle) |
| **P4** | `resin-grad` | Grad unit tests (CPU-visible buffers via GPU or future CPU ref — match current GPU tests) |
| **P5** | Compile/bind helper (params, sinks, write/read) | Shared test helpers ported |
| **P6** | Examples (`interp`, `mnist`); `resin-dataset` | `cargo run --example mnist` trains |
| **P7** | Delete Python package, pybind, Python CI primary path; README / AGENTS.md → Rust | `make check` / `cargo test` is the gate |

Commits should stay reviewable (one coherent story each), not a single dump.

## Test strategy

- Port tests **by behavior**, not by Python file layout necessarily:
  - accessor / view math
  - compose / prelude
  - IR param names
  - GPU interp trivial / compose / shared buffers / config
  - grad
  - compile binding
- Prefer **Rust snapshot tests** for WGSL and `IrProgram` / `WgpuProgram` where stable.
- GPU tests remain feature- or `#[ignore]`-gated the same way CI requires today.
- Optional short parity window: golden outputs from Python vs Rust **only if** a specific bug needs an oracle; not a standing dual compiler.

## Risks (accepted)

| Risk | Mitigation |
| --- | --- |
| Accidental structural eq / CSE | No `Hash`/`Eq` on `View` by value for memo; document `ptr_eq` |
| Refcount leaks from held `View`s | Normal; graphs are small; drop roots when done |
| Cycles later (custom nodes) | Introduce `Weak` only if needed |
| WGSL subtle bugs | Snapshots + GPU tests; mechanical port of codegen |
| Param tree ergonomics | Examples use explicit params until rung 6 |
| `Add` consuming vs borrowing | Impl for `View` and `&View`; document ownership |

## Open choices (resolve during P0–P1)

1. **`Arc` vs `Rc`** — default `Arc`.
2. **Workspace example layout** — root `examples/` vs per-crate examples; prefer workspace-level binaries that depend on published-in-workspace crates.
3. **`resin-grad` as crate vs module under `resin-dsl`** — separate crate keeps dependency edges clear; merge if it stays tiny.
4. **Single `resin` facade crate** re-exporting dsl+grad+ir for examples — optional sugar.

## Success criteria

- [ ] No PyO3 / maturin required to develop or run examples
- [ ] No msgpack on the admit hot path
- [ ] DSL composition and grad in Rust with Arc identity
- [ ] `resin-jit-wgpu` runs programs lowered from `resin-ir`
- [ ] MNIST-class example works
- [ ] Python product surface removed (docs notebooks optional)

## References (current tree)

- Python DSL: `src/resin/dsl/`
- IR: `src/resin/ir/`
- WGSL: `src/resin/wgpu/`
- Compile binding: `src/resin/runtime/compile.py`
- Runtime: `crates/resin-rt/`
- Seam: `crates/resin-rt-pybind/`, `WgpuProgram::to_msgpack` in Python `wgpu/spec.py`
