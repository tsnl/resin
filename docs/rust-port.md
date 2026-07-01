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

Python also threads **`PyTree[T]`** through compile, grad, opt, and `nn` modules. The Rust port needs an analogue from the start — not as a follow-on epic.

## Non-goals (this port)

- Tensor shape typing (rung 4): named dims / const-generic contracts — later epic.
- Etype-as-type-param (rung 2), rank types (rung 3), layout types (rung 5).
- Grad effect typing (rung 7).
- **`#[derive(ParamTree)]` proc macro** — hand-impl `Linear` / `Mlp` (and `Vec` / `Option`) for the port; derive is polish.
- A live **`enum ParamTree<T>`** AST (Object/Array/Leaf) — see [ParamTree as a trait](#paramtree-as-a-trait-python-pytree-analogue). Erased/serialized trees only if needed later for checkpoints or debug.
- True HKT — use GATs (`type Map<U>`) on the trait instead.
- Type-system work beyond graph identity, `ParamTree` module mapping, and a typed IR→JIT handoff.
- 3DGS / multi-output / custom-kernel roadmap features (implement once, on Rust, after the port).

## Goals

| Goal | Approach |
| --- | --- |
| No ser/de **execution** boundary | `admit(WgpuProgram)` (or IR→backend artifact) by value |
| DSL graph composition in Rust | `resin-dsl` builds an identity-keyed DAG |
| Identity like Python `eq=False` nodes | Encapsulated `Arc` / `Rc` inside `View`; pointer equality |
| Low verbosity / `std::ops` ergonomics | No arena lifetimes on the public API |
| **PyTree-shaped host APIs** | **`ParamTree` trait**: `map` / `zip_with` / `flatten` — no enum |
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

## ParamTree as a trait (Python PyTree analogue)

Python `PyTree[T]` is used **everywhere** on the host side: `compile_program` params/sinks, `grad(..., wrt=)`, `sgd`, `nn.Object` / `Linear` / `Mlp`. It is not optional for behavioral parity.

Python implements this as a **runtime union** (`dict` / `list` / `Object` / leaf) plus walkers. Rust does not need a matching sum type. **`ParamTree` is a trait**; structs and collections implement it. Containers are **impl details**, not a public AST.

### Trait sketch

```rust
// resin-core (sketch)

/// Host-side module / param tree — analogue of Python PyTree[T].
///
/// No live enum: shape lives in the concrete type (Linear, Mlp, Vec<...>).
pub trait ParamTree: Sized {
    type Leaf;
    /// Same module shape, new leaf type (GAT encoding of HKT — Rust has no HKT).
    type Map<U>: ParamTree<Leaf = U>;

    fn map<U>(self, f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U>;

    /// Dotted paths — Python `flatten_pytree_items` (e.g. `layers.0.weight`).
    fn flatten(&self) -> Vec<(String, &Self::Leaf)>;

    /// Pair with another tree of the same shape (`Self::Map<U>`) — sgd / zip.
    fn zip_with<U, O>(
        self,
        other: Self::Map<U>,
        f: impl FnMut(Self::Leaf, U) -> O,
    ) -> Self::Map<O>;
}
```

Optional later: callback-style `for_each_leaf` to avoid allocating path `String`s; `into_flatten` for owning moves.

### Blanket / std impls (collections)

```rust
impl<T: ParamTree> ParamTree for Vec<T> {
    type Leaf = T::Leaf;
    type Map<U> = Vec<T::Map<U>>;
    // map / flatten (index segments) / zip_with pairwise
}

impl<T: ParamTree> ParamTree for Option<T> {
    // None = empty subtree (Python bias=None)
}
```

### Module impls (hand-written in the port)

```rust
pub struct Linear<T> {
    pub weight: T,
    pub bias: Option<T>,
}

pub struct Mlp<T> {
    pub layers: Vec<Linear<T>>,
}

impl<T> ParamTree for Linear<T> {
    type Leaf = T;
    type Map<U> = Linear<U>;
    // field names → path segments; map/zip on weight and bias
}

impl<T> ParamTree for Mlp<T> {
    type Leaf = T;
    type Map<U> = Mlp<U>;
    // recurse into layers (Vec<Linear<T>> impl)
}
```

**Leaf vs nested module rule (for hand impls and a future derive):**

- Field of type `T` where `T` is this struct’s leaf type parameter → **leaf**
- Field whose type implements `ParamTree<Leaf = T>` → **recurse**

**`View` as a tree:** prefer Python-style overloads rather than forcing `View: ParamTree` with a tricky `Map<U> = U`:

- `grad(f, wrt: &View) -> View`
- `grad(f, wrt: M) -> M::Map<View>` where `M: ParamTree<Leaf = View>`

### What each operation serves

| Operation | Python today | Rust use |
| --- | --- | --- |
| `flatten` | `flatten_pytree_items` | `register_named_params`, `sink_tree`, `commit` paths |
| `map` | `map_pytree` | `grad(..., wrt=module)` → same-shaped grads |
| `zip_with` | `tree_map` / zip in `sgd` | `sgd(params, grads, lr)` |

### APIs that depend on `ParamTree` (port parity)

```rust
// compile / binding
for (path, view) in params.flatten() {
    builder.register_param(&path, view);
}
// sinks: flatten under a prefix, or ParamTree at each sink key

// grad
fn grad_wrt<M: ParamTree<Leaf = View>>(f: &View, wrt: M) -> M::Map<View> {
    let by_node = grad_by_node(f);
    wrt.map(|v| /* lookup by v.node_key() */)
}

// opt
fn sgd<M: ParamTree<Leaf = View>>(params: M, grads: M::Map<View>, lr: f32) -> M {
    params.zip_with(grads, |p, g| /* leaf update */)
}
```

### Explicitly not the centerpiece

- **`enum ParamTree<T> { Object, Array, Leaf, ... }`** — unused for live host APIs; optional later for wire/debug erasure.
- **`IsoParamTree: Into<Enum> + From<Enum>`** — unnecessary if the trait *is* the protocol.
- **`fn map(a: impl ParamTree<T>) -> impl ParamTree<U>`** — return type constructor is unknown without HKT; use **`M::Map<U>`** on a known `M` instead.
- **`dyn ParamTree`** — GAT trait is not object-safe; not needed for `Linear` / `Mlp` / generics.

### Future derive (out of port scope, in design scope)

`#[derive(ParamTree)]` (or `#[derive(Module)]`) generates `Leaf` / `Map` / `map` / `flatten` / `zip_with` for user structs. Port ships **hand impls** only.

## Crate layout

```text
resin (workspace)
├── crates/resin-core      # etype, accessor, ParamTree trait, Vec/Option impls
├── crates/resin-dsl       # View, NodeInner, ops, prelude (frontend)
├── crates/resin-nn        # Linear, Mlp, relu/softmax/...; ParamTree impls (or under dsl)
├── crates/resin-grad      # df_do, grad, accessor_adjoint (uses ParamTree)
├── crates/resin-ir        # IrProgram, kernels, builder (Arc DAG → IR)
├── crates/resin-jit-wgpu  # WGSL codegen + WgpuProgram + WgpuInterp (today's resin-rt)
├── crates/resin-jit-vk    # stub / later (consumes resin-ir)
├── crates/resin-dataset   # MNIST etc. (host I/O only)
└── examples/              # interp smoke, mnist, …
```

`resin-nn` may start as a module of `resin-dsl` if it stays tiny; separate crate keeps `ParamTree` impls for modules obvious.

**Dependency direction**

```text
resin-core          # ParamTree, etype, accessor
resin-dsl ──► resin-core
resin-nn  ──► resin-dsl (+ ParamTree impls on Linear/Mlp)
resin-grad ──► resin-dsl, resin-core
resin-ir ──► resin-dsl (flatten params/sinks via ParamTree at the compile facade)
resin-jit-wgpu ──► resin-ir (+ resin-core)
resin-jit-vk   ──► resin-ir
resin-dataset ──► (minimal; not on the hot path)
examples ──► dsl + nn + grad + ir + jit-wgpu + dataset
```

JIT backends consume **`resin-ir` values**, not msgpack. Optional: `resin-jit-wgpu` also exposes admitting a prebuilt `WgpuProgram` for tests.

**Delete after port**

- `src/resin/` (Python package)
- `crates/resin-rt-pybind/`
- Python half of `Makefile` / `pyproject.toml` as the primary workflow (or shrink to docs-only if notebooks remain)
- Dual Python `WgpuProgram` TypedDict mirror

**Rename:** today’s `crates/resin-rt` becomes (or merges into) `resin-jit-wgpu`. Keep history via `git mv` where practical.

## Layer responsibilities

### 1. Core — `resin-core`

- `ElementType`, operators, `Accessor` (runtime checks).
- **`ParamTree` trait** + `Vec` / `Option` impls + path conventions (`layers.0.weight`).

### 2. Frontend — `resin-dsl`

Port of `dsl/view.py`, `dsl/node.py` (and accessor/etype if not only in core).

- Public: `View`, `param`, `const`, `zeros` / `ones`, slicing / broadcast / permute, elementwise, reduce, matmul, remap.
- Private: `NodeInner`, `NodeKind`.
- Prelude: `F4`, `F2`, `U4` as values/constants (not type params in this port).

### 3. Modules — `resin-nn` (or `resin-dsl::nn`)

Port of `nn/__init__.py`: `Linear<T>`, `Mlp<T>`, relu, softmax, cross_entropy, mean — with **`ParamTree` impls**.

### 4. Transforms — `resin-grad`

Port of `grad/__init__.py`: `accessor_adjoint`, `df_do`, `grad` / `grad_by_node`.

- Accumulate grads **by node identity** (`ptr` keys), same as Python `grad_by_node`.
- **`grad_wrt<M: ParamTree<Leaf = View>>`** via `M::map` (parity with `grad(f, wrt=mlp)`).

### 5. IR — `resin-ir`

Port of `ir/ir.py` (+ RPN as needed).

- `IrProgramBuilder`: memo `*const NodeInner → IrBuffer`, `View` identity → `IrBufferView`.
- Kernels: elementwise RPN, matmul, reduction, remap (current Python surface).
- Output: owned `IrProgram` (index-based buffers/views/queue) — **serde-friendly**.
- Compile facade uses **`ParamTree::flatten`** for named params and sinks (port of `runtime/trees.py` / `compile.py`).

### 6. Backend JIT — `resin-jit-wgpu`

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

### 7. Host — `resin-dataset` + examples

MNIST loader and training loop as a Rust example (`Mlp<View>`, `grad_wrt`, `sgd`, `flatten` for buffer I/O). Datasets are **not** part of the DSL type story.

## Typing ladder (for orientation)

| Rung | This port |
| --- | --- |
| 0 Runtime shape/etype | **yes** |
| 1 Identity (Arc `ptr_eq`) | **yes** (no arena lifetime) |
| 2 Etype type params | no |
| 3 Rank types | no |
| 4 Shape / dim variables | no (later epic) |
| 5 Layout types | no |
| **6 ParamTree protocol** | **yes — as a trait** (`map` / `flatten` / `zip_with` + GAT `Map<U>`); **no** live enum; **no** derive required |
| 7 Grad effects | no |
| 8 IR/JIT as Rust values, no msgpack handoff | **yes** |

## Migration plan (big-bang)

Project is early; **no long dual-stack**. One branch replaces Python as the product surface.

| Phase | Work | Exit criteria |
| --- | --- | --- |
| **P0** | Crate skeleton; move/rename `resin-rt` → `resin-jit-wgpu`; `admit_program(WgpuProgram)`; msgpack optional | Rust tests admit a hand-built `WgpuProgram` without Python |
| **P1** | `resin-core`: accessor, etype, **`ParamTree` trait** + `Vec` / `Option`; `resin-dsl`: `View` + Arc, ops | Unit tests for accessor, small graphs, **flatten/map/zip on Linear/Vec** |
| **P2** | `resin-ir` builder + kernel mapping; register/sink via **`flatten`** | IR structural / snapshot tests; param path tests |
| **P3** | WGSL codegen + lowering in `resin-jit-wgpu` | Snapshot WGSL + end-to-end GPU smoke |
| **P4** | `resin-grad` + **`grad_wrt<M: ParamTree<…>>`** | Grad tests including tree-shaped `wrt` |
| **P5** | Compile/bind helper; **`sgd`** via `zip_with`; `resin-nn` (`Linear`, `Mlp`) | Shared test helpers ported |
| **P6** | Examples (`interp`, `mnist`); `resin-dataset` | `cargo run --example mnist` trains with `Mlp<View>` |
| **P7** | Delete Python package, pybind, Python CI primary path; README / AGENTS.md → Rust | `make check` / `cargo test` is the gate |

Commits should stay reviewable (one coherent story each), not a single dump.

## Test strategy

- Port tests **by behavior**, not by Python file layout necessarily:
  - accessor / view math
  - compose / prelude
  - **ParamTree** flatten paths / map / zip shape mismatch
  - IR param names
  - GPU interp trivial / compose / shared buffers / config
  - grad (scalar `wrt` and module `wrt`)
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
| `ParamTree` impl boilerplate | Hand-impl `Linear` / `Mlp` + blanket `Vec` / `Option`; derive later |
| `zip_with` shape errors | Runtime checks in blanket impls; same `M` / `M::Map<U>` at type level |
| `Add` consuming vs borrowing | Impl for `View` and `&View`; document ownership |
| GAT / compiler diagnostics | Keep trait methods few; prefer clear errors on `flatten` paths in tests |

## Open choices (resolve during P0–P1)

1. **`Arc` vs `Rc`** — default `Arc`.
2. **Workspace example layout** — root `examples/` vs per-crate examples; prefer workspace-level binaries that depend on published-in-workspace crates.
3. **`resin-grad` / `resin-nn` as crates vs modules** — separate crates keep edges clear; merge if tiny.
4. **Single `resin` facade crate** re-exporting dsl+nn+grad+ir for examples — optional sugar.
5. **Path segment rules** for tuples vs structs — match Python dotted paths where tests care.

## Success criteria

- [ ] No PyO3 / maturin required to develop or run examples
- [ ] No msgpack on the admit hot path
- [ ] DSL composition and grad in Rust with Arc identity
- [ ] **`ParamTree` trait** supports compile naming, `grad_wrt`, and `sgd` without a live tree enum
- [ ] `resin-jit-wgpu` runs programs lowered from `resin-ir`
- [ ] MNIST-class example works with `Mlp<View>` (or equivalent `ParamTree`)
- [ ] Python product surface removed (docs notebooks optional)

## References (current tree)

- Python DSL: `src/resin/dsl/`
- PyTree: `src/resin/core/pytree.py`
- NN modules: `src/resin/nn/__init__.py`
- Grad `wrt` trees: `src/resin/grad/__init__.py`
- Opt: `src/resin/opt/__init__.py`
- Compile / trees: `src/resin/runtime/compile.py`, `src/resin/runtime/trees.py`
- IR: `src/resin/ir/`
- WGSL: `src/resin/wgpu/`
- Runtime: `crates/resin-rt/`
- Seam: `crates/resin-rt-pybind/`, `WgpuProgram::to_msgpack` in Python `wgpu/spec.py`
