# Port resin to Rust

Big-bang rewrite of resin as a Rust-native stack. One language from DSL composition through IR and GPU execution. No Python product surface, no PyO3/msgpack handoff on the admit path.

## Architecture

```text
resin-dsl / resin-nn / resin-grad
            │
            ▼
        resin-ir          # owned IrProgram (index-based)
            │
            ├──────────────► resin-jit-wgpu   # WGSL + WgpuProgram + WgpuInterp
            └──────────────► resin-jit-vk     # later
```

Host I/O (datasets) and examples sit beside this stack; they are not part of the graph IR.

### Crates

| Crate | Responsibility |
| --- | --- |
| `resin-core` | `ElementType`, `Accessor`, **`ParamTree` trait**, `Vec` / `Option` impls |
| `resin-dsl` | `View`, `Node` / `NodeKind`, ops, prelude |
| `resin-nn` | `Linear`, `Mlp`, common nn ops; `ParamTree` impls (may live under `resin-dsl` if small) |
| `resin-grad` | `df_do`, `grad_by_node`, `grad_wrt` |
| `resin-ir` | `IrProgram`, kernels, builder (DAG → IR) |
| `resin-jit-wgpu` | WGSL codegen, `WgpuProgram`, interpreter (today’s `resin-rt`) |
| `resin-jit-vk` | Future Vulkan backend; consumes `resin-ir` |
| `resin-dataset` | MNIST and other host loaders |

**Dependencies:** `resin-dsl` → `resin-core`; `resin-nn` / `resin-grad` → `resin-dsl`; `resin-ir` → `resin-dsl`; `resin-jit-*` → `resin-ir`. Examples depend on dsl, nn, grad, ir, jit-wgpu, dataset as needed.

**Remove:** `src/resin/`, `crates/resin-rt-pybind/`, Python as the primary package/CI surface. Rename `crates/resin-rt` → `resin-jit-wgpu` (preserve history with `git mv` where practical).

Serde/`msgpack` on `IrProgram` / `WgpuProgram` remains for **debug dumps and expect tests only**, not for execution handoff.

## Graph identity: encapsulated `Arc`

Public `View` holds a private `Arc<Node>` plus an `Accessor`. Callers never see `Arc` or `Node`.

```rust
#[derive(Clone)]
pub struct View {
    node: Arc<Node>,
    accessor: Accessor,
}

struct Node {
    shape: Box<[u32]>,
    etype: ElementType,
    /// Operand views (empty for `Const` / `Param`).
    args: Vec<View>,
    kind: NodeKind,
}

/// Kind-specific payload only; operands live on `Node::args`.
enum NodeKind {
    Const { init: Box<[u8]> },
    Param { name: Arc<str> },
    Elementwise { op: ElementOperator },
    Matmul,
    Reduction {
        op: BinaryAssocElementOperator,
        axes: Box<[u32]>,
    },
    Remap { info: RemapInfo },
    // …
}
```

- Nodes are immutable after creation; the graph is a **DAG** (`args` point at existing nodes only).
- Identity for IR memo and grad is **pointer equality** (`Arc::ptr_eq` / `Arc::as_ptr`), not structural equality.
- `Clone` on `View` is cheap (refcount + copy accessor).
- Ops allocate new nodes internally (e.g. `std::ops::Add` on `View` / `&View`). Matmul is a method or dedicated trait — not `Mul` (`*` stays elementwise).
- Default to `Arc` (`Send`); switch to `Rc` only if needed.

Live graph serialization is not required. Prefer serializing lowered `IrProgram` / `WgpuProgram`. Debug dumps of a graph, if any, assign dense IDs by walking from sinks.

## ParamTree (host modules / params)

Host module and param bundles use a **`ParamTree` trait**. Module shape is the concrete type (`Linear`, `Mlp`, `Vec<…>`); collections are impls. No live Object/Array/Leaf enum.

```rust
pub trait ParamTree: Sized {
    type Leaf;
    type Map<U>: ParamTree<Leaf = U>;

    fn map<U>(self, f: impl FnMut(Self::Leaf) -> U) -> Self::Map<U>;

    /// Dotted paths (e.g. `layers.0.weight`).
    fn flatten(&self) -> impl Iterator<Item = (String, &Self::Leaf)> + '_;

    /// Same shape as `self`, other leaf type — needed for `sgd` (not derivable from `map` alone).
    fn zip_with<U, O>(
        self,
        other: Self::Map<U>,
        f: impl FnMut(Self::Leaf, U) -> O,
    ) -> Self::Map<O>;
}
```

| Op | Used for |
| --- | --- |
| `flatten` | Named params, sinks, commit buffer paths |
| `map` | `grad_wrt(module)` → same-shaped grads |
| `zip_with` | `sgd(params, grads, lr)` |

`map` only transforms one tree’s leaves. Pairing two trees (params + grads) needs a second structure-aligned walk — `zip_with` on the trait (or an equivalent free function with the same contract). Reconstructing a module from two independent `flatten` streams would require a separate unflatten/shape witness; prefer structural `zip_with`.

Blanket impls for `Vec<T: ParamTree>` and `Option<T: ParamTree>`. Hand-written impls for `Linear<T>` and `Mlp<T>` in the port; a derive macro is optional later.

**Leaf vs nested field:** type-parameter fields `T` are leaves; fields that implement `ParamTree<Leaf = T>` recurse.

**Grad entry points:** `grad(f, wrt: &View) -> View` and `grad_wrt<M: ParamTree<Leaf = View>>(f, wrt: M) -> M::Map<View>`.

Path spelling matches current Python dotted paths where tests care (`layers.0.weight`).

## Layers

### DSL (`resin-dsl`)

`View`, `param` / `const` / `zeros` / `ones`, slicing, broadcast, permute, elementwise, reduce, matmul, remap. Element types and shapes checked at **runtime**. Prelude exposes `F4` / `F2` / `U4` as values.

### NN (`resin-nn`)

`Linear<T>`, `Mlp<T>`, relu, softmax, cross_entropy, mean — with `ParamTree` impls.

### Grad (`resin-grad`)

`accessor_adjoint`, `df_do`, `grad_by_node` (map keyed by node identity), `grad_wrt` via `ParamTree::map`.

### IR (`resin-ir`)

`IrProgramBuilder` memos `*const Node → IrBuffer` and view identity → `IrBufferView`. Kernels: elementwise RPN, matmul, reduction, remap. Output: owned, index-based, serde-friendly `IrProgram`. Compile facade registers params/sinks through `ParamTree::flatten`.

### JIT (`resin-jit-wgpu`)

Lower `IrProgram` → `WgpuProgram` (WGSL + dispatch metadata). Interpreter admits a **`WgpuProgram` by value**:

```rust
fn admit_program(&mut self, program: WgpuProgram) -> Result<ProgramId, InterpError>;
```

`to_msgpack` / `from_msgpack` optional for tests and debugging.

### Dataset + examples

`resin-dataset` for loaders. Workspace examples (e.g. interp smoke, MNIST training with `Mlp<View>`, `grad_wrt`, `sgd`, buffer I/O via `flatten`).

## Out of scope

- Tensor shape / rank / etype-as-type-parameter systems
- Layout (pitch/offset) in the type system
- Grad effect typing
- `#[derive(ParamTree)]` (hand impls suffice)
- Live erased tree enum or `dyn ParamTree`
- 3DGS, multi-output nodes, custom-kernel protocol (build on Rust after the port)
- Long dual Python+Rust maintenance

## Delivery

Single branch, big-bang cutover. Reviewable commits; one concern per commit where practical.

| Phase | Deliverable |
| --- | --- |
| **P0** | Workspace layout; `resin-rt` → `resin-jit-wgpu`; `admit_program(WgpuProgram)` |
| **P1** | `resin-core` (`ParamTree`, accessor, etype) + `resin-dsl` (`View` / `Node` + ops) |
| **P2** | `resin-ir` + flatten-based param/sink registration |
| **P3** | WGSL codegen + lowering + GPU smoke tests |
| **P4** | `resin-grad` including tree-shaped `wrt` |
| **P5** | Compile/bind helpers, `sgd` (`zip_with`), `resin-nn` |
| **P6** | Examples + `resin-dataset` (MNIST runs) |
| **P7** | Remove Python package/pybind; Rust is the only product surface |

## Success criteria

- Develop and run examples without PyO3/maturin
- Admit path does not require msgpack
- Arc identity for graphs; `ParamTree` for compile naming, `grad_wrt`, and `sgd`
- End-to-end path: DSL → IR → `resin-jit-wgpu`
- MNIST-class example with `Mlp<View>` (or equivalent `ParamTree`)
- Python product surface gone (notebooks may remain as docs only)
