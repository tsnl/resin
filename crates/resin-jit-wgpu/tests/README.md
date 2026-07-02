# GPU interpreter integration tests

Goal: **pinpoint bugs with systematic, reusable tests**, not one-off CPU scripts.
When a test exposes a real defect, **land it** in this tree (with a short comment
naming the failure mode). Throwaway probes are fine while hunting; they graduate
here only if they protect a regression.

Run:

```sh
cargo test -p resin-jit-wgpu --test interp_single_node --test interp_rerun
# later tiers as they exist:
# cargo test -p resin-jit-wgpu --test interp_chain
# cargo test -p resin-jit-wgpu --test interp_grad
```

Helpers: `interp_helpers.rs` (`run_graph`, `run_graph_twice`, `approx_eq`). Prefer
extending helpers over copying admit/write/run/read boilerplate.

---

## Tier map (bottom → top)

```text
T0  Host-only (resin-dsl / resin-ir / resin-grad unit tests)
      shapes, accessors, IR memo, grad graph structure (no GPU)

T1  Single-node GPU execute          ← interp_single_node.rs (landed)
T2  Idempotent re-run (stale state)  ← interp_rerun.rs (landed)
T3  Small multi-node chains          ← interp_chain.rs (landed)
T4  Single-op / single-edge grads    ← interp_grad_ops.rs (G1–G5 landed)
T5  Short train-step contracts       ← interp_train_step.rs (S1–S4 landed)
T6  Full MNIST / long runs           ← examples only; not the primary oracle
```

**Rule:** do not use T6 to discover T1–T4 bugs. MNIST is a smoke/integration
canary *after* lower tiers are green.

---

## T1 — Single-node (or one kernel) execute

**Question:** does one lowered op write the mathematically expected buffer?

| Area | Cases | Status |
| --- | --- | --- |
| Const | 1d (2d optional) | landed |
| Elementwise binary | add, mul, sub, div | landed |
| Elementwise unary | neg, exp (floor/ceil/bitcast later) | partial |
| Reduce | sum, max (min later) | landed |
| Matmul | small dense; `x @ W.T` | landed |
| Broadcast | bias add `[B,N]+[N]` | landed |
| Remap gather | densify / `copy` on non-identity view | landed |
| Remap scatter-add | transpose-shaped adjoint layout | landed |
| Bitwise / U4 | and/or/xor, shifts | planned |
| Floor / ceil / bitcast | parity with Python tests | planned |

**Design notes**

- One sink `"out"`, explicit param names, fixed literals (no RNG).
- Tolerance: `1e-5` for f32 algebraic; slightly looser for exp/matmul if needed.
- Prefer **dense identity** graphs first; add **view/accessor** cases as their own tests (transpose, broadcast pitch 0).

---

## T2 — Re-run / stale state

**Question:** second `Interp::run` with the **same** inputs equals the first?

This is where **remap scatter-add** bit us: output not cleared → atomic add into
leftovers → double count / training blow-up.

| Area | Cases | Status |
| --- | --- | --- |
| Elementwise | add twice | landed |
| Matmul | twice | landed |
| Scatter-add transpose | twice must **not** be 2× | landed (caught clear bug) |
| Scatter-add identity | twice | landed |
| Gather | twice | landed |
| Scatter-add **N=3** runs | optional stress | easy add |
| Full small chain (linear+relu) | twice | planned (T3∩T2) |

**Design notes**

- Always **re-write params** between runs (params are inputs; outputs must not
  depend on previous output buffers unless intentional).
- For accumulate ops, assert equality to **single-run expectation**, not merely
  `r1 == r2` with both wrong.
- Comment the failure mode in the test when landing (`// without clear, second run is 2×`).

---

## T3 — Small multi-node chains (planned: `interp_chain.rs`)

**Question:** does composition of already-green T1 ops still match a host reference?

Keep chains **tiny** and **fully specified**:

1. **Linear no bias:** `y = x @ W.T` — values vs hand product.
2. **Linear + bias:** `y = x @ W.T + b`.
3. **Linear + ReLU.**
4. **Linear + softmax (axis 1)** — probs sum to 1; optional stable-max variant as its own case.
5. **Softmax + CE + mean** on fixed `x,y,W,b` vs host formula (same as old `debug_mnist_loss`, but as a test with fixed tensors—no dataset download).

**Host reference:** pure Rust in the test module (or `interp_helpers::cpu_*`), not a
throwaway binary. Download/MNIST belongs only in examples.

---

## T4 — Gradients, one edge at a time (planned: `interp_grad_ops.rs`)

**Question:** does `grad_wrt` + GPU execute match **finite difference** or a **closed form**?

Order matters (only add the next when the previous is green):

| # | Graph | Check |
| --- | --- | --- |
| G1 | `L = mean(y)`, `y = x @ W.T` | `∂L/∂W`, `∂L/∂x` vs analytic | **landed** |
| G2 | `y = x @ W.T + b`, `L = mean(y)` | `∂L/∂b`, `∂L/∂W` analytic | **landed** |
| G3 | `y = relu(x @ W.T + b)`, `L = mean(y)` | FD on one weight / bias | **landed** |
| G4 | `L = mean(CE(softmax(logits), y))`, logits linear | FD on `b` and `W[0]` (stable softmax) | **landed** |
| G4b | Same without max-sub softmax | FD on `W[0]` | **landed** |
| G5 | Two-layer `relu` then linear + G4 | FD on first/last `W` and last `b` | **landed** |
| G6 | Accessor adjoint alone | scatter-add transpose of a **known** `g` into `W` | T1 covers forward scatter |

**FD protocol (when used)**

- Central difference, `eps` in `{1e-3, 1e-4}`; report `L+`, `L-` if mismatch.
- Same admitted program; only rewrite the perturbed param buffer.
- Prefer **closed form** over FD when easy (G1–G2).

**Land** any G* test that fails on mainline before a fix; keep it after the fix.

---

## T5 — Train-step contracts (planned: `interp_train_step.rs`)

Not full MNIST. Fixed synthetic batch, **no download**.

| # | Contract |
| --- | --- |
| S1 | Graph `new_model = w - lr*g` bit-close to host `w - lr*g` (MLP + linear) | **landed** |
| S2 | **Same batch**, 20 host-SGD steps: loss finite, `<` CE floor, final `<` init − ε | **landed** |
| S3 | **Two different batches**, one step each: loss finite both times | **landed** |
| S4 | After multi-step, restore step-0 weights: re-run loss matches step-0 | **landed** |

MNIST example stays an **manual/slow canary**, not CI-default if download/GPU-heavy.

---

## T6 — Full MNIST / examples

- `examples/train_mnist.rs`, `debug_*` binaries: **investigation tools**.
- Promote a probe into T1–T5 when it isolates a mechanism (as with scatter-add re-run).
- Do not add new `debug_*.rs` without either deleting them later or turning the
  assertion into a `tests/` case.

---

## Suspected remaining issues → which tier

| Observation | Tier that should catch it |
| --- | --- |
| Scatter-add doubles on 2nd `run` | **T2** (landed) |
| Dead ReLU / zero weight grads (bad init) | T5/S2 with tiny init policy test; or document init in example only |
| Weight FD ≫ bias FD on full MLP | **T4/G4** (isolate last layer + softmax+CE, no MLP stack) |
| Same-batch SGD improves but multi-batch explodes | **T5/S2 vs S3** |
| Forward MLP CE ≠ CPU | **T3/chain softmax+CE** (landed spirit in debug_mnist_loss → graduate fixed tensors) |

---

## Implementation conventions

1. **`#[test]` + GPU:** skip or `return` only if adapter missing—prefer fail in CI with Vulkan soft GPU (workflow already installs mesa-vulkan).
2. **Names:** `op_scenario_expected` / `op_rerun_idempotent`.
3. **One behavior per test;** share setup via helpers.
4. **No network** in `tests/` (no MNIST download).
5. When fixing a bug: commit **fix + the failing test** together (or test first in TDD).

---

## Suggested implementation order (next work)

1. Graduate **G1** (matmul mean grad) from `debug_matmul_grad` into `interp_grad_ops.rs`.
2. Add **G4** last-layer softmax+CE with FD on `W[0,0]` and `b` — pins weight-scale bug if still present.
3. Graduate **S1** (graph vs host SGD equality) into `interp_train_step.rs`.
4. Add **S2** same-batch 20 steps with He init + small `lr` — must not hit `~16.12` floor.
5. Only then re-tune MNIST example / LR.

This file is the checklist; bars move from “planned” → “landed” as tests appear.
