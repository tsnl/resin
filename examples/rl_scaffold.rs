//! Whole-body RL controller scaffold over the batched physics engine.
//!
//! Loads an MJCF model with motors (default: `examples/mjcf/walker.xml`, a
//! planar biped), wraps it as a batched environment — every world in the
//! batch evaluates a different policy — and trains a linear policy with a
//! tiny evolution-strategies loop as a stand-in for a real algorithm.
//!
//!   cargo run --release --example rl_scaffold
//!   cargo run --release --example rl_scaffold -- path/to/model.xml
//!
//! # Handoff notes (for the GPU iteration)
//!
//! The pieces below are deliberately minimal; this is the intended path to
//! scale them:
//!
//! - **Backend.** Swap `CpuJit` for `WgpuJit` — `Env` is generic over the
//!   backend already. The compiled step is identical; only dispatch moves
//!   to the GPU.
//! - **Worlds.** `Env::new(…, worlds)` is the batch size. The tensor
//!   program's cost per step is one graph regardless of the count; worlds
//!   ride the trailing axis of every buffer (MuJoCo-Warp / Genesis style).
//!   ES below uses one world per perturbation; PPO would use worlds as
//!   vectorized rollout envs.
//! - **Policy on device.** The linear policy here runs on the host between
//!   steps, which forces a readback per step. The engine's step is an
//!   ordinary tensor graph, so the policy (matmuls + nonlinearity) belongs
//!   *inside* the traced step: trace `state -> obs -> action -> physics ->
//!   state`, jit once, and the whole rollout never leaves the device.
//! - **Gradients.** `Scene::trace_step_driven` is differentiable end to end
//!   (see `physics::tests::rollout_gradients_match_finite_differences`), so
//!   short-horizon analytic policy gradients (BPTT through the simulator)
//!   are available as an alternative or complement to ES/PPO.
//! - **Observations/rewards on device.** `observe`/`reward` below read
//!   state arrays on the host; both are expressible as tensor ops and can
//!   join the traced step.
//! - **Contact realism.** The engine supports spheres, cuboids, and the
//!   ground plane; capsule geoms (the usual whole-body collision proxy)
//!   are the first thing to add for humanoid models — a capsule-vs-plane
//!   contact is two sphere candidates at the segment ends.

use resin::jit::{Array, CpuJit, Jit};
use resin::dsl::Tensor;
use resin::jit::JittedFn;
use resin::physics::{Scene, State, mjcf};

/// The traced controlled step, and the same step bound to a backend.
type StepFn = Box<dyn Fn(&StepIn<Tensor>) -> State<Tensor> + Send + Sync>;
type CompiledStep = JittedFn<CpuJit, StepIn<Array>, State<Tensor>, StepFn>;

/// One batch of environments around a compiled, motor-driven step.
struct Env {
    scene: Scene,
    step: CompiledStep,
    state: State<Array>,
    worlds: usize,
    ctrl_lo: Vec<f32>,
    ctrl_hi: Vec<f32>,
}

#[derive(resin::Tree, Clone)]
struct StepIn<T> {
    state: State<T>,
    ctrl: T,
}

impl Env {
    fn new(loaded: &mjcf::Loaded, worlds: usize) -> Env {
        let scene = loaded.scene.clone();
        let traced = scene.clone();
        let step_fn: StepFn = Box::new(move |input: &StepIn<Tensor>| {
            traced.trace_step_driven(&input.state, Some(&input.ctrl))
        });
        let step = CpuJit.jit(step_fn);
        Env {
            state: scene.initial_state(worlds),
            worlds,
            ctrl_lo: loaded.actuators.iter().map(|a| a.ctrl_range[0]).collect(),
            ctrl_hi: loaded.actuators.iter().map(|a| a.ctrl_range[1]).collect(),
            scene,
            step,
        }
    }

    fn motors(&self) -> usize {
        self.scene.motors.len()
    }

    /// Bodies × (position + velocity) per world — a deliberately plain
    /// observation vector.
    fn observe(&self, world: usize) -> Vec<f32> {
        let n = self.scene.bodies.len();
        let mut obs = Vec::with_capacity(n * 6);
        obs.extend(&self.state.pos.data()[world * n * 3..(world + 1) * n * 3]);
        obs.extend(&self.state.vel.data()[world * n * 3..(world + 1) * n * 3]);
        obs
    }

    /// Walk forward (+x), stay tall: torso x-velocity plus a height bonus.
    fn reward(&self, world: usize, torso: usize) -> f32 {
        let n = self.scene.bodies.len();
        let vx = self.state.vel.data()[(world * n + torso) * 3];
        let height = self.state.pos.data()[(world * n + torso) * 3 + 1];
        vx + 0.2 * height
    }

    fn reset(&mut self) {
        self.state = self.scene.initial_state(self.worlds);
    }

    /// Apply per-world controls (`[worlds × motors]`, clamped to the
    /// actuator ranges) and advance one frame everywhere.
    fn step(&mut self, ctrl: &[f32]) {
        let m = self.motors();
        let clamped: Vec<f32> = ctrl
            .iter()
            .enumerate()
            .map(|(i, &c)| c.clamp(self.ctrl_lo[i % m], self.ctrl_hi[i % m]))
            .collect();
        self.state = self
            .step
            .call(&StepIn {
                state: self.state.clone(),
                ctrl: Array::from_f32(&[self.worlds, m], &clamped),
            })
            .expect("env step");
    }
}

/// Linear policy: `action = P · obs`, one matrix per candidate.
fn act(policy: &[f32], obs: &[f32], motors: usize) -> Vec<f32> {
    (0..motors)
        .map(|m| {
            obs.iter()
                .enumerate()
                .map(|(i, &o)| policy[m * obs.len() + i] * o)
                .sum()
        })
        .collect()
}

fn main() {
    let model_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "examples/mjcf/walker.xml".into());
    let loaded = mjcf::load_file(&model_path).expect("load MJCF");
    assert!(
        !loaded.scene.motors.is_empty(),
        "model has no motors to control"
    );
    let torso = loaded.body_ids.values().copied().min().unwrap(); // first body

    // One world per ES perturbation (antithetic pairs).
    let pairs = 4;
    let worlds = pairs * 2;
    let mut env = Env::new(&loaded, worlds);
    let (motors, obs_dim) = (env.motors(), env.observe(0).len());
    println!(
        "{} bodies, {motors} motors, obs {obs_dim}, {worlds} worlds",
        env.scene.bodies.len()
    );

    let mut policy = vec![0.0f32; motors * obs_dim];
    let mut rng = 0x9e3779b97f4a7c15u64;
    let mut random = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };

    let (sigma, alpha, episode) = (0.1f32, 0.05f32, 40usize);
    for iteration in 0..5 {
        // Perturb: worlds 2k and 2k+1 run θ ± σ·noise_k.
        let noise: Vec<Vec<f32>> = (0..pairs)
            .map(|_| (0..policy.len()).map(|_| random()).collect())
            .collect();
        env.reset();
        let mut returns = vec![0.0f32; worlds];
        for _ in 0..episode {
            let mut ctrl = Vec::with_capacity(worlds * motors);
            for world in 0..worlds {
                let (pair, sign) = (world / 2, if world % 2 == 0 { 1.0 } else { -1.0 });
                let candidate: Vec<f32> = policy
                    .iter()
                    .zip(&noise[pair])
                    .map(|(p, n)| p + sign * sigma * n)
                    .collect();
                ctrl.extend(act(&candidate, &env.observe(world), motors));
            }
            env.step(&ctrl);
            for (world, total) in returns.iter_mut().enumerate() {
                *total += env.reward(world, torso);
            }
        }

        // ES update: θ += α · Σ (R₊ − R₋)/2 · noise.
        for (pair, n) in noise.iter().enumerate() {
            let advantage = (returns[pair * 2] - returns[pair * 2 + 1]) / 2.0;
            for (p, v) in policy.iter_mut().zip(n) {
                *p += alpha * advantage * v / pairs as f32;
            }
        }
        let mean: f32 = returns.iter().sum::<f32>() / worlds as f32;
        println!("iteration {iteration}: mean return {mean:.3}");
    }
    println!("scaffold ran; see the handoff notes at the top of this file");
}
